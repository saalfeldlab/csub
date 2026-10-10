"""Render the spooled wrapper script that LSF runs on the compute node.

The wrapper runs *outside* the sandbox as the user. It must therefore never touch an
agent-chosen path: every path it writes to is derived from the broker-owned state dir and
``$LSB_JOBID``. All agent-controlled strings are ``shlex``-quoted and runtime values enter
the inner (in-sandbox) script only as positional parameters.

Wrapper-stage exit codes (so ``exit_code`` tells them apart from the command's):

  93  the per-job broker did not create its socket
  94  could not write the script body (shell mode)
  95  could not cd into the job's cwd
  96  could not create the in-sandbox job dir / HOME
  97  could not create the broker-owned job dir
  98  could not create the scratch dir
"""

from __future__ import annotations

import os
import shlex

from csub import __version__
from csub.broker.policy import SANDBOX_SCRIPTS
from csub.broker.resolve import ResolvedJob

SOCKET_NAME = "csub.sock"
EXIT_NO_BROKER = 93
EXIT_SCRIPT_WRITE = 94
EXIT_CD = 95
EXIT_INNER_MKDIR = 96
EXIT_JOB_DIR = 97
EXIT_SCRATCH = 98

q = shlex.quote


def mounts_env_value(job: ResolvedJob) -> str:
    return ",".join(f"{m.path}:{m.mode}" for m in job.mounts)


def render_inner(job: ResolvedJob, *, client_src: str | None = None) -> str:
    """The /bin/sh script executed inside the sandbox.

    Positional parameters: $1 LSB_JOBID, $2 CUDA_VISIBLE_DEVICES (maybe empty), $3 scratch
    dir (maybe empty), $4 LSB_DJOB_NUMPROC (maybe empty), $5 wrapper start epoch, $6 socket.

    ``client_src`` is the directory of the installed ``csub`` package, bind-mounted read-only
    by the wrapper so that any image gets `csub` / `import csub` (needs a python3 inside).
    """
    cwd = q(job.cwd)
    lines = [
        "set -u",
        f"export HOME={cwd}/.csub/home",
        f'J={cwd}/.csub/jobs/"$1"',
        f'export USER={q(job.user)} LOGNAME={q(job.user)} LSB_JOBID="$1"',
        'export CSUB_TRANSPORT=unix CSUB_SOCKET="$6" CSUB_JOB_ID="$1"',
        f"export CSUB_SESSION={q(job.session)} CSUB_IMAGE={q(job.image)}",
        f"export CSUB_MOUNTS={q(mounts_env_value(job))}",
        f"export CSUB_WALLTIME_MIN={job.walltime_min} "
        f"CSUB_DEADLINE_EPOCH=$(($5 + {job.walltime_min * 60}))",
        '[ -n "$2" ] && export CUDA_VISIBLE_DEVICES="$2"',
        '[ -n "$3" ] && export TMPDIR="$3"',
        '[ -n "$4" ] && export LSB_DJOB_NUMPROC="$4"',
    ]
    for key in sorted(job.env):
        lines.append(f"export {key}={q(job.env[key])}")
    lines += [
        f'mkdir -p "$J" "$HOME" || exit {EXIT_INNER_MKDIR}',
    ]
    if client_src:
        lines += [
            f"export PYTHONPATH={q(os.path.dirname(client_src))}${{PYTHONPATH:+:$PYTHONPATH}}",
            "mkdir -p \"$J/bin\" && printf '%s\\n' '#!/bin/sh' 'exec python3 -m csub.cli \"$@\"' "
            '>"$J/bin/csub" && chmod u+x "$J/bin/csub" || exit ' + str(EXIT_INNER_MKDIR),
            'export PATH="$J/bin:$PATH"',
        ]
    lines.append(f"cd {cwd} || exit {EXIT_CD}")
    if job.shell:
        assert isinstance(job.command, str)
        lines += [
            f'printf \'%s\' {q(job.command)} >"$J/script" && chmod u+x "$J/script" '
            f"|| exit {EXIT_SCRIPT_WRITE}",
            'exec "$J/script" >"$J/stdout" 2>"$J/stderr"',
        ]
    else:
        assert isinstance(job.command, tuple)
        lines.append(f'exec {shlex.join(job.command)} >"$J/stdout" 2>"$J/stderr"')
    return "\n".join(lines) + "\n"


def _allow_flags(hosts: tuple[str, ...]) -> list[str]:
    """podman-run.sh takes one --allow per host. Isolated so a flag-shape change is one line."""
    return [f"--allow {q(h)}" for h in hosts]


def sandbox_run_line(
    job: ResolvedJob, *, scripts_dir: str, inner: str, client_src: str | None = None
) -> str:
    """The podman-run.sh / sandbox-run.sh invocation as shell text, one flag per line.

    Some arguments are deliberately shell expressions ("$SCRATCH", "$JOB_DIR/...") that the
    wrapper expands at run time; everything agent-controlled is quoted. bwrap (sandbox-run.sh)
    takes the same flags minus image, identity and GPU, which only mean something to podman.
    """
    podman = job.sandbox == "podman"
    segments = [q(os.path.join(scripts_dir, SANDBOX_SCRIPTS[job.sandbox]))]
    if podman and job.keep_id:
        segments.append("--keep-id")
    if podman:
        segments.append(f"--image {q(job.image)}")
    for m in job.mounts:
        segments.append(f"--{m.mode} {q(m.path)}")
    if job.scratch:
        segments.append('--rw "$SCRATCH"')
    segments.append(f'--ro "$JOB_DIR/{SOCKET_NAME}"')
    if client_src:
        segments.append(f"--ro {q(client_src)}")
    if podman and job.gpus:
        segments.append("--gpu")
    if job.claude:
        segments.append("--claude")  # the launcher binds the CLI, config copy and credentials
    segments += _allow_flags(job.allow_hosts)
    segments.append(
        f"-- /bin/sh -c {q(inner)} csub-job "
        '"$LSB_JOBID" "${CUDA_VISIBLE_DEVICES:-}" "$SCRATCH" "${LSB_DJOB_NUMPROC:-}" '
        f'"$(date +%s)" "$JOB_DIR/{SOCKET_NAME}"'
    )
    return " \\\n    ".join(segments)


def render_wrapper(
    job: ResolvedJob,
    *,
    state_dir: str,
    scripts_dir: str,
    broker_cmd: str,
    home: str,
    client_src: str | None = None,
) -> str:
    """The script handed to ``bsub`` on stdin. ``broker_cmd`` is already shell-quoted.

    ``home`` is the user's real home: with ``bsub -env none`` nothing sets it, and
    podman-run.sh (``set -u``) reads ``$HOME`` and ``$USER``. The inner script overrides
    HOME for the job itself.
    """
    inner = render_inner(job, client_src=client_src)
    path = "/usr/local/bin:/usr/bin:/bin"
    if job.claude:
        # The launcher binds ~/.local/bin/claude; under bwrap PATH passes through, so name it.
        path = f"{home}/.local/bin:{path}"
    lines = [
        "#!/bin/sh",
        f"# generated by csub-broker {__version__} for job {job.job_name}; do not edit",
        f"export PATH={q(path)} USER={q(job.user)} HOME={q(home)}",
        f'JOB_DIR={q(state_dir)}/jobs/"$LSB_JOBID"',
        f'mkdir -p "$JOB_DIR" && cd "$JOB_DIR" || exit {EXIT_JOB_DIR}',
        "SCRATCH=''",
    ]
    if job.scratch:
        scratch_base = os.path.join(job.scratch_root, job.user, "csub")
        lines.append(
            f'SCRATCH={q(scratch_base)}/"$LSB_JOBID"; mkdir -p "$SCRATCH" || exit {EXIT_SCRATCH}'
        )
    lines += [
        f'{broker_cmd} --serve "$JOB_DIR/{SOCKET_NAME}" &',
        "BROKER_PID=$!",
        "trap 'kill \"$BROKER_PID\" 2>/dev/null' EXIT",
        f'i=0; while [ ! -S "$JOB_DIR/{SOCKET_NAME}" ]; do '
        f'i=$((i + 1)); [ "$i" -gt 100 ] && exit {EXIT_NO_BROKER}; sleep 0.1; done',
    ]
    if job.sandbox == "bwrap":
        # sandbox-run.sh binds $PWD read-write; that must be the job's cwd, never $JOB_DIR.
        lines.append(f"cd {q(job.cwd)} || exit {EXIT_CD}")
    lines += [
        sandbox_run_line(job, scripts_dir=scripts_dir, inner=inner, client_src=client_src),
        "rc=$?",
        '[ -n "$SCRATCH" ] && rm -rf "$SCRATCH"',
        'echo "$rc" > "$JOB_DIR/exit_code"',
        'exit "$rc"',
    ]
    return "\n".join(lines) + "\n"
