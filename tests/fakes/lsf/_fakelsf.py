#!/usr/bin/env python3
"""A small LSF impostor: bsub / bjobs / bkill backed by a JSON state file.

Stdlib only. Configured through environment variables:

  CSUB_FAKE_LSF_STATE           path of the JSON state file (required)
  CSUB_FAKE_LSF_BILLING_GROUP   printed as "This job will be billed to <group>" (testlab)
  CSUB_FAKE_LSF_QUEUES          comma list; -q outside it fails like LSF does
  CSUB_FAKE_LSF_BSUB_FAIL       if set, bsub prints it to stderr and exits 255
  CSUB_FAKE_LSF_PEND_S          minimum pending time before dispatch (0)
  CSUB_FAKE_LSF_MINUTE_S        seconds per -W minute (60) - small values test run-limit kills
  CSUB_FAKE_LSF_CLEAN_PERIOD_S  bjobs forgets finished jobs older than this (never)
  CSUB_FAKE_LSF_KILL_GRACE_S    seconds between kill signals (1)
  CSUB_FAKE_LSF_HOST            reported exec host (fakehost01)

Every job script is spooled to a file (like LSF) and run detached by a runner process
(this module with --run). Dependencies, -env none semantics, run limits and bkill are
modelled closely enough for csub's broker and integration tests.
"""

from __future__ import annotations

import fcntl
import json
import os
import pwd
import re
import shlex
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

VALUE_FLAGS = {
    "-q", "-n", "-W", "-We", "-gpu", "-J", "-g", "-o", "-oo", "-e", "-eo", "-w", "-env",
    "-cwd", "-R", "-P", "-M", "-app", "-m", "-u", "-sp", "-Jd", "-G", "-L",
}  # fmt: skip
BOOL_FLAGS = {"-Is", "-I", "-Ip", "-XF", "-K", "-H", "-N", "-B", "-r", "-x", "-h", "-V", "-a"}
DEP_RE = re.compile(r"^\s*(done|ended|exit)\((\d+)\)\s*$")
FIELD_WIDTHS = {
    "jobid": 7,
    "stat": 5,
    "exit_code": 10,
    "queue": 10,
    "exec_host": 11,
    "job_name": 10,
    "name": 10,
    "user": 7,
    "submit_time": 12,
}


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


def state_path() -> Path:
    p = os.environ.get("CSUB_FAKE_LSF_STATE")
    if not p:
        sys.stderr.write("fake lsf: CSUB_FAKE_LSF_STATE is not set\n")
        sys.exit(2)
    return Path(p)


def work_dir() -> Path:
    d = Path(str(state_path()) + ".d")
    (d / "spool").mkdir(parents=True, exist_ok=True)
    (d / "runner").mkdir(parents=True, exist_ok=True)
    return d


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {"next_id": 1001, "jobs": {}}


def _dump(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
    os.replace(tmp, path)


@contextmanager
def locked_state():
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = _load(path)
        try:
            yield state
            _dump(path, state)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def read_state() -> dict:
    with locked_state() as state:
        return json.loads(json.dumps(state))


def _user() -> str:
    return pwd.getpwuid(os.getuid()).pw_name


# --- bsub -------------------------------------------------------------------------------


def _die(msg: str, code: int = 255) -> int:
    sys.stderr.write(msg.rstrip("\n") + "\n")
    return code


def parse_bsub_argv(argv: list[str]) -> tuple[dict, list[str]]:
    opts: dict = {}
    positionals: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if positionals or not a.startswith("-") or a == "-":
            positionals.append(a)
        elif a in VALUE_FLAGS:
            if i + 1 >= len(argv):
                raise ValueError(f"bsub: option {a} requires an argument")
            opts.setdefault(a, []).append(argv[i + 1])
            i += 1
        elif a in BOOL_FLAGS:
            opts[a] = [True]
        else:
            raise ValueError(f"bsub: Unknown option {a}. Job not submitted.")
        i += 1
    return opts, positionals


def extract_directives(script: str) -> list[str]:
    out: list[str] = []
    # Like bsub: embedded options are honoured anywhere in the script, not only in a header
    # block before the first command (unlike Slurm's #SBATCH). Verified against LSF 10.1 on
    # the Janelia cluster, 2026-10-16: a held job whose only `#BSUB -J` came after an `echo`
    # was named by it. Command-line options still override embedded ones.
    for line in script.splitlines():
        s = line.strip()
        m = re.match(r"^#\s*BSUB\s+(.*)$", s)
        if m:
            out += shlex.split(m.group(1))
    return out


def parse_walltime(s: str) -> int:
    if ":" in s:
        h, m = s.split(":", 1)
        return int(h) * 60 + int(m)
    return int(s)


def parse_gpu(s: str) -> tuple[int, int | None]:
    num, gmem = None, None
    for part in s.split(":"):
        k, _, v = part.partition("=")
        if k == "num":
            num = int(v)
        elif k == "gmem":
            gmem = int(re.sub(r"[GgBb]+$", "", v))
    if num is None:
        raise ValueError("bsub: -gpu requires num=")
    return num, gmem


def parse_depends(expr: str) -> list[tuple[str, str]]:
    deps = []
    for term in expr.split("&&"):
        m = DEP_RE.match(term)
        if not m:
            raise ValueError("Bad dependency condition. Job not submitted.")
        deps.append((m.group(1), m.group(2)))
    return deps


def cmd_bsub(argv: list[str], stdin) -> int:
    fail = os.environ.get("CSUB_FAKE_LSF_BSUB_FAIL")
    if fail:
        return _die(fail)
    try:
        opts, positionals = parse_bsub_argv(argv)
    except ValueError as e:
        return _die(str(e))
    if positionals:
        script = "#!/bin/sh\n" + " ".join(positionals) + "\n"
        directives: list[str] = []
    else:
        if stdin.isatty():
            return _die("bsub: no command given. Job not submitted.")
        script = stdin.read()
        directives = extract_directives(script)
    if directives:
        try:
            dopts, _ = parse_bsub_argv(directives)
        except ValueError as e:
            return _die(str(e))
        merged = dict(dopts)
        for k, v in opts.items():
            merged[k] = v  # command line wins
        opts = merged

    def last(flag: str, default=None):
        return opts[flag][-1] if flag in opts else default

    queue = last("-q")
    default_queue = queue is None
    if default_queue:
        queue = "local"
    allowed = os.environ.get("CSUB_FAKE_LSF_QUEUES")
    if allowed and queue not in allowed.split(","):
        return _die(f"{queue}: Bad queue name. Job not submitted.")
    try:
        slots = max(int(x) for x in str(last("-n", "1")).split(","))
        walltime = parse_walltime(last("-W")) if last("-W") else None
        gpus, gmem = parse_gpu(last("-gpu")) if last("-gpu") else (0, None)
        deps = parse_depends(last("-w")) if last("-w") else []
    except ValueError as e:
        return _die(
            str(e) if str(e).startswith(("bsub", "Bad")) else f"bsub: {e}. Job not submitted."
        )
    if any(opts.get(f) for f in ("-Is", "-I", "-Ip")):
        return _die("bsub: interactive jobs are not supported by the fake. Job not submitted.")

    billing = os.environ.get("CSUB_FAKE_LSF_BILLING_GROUP", "testlab")
    wd = work_dir()
    with locked_state() as state:
        for _cond, dep_id in deps:
            if dep_id not in state["jobs"]:
                return _die("Dependency condition invalid or never satisfied. Job not submitted.")
        job_id = str(state["next_id"])
        state["next_id"] += 1
        spool = wd / "spool" / f"{job_id}.sh"
        spool.write_text(script)
        os.chmod(spool, 0o700)
        out = last("-oo") or last("-o")
        err = last("-eo") or last("-e")
        state["jobs"][job_id] = {
            "id": job_id,
            "name": last("-J", os.path.basename(positionals[0]) if positionals else "script"),
            "queue": queue,
            "default_queue": default_queue,
            "group": last("-g"),
            "slots": slots,
            "gpus": gpus,
            "gmem_gb": gmem,
            "walltime_min": walltime,
            "deps": deps,
            "out": out.replace("%J", job_id).replace("%I", "0") if out else None,
            "err": err.replace("%J", job_id).replace("%I", "0") if err else None,
            "out_overwrite": "-oo" in opts,
            "env_mode": last("-env", "all"),
            "cwd": last("-cwd", os.getcwd()),
            "argv": argv,
            "extra": {
                k: v for k, v in opts.items() if k in ("-R", "-P", "-M", "-app", "-m", "-We")
            },
            "spool": str(spool),
            "status": "PEND",
            "submit_time": time.time(),
            "start_time": None,
            "end_time": None,
            "exit_code": None,
            "term_reason": None,
            "runner_pid": None,
            "pgid": None,
            "exec_host": None,
            "kill_requested": False,
            "user": _user(),
        }
    log = open(wd / "runner" / f"{job_id}.log", "ab")
    proc = subprocess.Popen(
        [sys.executable, "-I", os.path.abspath(__file__), "--run", job_id],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
        start_new_session=True,
        close_fds=True,
    )
    with locked_state() as state:
        state["jobs"][job_id]["runner_pid"] = proc.pid
    sys.stdout.write(f"This job will be billed to {billing}\n")
    if default_queue:
        sys.stdout.write(f"Job <{job_id}> is submitted to default queue <{queue}>.\n")
    else:
        sys.stdout.write(f"Job <{job_id}> is submitted to queue <{queue}>.\n")
    sys.stdout.flush()
    return 0


# --- runner -------------------------------------------------------------------------------


def _dep_satisfied(cond: str, dep: dict | None) -> bool | None:
    """True = run, False = wait, None = never (stay pending forever)."""
    if dep is None:
        return None
    st = dep["status"]
    if st in ("PEND", "RUN"):
        return False
    if cond == "ended":
        return True
    if cond == "done":
        return True if st == "DONE" else None
    return True if st == "EXIT" else None


def _signal_sequence(pgid: int, first: int, grace: float) -> None:
    for sig in (first, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        deadline = time.time() + grace
        while time.time() < deadline:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.05)


def _job_env(job: dict, host: str) -> dict[str, str]:
    pw = pwd.getpwuid(os.getuid())
    if job["env_mode"] == "none":
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": pw.pw_dir,
            "USER": pw.pw_name,
            "LOGNAME": pw.pw_name,
            "SHELL": "/bin/sh",
        }
    else:
        env = dict(os.environ)
    env.update(
        {
            "LSB_JOBID": job["id"],
            "LSB_JOBNAME": job["name"],
            "LSB_QUEUE": job["queue"],
            "LSB_DJOB_NUMPROC": str(job["slots"]),
            "LSB_MAX_NUM_PROCESSORS": str(job["slots"]),
            "LSB_HOSTS": " ".join([host] * job["slots"]),
            "LSB_MCPU_HOSTS": f"{host} {job['slots']}",
            "LSB_JOBFILENAME": job["spool"],
            "LSB_SUB_HOST": "fakesubmit",
            "LSB_EXEC_CLUSTER": "fake",
            "LSB_JOBINDEX": "0",
        }
    )
    if job["group"]:
        env["LSB_JOBGROUP"] = job["group"]
    if job["out"]:
        env["LSB_OUTPUTFILE"] = job["out"]
    if job["gpus"]:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(job["gpus"]))
        env["CUDA_VISIBLE_DEVICES_ORIG"] = env["CUDA_VISIBLE_DEVICES"]
    # Test-only leak: the fakes themselves need their configuration inside the job.
    for k, v in os.environ.items():
        if k.startswith("CSUB_FAKE_"):
            env[k] = v
    return env


def run_job(job_id: str) -> int:
    host = os.environ.get("CSUB_FAKE_LSF_HOST", "fakehost01")
    pend_s = _env_float("CSUB_FAKE_LSF_PEND_S", 0.0)
    minute_s = _env_float("CSUB_FAKE_LSF_MINUTE_S", 60.0)
    grace = _env_float("CSUB_FAKE_LSF_KILL_GRACE_S", 1.0)
    t0 = time.time()
    while True:
        with locked_state() as state:
            job = state["jobs"][job_id]
            if job["kill_requested"]:
                job.update(
                    status="EXIT", exit_code=130, term_reason="TERM_OWNER", end_time=time.time()
                )
                return 0
            verdicts = [_dep_satisfied(c, state["jobs"].get(d)) for c, d in job["deps"]]
            ready = all(v is True for v in verdicts) and time.time() - t0 >= pend_s
            if ready:
                job["status"] = "RUN"
                job["start_time"] = time.time()
                job["exec_host"] = host
        if ready:
            break
        time.sleep(0.1)

    out_path = job["out"]
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out = open(out_path, "wb" if job["out_overwrite"] else "ab")
    else:
        out = open(os.devnull, "wb")
    err = out
    if job["err"]:
        Path(job["err"]).parent.mkdir(parents=True, exist_ok=True)
        err = open(job["err"], "ab")
    script = Path(job["spool"])
    cmd = [str(script)] if script.read_text().startswith("#!") else ["/bin/sh", str(script)]
    proc = subprocess.Popen(
        cmd,
        cwd=job["cwd"],
        env=_job_env(job, host),
        stdin=subprocess.DEVNULL,
        stdout=out,
        stderr=err,
        start_new_session=True,
    )
    with locked_state() as state:
        state["jobs"][job_id]["pgid"] = proc.pid
    deadline = time.time() + job["walltime_min"] * minute_s if job["walltime_min"] else None
    term_reason = None
    while proc.poll() is None:
        with locked_state() as state:
            killed = state["jobs"][job_id]["kill_requested"]
        if killed:
            term_reason = "TERM_OWNER"
            _signal_sequence(proc.pid, signal.SIGINT, grace)
            break
        if deadline and time.time() > deadline:
            term_reason = "TERM_RUNLIMIT"
            _signal_sequence(proc.pid, signal.SIGUSR2, grace)
            break
        time.sleep(0.1)
    rc = proc.wait()
    if rc < 0:
        rc = 128 - rc
    status = "DONE" if rc == 0 else "EXIT"
    epilogue = ["", "-" * 60]
    if term_reason == "TERM_OWNER":
        epilogue.append("TERM_OWNER: job killed by owner.")
    elif term_reason == "TERM_RUNLIMIT":
        epilogue.append("TERM_RUNLIMIT: job killed after reaching LSF run time limit.")
    epilogue.append("Successfully completed." if rc == 0 else f"Exited with exit code {rc}.")
    epilogue += ["", "Resource usage summary:", "", "    CPU time :   0.00 sec.", ""]
    if out_path:
        out.write(("\n".join(epilogue) + "\n").encode())
    out.close()
    with locked_state() as state:
        state["jobs"][job_id].update(
            status=status, exit_code=rc, end_time=time.time(), term_reason=term_reason
        )
    return 0


# --- bjobs --------------------------------------------------------------------------------


def _fmt_time(t: float | None) -> str:
    return time.strftime("%b %d %H:%M", time.localtime(t)) if t else "-"


def _field(job: dict, name: str) -> str:
    if name == "jobid":
        return job["id"]
    if name == "stat":
        return job["status"]
    if name == "exit_code":
        return str(job["exit_code"]) if job["status"] == "EXIT" else "-"
    if name == "queue":
        return job["queue"]
    if name == "exec_host":
        return job["exec_host"] or "-"
    if name in ("job_name", "name"):
        return job["name"]
    if name == "user":
        return job["user"]
    if name == "submit_time":
        return _fmt_time(job["submit_time"])
    if name == "from_host":
        return "fakesubmit"
    raise ValueError(f"bjobs: unknown field {name}")


def _parse_format(fmt: str) -> tuple[list[str], str | None]:
    m = re.search(r"""delimiter=(['"])(.*?)\1""", fmt)
    delim = None
    if m:
        delim = m.group(2)
        fmt = fmt[: m.start()] + fmt[m.end() :]
    fields = [f.split(":")[0].lower() for f in fmt.split()]
    return fields, delim


def cmd_bjobs(argv: list[str]) -> int:
    noheader = False
    fmt = None
    group = None
    show_all = False
    name = None
    as_json = False
    ids: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "-noheader":
            noheader = True
        elif a == "-o":
            fmt = argv[i + 1]
            i += 1
        elif a == "-g":
            group = argv[i + 1]
            i += 1
        elif a == "-J":
            name = argv[i + 1]
            i += 1
        elif a == "-u":
            i += 1
        elif a == "-a":
            show_all = True
        elif a == "-json":
            as_json = True
        elif a in ("-w", "-W", "-l"):
            pass
        elif a.startswith("-"):
            return _die(f"bjobs: Unknown option {a}")
        else:
            ids.append(a)
        i += 1
    clean = os.environ.get("CSUB_FAKE_LSF_CLEAN_PERIOD_S")
    now = time.time()
    state = read_state()

    def visible(j: dict) -> bool:
        if group and not (
            j["group"] == group or (j["group"] or "").startswith(group.rstrip("/") + "/")
        ):
            return False
        if name and j["name"] != name:
            return False
        if j["status"] in ("DONE", "EXIT"):
            if clean not in (None, "") and now - (j["end_time"] or now) > float(clean):
                return False
            if not ids and not show_all:
                return False
        return True

    missing: list[str] = []
    rows: list[dict] = []
    if ids:
        for jid in ids:
            j = state["jobs"].get(jid)
            if j is None or not visible(j):
                missing.append(jid)
            else:
                rows.append(j)
    else:
        rows = [j for j in state["jobs"].values() if visible(j)]
    rows.sort(key=lambda j: int(j["id"]))

    for jid in missing:
        sys.stderr.write(f"Job <{jid}> is not found\n")
    if not rows:
        if not missing:
            sys.stderr.write("No unfinished job found\n")
        return 255

    fields, delim = (
        _parse_format(fmt)
        if fmt
        else (
            ["jobid", "user", "stat", "queue", "from_host", "exec_host", "job_name", "submit_time"],
            None,
        )
    )
    if as_json:
        recs = [{f.upper(): _field(j, f) for f in fields} for j in rows]
        sys.stdout.write(
            json.dumps({"COMMAND": "bjobs", "JOBS": len(recs), "RECORDS": recs}) + "\n"
        )
        return 0

    def line(values: list[str]) -> str:
        if delim is not None:
            return delim.join(values)
        return "".join(
            v.ljust(FIELD_WIDTHS.get(f, 10) + 1)
            for v, f in zip(values, fields)  # noqa: B905
        ).rstrip()

    if not noheader:
        sys.stdout.write(line([f.upper() for f in fields]) + "\n")
    for j in rows:
        sys.stdout.write(line([_field(j, f) for f in fields]) + "\n")
    return 0


# --- bkill --------------------------------------------------------------------------------


def cmd_bkill(argv: list[str]) -> int:
    group = None
    name = None
    ids: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "-g":
            group = argv[i + 1]
            i += 1
        elif a == "-J":
            name = argv[i + 1]
            i += 1
        elif a in ("-s", "-u", "-q", "-m"):
            i += 1
        elif a.startswith("-"):
            return _die(f"bkill: Unknown option {a}")
        else:
            ids.append(a)
        i += 1
    if not ids and not name:
        return _die("bkill: job id or -J required")
    terminated = 0
    with locked_state() as state:
        jobs = state["jobs"]

        def in_scope(j: dict) -> bool:
            if group and not (
                j["group"] == group or (j["group"] or "").startswith(group.rstrip("/") + "/")
            ):
                return False
            return not name or j["name"] == name

        targets: list[str] = []
        if ids == ["0"] or (not ids and name):
            targets = [
                jid for jid, j in jobs.items() if in_scope(j) and j["status"] in ("PEND", "RUN")
            ]
            if not targets:
                sys.stderr.write("No matching job found\n")
                return 255
        else:
            for jid in ids:
                j = jobs.get(jid)
                if j is None or not in_scope(j):
                    sys.stderr.write(f"Job <{jid}>: No matching job found\n")
                elif j["status"] in ("DONE", "EXIT"):
                    sys.stderr.write(f"Job <{jid}>: Job has already finished\n")
                else:
                    targets.append(jid)
        for jid in targets:
            jobs[jid]["kill_requested"] = True
            sys.stdout.write(f"Job <{jid}> is being terminated\n")
            terminated += 1
    return 0 if terminated else 255


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--run"]:
        return run_job(argv[1])
    tool = (
        argv.pop(0)
        if argv and argv[0] in ("bsub", "bjobs", "bkill")
        else os.path.basename(sys.argv[0])
    )
    if tool == "bsub":
        return cmd_bsub(argv, sys.stdin)
    if tool == "bjobs":
        return cmd_bjobs(argv)
    if tool == "bkill":
        return cmd_bkill(argv)
    return _die(f"fake lsf: unknown tool {tool}", 2)


if __name__ == "__main__":
    sys.exit(main())
