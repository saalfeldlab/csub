"""``csub``: the command line surface over :class:`csub.client.Client`."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any, TextIO

from csub import __version__
from csub.client.api import Client, WaitResult
from csub.protocol import CsubError, JobSpec

EXIT_CODES = {
    "ok": 0,
    "job_failed": 1,
    "invalid_request": 2,
    "policy_violation": 3,
    "transport_error": 4,
    "lsf_error": 5,
    "not_found": 6,
    "broker_misconfigured": 7,
    "internal_error": 8,
    "timeout": 124,
}

_MEM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kKmMgGtT]?)[bB]?\s*$")


class UsageError(CsubError):
    def __init__(self, message: str):
        super().__init__("invalid_request", message)


def parse_mem(text: str) -> int:
    """'100G' -> 102400 (MB). Plain numbers are MB."""
    m = _MEM_RE.match(text)
    if not m:
        raise UsageError(f"--mem: cannot parse {text!r} (use e.g. 500M, 100G)")
    value, unit = float(m.group(1)), m.group(2).upper()
    factor = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}[unit]
    mb = int(round(value * factor))
    if mb < 0:
        raise UsageError("--mem: must be positive")
    return mb


def parse_walltime(text: str) -> int:
    """'90' -> 90, '01:30' -> 90 (minutes)."""
    text = text.strip()
    if re.fullmatch(r"\d+", text):
        return int(text)
    m = re.fullmatch(r"(\d+):(\d{1,2})", text)
    if m and int(m.group(2)) < 60:
        return int(m.group(1)) * 60 + int(m.group(2))
    raise UsageError(f"--walltime: cannot parse {text!r} (use minutes or HH:MM)")


def parse_env(items: Sequence[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise UsageError(f"--env: expected KEY=VALUE, got {item!r}")
        env[key] = value
    return env


def parse_depends(items: Sequence[str]) -> list[dict[str, str]]:
    out = []
    for item in items:
        job_id, _, when = item.partition(":")
        out.append({"job_id": job_id, "when": when or "done"})
    return out


# --- parser -------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="csub", description="Submit sandboxed jobs to the LSF cluster."
    )
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--version", action="version", version=f"csub {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser(
        "submit", help="submit a job", description="Submit a job. Size it in cpus/mem/walltime."
    )
    s.add_argument("--name")
    s.add_argument("--cpus", type=int, default=1)
    s.add_argument("--mem", help="memory, e.g. 500M or 100G (default MB)")
    s.add_argument("--gpus", type=int, default=0)
    s.add_argument("--gpu-mem-gb", type=int)
    s.add_argument("--walltime", help="minutes or HH:MM; prefer <= 60 (the short queue)")
    s.add_argument("-q", "--queue")
    s.add_argument("--image")
    s.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    s.add_argument(
        "--allow", action="append", default=[], metavar="HOST", help="allow HTTP(S) egress to HOST"
    )
    s.add_argument(
        "--scratch", action="store_true", help="private node-local scratch dir as TMPDIR"
    )
    s.add_argument(
        "--claude", action="store_true", help="run Claude Code in the job (policy permitting)"
    )
    s.add_argument("--depends-on", action="append", default=[], metavar="ID[:done|ended|exit]")
    s.add_argument("--cwd")
    s.add_argument("--session")
    s.add_argument("--shell", action="store_true", help="run a script body instead of an argv")
    s.add_argument(
        "--script", metavar="FILE|-", help="with --shell: read the body from FILE or stdin"
    )
    s.add_argument("--lsf-extra", action="append", default=[], metavar="FLAG")
    s.add_argument("--wait", action="store_true", help="wait for the job and show its output")
    s.add_argument("--timeout", type=float, help="with --wait: give up after this many seconds")
    s.add_argument("--tail", type=int, default=20, help="with --wait: lines of output to show")
    s.add_argument("command", nargs=argparse.REMAINDER, help="-- CMD [ARG...]")

    st = sub.add_parser("status", help="show job states")
    st.add_argument("job_ids", nargs="*", metavar="ID")

    w = sub.add_parser("wait", help="wait for jobs to finish")
    w.add_argument("job_ids", nargs="+", metavar="ID")
    w.add_argument("--timeout", type=float)
    w.add_argument("--tail", type=int, default=20)

    k = sub.add_parser("kill", help="kill jobs")
    k.add_argument("job_ids", nargs="*", metavar="ID")
    k.add_argument("--all", action="store_true", help="kill every job of this session")

    lg = sub.add_parser("logs", help="show a job's output")
    lg.add_argument("job_id", metavar="ID")
    g = lg.add_mutually_exclusive_group()
    g.add_argument("--stdout", action="store_true")
    g.add_argument("--stderr", action="store_true")
    lg.add_argument("--tail", type=int)
    lg.add_argument("--follow", "-f", action="store_true", help="keep printing until the job ends")
    lg.add_argument("--cwd", help="the job's working directory (default: ask the broker)")

    sub.add_parser("probe", help="show what the broker allows: queues, limits, images, hosts")
    return p


# --- commands -----------------------------------------------------------------------------


def spec_from_args(ns: argparse.Namespace, stdin: TextIO) -> JobSpec:
    command = list(ns.command)
    if command and command[0] == "--":
        command = command[1:]
    d: dict[str, Any] = {
        "cpus": ns.cpus,
        "gpus": ns.gpus,
        "scratch": ns.scratch,
        "claude": ns.claude,
        "shell": ns.shell,
    }
    if ns.shell:
        if ns.script:
            body = stdin.read() if ns.script == "-" else open(ns.script).read()
        elif command:
            body = " ".join(command) + "\n"
        else:
            raise UsageError("--shell needs --script FILE|- or a command")
        d["command"] = body
    else:
        if ns.script:
            raise UsageError("--script requires --shell")
        if not command:
            raise UsageError("no command given (use: csub submit [options] -- CMD ARG...)")
        d["command"] = command
    if ns.mem:
        d["mem_mb"] = parse_mem(ns.mem)
    if ns.walltime:
        d["walltime_min"] = parse_walltime(ns.walltime)
    for key in ("name", "queue", "image", "session", "cwd"):
        if getattr(ns, key):
            d[key] = getattr(ns, key)
    if ns.gpu_mem_gb is not None:
        d["gpu_mem_gb"] = ns.gpu_mem_gb
    if ns.env:
        d["env"] = parse_env(ns.env)
    if ns.allow:
        d["allow_hosts"] = ns.allow
    if ns.depends_on:
        d["depends_on"] = parse_depends(ns.depends_on)
    if ns.lsf_extra:
        d["lsf_extra"] = ns.lsf_extra
    if d.get("cwd"):
        d["cwd"] = os.path.realpath(d["cwd"])
    return JobSpec.from_dict(d)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def format_status_table(entries) -> str:
    rows = [("JOBID", "STATE", "EXIT", "QUEUE", "HOST", "NAME")]
    for e in entries:
        rows.append(
            (
                e.job_id,
                e.state,
                "-" if e.exit_code is None else str(e.exit_code),
                e.queue or "-",
                e.exec_host or "-",
                e.name or "-",
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = []
    for r in rows:
        lines.append("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())  # noqa: B905
    return "\n".join(lines)


def format_wait(result: WaitResult, out: TextIO) -> None:
    for e in result.jobs:
        s = e.status
        code = "" if s.exit_code is None else f" (exit {s.exit_code})"
        out.write(f"Job {s.job_id}: {s.state}{code}\n")
        for label, text in (("stdout", e.stdout_tail), ("stderr", e.stderr_tail)):
            if text:
                out.write(
                    f"--- {label} "
                    f"({e.stdout_path if label == 'stdout' else e.stderr_path}) ---\n{text}"
                )
                if not text.endswith("\n"):
                    out.write("\n")
    if result.timed_out:
        out.write(f"Timed out after {result.elapsed_s:.0f}s; jobs are still running.\n")


def wait_exit_code(result: WaitResult) -> int:
    if result.timed_out:
        return EXIT_CODES["timeout"]
    states = [e.status for e in result.jobs]
    if all(s.state == "DONE" for s in states):
        return 0
    if len(states) == 1 and states[0].exit_code and 1 <= states[0].exit_code <= 125:
        return states[0].exit_code
    return EXIT_CODES["job_failed"]


def format_probe(p, out: TextIO) -> None:
    out.write(f"broker {p.broker_version} for {p.user}\n\n")
    out.write(
        f"{'QUEUE':<14}{'GPU':<5}{'MEM/SLOT':>9}{'SLOTS/GPU':>10}{'MAX WALL':>10}{'$/GPU·h':>9}\n"
    )
    for name, q in p.queues.items():
        out.write(
            f"{name:<14}{'yes' if q.get('gpu') else '-':<5}"
            f"{q.get('mem_per_slot_mb', 0) // 1024:>7} GB"
            f"{q.get('slots_per_gpu') or '-':>10}{q.get('max_walltime_min', 0) // 60:>8} h"
            f"{q.get('gpu_price_usd_per_hour') or 0:>9.2f}\n"
        )
    lim = p.limits
    out.write(
        f"\nlimits: {lim.get('max_slots')} slots, {lim.get('max_gpus')} GPUs, "
        f"{lim.get('max_walltime_min', 0) // 1440} days, "
        f"cost cap ${lim.get('max_estimated_cost_usd', 0):.2f}; "
        f"slot price ${lim.get('slot_price_usd_per_hour', 0):.2f}/h\n"
    )
    out.write(f"default image: {p.default_image}\n")
    out.write("allowed images: " + (", ".join(p.allowed_images) or "(default only)") + "\n")
    out.write("allowed roots: " + ", ".join(p.allowed_roots) + "\n")
    out.write("allowed hosts: " + (", ".join(p.allowed_hosts) or "(none)") + "\n")
    out.write(
        f"sandbox: {p.sandbox}; scratch: {'yes' if p.scratch else 'no'}; "
        f"env: {', '.join(p.env_allow) or '(none)'}\n"
    )


def cmd_submit(ns, client: Client, out: TextIO, stdin: TextIO) -> int:
    spec = spec_from_args(ns, stdin)
    r = client.submit(spec)
    if ns.json:
        if not ns.wait:
            out.write(json.dumps(r.to_dict(), indent=1) + "\n")
    else:
        billed = f" billed to {r.billing_group}" if r.billing_group else ""
        via = f" via {r.sandbox}" if r.sandbox else ""
        out.write(
            f"Job {r.job_id} submitted to {r.queue}{via} "
            f"({_plural(r.slots, 'slot')}, {r.walltime_min} min, "
            f"est. max ${r.estimated_max_cost_usd:.2f}){billed}\n"
        )
    if not ns.wait:
        return 0
    result = client.wait([r.job_id], timeout_s=ns.timeout, tail_lines=ns.tail)
    if ns.json:
        out.write(json.dumps({"submit": r.to_dict(), "wait": result.to_dict()}, indent=1) + "\n")
    else:
        format_wait(result, out)
    return wait_exit_code(result)


def cmd_status(ns, client: Client, out: TextIO) -> int:
    entries = client.status(ns.job_ids or None)
    if ns.json:
        out.write(json.dumps([e.to_dict() for e in entries], indent=1) + "\n")
    elif entries:
        out.write(format_status_table(entries) + "\n")
    else:
        out.write("no jobs\n")
    return 0


def cmd_wait(ns, client: Client, out: TextIO) -> int:
    result = client.wait(ns.job_ids, timeout_s=ns.timeout, tail_lines=ns.tail)
    if ns.json:
        out.write(json.dumps(result.to_dict(), indent=1) + "\n")
    else:
        format_wait(result, out)
    return wait_exit_code(result)


def cmd_kill(ns, client: Client, out: TextIO) -> int:
    if bool(ns.job_ids) == ns.all:
        raise UsageError("give job ids or --all (not both)")
    r = client.kill(ns.job_ids or None)
    if ns.json:
        out.write(json.dumps(r.to_dict(), indent=1) + "\n")
    else:
        for jid in r.killed:
            out.write(f"Job {jid} is being terminated\n")
        for jid in r.already_finished:
            out.write(f"Job {jid} had already finished\n")
        if not r.killed and not r.already_finished:
            out.write("nothing to kill\n")
    return 0


def cmd_logs(ns, client: Client, out: TextIO, sleep: Callable[[float], None] = time.sleep) -> int:
    stream = "stdout" if ns.stdout else "stderr" if ns.stderr else "both"
    cwd = ns.cwd
    if ns.follow:
        shown = {"stdout": "", "stderr": ""}
        while True:
            st = client.status([ns.job_id])[0]
            cwd = cwd or st.cwd
            if cwd:
                try:
                    lg = client.logs(ns.job_id, stream=stream, cwd=cwd)
                except CsubError as e:
                    if e.code != "not_found":
                        raise
                    lg = None
                if lg:
                    for label in ("stdout", "stderr"):
                        text = getattr(lg, label)
                        if text and text.startswith(shown[label]) and len(text) > len(shown[label]):
                            out.write(text[len(shown[label]) :])
                            out.flush()
                            shown[label] = text
            if st.terminal:
                return 0
            sleep(2.0)
    lg = client.logs(ns.job_id, stream=stream, tail=ns.tail, cwd=cwd)
    if ns.json:
        out.write(json.dumps(lg.__dict__, indent=1) + "\n")
        return 0
    if stream == "both":
        if lg.stdout:
            out.write(lg.stdout if lg.stdout.endswith("\n") else lg.stdout + "\n")
        if lg.stderr:
            out.write(f"--- stderr ---\n{lg.stderr}")
    else:
        out.write(lg.stdout if stream == "stdout" else lg.stderr or "")
    return 0


def cmd_probe(ns, client: Client, out: TextIO) -> int:
    p = client.probe()
    if ns.json:
        out.write(json.dumps(p.to_dict(), indent=1) + "\n")
    else:
        format_probe(p, out)
    return 0


def main(
    argv: list[str] | None = None,
    *,
    client_factory: Callable[[], Client] = Client.from_env,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    stdin: TextIO | None = None,
) -> int:
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    inp = stdin or sys.stdin
    parser = build_parser()
    ns = parser.parse_args(argv)
    try:
        client = client_factory()
        if ns.cmd == "submit":
            return cmd_submit(ns, client, out, inp)
        if ns.cmd == "status":
            return cmd_status(ns, client, out)
        if ns.cmd == "wait":
            return cmd_wait(ns, client, out)
        if ns.cmd == "kill":
            return cmd_kill(ns, client, out)
        if ns.cmd == "logs":
            return cmd_logs(ns, client, out)
        if ns.cmd == "probe":
            return cmd_probe(ns, client, out)
        parser.error(f"unknown command {ns.cmd}")
    except CsubError as e:
        if ns.json:
            out.write(json.dumps({"error": {"code": e.code, "message": e.message}}) + "\n")
        err.write(f"csub: {e.code}: {e.message}\n")
        return EXIT_CODES.get(e.code, EXIT_CODES["internal_error"])
    except KeyboardInterrupt:
        err.write("csub: interrupted\n")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
