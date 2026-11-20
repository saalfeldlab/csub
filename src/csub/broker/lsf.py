"""Run and parse LSF's command line tools. Standard library only.

Everything LSF-specific about *output formats* lives here so the rest of the broker deals
in plain data. All commands are run through a bash wrapper that sources the LSF profile
first (a forced SSH command has no login environment).
"""

from __future__ import annotations

import re
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from csub.broker.policy import LsfConfig
from csub.protocol import CsubError

BSUB_SUBMITTED_RE = re.compile(r"Job <(\d+)> is submitted to (?:default )?queue <([^>]+)>\.")
BSUB_BILLING_RE = re.compile(r"This job will be billed to (\S+)")
BJOBS_FIELDS = (
    "jobid", "stat", "exit_code", "queue", "exec_host", "job_name",
    "cpu_used", "max_mem", "avg_mem", "run_time", "memlimit",
)  # fmt: skip
BJOBS_NOT_FOUND_RE = re.compile(r"(No (unfinished |matching |)job|is not found|No job group)", re.I)
BKILL_TERMINATED_RE = re.compile(r"Job <(\d+)> is being terminated")
BKILL_FINISHED_RE = re.compile(r"Job <(\d+)>: Job has already finished")
BKILL_NOT_FOUND_RE = re.compile(r"Job <(\d+)>: No matching job found")

# What a job used, as bjobs -o prints it: "0.7 second(s)", "311 Mbytes", "6.8 Gbytes", "15 G";
# "-" where LSF has no figure (yet). LSF's job summary in lsf.out says "1.09 sec.", "47 MB".
_AMOUNT_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]*)")
_MB_PER_UNIT = {"K": 1 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024}
_MEM_UNIT_RE = re.compile(r"^([KMGT])(B|bytes)?$", re.I)


def parse_seconds(text: str | None) -> float | None:
    """'33 second(s)' or '1.09 sec.' -> seconds; None for '-' or anything else."""
    m = _AMOUNT_RE.match(text or "")
    return float(m.group(1)) if m and m.group(2).lower().startswith("sec") else None


def parse_mb(text: str | None) -> float | None:
    """'311 Mbytes', '6.8 Gbytes', '15 G' or '47 MB' -> megabytes; None for '-'."""
    m = _AMOUNT_RE.match(text or "")
    unit = _MEM_UNIT_RE.match(m.group(2)) if m else None
    if not unit:
        return None
    return round(float(m.group(1)) * _MB_PER_UNIT[unit.group(1).upper()], 1)


_SUMMARY_FIELDS = {
    "CPU time": ("cpu_time_s", parse_seconds),
    "Max Memory": ("max_mem_mb", parse_mb),
    "Average Memory": ("avg_mem_mb", parse_mb),
    "Total Requested Memory": ("mem_limit_mb", parse_mb),
    "Run time": ("run_time_s", parse_seconds),
}


def parse_usage_summary(text: str) -> dict[str, float | None]:
    """The figures of LSF's "Resource usage summary" at the end of a job's output file. Only
    lines after that heading count, so nothing the job itself printed can pass for a figure."""
    out: dict[str, float | None] = {name: None for name, _ in _SUMMARY_FIELDS.values()}
    _, found, summary = text.rpartition("Resource usage summary:")
    if not found:
        return out
    for line in summary.splitlines():
        label, sep, value = line.partition(" :")
        field = _SUMMARY_FIELDS.get(label.strip())
        if sep and field:
            out[field[0]] = field[1](value)
    return out


STATE_MAP = {
    "PEND": "PEND",
    "RUN": "RUN",
    "DONE": "DONE",
    "EXIT": "EXIT",
    "PSUSP": "SUSP",
    "USUSP": "SUSP",
    "SSUSP": "SUSP",
}


class LsfError(CsubError):
    def __init__(self, message: str):
        super().__init__("lsf_error", message)


@dataclass(frozen=True)
class BsubResult:
    job_id: str
    queue: str
    billing_group: str | None
    raw_stdout: str


@dataclass(frozen=True)
class BjobsRow:
    job_id: str
    stat: str  # raw LSF state
    exit_code: int | None
    queue: str | None
    exec_host: str | None
    job_name: str | None
    cpu_time_s: float | None = None
    max_mem_mb: float | None = None
    avg_mem_mb: float | None = None
    run_time_s: float | None = None
    mem_limit_mb: float | None = None

    def usage(self) -> dict[str, float | None]:
        return {
            "cpu_time_s": self.cpu_time_s,
            "max_mem_mb": self.max_mem_mb,
            "avg_mem_mb": self.avg_mem_mb,
            "run_time_s": self.run_time_s,
            "mem_limit_mb": self.mem_limit_mb,
        }

    @property
    def state(self) -> str:
        return STATE_MAP.get(self.stat, "UNKNOWN")


def _tail(text: str, n: int = 20) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-n:])


class LsfRunner:
    def __init__(
        self,
        cfg: LsfConfig,
        *,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ):
        self.cfg = cfg
        self._run = run

    # --- plumbing ---

    def argv(self, tool: str, args: list[str]) -> list[str]:
        if self.cfg.profile:
            script = f'. {shlex.quote(self.cfg.profile)} >/dev/null 2>&1 || exit 97; exec "$0" "$@"'
            # Under sshd, bash sources ~/.bashrc even for `bash -c`; with lsf.norc the policy
            # can skip that (seconds per LSF call when .bashrc runs a conda hook or similar).
            bash = ["bash", "--noprofile", "--norc"] if self.cfg.norc else ["bash"]
            return [*bash, "-c", script, tool, *args]
        return [tool, *args]

    def _exec(
        self, tool: str, args: list[str], *, input: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        try:
            return self._run(
                self.argv(tool, args),
                input=input,
                capture_output=True,
                text=True,
                timeout=self.cfg.timeout_s,
            )
        except FileNotFoundError as e:
            raise LsfError(f"{tool}: not found ({e})") from None
        except subprocess.TimeoutExpired:
            raise LsfError(f"{tool} did not finish within {self.cfg.timeout_s}s") from None

    # --- commands ---

    def bsub(self, args: list[str], script: str) -> BsubResult:
        cp = self._exec(self.cfg.bsub, args, input=script)
        m = BSUB_SUBMITTED_RE.search(cp.stdout)
        if cp.returncode != 0 or not m:
            detail = _tail(cp.stderr) or _tail(cp.stdout) or f"exit status {cp.returncode}"
            raise LsfError(f"bsub failed: {detail}")
        billing = BSUB_BILLING_RE.search(cp.stdout)
        return BsubResult(
            job_id=m.group(1),
            queue=m.group(2),
            billing_group=billing.group(1) if billing else None,
            raw_stdout=cp.stdout,
        )

    def bjobs(self, group: str, job_ids: list[str] | None = None) -> list[BjobsRow]:
        fmt = " ".join(BJOBS_FIELDS) + ' delimiter="|"'
        args = ["-a", "-noheader", "-o", fmt, "-g", group, *(job_ids or [])]
        cp = self._exec(self.cfg.bjobs, args)
        rows: list[BjobsRow] = []
        for line in cp.stdout.splitlines():
            parts = line.rstrip("\n").split("|")
            if len(parts) != len(BJOBS_FIELDS) or not parts[0].isdigit():
                continue
            jobid, stat, exit_code, queue, exec_host, job_name = parts[:6]
            cpu_used, max_mem, avg_mem, run_time, memlimit = parts[6:]
            rows.append(
                BjobsRow(
                    cpu_time_s=parse_seconds(cpu_used),
                    max_mem_mb=parse_mb(max_mem),
                    avg_mem_mb=parse_mb(avg_mem),
                    run_time_s=parse_seconds(run_time),
                    mem_limit_mb=parse_mb(memlimit),
                    job_id=jobid,
                    stat=stat,
                    exit_code=int(exit_code) if exit_code.isdigit() else None,
                    queue=None if queue == "-" else queue,
                    exec_host=None if exec_host == "-" else exec_host,
                    job_name=None if job_name == "-" else job_name,
                )
            )
        if cp.returncode != 0 and not rows and not BJOBS_NOT_FOUND_RE.search(cp.stderr + cp.stdout):
            raise LsfError(f"bjobs failed: {_tail(cp.stderr) or f'exit status {cp.returncode}'}")
        return rows

    def bkill(self, group: str, job_ids: list[str] | None) -> dict[str, str]:
        """Return job_id -> 'killed' | 'finished' | 'not_found'. None/[] kills the whole group."""
        args = ["-g", group, *(job_ids or ["0"])]
        cp = self._exec(self.cfg.bkill, args)
        out: dict[str, str] = {}
        for m in BKILL_TERMINATED_RE.finditer(cp.stdout):
            out[m.group(1)] = "killed"
        for m in BKILL_FINISHED_RE.finditer(cp.stderr):
            out[m.group(1)] = "finished"
        for m in BKILL_NOT_FOUND_RE.finditer(cp.stderr):
            out[m.group(1)] = "not_found"
        if cp.returncode != 0 and not out and not BJOBS_NOT_FOUND_RE.search(cp.stderr):
            raise LsfError(f"bkill failed: {_tail(cp.stderr) or f'exit status {cp.returncode}'}")
        return out
