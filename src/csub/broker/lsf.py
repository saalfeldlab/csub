"""Run and parse LSF's command line tools. Standard library only.

Everything LSF-specific about *output formats* lives here so the rest of the broker deals
in plain data. All commands are run through a bash wrapper that sources the LSF profile
first (a forced SSH command has no login environment).

With ``lsf.submit_host`` set, that same wrapper is run over ssh on the submit host, so the broker
can sit on a workstation that shares the cluster filesystem but is not an LSF host. The LSF
binaries and profile may well be visible there too (a shared /misc or /opt), so presence of
files proves nothing; what matters is whether this host may talk to the cluster. Two cases are
known to be fine and keep calling bsub directly: inside an LSF job (the per-job broker the
wrapper starts on a compute node; LSF sets LSB_JOBID there), and on the submit host itself.
"""

from __future__ import annotations

import os
import re
import shlex
import socket
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass

from csub.broker.policy import LsfConfig
from csub.protocol import CsubError
from csub.transport.ssh import default_control_dir

BSUB_SUBMITTED_RE = re.compile(r"Job <(\d+)> is submitted to (?:default )?queue <([^>]+)>\.")
BSUB_BILLING_RE = re.compile(r"This job will be billed to (\S+)")
BJOBS_FIELDS = ("jobid", "stat", "exit_code", "queue", "exec_host", "job_name")
BJOBS_NOT_FOUND_RE = re.compile(r"(No (unfinished |matching |)job|is not found|No job group)", re.I)
BKILL_TERMINATED_RE = re.compile(r"Job <(\d+)> is being terminated")
BKILL_FINISHED_RE = re.compile(r"Job <(\d+)>: Job has already finished")
BKILL_NOT_FOUND_RE = re.compile(r"Job <(\d+)>: No matching job found")

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

    @property
    def state(self) -> str:
        return STATE_MAP.get(self.stat, "UNKNOWN")


def is_this_host(name: str) -> bool:
    """True when ``name`` resolves to one of this machine's addresses (aliases included)."""
    try:
        theirs = {ai[4][0] for ai in socket.getaddrinfo(name, None)}
        mine = {ai[4][0] for ai in socket.getaddrinfo(socket.gethostname(), None)}
    except OSError:
        return False
    return bool(theirs & mine)


def _tail(text: str, n: int = 20) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-n:])


class LsfRunner:
    def __init__(
        self,
        cfg: LsfConfig,
        *,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        own_prefix: str | None = None,
    ):
        self.cfg = cfg
        self._run = run
        self._own_prefix = os.path.realpath(own_prefix or sys.prefix)
        self._checked: set[str] = set()
        self._remote = bool(cfg.submit_host) and not (
            "LSB_JOBID" in os.environ or is_this_host(cfg.submit_host)
        )

    # --- plumbing ---

    def argv(self, tool: str, args: list[str]) -> list[str]:
        if self.cfg.profile:
            script = f'. {shlex.quote(self.cfg.profile)} >/dev/null 2>&1 || exit 97; exec "$0" "$@"'
            # Under sshd, bash sources ~/.bashrc even for `bash -c`; with lsf.norc the policy
            # can skip that (seconds per LSF call when .bashrc runs a conda hook or similar).
            bash = ["bash", "--noprofile", "--norc"] if self.cfg.norc else ["bash"]
            local = [*bash, "-c", script, tool, *args]
        else:
            local = [tool, *args]
        if not self._remote:
            return local
        # The user's own ~/.ssh config supplies user, key and known_hosts. One multiplexed
        # connection serves every call, so polling costs one round trip, not one handshake.
        control_dir = default_control_dir()
        os.makedirs(control_dir, mode=0o700, exist_ok=True)
        return [
            "ssh", "-T",
            "-o", "BatchMode=yes",
            "-o", "ControlMaster=auto",
            "-o", "ControlPersist=120",
            "-o", f"ControlPath={control_dir}/cm-%C",
            "-o", "ConnectTimeout=20",
            "-o", "LogLevel=ERROR",
            self.cfg.submit_host, shlex.join(local),
        ]  # fmt: skip

    def _check_not_self(self, tool: str) -> None:
        """Refuse to call a `bsub` that lives in our own install prefix (a shim)."""
        if tool in self._checked or "/" in tool or self._remote:
            return
        self._checked.add(tool)
        try:
            cp = self._run(
                self.argv("bash", ["-c", "command -v " + shlex.quote(tool)]),
                capture_output=True,
                text=True,
                timeout=self.cfg.timeout_s,
            )
        except (OSError, subprocess.TimeoutExpired):
            return
        path = cp.stdout.strip()
        if path and os.path.realpath(path).startswith(self._own_prefix + "/"):
            raise CsubError(
                "broker_misconfigured",
                f"{tool} resolves to {path}, inside csub's own install prefix; "
                f"LSF's {tool} is shadowed",
            )

    def _exec(
        self, tool: str, args: list[str], *, input: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        self._check_not_self(tool)
        try:
            return self._run(
                self.argv(tool, args),
                input=input,
                capture_output=True,
                text=True,
                timeout=self.cfg.timeout_s,
            )
        except FileNotFoundError as e:
            raise LsfError(f"{self.argv(tool, [])[0]}: not found ({e})") from None
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
            jobid, stat, exit_code, queue, exec_host, job_name = parts
            rows.append(
                BjobsRow(
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
