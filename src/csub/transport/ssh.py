"""Reach the broker on the submit host through its SSH forced command."""

from __future__ import annotations

import os
import subprocess
from typing import Any

from csub.protocol import PROTOCOL_VERSION
from csub.transport.base import TransportError, encode, finish


def default_control_dir() -> str:
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg and os.path.isdir(xdg):
        return os.path.join(xdg, "csub")
    ssh_dir = os.path.expanduser("~/.ssh")
    if os.path.isdir(ssh_dir):
        return os.path.join(ssh_dir, "csub")
    return f"/tmp/csub-{os.getuid()}"


class SshTransport:
    def __init__(
        self,
        host: str,
        *,
        user: str | None = None,
        key: str | None = None,
        port: int = 22,
        broker_cmd: str = "csub-broker",
        opts: tuple[str, ...] = (),
        control_dir: str | None = None,
        connect_timeout_s: int = 20,
        timeout_s: float = 600.0,
    ):
        self.host = host
        self.user = user
        self.key = os.path.expanduser(key) if key else None
        self.port = port
        self.broker_cmd = broker_cmd
        self.opts = tuple(opts)
        self.control_dir = control_dir or default_control_dir()
        self.connect_timeout_s = connect_timeout_s
        self.timeout_s = timeout_s

    @property
    def argv(self) -> list[str]:
        return self._argv()

    def _argv(self, *, diagnose: bool = False) -> list[str]:
        argv = ["ssh", "-T", "-p", str(self.port)]
        if diagnose:
            # ssh sends the ProxyCommand's stderr to /dev/null when ControlPersist is on, so a
            # failing proxy exits 255 with no message; this variant shows it.
            argv += ["-v", "-o", "ControlMaster=no"]
        if self.key:
            argv += ["-i", self.key, "-o", "IdentitiesOnly=yes"]
        argv += [
            "-o", "BatchMode=yes",
            "-o", "ControlMaster=auto",
            "-o", "ControlPersist=120",
            "-o", f"ControlPath={self.control_dir}/cm-%C",
            "-o", f"ConnectTimeout={self.connect_timeout_s}",
            "-o", "ServerAliveInterval=15",
            "-o", "LogLevel=ERROR",
        ]  # fmt: skip
        for opt in self.opts:
            argv += ["-o", opt] if not opt.startswith("-") else [opt]
        if self.user:
            argv += ["-l", self.user]
        argv += [self.host, self.broker_cmd]
        return argv

    def _diagnose(self, request: dict[str, Any]) -> str:
        """A silent exit 255 may be a dead proxy, or a connection lost after the broker read the
        request. Never replay the request (a submit would run twice): send a probe verbosely and
        report its outcome next to the warning that the original outcome is unknown."""
        if request.get("op") == "submit":
            head = (
                f"ssh {self.host} failed; submission outcome is unknown. The request may have "
                "reached the broker. Check csub status and bjobs -a before resubmitting."
            )
        else:
            head = (
                f"ssh {self.host} failed (exit status 255); the request may have reached "
                "the broker."
            )
        try:
            cp = subprocess.run(
                self._argv(diagnose=True),
                input=encode({"protocol": PROTOCOL_VERSION, "op": "probe"}),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            return f"{head} Diagnostic probe did not answer within {self.timeout_s}s."
        if cp.returncode == 0:
            return f"{head} Diagnostic probe succeeded: the broker is reachable now."
        tail = "\n".join(cp.stderr.strip().splitlines()[-5:]) or f"exit status {cp.returncode}"
        return f"{head} Diagnostic probe failed: {tail}"

    def call(self, request: dict[str, Any]) -> dict[str, Any]:
        os.makedirs(self.control_dir, mode=0o700, exist_ok=True)
        try:
            cp = subprocess.run(
                self.argv,
                input=encode(request),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except FileNotFoundError:
            raise TransportError("ssh is not installed") from None
        except subprocess.TimeoutExpired:
            raise TransportError(
                f"ssh to {self.host} did not answer within {self.timeout_s}s"
            ) from None
        if cp.returncode == 255 and not cp.stderr.strip():
            raise TransportError(self._diagnose(request))
        if cp.returncode not in (0, 1) and "command not found" in cp.stderr:
            raise TransportError(
                f"ssh {self.host}: {cp.stderr.strip().splitlines()[-1]}. The key is not bound to "
                "a forced command (restrict,command=... in authorized_keys), so the remote shell "
                "ran the command literally. Either set that up, or point CSUB_BROKER_CMD at the "
                "absolute path (e.g. ~/.local/bin/csub-broker); on the submit host itself use "
                "CSUB_TRANSPORT=local."
            )
        return finish(cp.stdout, cp.returncode, cp.stderr, f"ssh {self.host}")
