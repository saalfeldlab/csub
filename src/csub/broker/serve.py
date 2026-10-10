"""Accept one request per connection on a Unix socket.

Used by the per-job broker (``csub-broker --serve``, started by the job wrapper outside the
sandbox; the socket file is bind-mounted into the container) and by ``csub relay``. Each
connection sends one JSON request and half-closes; we answer and close.
Requests are handled sequentially — a job's own submissions are rarely concurrent and the
handler re-reads the policy per request, so there is nothing to share.
"""

from __future__ import annotations

import os
import signal
import socket
from collections.abc import Callable
from typing import TextIO

from csub.protocol import LIMITS

READ_TIMEOUT_S = 120.0


def _listening(path: str) -> bool:
    """True if a process accepts connections on the socket at path."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
        try:
            probe.connect(path)
        except OSError:  # missing, or a stale file nobody listens on
            return False
    return True


def serve(
    path: str,
    handler: Callable[[bytes], dict],
    *,
    stderr: TextIO,
    ready: Callable[[], None] | None = None,
) -> int:
    import json

    if _listening(path):
        stderr.write(f"csub-broker --serve: a broker is already serving {path}\n")
        return 1
    try:
        os.unlink(path)  # stale socket from a broker that died
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    os.chmod(path, 0o600)
    srv.listen(16)
    stop = False

    def _stop(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    if ready:
        ready()
    srv.settimeout(0.5)
    try:
        while not stop:
            try:
                conn, _ = srv.accept()
            except socket.timeout:  # TimeoutError on 3.10+, socket.timeout on 3.9
                continue
            except OSError:
                if stop:
                    break
                raise
            with conn:
                try:
                    conn.settimeout(READ_TIMEOUT_S)
                    chunks: list[bytes] = []
                    total = 0
                    while True:
                        chunk = conn.recv(65536)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        total += len(chunk)
                        if total > LIMITS["max_request_bytes"]:
                            break
                    resp = handler(b"".join(chunks))
                    conn.sendall(json.dumps(resp, separators=(",", ":")).encode() + b"\n")
                except OSError as e:
                    stderr.write(f"csub-broker --serve: connection error: {e}\n")
    finally:
        srv.close()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    return 0
