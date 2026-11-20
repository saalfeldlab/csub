"""Read job output from the shared filesystem (no broker round trip)."""

from __future__ import annotations

import os
from pathlib import Path

from csub.protocol import CsubError


def job_dir(cwd: str, job_id: str, index: int | None = None) -> Path:
    """``index``: the element of an array job, whose output is one directory further down."""
    d = Path(cwd) / ".csub" / "jobs" / str(job_id)
    return d / str(int(index)) if index else d


def tail_file(path: Path, lines: int | None) -> str:
    """Last ``lines`` lines of ``path`` ('' if missing). ``None`` returns everything."""
    try:
        if lines is None:
            return path.read_text(errors="replace")
        if lines <= 0:
            return ""
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 65536
            data = b""
            while size > 0 and data.count(b"\n") <= lines:
                step = min(block, size)
                size -= step
                f.seek(size)
                data = f.read(step) + data
            text = data.decode(errors="replace")
            return "\n".join(text.splitlines()[-lines:]) + ("\n" if text.endswith("\n") else "")
    except FileNotFoundError:
        return ""


def read_logs(
    cwd: str,
    job_id: str,
    *,
    stream: str = "both",
    tail: int | None = None,
    index: int | None = None,
) -> dict[str, str]:
    d = job_dir(cwd, job_id, index)
    if not d.is_dir():
        what = f"element {index} of job {job_id}" if index else f"job {job_id}"
        raise CsubError("not_found", f"no output directory for {what} at {d}")
    if not index and not (d / "stdout").exists() and any(e.name.isdigit() for e in d.iterdir()):
        raise CsubError(
            "invalid_request", f"job {job_id} is an array job: say which element (index)"
        )
    out: dict[str, str] = {"job_dir": str(d)}
    if stream in ("stdout", "both"):
        out["stdout"] = tail_file(d / "stdout", tail)
    if stream in ("stderr", "both"):
        out["stderr"] = tail_file(d / "stderr", tail)
    return out
