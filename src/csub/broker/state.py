"""Broker-owned on-disk state: ``<state_dir>/jobs/<id>/`` and ``broker.log``.

Nothing here is ever reachable from inside a sandbox (the state dir lives under $HOME or
outside every allowed root), so plain file operations are safe. Files are created 0600.
"""

from __future__ import annotations

import json
import os
import secrets
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

META_NAME = "meta.json"
REQUEST_NAME = "request.json"
WRAPPER_NAME = "wrapper.sh"
EXIT_CODE_NAME = "exit_code"


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    user: str
    session: str
    name: str
    queue: str
    cwd: str
    image: str
    submitted_at: float
    array_size: int = 0  # elements of an array job; 0 for an ordinary job

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StateDir:
    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(os.path.realpath(os.fspath(root)))
        self.jobs_dir = self.root / "jobs"
        self.pending_dir = self.jobs_dir / ".pending"
        self.log_path = self.root / "broker.log"
        for d in (self.root, self.jobs_dir, self.pending_dir):
            d.mkdir(mode=0o700, parents=True, exist_ok=True)

    # --- submission lifecycle ---

    def new_pending(self) -> tuple[str, Path]:
        token = secrets.token_hex(8)
        d = self.pending_dir / token
        d.mkdir(mode=0o700)
        return token, d

    @staticmethod
    def write_private(path: Path, text: str) -> None:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(text)

    def finalize(self, token: str, job_id: str) -> Path:
        """Move the pending files into jobs/<id>/ (which the wrapper may already have made)."""
        src = self.pending_dir / token
        dest = self.job_dir(job_id)
        dest.mkdir(mode=0o700, exist_ok=True)
        for entry in src.iterdir():
            os.replace(entry, dest / entry.name)
        src.rmdir()
        return dest

    def discard_pending(self, token: str) -> None:
        d = self.pending_dir / token
        if d.exists():
            for entry in d.iterdir():
                entry.unlink()
            d.rmdir()

    # --- lookups ---

    def job_dir(self, job_id: str) -> Path:
        if not job_id.isdigit():
            raise ValueError(f"not a job id: {job_id!r}")
        return self.jobs_dir / job_id

    def record(self, job_id: str) -> JobRecord | None:
        try:
            data = json.loads((self.job_dir(job_id) / META_NAME).read_text())
        except (FileNotFoundError, ValueError):
            return None
        return JobRecord(**data)

    def lookup(self, job_id: str) -> tuple[str, str] | None:
        rec = self.record(job_id)
        return (rec.user, rec.session) if rec else None

    def read_exit_code(self, job_id: str, index: int = 0) -> int | None:
        """``index`` is the element of an array job (its files are in jobs/<id>/<index>/)."""
        d = self.job_dir(job_id)
        if index:
            d = d / str(index)
        try:
            text = (d / EXIT_CODE_NAME).read_text().strip()
            return int(text)
        except (FileNotFoundError, ValueError):
            return None

    def read_lsf_out(self, job_id: str, index: int = 0) -> str:
        """The tail of the job's LSF output file ('' if there is none), which ends with LSF's
        resource usage summary once the job has finished. ``index`` is the element of an array
        job (its file is lsf.<index>.out)."""
        name = f"lsf.{index}.out" if index else "lsf.out"
        try:
            with open(self.job_dir(job_id) / name, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 8192))
                return f.read().decode(errors="replace")
        except (FileNotFoundError, ValueError):
            return ""

    def list_jobs(self, user: str, session: str | None = None) -> list[JobRecord]:
        out: list[JobRecord] = []
        for entry in self.jobs_dir.iterdir():
            if not entry.name.isdigit():
                continue
            rec = self.record(entry.name)
            if rec and rec.user == user and (session is None or rec.session == session):
                out.append(rec)
        out.sort(key=lambda r: int(r.job_id))
        return out

    # --- diagnostics ---

    def log_exception(self, context: str) -> str:
        """Append the current traceback to broker.log; return a short reference id."""
        ref = time.strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(3)
        try:
            with open(self.log_path, "a") as f:
                os.chmod(self.log_path, 0o600)
                f.write(f"--- {ref} {context}\n{traceback.format_exc()}\n")
        except OSError:
            pass
        return ref
