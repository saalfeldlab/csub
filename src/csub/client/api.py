"""The Python API. Everything else (CLI, MCP) is a thin layer over :class:`Client`."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from csub.client import logs as _logs
from csub.client.config import (
    ClientConfig,
    ConfigError,
    derive_session,
    load_client_config,
    make_transport,
)
from csub.protocol import (
    CsubError,
    JobSpec,
    JobStatus,
    KillResult,
    ProbeResult,
    SubmitResult,
    raise_for_response,
)
from csub.transport import Transport


@dataclass(frozen=True)
class Logs:
    job_id: str
    job_dir: str
    stdout: str | None = None
    stderr: str | None = None


@dataclass(frozen=True)
class WaitEntry:
    status: JobStatus
    stdout_tail: str = ""
    stderr_tail: str = ""
    stdout_path: str | None = None
    stderr_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.status.to_dict(),
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
        }


@dataclass(frozen=True)
class WaitResult:
    jobs: list[WaitEntry]
    timed_out: bool
    elapsed_s: float

    @property
    def all_done(self) -> bool:
        return not self.timed_out and all(e.status.state == "DONE" for e in self.jobs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "jobs": [e.to_dict() for e in self.jobs],
            "timed_out": self.timed_out,
            "elapsed_s": round(self.elapsed_s, 1),
        }


@dataclass(frozen=True)
class SelfJob:
    """The job the current process runs in (from the CSUB_* variables the wrapper sets)."""

    job_id: str
    session: str | None
    walltime_min: int | None
    deadline_epoch: int | None
    image: str | None
    mounts: str | None

    def seconds_left(self, now: float | None = None) -> float | None:
        if self.deadline_epoch is None:
            return None
        return self.deadline_epoch - (time.time() if now is None else now)


def self_job(environ: Mapping[str, str] | None = None) -> SelfJob | None:
    env = os.environ if environ is None else environ
    job_id = env.get("CSUB_JOB_ID")
    if not job_id:
        return None

    def _int(key: str) -> int | None:
        v = env.get(key)
        return int(v) if v and v.isdigit() else None

    return SelfJob(
        job_id=job_id,
        session=env.get("CSUB_SESSION"),
        walltime_min=_int("CSUB_WALLTIME_MIN"),
        deadline_epoch=_int("CSUB_DEADLINE_EPOCH"),
        image=env.get("CSUB_IMAGE"),
        mounts=env.get("CSUB_MOUNTS"),
    )


BACKOFF = (1.0, 2.0, 4.0, 8.0, 15.0, 30.0)


class Client:
    def __init__(self, config: ClientConfig | None = None, transport: Transport | None = None):
        self.config = config or load_client_config()
        self.transport = transport or make_transport(self.config)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Client:
        return cls(load_client_config(environ))

    # --- plumbing ---

    def _call(self, request: dict[str, Any]) -> dict[str, Any]:
        return raise_for_response(self.transport.call({"protocol": 1, **request}))

    def _session(self) -> str:
        if self.config.session:
            return self.config.session
        if self.config.mounts:
            return derive_session(self.config.mounts)
        raise ConfigError("no session: set CSUB_SESSION or CSUB_MOUNTS")

    def complete(self, spec: JobSpec, *, cwd: str | None = None) -> JobSpec:
        """Fill cwd / mounts / session / image from the environment where the spec left them out."""
        changes: dict[str, Any] = {}
        if spec.cwd is None:
            changes["cwd"] = os.path.realpath(cwd or os.getcwd())
        if not spec.mounts:
            if not self.config.mounts:
                raise ConfigError(
                    "no mounts: CSUB_MOUNTS must list the sandbox's bind mounts, or without a "
                    "sandbox the directories jobs may use "
                    '(e.g. "/groups/lab/project:rw,/nrs/lab/data:ro")'
                )
            changes["mounts"] = self.config.mounts
        if spec.session is None:
            changes["session"] = self._session()
        if spec.image is None and self.config.image:
            changes["image"] = self.config.image
        return spec.replace(**changes) if changes else spec

    # --- operations ---

    def submit(
        self,
        spec: JobSpec | Mapping[str, Any] | None = None,
        *,
        cwd: str | None = None,
        **fields: Any,
    ) -> SubmitResult:
        if spec is None:
            spec = JobSpec.from_dict(dict(fields))
        elif isinstance(spec, Mapping):
            spec = JobSpec.from_dict({**spec, **fields})
        elif fields:
            spec = spec.replace(**fields)
        spec = self.complete(spec, cwd=cwd)
        resp = self._call({"op": "submit", "job": spec.to_dict()})
        return SubmitResult.from_dict(resp)

    def status(self, job_ids: Iterable[str] | None = None) -> list[JobStatus]:
        ids = [str(j) for j in job_ids] if job_ids is not None else []
        req: dict[str, Any] = {"op": "status"}
        if ids:
            req["job_ids"] = ids
        else:
            req["session"] = self._session()
        return [JobStatus.from_dict(e) for e in self._call(req)["jobs"]]

    def kill(self, job_ids: Iterable[str] | None = None) -> KillResult:
        ids = [str(j) for j in job_ids] if job_ids is not None else []
        req: dict[str, Any] = {"op": "kill"}
        if ids:
            req["job_ids"] = ids
        else:
            req["session"] = self._session()
        return KillResult.from_dict(self._call(req))

    def probe(self) -> ProbeResult:
        return ProbeResult.from_dict(self._call({"op": "probe"}))

    def logs(
        self,
        job_id: str,
        *,
        stream: str = "both",
        tail: int | None = None,
        cwd: str | None = None,
        index: int | None = None,
    ) -> Logs:
        """``index``: for an array job, the element whose output to read."""
        job_id = str(job_id)
        if cwd is None:
            cwd = self.status([job_id])[0].cwd
            if cwd is None:
                raise CsubError("not_found", f"job {job_id} has no recorded working directory")
        data = _logs.read_logs(cwd, job_id, stream=stream, tail=tail, index=index)
        return Logs(
            job_id=job_id,
            job_dir=data["job_dir"],
            stdout=data.get("stdout"),
            stderr=data.get("stderr"),
        )

    def wait(
        self,
        job_ids: Iterable[str],
        *,
        timeout_s: float | None = None,
        tail_lines: int = 50,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> WaitResult:
        """Poll until every job is terminal; never raises on timeout (``timed_out`` is set)."""
        ids = [str(j) for j in job_ids]
        start = clock()
        attempt = 0
        while True:
            statuses = self.status(ids)
            if all(s.terminal for s in statuses):
                timed_out = False
                break
            elapsed = clock() - start
            if timeout_s is not None and elapsed >= timeout_s:
                timed_out = True
                break
            delay = min(BACKOFF[min(attempt, len(BACKOFF) - 1)], self.config.wait_max_poll_s)
            if timeout_s is not None:
                delay = min(delay, max(0.0, timeout_s - elapsed))
            sleep(delay)
            attempt += 1
        entries = []
        for s in statuses:
            out = err = ""
            out_path = err_path = None
            if s.cwd and s.array:
                # An array has no single output: show the first failed element's, if any.
                failed = s.array.get("failed") or []
                if failed:
                    d = _logs.job_dir(s.cwd, s.job_id, failed[0])
                    out_path, err_path = str(d / "stdout"), str(d / "stderr")
                    out = _logs.tail_file(d / "stdout", tail_lines)
                    err = _logs.tail_file(d / "stderr", tail_lines)
            elif s.cwd:
                d = _logs.job_dir(s.cwd, s.job_id)
                out_path, err_path = str(d / "stdout"), str(d / "stderr")
                out = _logs.tail_file(d / "stdout", tail_lines)
                err = _logs.tail_file(d / "stderr", tail_lines)
            entries.append(WaitEntry(s, out, err, out_path, err_path))
        return WaitResult(jobs=entries, timed_out=timed_out, elapsed_s=clock() - start)


# --- module-level convenience (lazy default client) --------------------------------------

_default: Client | None = None


def default_client() -> Client:
    global _default
    if _default is None:
        _default = Client.from_env()
    return _default


def submit(spec: JobSpec | Mapping[str, Any] | None = None, **fields: Any) -> SubmitResult:
    return default_client().submit(spec, **fields)


def status(job_ids: Iterable[str] | None = None) -> list[JobStatus]:
    return default_client().status(job_ids)


def wait(job_ids: Iterable[str], **kw: Any) -> WaitResult:
    return default_client().wait(job_ids, **kw)


def kill(job_ids: Iterable[str] | None = None) -> KillResult:
    return default_client().kill(job_ids)


def logs(job_id: str, **kw: Any) -> Logs:
    return default_client().logs(job_id, **kw)


def probe() -> ProbeResult:
    return default_client().probe()
