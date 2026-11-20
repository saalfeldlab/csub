"""The broker's operations: submit / status / kill / probe."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import csub
from csub import __version__
from csub.broker.lsf import LsfRunner, parse_usage_summary
from csub.broker.policy import Policy
from csub.broker.resolve import Parent, ResolveContext, ResolveError, parent_probe, resolve
from csub.broker.state import META_NAME, REQUEST_NAME, WRAPPER_NAME, JobRecord, StateDir
from csub.broker.wrapper import render_wrapper
from csub.protocol import PROTOCOL_VERSION, CsubError, Request, ok_response


@dataclass(frozen=True)
class BrokerContext:
    policy: Policy
    user: str
    home: str
    state: StateDir
    lsf: LsfRunner
    protected_dirs: tuple[str, ...]
    broker_cmd: str  # shell-quoted command that starts this broker (for --serve in the wrapper)
    parent: Parent | None = None  # per-job broker: the job it serves; requests stay inside it

    def resolve_context(self) -> ResolveContext:
        return ResolveContext(
            user=self.user,
            home=os.path.realpath(self.home),
            protected_dirs=self.protected_dirs,
            lookup_job=self.state.lookup,
            is_array=lambda job_id: bool((rec := self.state.record(job_id)) and rec.array_size),
            parent=self.parent,
        )


def client_source_dir() -> str:
    """The installed csub package, mounted read-only into jobs so any image has the client."""
    return os.path.realpath(os.path.dirname(os.path.abspath(csub.__file__)))


def _session_group(ctx: BrokerContext, session: str) -> str:
    return f"/csub/{ctx.user}/{session}"


def op_probe(req: Request, ctx: BrokerContext) -> dict:
    summary = ctx.policy.probe_summary()
    if ctx.parent is not None:  # advertise what a child can actually get
        summary.update(parent_probe(ctx.parent, ctx.policy))
    return ok_response(
        broker_version=__version__, protocol=PROTOCOL_VERSION, user=ctx.user, **summary
    )


def op_submit(req: Request, ctx: BrokerContext) -> dict:
    assert req.job is not None
    job = resolve(req.job, ctx.policy, ctx.resolve_context())

    cap = ctx.policy.limits.max_pending_per_session
    if cap:
        # Jobs, not elements: an array with pending elements is one pending job. Its width is
        # bounded by limits.max_array_size.
        pending = len({r.job_id for r in ctx.lsf.bjobs(job.job_group) if r.state == "PEND"})
        if pending >= cap:
            raise ResolveError(
                "policy_violation",
                f"session already has {pending} pending jobs (limit {cap}); wait for some to start",
            )

    wrapper = render_wrapper(
        job,
        state_dir=str(ctx.state.root),
        scripts_dir=ctx.policy.sandbox_scripts_dir,
        broker_cmd=ctx.broker_cmd,
        home=ctx.home,
        client_src=client_source_dir() if ctx.policy.inject_client else None,
        record_timeout_s=ctx.policy.lsf.timeout_s,
        umask=ctx.policy.lsf.umask,
    )
    token, pdir = ctx.state.new_pending()
    try:
        ctx.state.write_private(
            pdir / REQUEST_NAME,
            json.dumps({"spec": req.job.to_dict(), "resolved": job.to_response()}, indent=1),
        )
        ctx.state.write_private(pdir / WRAPPER_NAME, wrapper)
        result = ctx.lsf.bsub(job.bsub_args(str(ctx.state.root)), wrapper)
    except BaseException:
        ctx.state.discard_pending(token)
        raise
    dest = ctx.state.finalize(token, result.job_id)
    record = JobRecord(
        job_id=result.job_id,
        user=ctx.user,
        session=job.session,
        name=job.name,
        queue=result.queue,
        cwd=job.cwd,
        image=job.image,
        submitted_at=time.time(),
        array_size=job.array or 0,
    )
    ctx.state.write_private(dest / META_NAME, json.dumps(record.to_dict()))
    return ok_response(
        job_id=result.job_id,
        billing_group=result.billing_group,
        job_dir=f"{job.cwd}/.csub/jobs/{result.job_id}",
        broker_job_dir=str(dest),
        **{**job.to_response(), "queue": result.queue},
    )


def _records_for(req: Request, ctx: BrokerContext) -> dict[str, JobRecord]:
    """Explicit ids must all be jobs this broker submitted for this user (and, in a per-job
    broker, of the parent's session); else not_found."""
    assert req.job_ids
    records: dict[str, JobRecord] = {}
    missing: list[str] = []
    for jid in req.job_ids:
        rec = ctx.state.record(jid)
        if (
            rec is None
            or rec.user != ctx.user
            or (
                ctx.parent is not None
                and ctx.parent.session is not None
                and rec.session != ctx.parent.session
            )
        ):
            missing.append(jid)
        else:
            records[jid] = rec
    if missing:
        raise CsubError("not_found", "not csub jobs of this user: " + ", ".join(missing))
    return records


FAILED_INDICES_SHOWN = 100


def _array_entry(rec: JobRecord | None, rows: list, ctx: BrokerContext, job_id: str) -> dict:
    """One entry for a whole array job: the state of the array, counts per state, and which
    elements failed. An element bjobs no longer lists is read from its exit_code file.

    What the array used is one set of figures: the highest peak memory and the longest run time
    of any element, the CPU time of all of them together, and the memory each was given. There
    is no average memory for an array.

    ``rec`` is None in the moment between bsub accepting the array and the broker writing its
    record: the array is then known from bjobs alone, which lists every element."""
    by_index = {r.index: r for r in rows if r.index}
    counts: dict[str, int] = {}
    failed: list[int] = []
    worst = 0
    used: list[dict] = []
    queue = rec.queue if rec else None
    size = rec.array_size if rec else max(by_index)
    for i in range(1, size + 1):
        row = by_index.get(i)
        code: int | None
        if row is not None:
            state = row.state
            code = 0 if state == "DONE" else row.exit_code
            if state == "EXIT" and code is None:
                code = ctx.state.read_exit_code(job_id, i)
            queue = row.queue or queue
            used.append(row.usage())
        else:
            code = ctx.state.read_exit_code(job_id, i)
            state = "UNKNOWN" if code is None else ("DONE" if code == 0 else "EXIT")
            if code is not None:
                used.append(parse_usage_summary(ctx.state.read_lsf_out(job_id, i)))
        counts[state] = counts.get(state, 0) + 1
        if state == "EXIT":
            failed.append(i)
            worst = max(worst, code or 1)
    if any(counts.get(s) for s in ("RUN", "PEND", "SUSP")):
        # Still going. RUN as soon as anything runs; otherwise what the waiting elements are.
        state = next(s for s in ("RUN", "PEND", "SUSP") if counts.get(s))
        exit_code = None
    elif counts.get("EXIT"):
        state, exit_code = "EXIT", worst
    elif counts.get("UNKNOWN"):
        state, exit_code = "UNKNOWN", None
    else:
        state, exit_code = "DONE", 0
    return {
        "job_id": job_id,
        "state": state,
        "exit_code": exit_code,
        "queue": queue,
        "exec_host": None,
        # bjobs names an element "<job name>[<index>]"
        "name": rec.name if rec else (rows[0].job_name or "").partition("[")[0] or None,
        "cwd": rec.cwd if rec else None,
        "session": rec.session if rec else None,
        "source": "bjobs" if by_index else "exit_code",
        "array": {
            "size": size,
            "counts": counts,
            "failed": failed[:FAILED_INDICES_SHOWN],
        },
        **_array_usage(used),
    }


def _array_usage(used: list[dict]) -> dict[str, float | None]:
    def over(field: str, fn) -> float | None:
        values = [u[field] for u in used if u.get(field) is not None]
        return round(fn(values), 1) if values else None

    return {
        "cpu_time_s": over("cpu_time_s", sum),
        "max_mem_mb": over("max_mem_mb", max),
        "avg_mem_mb": None,
        "run_time_s": over("run_time_s", max),
        "mem_limit_mb": over("mem_limit_mb", max),
    }


def _entry(rec: JobRecord | None, rows: list, ctx: BrokerContext, job_id: str) -> dict:
    if (rec is not None and rec.array_size) or (rec is None and any(r.index for r in rows)):
        return _array_entry(rec, rows, ctx, job_id)
    return _status_entry(rec, rows[-1] if rows else None, ctx, job_id)


def _rows_by_job(rows) -> dict[str, list]:
    out: dict[str, list] = {}
    for row in rows:
        out.setdefault(row.job_id, []).append(row)
    return out


def _status_entry(rec: JobRecord | None, row, ctx: BrokerContext, job_id: str) -> dict:
    if row is not None:
        state = row.state
        if state == "EXIT":
            exit_code = (
                row.exit_code if row.exit_code is not None else ctx.state.read_exit_code(job_id)
            )
        elif state == "DONE":
            exit_code = 0
        else:
            exit_code = None
        source = "bjobs"
        queue, exec_host = row.queue, row.exec_host
        usage = row.usage()
    else:
        exit_code = ctx.state.read_exit_code(job_id)
        if exit_code is None:
            state, source = "UNKNOWN", "record"
        else:
            state, source = ("DONE" if exit_code == 0 else "EXIT"), "exit_code"
        queue = rec.queue if rec else None
        exec_host = None
        # bjobs has forgotten the job: LSF's own summary in the job's output file remains.
        usage = parse_usage_summary(ctx.state.read_lsf_out(job_id))
    return {
        "job_id": job_id,
        "state": state,
        "exit_code": exit_code,
        "queue": queue,
        "exec_host": exec_host,
        "name": rec.name if rec else (row.job_name if row else None),
        "cwd": rec.cwd if rec else None,
        "session": rec.session if rec else None,
        "source": source,
        **usage,
    }


def op_status(req: Request, ctx: BrokerContext) -> dict:
    entries: list[dict] = []
    if req.job_ids:
        records = _records_for(req, ctx)
        by_session: dict[str, list[str]] = {}
        for jid, rec in records.items():
            by_session.setdefault(rec.session, []).append(jid)
        found: list = []
        for session, ids in by_session.items():
            found += ctx.lsf.bjobs(_session_group(ctx, session), ids)
        rows = _rows_by_job(found)
        for jid, rec in records.items():
            entries.append(_entry(rec, rows.get(jid, []), ctx, jid))
    else:
        assert req.session is not None
        rows = _rows_by_job(ctx.lsf.bjobs(_session_group(ctx, req.session)))
        records = {r.job_id: r for r in ctx.state.list_jobs(ctx.user, req.session)}
        for jid in set(rows) | set(records):
            entries.append(_entry(records.get(jid), rows.get(jid, []), ctx, jid))
    entries.sort(key=lambda e: int(e["job_id"]))
    return ok_response(jobs=entries)


def op_kill(req: Request, ctx: BrokerContext) -> dict:
    killed: list[str] = []
    finished: list[str] = []
    if req.job_ids:
        records = _records_for(req, ctx)
        by_session: dict[str, list[str]] = {}
        for jid, rec in records.items():
            by_session.setdefault(rec.session, []).append(jid)
        for session, ids in by_session.items():
            result = ctx.lsf.bkill(_session_group(ctx, session), ids)
            for jid in ids:
                (killed if result.get(jid) == "killed" else finished).append(jid)
    else:
        assert req.session is not None
        result = ctx.lsf.bkill(_session_group(ctx, req.session), None)
        killed = [j for j, v in result.items() if v == "killed"]
        finished = [j for j, v in result.items() if v != "killed"]
    return ok_response(killed=sorted(killed, key=int), already_finished=sorted(finished, key=int))


OPS = {"submit": op_submit, "status": op_status, "kill": op_kill, "probe": op_probe}


def handle(req: Request, ctx: BrokerContext) -> dict:
    if (
        ctx.parent is not None
        and ctx.parent.session is not None
        and req.session not in (None, ctx.parent.session)
    ):
        raise CsubError(
            "policy_violation", f"session {req.session!r}: child jobs stay in the parent's session"
        )
    return OPS[req.op](req, ctx)
