"""The broker's operations: submit / status / kill / probe."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import csub
from csub import __version__
from csub.broker.lsf import LsfRunner
from csub.broker.policy import Policy
from csub.broker.resolve import ResolveContext, ResolveError, resolve
from csub.broker.state import META_NAME, REQUEST_NAME, WRAPPER_NAME, JobRecord, StateDir
from csub.broker.wrapper import check_no_directives, render_wrapper
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

    def resolve_context(self) -> ResolveContext:
        return ResolveContext(
            user=self.user,
            home=os.path.realpath(self.home),
            protected_dirs=self.protected_dirs,
            lookup_job=self.state.lookup,
        )


def client_source_dir() -> str:
    """The installed csub package, mounted read-only into jobs so any image has the client."""
    return os.path.realpath(os.path.dirname(os.path.abspath(csub.__file__)))


def _session_group(ctx: BrokerContext, session: str) -> str:
    return f"/csub/{ctx.user}/{session}"


def op_probe(req: Request, ctx: BrokerContext) -> dict:
    return ok_response(
        broker_version=__version__,
        protocol=PROTOCOL_VERSION,
        user=ctx.user,
        **ctx.policy.probe_summary(),
    )


def op_submit(req: Request, ctx: BrokerContext) -> dict:
    assert req.job is not None
    job = resolve(req.job, ctx.policy, ctx.resolve_context())

    cap = ctx.policy.limits.max_pending_per_session
    if cap:
        pending = sum(1 for r in ctx.lsf.bjobs(job.job_group) if r.state == "PEND")
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
    )
    check_no_directives(wrapper)
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
    """Explicit ids must all be jobs this broker submitted for this user; else not_found."""
    assert req.job_ids
    records: dict[str, JobRecord] = {}
    missing: list[str] = []
    for jid in req.job_ids:
        rec = ctx.state.record(jid)
        if rec is None or rec.user != ctx.user:
            missing.append(jid)
        else:
            records[jid] = rec
    if missing:
        raise CsubError("not_found", "not csub jobs of this user: " + ", ".join(missing))
    return records


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
    else:
        exit_code = ctx.state.read_exit_code(job_id)
        if exit_code is None:
            state, source = "UNKNOWN", "record"
        else:
            state, source = ("DONE" if exit_code == 0 else "EXIT"), "exit_code"
        queue = rec.queue if rec else None
        exec_host = None
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
    }


def op_status(req: Request, ctx: BrokerContext) -> dict:
    entries: list[dict] = []
    if req.job_ids:
        records = _records_for(req, ctx)
        by_session: dict[str, list[str]] = {}
        for jid, rec in records.items():
            by_session.setdefault(rec.session, []).append(jid)
        rows = {}
        for session, ids in by_session.items():
            for row in ctx.lsf.bjobs(_session_group(ctx, session), ids):
                rows[row.job_id] = row
        for jid, rec in records.items():
            entries.append(_status_entry(rec, rows.get(jid), ctx, jid))
    else:
        assert req.session is not None
        rows = {r.job_id: r for r in ctx.lsf.bjobs(_session_group(ctx, req.session))}
        records = {r.job_id: r for r in ctx.state.list_jobs(ctx.user, req.session)}
        for jid in set(rows) | set(records):
            entries.append(_status_entry(records.get(jid), rows.get(jid), ctx, jid))
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
    return OPS[req.op](req, ctx)
