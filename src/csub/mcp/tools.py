"""Tool definitions: schema (from protocol.py), agent-steering description, handler."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from csub.client.api import Client
from csub.protocol import (
    KILL_ARGS_SCHEMA,
    LOGS_ARGS_SCHEMA,
    PROBE_ARGS_SCHEMA,
    STATUS_ARGS_SCHEMA,
    WAIT_ARGS_SCHEMA,
    CsubError,
    JobSpec,
    jobspec_schema,
)

INSTRUCTIONS = """\
csub submits jobs to the LSF compute cluster. Every job runs in a sandbox with your bind
mounts at the same paths and no network unless allow_hosts says otherwise; its image is
CSUB_IMAGE if set (inside a csub job: that job's image), else the policy default.
Workflow: csub_probe once (queues, limits, prices) -> csub_submit -> csub_wait -> csub_logs.
Size jobs in cpus / mem_mb / walltime_min; the broker turns that into LSF slots and picks
the queue. Job output is in <cwd>/.csub/jobs/<job_id>/{stdout,stderr}; where the cluster
filesystem is mounted you can read it directly too.
"""


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[Client, dict[str, Any]], dict[str, Any]]
    read_only: bool = False
    destructive: bool = False
    idempotent: bool = False


def _probe(client: Client, args: dict[str, Any]) -> dict[str, Any]:
    return client.probe().to_dict()


def _submit(client: Client, args: dict[str, Any]) -> dict[str, Any]:
    return client.submit(JobSpec.from_dict(args)).to_dict()


def _status(client: Client, args: dict[str, Any]) -> dict[str, Any]:
    return {"jobs": [s.to_dict() for s in client.status(args.get("job_ids"))]}


def _wait(client: Client, args: dict[str, Any]) -> dict[str, Any]:
    return client.wait(
        args["job_ids"],
        timeout_s=args.get("timeout_s", 600),
        tail_lines=args.get("tail_lines", 50),
    ).to_dict()


def _kill(client: Client, args: dict[str, Any]) -> dict[str, Any]:
    ids = args.get("job_ids")
    if not ids and not args.get("all"):
        raise CsubError("invalid_request", "give job_ids or all=true")
    return client.kill(ids or None).to_dict()


def _logs(client: Client, args: dict[str, Any]) -> dict[str, Any]:
    lg = client.logs(
        args["job_id"], stream=args.get("stream", "both"), tail=args.get("tail_lines", 200)
    )
    return {"job_id": lg.job_id, "job_dir": lg.job_dir, "stdout": lg.stdout, "stderr": lg.stderr}


TOOLS: dict[str, ToolDef] = {
    t.name: t
    for t in [
        ToolDef(
            "csub_probe",
            "Call this first. Returns what the cluster broker allows: queues with memory per "
            "slot, slots per GPU, maximum walltime and GPU prices; limits (slots, GPUs, cost cap); "
            "the default and allowed container images; hosts that may be reached from a job.",
            PROBE_ARGS_SCHEMA,
            _probe,
            read_only=True,
            idempotent=True,
        ),
        ToolDef(
            "csub_submit",
            "Submit a job that runs the given command inside the cluster sandbox (same image and "
            "paths as here). Size it in cpus, mem_mb and walltime_min - never in slots or bsub "
            "flags; the broker converts memory into slot-sized chunks and picks the queue. Prefer "
            "walltime_min <= 60: those jobs schedule fastest and cost least; failed jobs are still "
            "billed. GPU jobs must set gpus >= 1 and a GPU queue from csub_probe (gpu_short for "
            "<= 1 h). There is no network unless allow_hosts lists hosts from probe.allowed_hosts. "
            "Returns the job_id and the resolved request including estimated_max_cost_usd. "
            "Output appears in <cwd>/.csub/jobs/<job_id>/{stdout,stderr}. From inside a job you "
            "can submit further jobs within this job's mounts, hosts and image; depends_on with "
            "when='ended' lets a job queue its successor.",
            jobspec_schema(exclude=("mounts", "session")),
            _submit,
        ),
        ToolDef(
            "csub_status",
            "States of jobs (PEND, RUN, DONE, EXIT with exit_code, SUSP, UNKNOWN). Omit job_ids "
            "for every job of this session. Prefer csub_wait over polling this.",
            STATUS_ARGS_SCHEMA,
            _status,
            read_only=True,
        ),
        ToolDef(
            "csub_wait",
            "Block until the given jobs finish (or timeout_s passes), then return their final "
            "states plus the last lines of stdout/stderr. On timeout returns timed_out=true (not "
            "an error): call again if you want to keep waiting.",
            WAIT_ARGS_SCHEMA,
            _wait,
            read_only=True,
            idempotent=True,
        ),
        ToolDef(
            "csub_kill",
            "Kill running or pending jobs by id, or all jobs of this session with all=true.",
            KILL_ARGS_SCHEMA,
            _kill,
            destructive=True,
        ),
        ToolDef(
            "csub_logs",
            "Read a job's stdout/stderr (last tail_lines lines) from the shared filesystem.",
            LOGS_ARGS_SCHEMA,
            _logs,
            read_only=True,
            idempotent=True,
        ),
    ]
}
