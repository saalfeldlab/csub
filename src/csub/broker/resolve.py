"""Turn a JobSpec into a ResolvedJob under a Policy, or reject it.

Pure: filesystem probes and job lookups are injected through ``ResolveContext`` so every
rule is unit-testable. Every rejection carries a protocol error code:

* ``invalid_request``   - the request cannot mean anything sensible
* ``policy_violation``  - the request is meaningful but the policy forbids it
* ``not_found``         - a dependency does not exist in this user's session
* ``broker_misconfigured`` - a broker-owned path would be exposed to the agent
"""

from __future__ import annotations

import fnmatch
import math
import os
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from csub._paths import glob_to_regex, image_registry, is_under
from csub.broker.policy import Policy, QueuePolicy
from csub.protocol import CsubError, Dependency, JobSpec, contains_bsub_directive

RESERVED_ENV_KEYS = frozenset(
    {"HOME", "PATH", "TMPDIR", "USER", "LOGNAME", "SHELL", "CUDA_VISIBLE_DEVICES"}
)
RESERVED_ENV_PREFIXES = ("LSB_", "LSF_", "CSUB_")

# What Claude Code needs to reach; added to a job's allow_hosts when it asks for claude.
CLAUDE_HOSTS = ("api.anthropic.com", "claude.ai", "platform.claude.com")

# Flags the broker emits itself; passthrough of these would let the agent override policy.
OWNED_BSUB_FLAGS = frozenset(
    {
        "-q",
        "-n",
        "-W",
        "-gpu",
        "-J",
        "-g",
        "-o",
        "-oo",
        "-e",
        "-eo",
        "-w",
        "-cwd",
        "-i",
        "-is",
        "-env",
    }
)
# Flags whose semantics csub does not support at all.
UNSUPPORTED_BSUB_FLAGS = frozenset(
    {"-I", "-Ip", "-Is", "-IS", "-ISp", "-ISs", "-IX", "-XF", "-K", "-app", "-m", "-tty", "-env"}
)

_NAME_BAD_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
_NAME_SPLIT = re.compile(r"[_.-]+")


class ResolveError(CsubError):
    pass


def _invalid(msg: str) -> ResolveError:
    return ResolveError("invalid_request", msg)


def _policy(msg: str) -> ResolveError:
    return ResolveError("policy_violation", msg)


@dataclass(frozen=True)
class ResolveContext:
    user: str
    home: str  # realpath'd
    protected_dirs: tuple[str, ...] = ()  # realpaths the agent must never be able to write
    realpath: Callable[[str], str] = os.path.realpath
    isdir: Callable[[str], bool] = os.path.isdir
    # job_id -> (user, session) of a job this broker submitted, or None
    lookup_job: Callable[[str], tuple[str, str] | None] = lambda _job_id: None


@dataclass(frozen=True)
class ResolvedMount:
    path: str
    mode: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "mode": self.mode}


@dataclass(frozen=True)
class ResolvedJob:
    user: str
    session: str
    job_group: str
    job_name: str
    name: str
    queue: str
    slots: int
    walltime_min: int
    gpus: int
    gpu_mem_gb: int | None
    image: str
    mounts: tuple[ResolvedMount, ...]
    cwd: str
    command: tuple[str, ...] | str
    shell: bool
    env: dict[str, str]
    allow_hosts: tuple[str, ...]
    scratch: bool
    claude: bool
    scratch_root: str
    keep_id: bool
    sandbox: str  # "podman" or "bwrap"
    depends_on: tuple[Dependency, ...]
    lsf_extra_argv: tuple[str, ...]
    estimated_max_cost_usd: float

    def bsub_args(self, state_dir: str) -> list[str]:
        args = ["-q", self.queue, "-n", str(self.slots), "-W", str(self.walltime_min)]
        if self.gpus:
            gpu = f"num={self.gpus}"
            if self.gpu_mem_gb:
                gpu += f":gmem={self.gpu_mem_gb}G"
            args += ["-gpu", gpu]
        args += [
            "-J", self.job_name,
            "-g", self.job_group,
            "-o", os.path.join(state_dir, "jobs", "%J", "lsf.out"),
            "-env", "none",
        ]  # fmt: skip
        if self.depends_on:
            args += ["-w", " && ".join(f"{d.when}({d.job_id})" for d in self.depends_on)]
        args += list(self.lsf_extra_argv)
        return args

    def to_response(self) -> dict[str, Any]:
        return {
            "queue": self.queue,
            "slots": self.slots,
            "walltime_min": self.walltime_min,
            "gpus": self.gpus,
            "gpu_mem_gb": self.gpu_mem_gb,
            "image": self.image,
            "sandbox": self.sandbox,
            "mounts": [m.to_dict() for m in self.mounts],
            "allow_hosts": list(self.allow_hosts),
            "scratch": self.scratch,
            "claude": self.claude,
            "name": self.name,
            "job_group": self.job_group,
            "session": self.session,
            "cwd": self.cwd,
            "depends_on": [d.to_dict() for d in self.depends_on],
            "estimated_max_cost_usd": ceil_cents(self.estimated_max_cost_usd),
        }


# --- individual rules ------------------------------------------------------------------


def ceil_cents(usd: float) -> float:
    """Round an estimated *maximum* up to the next cent (never show $0.00 for a real cost)."""
    return math.ceil(round(usd * 100, 6)) / 100


def compute_slots(cpus: int, mem_mb: int, mem_per_slot_mb: int) -> int:
    return max(cpus, math.ceil(mem_mb / mem_per_slot_mb))


def select_queue(spec: JobSpec, policy: Policy) -> tuple[QueuePolicy, int]:
    """Pick the queue and the effective walltime."""
    lim = policy.limits
    if spec.queue is not None:
        q = policy.queue(spec.queue)
        if q is None:
            raise _policy(
                f"queue {spec.queue!r} is unknown or not permitted; allowed: "
                + ", ".join(sorted(policy.queues))
            )
        if spec.gpus and not q.gpu:
            raise _policy(
                f"queue {q.name!r} has no GPUs; GPU queues: " + ", ".join(_gpu_queues(policy))
            )
        if not spec.gpus and q.gpu:
            raise _policy(f"queue {q.name!r} is a GPU queue; request gpus >= 1 or use a CPU queue")
    else:
        if spec.gpus:
            raise _invalid(
                "GPU jobs must name a queue; GPU queues: " + ", ".join(_gpu_queues(policy))
            )
        short = policy.queues[lim.cpu_short_queue]
        wt = spec.walltime_min if spec.walltime_min is not None else lim.default_cpu_walltime_min
        q = short if wt <= short.max_walltime_min else policy.queues[lim.cpu_default_queue]

    default = lim.default_gpu_walltime_min if q.gpu else lim.default_cpu_walltime_min
    walltime = (
        spec.walltime_min if spec.walltime_min is not None else min(default, q.max_walltime_min)
    )
    if walltime > q.max_walltime_min:
        raise _policy(
            f"walltime {walltime} min exceeds queue {q.name!r} maximum of {q.max_walltime_min} min"
        )
    if walltime > lim.max_walltime_min:
        raise _policy(
            f"walltime {walltime} min exceeds policy maximum of {lim.max_walltime_min} min"
        )
    return q, walltime


def _gpu_queues(policy: Policy) -> list[str]:
    return sorted(n for n, q in policy.queues.items() if q.gpu)


def estimate_cost(
    slots: int, gpus: int, walltime_min: int, queue: QueuePolicy, policy: Policy
) -> float:
    hours = walltime_min / 60.0
    return (
        slots * hours * policy.limits.slot_price_usd_per_hour
        + gpus * hours * queue.gpu_price_usd_per_hour
    )


def sanitize_name(raw: str | None, command: tuple[str, ...] | str, *, max_len: int) -> str:
    if raw is None or not raw.strip():
        raw = os.path.basename(command[0]) if isinstance(command, tuple) else "script"
    name = _NAME_BAD_CHARS.sub("_", raw.strip())
    name = re.sub(r"_+", "_", name).strip("_.-")
    if len(name) > max_len:
        name = name[:max_len].rstrip("_.-")
    return name or "job"


def check_name(name: str, policy: Policy, user: str) -> None:
    lowered = name.lower()
    if user and user.lower() in lowered:
        raise _policy(f"job name {name!r} must not contain the username")
    components = {c for c in _NAME_SPLIT.split(lowered) if c}
    bad = sorted(components & set(policy.forbidden_name_tokens))
    if bad:
        raise _policy(f"job name {name!r} contains forbidden token(s): {', '.join(bad)}")


def check_image(image: str | None, policy: Policy, *, sandbox: str = "podman") -> str:
    if sandbox == "bwrap":
        # bwrap runs on the node's own toolchain; there is no image to honour. The default
        # image is accepted as a no-op so a client.toml `image` does not break jobs.
        if image is not None and image != policy.default_image:
            raise _policy(
                f"image {image!r}: this queue runs jobs under bwrap on the node's own toolchain, "
                "which takes no image; omit image (see csub probe)"
            )
        return ""
    if image is None:
        return policy.default_image
    registry = image_registry(image)
    if registry is None:
        raise _invalid(f"image {image!r} must be registry-qualified (e.g. ghcr.io/org/name:tag)")
    if registry == "localhost":
        raise _policy(f"image {image!r}: localhost images are not visible on compute nodes")
    if image == policy.default_image:
        return image
    for pat in policy.allowed_images:
        if glob_to_regex(pat).fullmatch(image):
            return image
    raise _policy(
        f"image {image!r} is not in the allowed images: " + ", ".join(policy.allowed_images)
    )


def check_mounts(spec: JobSpec, policy: Policy, ctx: ResolveContext) -> tuple[ResolvedMount, ...]:
    if not spec.mounts:
        raise _invalid("no mounts given; is CSUB_MOUNTS set in the submitting container?")
    home = ctx.home
    resolved: list[ResolvedMount] = []
    for m in spec.mounts:
        real = ctx.realpath(m.path)
        if real != m.path:
            raise _policy(
                f"mount {m.path} is not canonical (resolves to {real}); mount the real path"
            )
        if not ctx.isdir(m.path):
            raise _invalid(f"mount {m.path} is not a directory on the submit host")
        if is_under(m.path, home) or is_under(home, m.path):
            raise _policy(
                f"mount {m.path}: the home directory may not be mounted or contain a mount"
            )
        for root in policy.denied_roots:
            if is_under(m.path, root):
                raise _policy(f"mount {m.path} is under denied root {root}")
        if not any(is_under(m.path, root) for root in policy.allowed_roots):
            raise _policy(
                f"mount {m.path} is outside the allowed roots: " + ", ".join(policy.allowed_roots)
            )
        mode = m.mode
        if mode == "rw" and any(is_under(m.path, root) for root in policy.readonly_roots):
            mode = "ro"
        for d in ctx.protected_dirs:
            if is_under(d, m.path):
                raise ResolveError(
                    "broker_misconfigured",
                    f"broker-owned path {d} lies under requested mount {m.path}; "
                    "fix the broker policy",
                )
            if is_under(m.path, d):
                raise _policy(f"mount {m.path} lies inside broker-owned path {d}")
        for other in resolved:
            if is_under(m.path, other.path) or is_under(other.path, m.path):
                raise _invalid(f"mounts {other.path} and {m.path} overlap")
        resolved.append(ResolvedMount(path=m.path, mode=mode))
    return tuple(resolved)


def check_cwd(spec: JobSpec, mounts: tuple[ResolvedMount, ...], ctx: ResolveContext) -> str:
    if spec.cwd is None:
        raise _invalid("cwd is required")
    cwd = ctx.realpath(spec.cwd)
    if not ctx.isdir(cwd):
        raise _invalid(f"cwd {spec.cwd} is not a directory")
    for m in mounts:
        if is_under(cwd, m.path):
            if m.mode != "rw":
                raise _policy(
                    f"cwd {cwd} is inside read-only mount {m.path}; "
                    "the job must be able to write there"
                )
            return cwd
    raise _policy(f"cwd {cwd} is not inside any read-write mount")


def check_env(env: dict[str, str], policy: Policy) -> dict[str, str]:
    out: dict[str, str] = {}
    for key in sorted(env):
        if key in RESERVED_ENV_KEYS or key.startswith(RESERVED_ENV_PREFIXES):
            raise _invalid(f"env {key} is reserved and set by csub")
        if not any(fnmatch.fnmatchcase(key, pat) for pat in policy.env_allow):
            raise _policy(
                f"env {key} is not permitted; allowed patterns: " + ", ".join(policy.env_allow)
            )
        if contains_bsub_directive(env[key]):
            raise _invalid(f"env {key}: value contains a #BSUB line")
        out[key] = env[key]
    return out


def check_hosts(hosts: tuple[str, ...], policy: Policy) -> tuple[str, ...]:
    out: list[str] = []
    for h in hosts:
        if not any(h == pat or fnmatch.fnmatchcase(h, pat) for pat in policy.allowed_hosts):
            raise _policy(f"host {h} is not permitted; allowed: " + ", ".join(policy.allowed_hosts))
        if h not in out:
            out.append(h)
    return tuple(sorted(out))


def check_depends(
    deps: tuple[Dependency, ...], ctx: ResolveContext, session: str
) -> tuple[Dependency, ...]:
    for d in deps:
        rec = ctx.lookup_job(d.job_id)
        if rec is None or rec != (ctx.user, session):
            raise ResolveError("not_found", f"dependency {d.job_id} is not a job of this session")
    return deps


def check_lsf_extra(extra: tuple[str, ...], policy: Policy) -> tuple[str, ...]:
    if not extra:
        return ()
    if not policy.lsf_extra_allow:
        raise _policy("lsf_extra is not permitted by policy")
    argv: list[str] = []
    for item in extra:
        if not any(p.fullmatch(item) for p in policy.lsf_extra_allow):
            raise _policy(f"lsf_extra entry {item!r} does not match any permitted pattern")
        try:
            parts = shlex.split(item)
        except ValueError as e:
            raise _invalid(f"lsf_extra entry {item!r}: {e}") from None
        if not parts:
            raise _invalid("lsf_extra entry is empty")
        if parts[0] in UNSUPPORTED_BSUB_FLAGS:
            raise _invalid(f"bsub flag {parts[0]} is not supported")
        if parts[0] in OWNED_BSUB_FLAGS:
            raise _policy(f"bsub flag {parts[0]} is set by the broker and cannot be overridden")
        if policy.lsf.project and parts[0].startswith("-P"):
            # LSF takes the last -P; with lsf.project set, the policy's must be the only one.
            raise _policy(
                "bsub flag -P is set by the policy (lsf.project) and cannot be overridden"
            )
        if not parts[0].startswith("-"):
            raise _invalid(f"lsf_extra entry {item!r} must start with a flag")
        argv += parts
    return tuple(argv)


def check_no_directives(spec: JobSpec, name: str) -> None:
    texts = [spec.command] if isinstance(spec.command, str) else list(spec.command)
    texts.append(name)
    for t in texts:
        if contains_bsub_directive(t):
            raise _invalid(
                "command contains a #BSUB line; LSF would read it as a submission option"
            )


# --- the whole thing --------------------------------------------------------------------


def resolve(spec: JobSpec, policy: Policy, ctx: ResolveContext) -> ResolvedJob:
    if spec.session is None:
        raise _invalid("session is required")
    session = spec.session
    job_group = f"/csub/{ctx.user}/{session}"

    mounts = check_mounts(spec, policy, ctx)
    cwd = check_cwd(spec, mounts, ctx)

    queue, walltime = select_queue(spec, policy)
    lim = policy.limits
    slots = compute_slots(spec.cpus, spec.mem_mb, queue.mem_per_slot_mb)
    if slots > lim.max_slots:
        raise _policy(
            f"{spec.cpus} cpus / {spec.mem_mb} MB need {slots} slots "
            f"of {queue.mem_per_slot_mb} MB; "
            f"the maximum is {lim.max_slots}"
        )
    if spec.gpus:
        max_gpus = min(lim.max_gpus, queue.max_gpus or lim.max_gpus)
        if spec.gpus > max_gpus:
            raise _policy(
                f"{spec.gpus} GPUs exceeds the maximum of {max_gpus} on queue {queue.name!r}"
            )
        assert queue.slots_per_gpu is not None
        if slots > queue.slots_per_gpu * spec.gpus:
            need = math.ceil(slots / queue.slots_per_gpu)
            raise _policy(
                f"{slots} slots with {spec.gpus} GPU(s) would strand GPUs on queue {queue.name!r} "
                f"({queue.slots_per_gpu} slots per GPU); "
                f"request >= {need} GPUs or reduce cpus/mem_mb"
            )
        if spec.gpu_mem_gb is not None and not queue.gmem_allowed:
            raise _policy(
                f"gpu_mem_gb is only meaningful on mixed-model queues, not {queue.name!r}"
            )

    cost = estimate_cost(slots, spec.gpus, walltime, queue, policy)
    if lim.max_estimated_cost_usd and cost > lim.max_estimated_cost_usd:
        raise _policy(
            f"estimated maximum cost ${cost:.2f} ({slots} slots, {spec.gpus} GPUs, {walltime} min) "
            f"exceeds the per-job cap of ${lim.max_estimated_cost_usd:.2f}"
        )

    sandbox = policy.sandbox
    if spec.gpus and sandbox == "bwrap":
        raise _policy("GPU jobs need podman but this broker runs jobs under bwrap (policy sandbox)")
    image = check_image(spec.image, policy, sandbox=sandbox)
    env = check_env(spec.env, policy)
    if spec.claude and not policy.allow_claude:
        raise _policy("claude is not permitted by policy")
    hosts = check_hosts(spec.allow_hosts + (CLAUDE_HOSTS if spec.claude else ()), policy)
    if spec.scratch and not policy.scratch:
        raise _policy("scratch is not permitted by policy")

    name = sanitize_name(spec.name, spec.command, max_len=policy.max_name_len)
    check_name(name, policy, ctx.user)
    check_no_directives(spec, name)
    deps = check_depends(spec.depends_on, ctx, session)
    extra = check_lsf_extra(spec.lsf_extra, policy)

    return ResolvedJob(
        user=ctx.user,
        session=session,
        job_group=job_group,
        job_name=f"csub-{name}",
        name=name,
        queue=queue.name,
        slots=slots,
        walltime_min=walltime,
        gpus=spec.gpus,
        gpu_mem_gb=spec.gpu_mem_gb,
        image=image,
        mounts=mounts,
        cwd=cwd,
        command=spec.command,
        shell=spec.shell,
        env=env,
        allow_hosts=hosts,
        scratch=spec.scratch,
        claude=spec.claude,
        scratch_root=policy.scratch_root,
        keep_id=policy.keep_id,
        sandbox=sandbox,
        depends_on=deps,
        lsf_extra_argv=(("-P", policy.lsf.project) if policy.lsf.project else ()) + extra,
        estimated_max_cost_usd=cost,
    )
