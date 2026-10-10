"""Broker policy: strict TOML loading and validation. Standard library only.

The policy is the only thing the agent cannot influence; everything that is
security-relevant is decided from it. Unknown keys are errors so that a typo never
silently disables a rule.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from csub._paths import image_registry, is_under, normalize_abs
from csub.protocol import CsubError

DEFAULT_FORBIDDEN_NAME_TOKENS = ("spark", "janelia", "master", "int")


class PolicyError(CsubError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__("broker_misconfigured", message, details)


SANDBOX_SCRIPTS = {"podman": "podman-run.sh", "bwrap": "sandbox-run.sh"}
SANDBOXES = tuple(SANDBOX_SCRIPTS)


@dataclass(frozen=True)
class QueuePolicy:
    name: str
    mem_per_slot_mb: int
    max_walltime_min: int
    gpu: bool = False
    slots_per_gpu: int | None = None
    gpu_price_usd_per_hour: float = 0.0
    gmem_allowed: bool = False
    max_gpus: int | None = None

    def summary(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("name")
        return d


@dataclass(frozen=True)
class Limits:
    max_slots: int = 128
    max_gpus: int = 8
    max_walltime_min: int = 20160
    max_estimated_cost_usd: float = 200.0
    max_pending_per_session: int = 0
    default_cpu_walltime_min: int = 60
    default_gpu_walltime_min: int = 120
    cpu_short_queue: str = "short"
    cpu_default_queue: str = "local"
    slot_price_usd_per_hour: float = 0.05


@dataclass(frozen=True)
class LsfConfig:
    profile: str = "/etc/profile.d/profile.lsf.sh"
    bsub: str = "bsub"
    bjobs: str = "bjobs"
    bkill: str = "bkill"
    timeout_s: int = 120
    norc: bool = False  # run LSF commands via `bash --noprofile --norc`, skipping ~/.bashrc
    project: str = ""  # bsub -P, the lab or project every job is billed to
    submit_host: str = ""  # run bsub/bjobs/bkill over ssh here when this machine has no LSF


@dataclass(frozen=True)
class Policy:
    home: str
    source: str
    state_dir: str
    sandbox_scripts_dir: str
    default_image: str
    allowed_roots: tuple[str, ...]
    allowed_images: tuple[str, ...] = ()
    denied_roots: tuple[str, ...] = ()
    readonly_roots: tuple[str, ...] = ()
    allowed_hosts: tuple[str, ...] = ()
    scratch: bool = True
    scratch_root: str = "/scratch"
    keep_id: bool = True
    inject_client: bool = True
    sandbox: str = "podman"  # bwrap cannot pass GPUs through: GPU requests are then refused
    env_allow: tuple[str, ...] = ()
    lsf_extra_allow: tuple[re.Pattern[str], ...] = ()
    forbidden_name_tokens: tuple[str, ...] = DEFAULT_FORBIDDEN_NAME_TOKENS
    max_name_len: int = 64
    limits: Limits = field(default_factory=Limits)
    queues: dict[str, QueuePolicy] = field(default_factory=dict)
    lsf: LsfConfig = field(default_factory=LsfConfig)

    def queue(self, name: str) -> QueuePolicy | None:
        return self.queues.get(name)

    def probe_summary(self) -> dict[str, Any]:
        return {
            "queues": {name: q.summary() for name, q in self.queues.items()},
            "limits": asdict(self.limits),
            "default_image": self.default_image,
            "allowed_images": list(self.allowed_images),
            "allowed_roots": list(self.allowed_roots),
            "readonly_roots": list(self.readonly_roots),
            "denied_roots": list(self.denied_roots),
            "allowed_hosts": list(self.allowed_hosts),
            "scratch": self.scratch,
            "keep_id": self.keep_id,
            "inject_client": self.inject_client,
            "sandbox": self.sandbox,
            "env_allow": list(self.env_allow),
            "lsf_extra_enabled": bool(self.lsf_extra_allow),
        }


# --- schema -----------------------------------------------------------------------

# key -> (type, required)
_BROKER_KEYS: dict[str, tuple[str, bool]] = {
    "state_dir": ("str", False),
    "sandbox_scripts_dir": ("str", False),
    "default_image": ("str", True),
    "allowed_images": ("list[str]", False),
    "allowed_roots": ("list[str]", True),
    "denied_roots": ("list[str]", False),
    "readonly_roots": ("list[str]", False),
    "allowed_hosts": ("list[str]", False),
    "scratch": ("bool", False),
    "scratch_root": ("str", False),
    "keep_id": ("bool", False),
    "inject_client": ("bool", False),
    "sandbox": ("str", False),
    "env_allow": ("list[str]", False),
    "lsf_extra_allow": ("list[str]", False),
    "forbidden_name_tokens": ("list[str]", False),
    "max_name_len": ("int", False),
}
_LIMITS_KEYS: dict[str, tuple[str, bool]] = {
    "max_slots": ("int", False),
    "max_gpus": ("int", False),
    "max_walltime_min": ("int", False),
    "max_estimated_cost_usd": ("float", False),
    "max_pending_per_session": ("int", False),
    "default_cpu_walltime_min": ("int", False),
    "default_gpu_walltime_min": ("int", False),
    "cpu_short_queue": ("str", False),
    "cpu_default_queue": ("str", False),
    "slot_price_usd_per_hour": ("float", False),
}
_QUEUE_KEYS: dict[str, tuple[str, bool]] = {
    "mem_per_slot_mb": ("int", True),
    "max_walltime_min": ("int", True),
    "gpu": ("bool", False),
    "slots_per_gpu": ("int", False),
    "gpu_price_usd_per_hour": ("float", False),
    "gmem_allowed": ("bool", False),
    "max_gpus": ("int", False),
}
_LSF_KEYS: dict[str, tuple[str, bool]] = {
    "profile": ("str", False),
    "bsub": ("str", False),
    "bjobs": ("str", False),
    "bkill": ("str", False),
    "timeout_s": ("int", False),
    "norc": ("bool", False),
    "project": ("str", False),
    "submit_host": ("str", False),
}
_TOP_KEYS = {"broker", "limits", "queues", "lsf"}

_QUEUE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_ENV_PATTERN_RE = re.compile(r"^[A-Za-z_*?][A-Za-z0-9_*?]*$")


def _check_type(value: Any, typ: str, where: str) -> Any:
    ok = {
        "str": lambda v: isinstance(v, str),
        "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "bool": lambda v: isinstance(v, bool),
        "list[str]": lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v),
    }[typ]
    if not ok(value):
        raise PolicyError(f"{where}: expected {typ}, got {type(value).__name__}")
    return float(value) if typ == "float" else value


def _section(
    data: dict[str, Any], name: str, keys: dict[str, tuple[str, bool]], source: str
) -> dict[str, Any]:
    raw = data.get(name, {})
    if not isinstance(raw, dict):
        raise PolicyError(f"{source}: [{name}] must be a table")
    unknown = sorted(set(raw) - set(keys))
    if unknown:
        raise PolicyError(f"{source}: [{name}] unknown key(s): {', '.join(unknown)}")
    out: dict[str, Any] = {}
    for key, (typ, required) in keys.items():
        if key in raw:
            out[key] = _check_type(raw[key], typ, f"{source}: {name}.{key}")
        elif required:
            raise PolicyError(f"{source}: {name}.{key} is required")
    return out


def _expand(path: str, home: str, where: str) -> str:
    if path == "~":
        path = home
    elif path.startswith("~/"):
        path = home + path[1:]
    try:
        return normalize_abs(path)
    except ValueError as e:
        raise PolicyError(f"{where}: {e}") from None


def _positive(d: dict[str, Any], key: str, where: str, *, minimum: int | float = 1) -> None:
    if key in d and d[key] < minimum:
        raise PolicyError(f"{where}.{key}: must be >= {minimum}")


# --- loading --------------------------------------------------------------------------


def default_policy_path(home: str) -> str:
    return os.path.join(home, ".config", "csub", "broker.toml")


def load_policy(path: str | os.PathLike[str], *, home: str, check_fs: bool = True) -> Policy:
    path = os.fspath(path)
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        raise PolicyError(f"policy file not found: {path}") from None
    except PermissionError:
        raise PolicyError(f"policy file not readable: {path}") from None
    except tomllib.TOMLDecodeError as e:
        raise PolicyError(f"{path}: invalid TOML: {e}") from None
    return parse_policy(data, home=home, source=path, check_fs=check_fs)


def parse_policy(  # noqa: C901 - one validator
    data: dict[str, Any], *, home: str, source: str = "<policy>", check_fs: bool = True
) -> Policy:
    if not isinstance(data, dict):
        raise PolicyError(f"{source}: top level must be a table")
    unknown = sorted(set(data) - _TOP_KEYS)
    if unknown:
        raise PolicyError(f"{source}: unknown table(s): {', '.join(unknown)}")
    home = normalize_abs(home)

    b = _section(data, "broker", _BROKER_KEYS, source)
    lim = _section(data, "limits", _LIMITS_KEYS, source)
    lsf = _section(data, "lsf", _LSF_KEYS, source)

    # --- broker ---
    where = f"{source}: broker"
    state_dir = _expand(b.get("state_dir", "~/.csub"), home, f"{where}.state_dir")
    scripts_dir = _expand(
        b.get("sandbox_scripts_dir", "~/.local/share/csub/agentic-sandbox/scripts"),
        home,
        f"{where}.sandbox_scripts_dir",
    )
    scratch_root = _expand(b.get("scratch_root", "/scratch"), home, f"{where}.scratch_root")

    default_image = b["default_image"]
    if not default_image or any(c.isspace() for c in default_image):
        raise PolicyError(f"{where}.default_image: invalid image reference {default_image!r}")
    registry = image_registry(default_image)
    if registry is None:
        raise PolicyError(f"{where}.default_image: must be registry-qualified: {default_image!r}")
    if registry == "localhost":
        raise PolicyError(
            f"{where}.default_image: localhost images are not visible on compute nodes"
        )

    for key in ("allowed_images",):
        for pat in b.get(key, []):
            if not pat or any(c.isspace() for c in pat):
                raise PolicyError(f"{where}.{key}: invalid pattern {pat!r}")

    def roots(key: str) -> tuple[str, ...]:
        out = []
        for p in b.get(key, []):
            out.append(_expand(p, home, f"{where}.{key}"))
        return tuple(out)

    allowed_roots = roots("allowed_roots")
    if not allowed_roots:
        raise PolicyError(f"{where}.allowed_roots: must not be empty")
    denied_roots = roots("denied_roots")
    readonly_roots = roots("readonly_roots")

    sandbox = b.get("sandbox", "podman")
    if sandbox not in SANDBOXES:
        raise PolicyError(
            f"{where}.sandbox: expected one of {', '.join(SANDBOXES)}, got {sandbox!r}"
        )

    for h in b.get("allowed_hosts", []):
        if not h or any(c.isspace() for c in h) or h != h.lower():
            raise PolicyError(f"{where}.allowed_hosts: invalid hostname pattern {h!r}")

    for pat in b.get("env_allow", []):
        if not _ENV_PATTERN_RE.match(pat):
            raise PolicyError(f"{where}.env_allow: invalid pattern {pat!r}")

    compiled: list[re.Pattern[str]] = []
    for pat in b.get("lsf_extra_allow", []):
        try:
            compiled.append(re.compile(pat))
        except re.error as e:
            raise PolicyError(f"{where}.lsf_extra_allow: invalid regex {pat!r}: {e}") from None

    tokens = tuple(
        t.lower() for t in b.get("forbidden_name_tokens", list(DEFAULT_FORBIDDEN_NAME_TOKENS))
    )
    for t in tokens:
        if not t or not re.match(r"^[a-z0-9]+$", t):
            raise PolicyError(f"{where}.forbidden_name_tokens: invalid token {t!r}")

    max_name_len = b.get("max_name_len", 64)
    if max_name_len < 8:
        raise PolicyError(f"{where}.max_name_len: must be >= 8")

    # --- limits ---
    lwhere = f"{source}: limits"
    for key in (
        "max_slots",
        "max_gpus",
        "max_walltime_min",
        "default_cpu_walltime_min",
        "default_gpu_walltime_min",
    ):
        _positive(lim, key, lwhere)
    _positive(lim, "max_estimated_cost_usd", lwhere, minimum=0)
    _positive(lim, "max_pending_per_session", lwhere, minimum=0)
    _positive(lim, "slot_price_usd_per_hour", lwhere, minimum=0)
    limits = Limits(**lim)

    # --- queues ---
    raw_queues = data.get("queues", {})
    if not isinstance(raw_queues, dict) or not raw_queues:
        raise PolicyError(f"{source}: [queues.<name>] tables are required")
    queues: dict[str, QueuePolicy] = {}
    for qname, _ in raw_queues.items():
        if not _QUEUE_NAME_RE.match(qname):
            raise PolicyError(f"{source}: invalid queue name {qname!r}")
        if qname.endswith("_parallel"):
            raise PolicyError(f"{source}: queues.{qname}: parallel queues are not supported")
        q = _section(raw_queues, qname, _QUEUE_KEYS, f"{source}: queues")
        qwhere = f"{source}: queues.{qname}"
        _positive(q, "mem_per_slot_mb", qwhere)
        _positive(q, "max_walltime_min", qwhere)
        gpu = q.get("gpu", False)
        if gpu:
            if "slots_per_gpu" not in q:
                raise PolicyError(f"{qwhere}.slots_per_gpu: required for GPU queues")
            _positive(q, "slots_per_gpu", qwhere)
            _positive(q, "gpu_price_usd_per_hour", qwhere, minimum=0)
            _positive(q, "max_gpus", qwhere)
        else:
            for key in ("slots_per_gpu", "gpu_price_usd_per_hour", "gmem_allowed", "max_gpus"):
                if key in q:
                    raise PolicyError(f"{qwhere}.{key}: only valid for GPU queues")
        queues[qname] = QueuePolicy(name=qname, **q)

    for key in ("cpu_short_queue", "cpu_default_queue"):
        qn = getattr(limits, key)
        if qn not in queues:
            raise PolicyError(f"{lwhere}.{key}: queue {qn!r} is not defined")
        if queues[qn].gpu:
            raise PolicyError(f"{lwhere}.{key}: queue {qn!r} must be a CPU queue")

    # --- lsf ---
    lsf_cfg = LsfConfig(**lsf)
    for key in ("bsub", "bjobs", "bkill"):
        if not getattr(lsf_cfg, key):
            raise PolicyError(f"{source}: lsf.{key}: must not be empty")
    if lsf_cfg.timeout_s < 1:
        raise PolicyError(f"{source}: lsf.timeout_s: must be >= 1")

    # --- trust-boundary placement of broker-owned directories ---
    home_real = os.path.realpath(home)
    roots_real = [os.path.realpath(r) for r in allowed_roots]
    for label, d in (
        ("state_dir", state_dir),
        ("sandbox_scripts_dir", scripts_dir),
        ("scratch_root", scratch_root),
    ):
        d_real = os.path.realpath(d)
        if is_under(d_real, home_real):
            continue
        hit = [r for r in roots_real if is_under(d_real, r)]
        if hit:
            raise PolicyError(
                f"{where}.{label}: {d} lies under allowed root {hit[0]} but not under $HOME; "
                "an agent could write to it"
            )
    if check_fs:
        script = os.path.join(scripts_dir, SANDBOX_SCRIPTS[sandbox])
        if not os.path.isfile(script) or not os.access(script, os.X_OK):
            raise PolicyError(f"{where}.sandbox_scripts_dir: {script} is missing or not executable")
        if (
            lsf_cfg.profile
            and not lsf_cfg.submit_host  # the profile is sourced on the remote host
            and not os.access(lsf_cfg.profile, os.R_OK)
        ):
            raise PolicyError(f"{source}: lsf.profile: {lsf_cfg.profile} is not readable")

    return Policy(
        home=home,
        source=source,
        state_dir=state_dir,
        sandbox_scripts_dir=scripts_dir,
        default_image=default_image,
        allowed_images=tuple(b.get("allowed_images", [])),
        allowed_roots=allowed_roots,
        denied_roots=denied_roots,
        readonly_roots=readonly_roots,
        allowed_hosts=tuple(b.get("allowed_hosts", [])),
        scratch=b.get("scratch", True),
        scratch_root=scratch_root,
        keep_id=b.get("keep_id", True),
        inject_client=b.get("inject_client", True),
        sandbox=sandbox,
        env_allow=tuple(b.get("env_allow", [])),
        lsf_extra_allow=tuple(compiled),
        forbidden_name_tokens=tokens,
        max_name_len=max_name_len,
        limits=limits,
        queues=queues,
        lsf=lsf_cfg,
    )
