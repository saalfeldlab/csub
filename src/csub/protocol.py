"""Wire protocol shared by client and broker. Standard library only.

One JSON request, one JSON response, over any transport. ``JobSpec.from_dict`` is the
single structural validator: everything that is wrong *regardless of policy* (types,
ranges, shapes) is rejected here with ``invalid_request``; everything that depends on the
broker's policy is decided in :mod:`csub.broker.resolve`.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from csub._paths import normalize_abs

PROTOCOL_VERSION = 1
OPS = ("submit", "status", "kill", "probe")
ERROR_CODES = (
    "policy_violation",
    "invalid_request",
    "lsf_error",
    "not_found",
    "broker_misconfigured",
    "internal_error",
)
CLIENT_ERROR_CODES = ("transport_error",)
STATES = ("PEND", "RUN", "DONE", "EXIT", "SUSP", "UNKNOWN")
TERMINAL_STATES = ("DONE", "EXIT", "UNKNOWN")
DEPENDENCY_CONDITIONS = ("done", "ended", "exit")
MOUNT_MODES = ("rw", "ro")

LIMITS = {
    "max_argv": 4096,
    "max_arg_len": 65536,
    "max_body_len": 1 << 20,
    "max_env": 256,
    "max_env_value_len": 65536,
    "max_mounts": 64,
    "max_hosts": 64,
    "max_job_ids": 1000,
    "max_depends": 64,
    "max_name_len": 256,
    "max_lsf_extra": 32,
    "max_request_bytes": 4 << 20,
}

ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
HOSTNAME_RE = re.compile(
    r"^(?=.{1,253}$)[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)*$"
)
JOB_ID_RE = re.compile(r"^[0-9]{1,20}$")
ARRAY_JOB_ID_RE = re.compile(r"^[0-9]+\[")
SESSION_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
QUEUE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
# LSF reads embedded submission options from spooled scripts; agent text must never
# contain such a line.
BSUB_DIRECTIVE_RE = re.compile(r"^[ \t]*#[ \t]*BSUB\b", re.MULTILINE)


# --- errors -------------------------------------------------------------------


class CsubError(Exception):
    """An error with a protocol error code. Broker and client both raise these."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        if code not in ERROR_CODES + CLIENT_ERROR_CODES:
            raise ValueError(f"unknown error code {code!r}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def to_response(self) -> dict[str, Any]:
        return error_response(self.code, self.message, self.details)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


class ProtocolError(CsubError):
    """Structurally invalid request or response (``invalid_request``)."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__("invalid_request", message, details)


def contains_bsub_directive(text: str) -> bool:
    return BSUB_DIRECTIVE_RE.search(text) is not None


# --- validation helpers ---------------------------------------------------------


def _tn(value: Any) -> str:
    return type(value).__name__


def _str(value: Any, where: str, *, min_len: int = 0, max_len: int | None = None) -> str:
    if not isinstance(value, str):
        raise ProtocolError(f"{where}: expected string, got {_tn(value)}")
    if "\0" in value:
        raise ProtocolError(f"{where}: contains NUL")
    if len(value) < min_len:
        raise ProtocolError(f"{where}: must not be empty")
    if max_len is not None and len(value) > max_len:
        raise ProtocolError(f"{where}: longer than {max_len} characters")
    return value


def _int(value: Any, where: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{where}: expected integer, got {_tn(value)}")
    if value < minimum:
        raise ProtocolError(f"{where}: must be >= {minimum}")
    return value


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ProtocolError(f"{where}: expected boolean, got {_tn(value)}")
    return value


def _list(value: Any, where: str, *, max_len: int) -> list[Any]:
    if not isinstance(value, list):
        raise ProtocolError(f"{where}: expected list, got {_tn(value)}")
    if len(value) > max_len:
        raise ProtocolError(f"{where}: more than {max_len} entries")
    return value


def _no_unknown(d: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(d) - allowed)
    if unknown:
        raise ProtocolError(f"{where}: unknown field(s): {', '.join(unknown)}")


def _job_id(value: Any, where: str) -> str:
    s = _str(value, where, min_len=1)
    if ARRAY_JOB_ID_RE.match(s):
        raise ProtocolError(f"{where}: array jobs are not supported")
    if not JOB_ID_RE.match(s):
        raise ProtocolError(f"{where}: not a job id: {s!r}")
    return s


# --- job specification ----------------------------------------------------------


@dataclass(frozen=True)
class Mount:
    path: str
    mode: str = "rw"

    @classmethod
    def from_dict(cls, d: Any, where: str = "mount") -> Mount:
        if not isinstance(d, dict):
            raise ProtocolError(f"{where}: expected object, got {_tn(d)}")
        _no_unknown(d, {"path", "mode"}, where)
        if "path" not in d:
            raise ProtocolError(f"{where}.path: required")
        try:
            path = normalize_abs(_str(d["path"], f"{where}.path", min_len=1))
        except ValueError as e:
            raise ProtocolError(f"{where}.path: {e}") from None
        mode = d.get("mode", "rw")
        if mode not in MOUNT_MODES:
            raise ProtocolError(f"{where}.mode: expected 'rw' or 'ro', got {mode!r}")
        return cls(path=path, mode=mode)

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "mode": self.mode}


@dataclass(frozen=True)
class Dependency:
    job_id: str
    when: str = "done"

    @classmethod
    def from_dict(cls, d: Any, where: str = "dependency") -> Dependency:
        if isinstance(d, str):
            return cls(job_id=_job_id(d, where), when="done")
        if not isinstance(d, dict):
            raise ProtocolError(f"{where}: expected job id or object, got {_tn(d)}")
        _no_unknown(d, {"job_id", "when"}, where)
        if "job_id" not in d:
            raise ProtocolError(f"{where}.job_id: required")
        when = d.get("when", "done")
        if when not in DEPENDENCY_CONDITIONS:
            raise ProtocolError(
                f"{where}.when: expected one of {', '.join(DEPENDENCY_CONDITIONS)}, got {when!r}"
            )
        return cls(job_id=_job_id(d["job_id"], f"{where}.job_id"), when=when)

    def to_dict(self) -> dict[str, str]:
        return {"job_id": self.job_id, "when": self.when}


@dataclass(frozen=True)
class JobSpec:
    """What the agent asks for. Paths are container paths (== host paths)."""

    command: tuple[str, ...] | str
    shell: bool = False
    cwd: str | None = None
    name: str | None = None
    cpus: int = 1
    mem_mb: int = 0
    gpus: int = 0
    gpu_mem_gb: int | None = None
    walltime_min: int | None = None
    queue: str | None = None
    image: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    allow_hosts: tuple[str, ...] = ()
    scratch: bool = False
    claude: bool = False
    depends_on: tuple[Dependency, ...] = ()
    mounts: tuple[Mount, ...] = ()
    session: str | None = None
    lsf_extra: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, d: Any, where: str = "job") -> JobSpec:  # noqa: C901 - one validator
        if not isinstance(d, dict):
            raise ProtocolError(f"{where}: expected object, got {_tn(d)}")
        _no_unknown(d, {f.name for f in fields(cls)}, where)
        d = {k: v for k, v in d.items() if v is not None}

        shell = _bool(d.get("shell", False), f"{where}.shell")
        if "command" not in d:
            raise ProtocolError(f"{where}.command: required")
        raw_cmd = d["command"]
        command: tuple[str, ...] | str
        if shell:
            if not isinstance(raw_cmd, str):
                raise ProtocolError(
                    f"{where}.command: with shell=true, command must be a script string"
                )
            command = _str(raw_cmd, f"{where}.command", min_len=1, max_len=LIMITS["max_body_len"])
        else:
            if isinstance(raw_cmd, str):
                raise ProtocolError(
                    f"{where}.command: expected an argv list; use shell=true for a script string"
                )
            items = _list(raw_cmd, f"{where}.command", max_len=LIMITS["max_argv"])
            if not items:
                raise ProtocolError(f"{where}.command: must not be empty")
            command = tuple(
                _str(a, f"{where}.command[{i}]", max_len=LIMITS["max_arg_len"])
                for i, a in enumerate(items)
            )
            if not command[0]:
                raise ProtocolError(f"{where}.command[0]: must not be empty")

        cwd = None
        if "cwd" in d:
            try:
                cwd = normalize_abs(_str(d["cwd"], f"{where}.cwd", min_len=1))
            except ValueError as e:
                raise ProtocolError(f"{where}.cwd: {e}") from None

        name = (
            _str(d["name"], f"{where}.name", max_len=LIMITS["max_name_len"])
            if "name" in d
            else None
        )
        cpus = _int(d.get("cpus", 1), f"{where}.cpus", minimum=1)
        mem_mb = _int(d.get("mem_mb", 0), f"{where}.mem_mb", minimum=0)
        gpus = _int(d.get("gpus", 0), f"{where}.gpus", minimum=0)
        gpu_mem_gb = (
            _int(d["gpu_mem_gb"], f"{where}.gpu_mem_gb", minimum=1) if "gpu_mem_gb" in d else None
        )
        if gpu_mem_gb is not None and gpus < 1:
            raise ProtocolError(f"{where}.gpu_mem_gb: requires gpus >= 1")
        walltime_min = (
            _int(d["walltime_min"], f"{where}.walltime_min", minimum=1)
            if "walltime_min" in d
            else None
        )

        queue = None
        if "queue" in d:
            queue = _str(d["queue"], f"{where}.queue", min_len=1)
            if not QUEUE_RE.match(queue):
                raise ProtocolError(f"{where}.queue: invalid queue name {queue!r}")

        image = None
        if "image" in d:
            image = _str(d["image"], f"{where}.image", min_len=1, max_len=512)
            if any(c.isspace() for c in image):
                raise ProtocolError(f"{where}.image: must not contain whitespace")

        env: dict[str, str] = {}
        if "env" in d:
            raw_env = d["env"]
            if not isinstance(raw_env, dict):
                raise ProtocolError(f"{where}.env: expected object, got {_tn(raw_env)}")
            if len(raw_env) > LIMITS["max_env"]:
                raise ProtocolError(f"{where}.env: more than {LIMITS['max_env']} entries")
            for k, v in raw_env.items():
                if not isinstance(k, str) or not ENV_KEY_RE.match(k):
                    raise ProtocolError(f"{where}.env: invalid variable name {k!r}")
                env[k] = _str(v, f"{where}.env.{k}", max_len=LIMITS["max_env_value_len"])

        hosts: list[str] = []
        if "allow_hosts" in d:
            for i, h in enumerate(
                _list(d["allow_hosts"], f"{where}.allow_hosts", max_len=LIMITS["max_hosts"])
            ):
                h = _str(h, f"{where}.allow_hosts[{i}]", min_len=1).strip().lower()
                if not HOSTNAME_RE.match(h):
                    raise ProtocolError(f"{where}.allow_hosts[{i}]: not a hostname: {h!r}")
                if h not in hosts:
                    hosts.append(h)

        scratch = _bool(d.get("scratch", False), f"{where}.scratch")
        claude = _bool(d.get("claude", False), f"{where}.claude")

        depends: list[Dependency] = []
        if "depends_on" in d:
            for i, dep in enumerate(
                _list(d["depends_on"], f"{where}.depends_on", max_len=LIMITS["max_depends"])
            ):
                depends.append(Dependency.from_dict(dep, f"{where}.depends_on[{i}]"))

        mounts: list[Mount] = []
        if "mounts" in d:
            for i, m in enumerate(
                _list(d["mounts"], f"{where}.mounts", max_len=LIMITS["max_mounts"])
            ):
                mount = Mount.from_dict(m, f"{where}.mounts[{i}]")
                if any(mount.path == x.path for x in mounts):
                    raise ProtocolError(f"{where}.mounts[{i}]: duplicate path {mount.path}")
                mounts.append(mount)

        session = None
        if "session" in d:
            session = _str(d["session"], f"{where}.session", min_len=1)
            if not SESSION_RE.match(session):
                raise ProtocolError(f"{where}.session: invalid session id {session!r}")

        lsf_extra: list[str] = []
        if "lsf_extra" in d:
            for i, x in enumerate(
                _list(d["lsf_extra"], f"{where}.lsf_extra", max_len=LIMITS["max_lsf_extra"])
            ):
                lsf_extra.append(_str(x, f"{where}.lsf_extra[{i}]", min_len=1, max_len=1024))

        return cls(
            command=command,
            shell=shell,
            cwd=cwd,
            name=name,
            cpus=cpus,
            mem_mb=mem_mb,
            gpus=gpus,
            gpu_mem_gb=gpu_mem_gb,
            walltime_min=walltime_min,
            queue=queue,
            image=image,
            env=env,
            allow_hosts=tuple(hosts),
            scratch=scratch,
            claude=claude,
            depends_on=tuple(depends),
            mounts=tuple(mounts),
            session=session,
            lsf_extra=tuple(lsf_extra),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "command": list(self.command) if isinstance(self.command, tuple) else self.command,
            "shell": self.shell,
            "cpus": self.cpus,
            "mem_mb": self.mem_mb,
            "gpus": self.gpus,
            "scratch": self.scratch,
            "claude": self.claude,
            "env": dict(self.env),
            "allow_hosts": list(self.allow_hosts),
            "depends_on": [x.to_dict() for x in self.depends_on],
            "mounts": [m.to_dict() for m in self.mounts],
            "lsf_extra": list(self.lsf_extra),
        }
        for key in ("cwd", "name", "gpu_mem_gb", "walltime_min", "queue", "image", "session"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out

    def replace(self, **changes: Any) -> JobSpec:
        from dataclasses import replace

        return replace(self, **changes)


# --- request envelope -----------------------------------------------------------


@dataclass(frozen=True)
class Request:
    op: str
    job: JobSpec | None = None
    job_ids: tuple[str, ...] | None = None
    session: str | None = None
    protocol: int = PROTOCOL_VERSION

    @classmethod
    def from_dict(cls, d: Any) -> Request:
        if not isinstance(d, dict):
            raise ProtocolError(f"request: expected object, got {_tn(d)}")
        _no_unknown(d, {"protocol", "op", "job", "job_ids", "session"}, "request")
        proto = d.get("protocol")
        if isinstance(proto, bool) or not isinstance(proto, int) or proto != PROTOCOL_VERSION:
            raise ProtocolError(f"request.protocol: expected {PROTOCOL_VERSION}, got {proto!r}")
        op = d.get("op")
        if op not in OPS:
            raise ProtocolError(f"request.op: expected one of {', '.join(OPS)}, got {op!r}")
        job = None
        job_ids = None
        session = None
        if d.get("session") is not None:
            session = _str(d["session"], "request.session", min_len=1)
            if not SESSION_RE.match(session):
                raise ProtocolError(f"request.session: invalid session id {session!r}")
        if op == "submit":
            if "job" not in d:
                raise ProtocolError("request.job: required for submit")
            job = JobSpec.from_dict(d["job"], "job")
            if "job_ids" in d:
                raise ProtocolError("request.job_ids: not allowed for submit")
        elif op in ("status", "kill"):
            if "job" in d:
                raise ProtocolError(f"request.job: not allowed for {op}")
            if d.get("job_ids") is not None:
                ids = _list(d["job_ids"], "request.job_ids", max_len=LIMITS["max_job_ids"])
                job_ids = tuple(_job_id(x, f"request.job_ids[{i}]") for i, x in enumerate(ids))
            if not job_ids and session is None:
                raise ProtocolError(f"request: {op} needs job_ids or a session")
        else:  # probe
            for k in ("job", "job_ids"):
                if k in d:
                    raise ProtocolError(f"request.{k}: not allowed for probe")
        return cls(op=op, job=job, job_ids=job_ids, session=session, protocol=proto)

    @classmethod
    def from_json(cls, text: str | bytes) -> Request:
        if isinstance(text, bytes):
            try:
                text = text.decode("utf-8")
            except UnicodeDecodeError:
                raise ProtocolError("request: not valid UTF-8") from None
        if len(text) > LIMITS["max_request_bytes"]:
            raise ProtocolError("request: too large")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ProtocolError(f"request: not valid JSON ({e.msg} at {e.pos})") from None
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"protocol": self.protocol, "op": self.op}
        if self.job is not None:
            out["job"] = self.job.to_dict()
        if self.job_ids is not None:
            out["job_ids"] = list(self.job_ids)
        if self.session is not None:
            out["session"] = self.session
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))


# --- responses --------------------------------------------------------------------


def ok_response(**payload: Any) -> dict[str, Any]:
    return {"protocol": PROTOCOL_VERSION, "ok": True, **payload}


def error_response(
    code: str, message: str, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if details:
        err["details"] = details
    return {"protocol": PROTOCOL_VERSION, "ok": False, "error": err}


def parse_response(text: str | bytes) -> dict[str, Any]:
    """Client side: validate the envelope of a broker response. Raises ProtocolError."""
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise ProtocolError(f"response: not valid JSON: {text[:200]!r}") from None
    if not isinstance(data, dict):
        raise ProtocolError("response: expected object")
    if data.get("protocol") != PROTOCOL_VERSION:
        raise ProtocolError(f"response: unexpected protocol {data.get('protocol')!r}")
    if not isinstance(data.get("ok"), bool):
        raise ProtocolError("response: missing 'ok'")
    if not data["ok"]:
        err = data.get("error")
        if (
            not isinstance(err, dict)
            or not isinstance(err.get("code"), str)
            or not isinstance(err.get("message"), str)
        ):
            raise ProtocolError("response: malformed error")
    return data


def raise_for_response(data: dict[str, Any]) -> dict[str, Any]:
    """Turn an ``ok: false`` response into a CsubError; return the payload otherwise."""
    if not data["ok"]:
        err = data["error"]
        code = err["code"] if err["code"] in ERROR_CODES else "internal_error"
        raise CsubError(code, err["message"], err.get("details"))
    return data


# --- typed results (lenient: unknown keys are ignored for forward compatibility) ----


def _lenient(cls: Any, d: dict[str, Any]) -> Any:
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


@dataclass(frozen=True)
class SubmitResult:
    job_id: str
    queue: str
    slots: int
    walltime_min: int
    gpus: int = 0
    gpu_mem_gb: int | None = None
    image: str = ""
    sandbox: str = ""  # "podman" or "bwrap"; the latter runs on the node's toolchain, no image
    mounts: list[dict[str, str]] = field(default_factory=list)
    allow_hosts: list[str] = field(default_factory=list)
    scratch: bool = False
    claude: bool = False
    name: str = ""
    job_group: str = ""
    session: str = ""
    cwd: str = ""
    job_dir: str = ""
    depends_on: list[dict[str, str]] = field(default_factory=list)
    estimated_max_cost_usd: float = 0.0
    billing_group: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SubmitResult:
        return _lenient(cls, d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JobStatus:
    job_id: str
    state: str
    exit_code: int | None = None
    queue: str | None = None
    exec_host: str | None = None
    name: str | None = None
    cwd: str | None = None
    session: str | None = None
    source: str = "bjobs"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JobStatus:
        return _lenient(cls, d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES


@dataclass(frozen=True)
class KillResult:
    killed: list[str] = field(default_factory=list)
    already_finished: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KillResult:
        return _lenient(cls, d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProbeResult:
    broker_version: str = ""
    protocol: int = PROTOCOL_VERSION
    user: str = ""
    queues: dict[str, dict[str, Any]] = field(default_factory=dict)
    limits: dict[str, Any] = field(default_factory=dict)
    default_image: str = ""
    allowed_images: list[str] = field(default_factory=list)
    allowed_roots: list[str] = field(default_factory=list)
    readonly_roots: list[str] = field(default_factory=list)
    denied_roots: list[str] = field(default_factory=list)
    allowed_hosts: list[str] = field(default_factory=list)
    scratch: bool = False
    keep_id: bool = True
    allow_claude: bool = False
    sandbox: str = "podman"
    env_allow: list[str] = field(default_factory=list)
    lsf_extra_enabled: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProbeResult:
        return _lenient(cls, d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- JSON Schema (hand-written; descriptions double as agent guidance) --------------

JOBSPEC_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "JobSpec",
    "type": "object",
    "additionalProperties": False,
    "required": ["command"],
    "properties": {
        "command": {
            "description": (
                'The program to run. An argv list (recommended), e.g. ["python", "train.py"]. '
                "With shell=true: a script body instead (a shebang is honoured)."
            ),
            "oneOf": [
                {"type": "array", "items": {"type": "string"}, "minItems": 1},
                {"type": "string", "minLength": 1},
            ],
        },
        "shell": {
            "description": (
                "If true, command is a script body written into the job dir and executed."
            ),
            "type": "boolean",
            "default": False,
        },
        "cwd": {
            "description": (
                "Absolute working directory for the job; must lie inside a read-write mount. "
                "Defaults to the submitter's current directory. Job output lands in "
                "<cwd>/.csub/jobs/<job_id>/{stdout,stderr}."
            ),
            "type": "string",
        },
        "name": {
            "description": "Short job name ([A-Za-z0-9_.-]); defaults to the program name.",
            "type": "string",
        },
        "cpus": {
            "description": (
                "CPU cores wanted (default 1). The broker converts cpus/mem_mb into LSF slots."
            ),
            "type": "integer",
            "minimum": 1,
            "default": 1,
        },
        "mem_mb": {
            "description": (
                "Memory wanted in MB. Slots are max(cpus, ceil(mem_mb / memory_per_slot)); "
                "memory per slot is 15 GB on CPU/L4 queues, 40 GB on A100/H100/H200/B300 queues."
            ),
            "type": "integer",
            "minimum": 0,
            "default": 0,
        },
        "gpus": {
            "description": "GPUs wanted. GPU jobs must name a GPU queue (see csub_probe).",
            "type": "integer",
            "minimum": 0,
            "default": 0,
        },
        "gpu_mem_gb": {
            "description": (
                "Minimum GPU memory per GPU in GB; only on mixed-model queues such as gpu_short."
            ),
            "type": "integer",
            "minimum": 1,
        },
        "walltime_min": {
            "description": (
                "Hard runtime limit in minutes. Prefer <= 60: those jobs go to the 'short' queue, "
                "schedule fastest and cost least. Default: 60 (CPU) / 120 (GPU), capped by the "
                "queue. Jobs are killed at the limit; use CSUB_DEADLINE_EPOCH inside the job to "
                "checkpoint."
            ),
            "type": "integer",
            "minimum": 1,
        },
        "queue": {
            "description": (
                "LSF queue. Optional for CPU jobs (chosen from walltime); required for GPU jobs: "
                "gpu_short (<= 1 h, any GPU), gpu_l4, gpu_a100, gpu_h100, gpu_h200, ... — "
                "see csub_probe."
            ),
            "type": "string",
        },
        "image": {
            "description": (
                "Container image; defaults to the image this container runs in. "
                "Must be allowed by policy. Queues that run under bwrap (see csub_probe) "
                "take no image; the response's `sandbox` says which backend ran."
            ),
            "type": "string",
        },
        "env": {
            "description": "Environment variables for the job (policy allowlist applies).",
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "allow_hosts": {
            "description": (
                "Hostnames the job may reach over HTTP(S). Default: no network at all. "
                "Must be a subset of the policy's allowed hosts (see csub_probe)."
            ),
            "type": "array",
            "items": {"type": "string"},
        },
        "scratch": {
            "description": (
                "Give the job a private node-local scratch directory, exported as TMPDIR."
            ),
            "type": "boolean",
            "default": False,
        },
        "claude": {
            "description": (
                "Run Claude Code inside the job: the CLI, the submitter's settings and "
                "credentials are provided, and the Anthropic API hosts are added to allow_hosts. "
                "Needs the policy's allow_claude (see csub_probe). For agents that start agents."
            ),
            "type": "boolean",
            "default": False,
        },
        "depends_on": {
            "description": (
                "Start only after these jobs: {job_id, when} with when = done (default, exit 0), "
                "ended (any exit, incl. killed) or exit (non-zero). A plain job id means done."
            ),
            "type": "array",
            "items": {
                "oneOf": [
                    {"type": "string"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["job_id"],
                        "properties": {
                            "job_id": {"type": "string"},
                            "when": {"type": "string", "enum": list(DEPENDENCY_CONDITIONS)},
                        },
                    },
                ]
            },
        },
        "mounts": {
            "description": "Bind mounts {path, mode}; filled in by the client from CSUB_MOUNTS.",
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path"],
                "properties": {
                    "path": {"type": "string"},
                    "mode": {"type": "string", "enum": list(MOUNT_MODES)},
                },
            },
        },
        "session": {
            "description": "Job group component; filled in by the client.",
            "type": "string",
        },
        "lsf_extra": {
            "description": (
                "Extra bsub flags (one complete flag per entry); normally not permitted by policy."
            ),
            "type": "array",
            "items": {"type": "string"},
        },
    },
}


def jobspec_schema(exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    schema = copy.deepcopy(JOBSPEC_SCHEMA)
    for name in exclude:
        schema["properties"].pop(name, None)
    schema["required"] = [r for r in schema["required"] if r not in exclude]
    return schema


_JOB_IDS = {"type": "array", "items": {"type": "string", "pattern": "^[0-9]+$"}}

STATUS_ARGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "job_ids": {**_JOB_IDS, "description": "Job ids; omit for all jobs of this session."},
    },
}

KILL_ARGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "job_ids": {**_JOB_IDS, "description": "Job ids to kill."},
        "all": {"type": "boolean", "description": "Kill every job of this session."},
    },
}

WAIT_ARGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["job_ids"],
    "properties": {
        "job_ids": {**_JOB_IDS, "minItems": 1, "description": "Jobs to wait for."},
        "timeout_s": {
            "type": "integer",
            "minimum": 1,
            "maximum": 3600,
            "default": 600,
            "description": (
                "Give up after this many seconds (returns timed_out=true, not an error)."
            ),
        },
        "tail_lines": {"type": "integer", "minimum": 0, "default": 50},
    },
}

LOGS_ARGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["job_id"],
    "properties": {
        "job_id": {"type": "string", "pattern": "^[0-9]+$"},
        "stream": {"type": "string", "enum": ["stdout", "stderr", "both"], "default": "both"},
        "tail_lines": {"type": "integer", "minimum": 0, "default": 200},
    },
}

PROBE_ARGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {},
}
