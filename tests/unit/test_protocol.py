import dataclasses
import json

import pytest

from csub.protocol import (
    JOBSPEC_SCHEMA,
    CsubError,
    Dependency,
    JobSpec,
    Mount,
    ProtocolError,
    Request,
    error_response,
    jobspec_schema,
    ok_response,
    parse_response,
    raise_for_response,
)


def minimal(**kw):
    d = {"command": ["echo", "hi"]}
    d.update(kw)
    return d


def test_roundtrip_defaults():
    spec = JobSpec.from_dict(minimal())
    assert spec.command == ("echo", "hi")
    assert spec.cpus == 1 and spec.mem_mb == 0 and spec.gpus == 0
    assert spec.shell is False and spec.scratch is False
    again = JobSpec.from_dict(spec.to_dict())
    assert again == spec


def test_full_roundtrip():
    d = minimal(
        cwd="/g/lab/p/",
        name="x",
        cpus=4,
        mem_mb=1000,
        gpus=1,
        gpu_mem_gb=40,
        walltime_min=30,
        queue="gpu_short",
        image="ghcr.io/x/y:z",
        env={"A": "1"},
        allow_hosts=["PyPI.org", "pypi.org"],
        scratch=True,
        depends_on=["12", {"job_id": "13", "when": "ended"}],
        mounts=[{"path": "/g/lab/p", "mode": "rw"}, {"path": "/n/d", "mode": "ro"}],
        session="s1",
        lsf_extra=["-P proj"],
    )
    spec = JobSpec.from_dict(d)
    assert spec.cwd == "/g/lab/p"  # normalized
    assert spec.allow_hosts == ("pypi.org",)  # lowercased, deduped
    assert spec.depends_on == (Dependency("12", "done"), Dependency("13", "ended"))
    assert spec.mounts[1] == Mount("/n/d", "ro")
    assert JobSpec.from_dict(spec.to_dict()) == spec


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"bogus": 1}, "unknown field"),
        ({"command": "echo hi"}, "argv list"),
        ({"command": ["echo"], "shell": True}, "script string"),
        ({"command": []}, "must not be empty"),
        ({"command": [""]}, "command\\[0\\]"),
        ({"command": ["echo", 3]}, "expected string"),
        ({"cpus": "4"}, "expected integer"),
        ({"cpus": True}, "expected integer"),
        ({"cpus": 0}, ">= 1"),
        ({"mem_mb": -1}, ">= 0"),
        ({"gpu_mem_gb": 40}, "requires gpus"),
        ({"walltime_min": 0}, ">= 1"),
        ({"cwd": "relative"}, "absolute"),
        ({"queue": "bad queue"}, "invalid queue"),
        ({"image": "a b"}, "whitespace"),
        ({"env": ["A=1"]}, "expected object"),
        ({"env": {"1A": "x"}}, "invalid variable name"),
        ({"env": {"A": 1}}, "expected string"),
        ({"allow_hosts": ["not a host"]}, "not a hostname"),
        ({"allow_hosts": ["http://x.org"]}, "not a hostname"),
        ({"depends_on": ["12[3]"]}, "element of an array job"),
        ({"depends_on": ["abc"]}, "not a job id"),
        ({"depends_on": [{"job_id": "1", "when": "later"}]}, "when"),
        ({"mounts": [{"path": "/a"}, {"path": "/a", "mode": "ro"}]}, "duplicate"),
        ({"mounts": [{"path": "/a", "mode": "rwx"}]}, "mode"),
        ({"mounts": [{"path": "a"}]}, "absolute"),
        ({"mounts": [{"path": "/a\nb"}]}, "control character"),
        ({"mounts": [{"path": "/a:b"}]}, "':' or ','"),
        ({"mounts": [{"path": "/a,b"}]}, "':' or ','"),
        ({"cwd": "/a\tb"}, "control character"),
        ({"cwd": "/a\x7fb"}, "control character"),
        ({"session": "bad session!"}, "invalid session"),
        ({"shell": "yes"}, "expected boolean"),
        ({"lsf_extra": [""]}, "must not be empty"),
    ],
)
def test_rejections(overrides, match):
    with pytest.raises(ProtocolError, match=match) as e:
        JobSpec.from_dict(minimal(**overrides))
    assert e.value.code == "invalid_request"


def test_none_means_missing():
    spec = JobSpec.from_dict(minimal(cwd=None, name=None, queue=None))
    assert spec.cwd is None and spec.queue is None


def test_shell_body():
    spec = JobSpec.from_dict({"command": "#!/bin/bash\necho hi\n", "shell": True})
    assert isinstance(spec.command, str)
    assert JobSpec.from_dict(spec.to_dict()) == spec


# --- Request ---


def test_request_submit():
    req = Request.from_json(json.dumps({"protocol": 1, "op": "submit", "job": minimal()}))
    assert req.op == "submit" and req.job is not None
    assert Request.from_json(req.to_json()) == req


def test_request_status_kill():
    req = Request.from_dict({"protocol": 1, "op": "status", "job_ids": ["1", "22"]})
    assert req.job_ids == ("1", "22")
    req = Request.from_dict({"protocol": 1, "op": "kill", "session": "abc"})
    assert req.session == "abc" and req.job_ids is None
    with pytest.raises(ProtocolError, match="job_ids or a session"):
        Request.from_dict({"protocol": 1, "op": "status"})
    with pytest.raises(ProtocolError, match="not a job id"):
        Request.from_dict({"protocol": 1, "op": "kill", "job_ids": ["x"]})


def test_request_probe_and_errors():
    assert Request.from_dict({"protocol": 1, "op": "probe"}).op == "probe"
    for bad, match in [
        ("not json", "not valid JSON"),
        ("[]", "expected object"),
        ('{"protocol": 2, "op": "probe"}', "protocol"),
        ('{"protocol": 1, "op": "dance"}', "op"),
        ('{"protocol": 1, "op": "submit"}', "job: required"),
        ('{"protocol": 1, "op": "probe", "job_ids": []}', "not allowed"),
        ('{"protocol": 1, "op": "probe", "extra": 1}', "unknown field"),
    ]:
        with pytest.raises(ProtocolError, match=match):
            Request.from_json(bad)


def test_request_bytes_and_size():
    assert Request.from_json(b'{"protocol": 1, "op": "probe"}').op == "probe"
    with pytest.raises(ProtocolError, match="UTF-8"):
        Request.from_json(b"\xff\xfe")


# --- responses ---


def test_responses():
    ok = ok_response(job_id="1")
    assert parse_response(json.dumps(ok)) == ok
    assert raise_for_response(ok)["job_id"] == "1"
    err = error_response("policy_violation", "nope", {"x": 1})
    with pytest.raises(CsubError) as e:
        raise_for_response(parse_response(json.dumps(err)))
    assert e.value.code == "policy_violation" and e.value.details == {"x": 1}
    for bad in [
        "garbage",
        "[]",
        '{"protocol": 1}',
        '{"protocol": 1, "ok": false}',
        '{"protocol": 9, "ok": true}',
    ]:
        with pytest.raises(ProtocolError):
            parse_response(bad)


def test_unknown_error_code_from_broker_becomes_internal():
    data = parse_response(
        json.dumps({"protocol": 1, "ok": False, "error": {"code": "weird", "message": "m"}})
    )
    with pytest.raises(CsubError) as e:
        raise_for_response(data)
    assert e.value.code == "internal_error"


def test_transport_error_from_a_relay_is_kept():
    err = {"code": "transport_error", "message": "m"}
    data = parse_response(json.dumps({"protocol": 1, "ok": False, "error": err}))
    with pytest.raises(CsubError) as e:
        raise_for_response(data)
    assert e.value.code == "transport_error"


def test_csub_error_rejects_unknown_code():
    with pytest.raises(ValueError):
        CsubError("nonsense", "x")
    assert str(CsubError("not_found", "gone")) == "not_found: gone"


# --- schema ---


def test_schema_matches_dataclass():
    names = {f.name for f in dataclasses.fields(JobSpec)}
    assert set(JOBSPEC_SCHEMA["properties"]) == names
    assert JOBSPEC_SCHEMA["required"] == ["command"]
    for name, prop in JOBSPEC_SCHEMA["properties"].items():
        assert prop.get("description"), f"{name} lacks a description"


def test_jobspec_schema_exclude():
    s = jobspec_schema(exclude=("mounts", "session", "command"))
    assert "mounts" not in s["properties"] and "session" not in s["properties"]
    assert s["required"] == []
    assert "mounts" in JOBSPEC_SCHEMA["properties"]  # original untouched


def test_schema_is_valid_jsonschema():
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(JOBSPEC_SCHEMA)
    jsonschema.validate(minimal(cpus=2), JOBSPEC_SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(minimal(bogus=1), JOBSPEC_SCHEMA)


def test_array_fields():
    spec = JobSpec.from_dict({"command": ["true"], "array": 20, "array_concurrent": 4})
    assert spec.array == 20 and spec.array_concurrent == 4
    d = spec.to_dict()
    assert d["array"] == 20 and d["array_concurrent"] == 4
    assert JobSpec.from_dict(d) == spec
    plain = JobSpec.from_dict({"command": ["true"]})
    assert plain.array is None and "array" not in plain.to_dict()
    for bad, match in (
        ({"array": 0}, "array: must be >= 1"),
        ({"array": "3"}, "array: expected integer"),
        ({"array": True}, "array: expected integer"),
        ({"array": 1_000_001}, "more than 1000000 elements"),
        ({"array_concurrent": 2}, "array_concurrent: requires array"),
        ({"array": 3, "array_concurrent": 0}, "array_concurrent: must be >= 1"),
    ):
        with pytest.raises(ProtocolError, match=match):
            JobSpec.from_dict({"command": ["true"], **bad})
