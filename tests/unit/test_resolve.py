"""Every policy rule has a rejecting test; the spec's worked examples are checked exactly."""

import os
from dataclasses import replace

import pytest

from csub.broker.resolve import ResolveContext, ceil_cents, compute_slots, resolve, sanitize_name
from csub.protocol import CsubError
from tests.conftest import TEST_USER

GB = 1024


def reject(code, match, spec, policy, ctx):
    with pytest.raises(CsubError, match=match) as e:
        resolve(spec, policy, ctx)
    assert e.value.code == code, e.value
    return e.value


# --- slots, queues, walltime, GPUs, cost (spec vectors) --------------------------------


def test_compute_slots():
    assert compute_slots(4, 100 * GB, 15 * GB) == 7
    assert compute_slots(2, 0, 15 * GB) == 2
    assert compute_slots(1, 15 * GB, 15 * GB) == 1
    assert compute_slots(1, 15 * GB + 1, 15 * GB) == 2


@pytest.mark.parametrize(
    "fields, queue, slots, walltime",
    [
        ({"cpus": 4, "mem_mb": 100 * GB, "queue": "local"}, "local", 7, 60),
        ({"cpus": 4, "mem_mb": 100 * GB, "walltime_min": 90}, "local", 7, 90),
        ({"cpus": 2, "gpus": 1, "queue": "gpu_a100"}, "gpu_a100", 2, 120),
        ({"cpus": 1, "mem_mb": 80 * GB, "gpus": 1, "queue": "gpu_a100"}, "gpu_a100", 2, 120),
        ({"walltime_min": 90}, "local", 1, 90),
        ({"walltime_min": 30}, "short", 1, 30),
        ({"walltime_min": 60}, "short", 1, 60),
        ({"walltime_min": 61}, "local", 1, 61),
        ({}, "short", 1, 60),
        ({"gpus": 1, "queue": "gpu_l4"}, "gpu_l4", 1, 120),
        ({"gpus": 1, "queue": "gpu_short"}, "gpu_short", 1, 60),  # default clamped to queue max
        ({"gpus": 2, "cpus": 16, "queue": "gpu_l4"}, "gpu_l4", 16, 120),
    ],
)
def test_queue_slots_walltime(make_spec, policy, ctx, fields, queue, slots, walltime):
    job = resolve(make_spec(**fields), policy, ctx)
    assert (job.queue, job.slots, job.walltime_min) == (queue, slots, walltime)


def test_local_default_walltime_without_queue_is_short(make_spec, policy, ctx):
    job = resolve(make_spec(queue="local"), policy, ctx)
    assert job.walltime_min == 60  # default_cpu_walltime_min, queue explicitly chosen


@pytest.mark.parametrize(
    "fields, code, match",
    [
        ({"cpus": 100, "gpus": 1, "queue": "gpu_l4"}, "policy_violation", "strand GPUs"),
        ({"cpus": 17, "gpus": 2, "queue": "gpu_l4"}, "policy_violation", "strand GPUs"),
        ({"cpus": 129}, "policy_violation", "maximum is 128"),
        ({"gpus": 2, "queue": "gpu_rtx6000"}, "policy_violation", "maximum of 1"),
        ({"gpus": 9, "queue": "gpu_l4"}, "policy_violation", "maximum of 8"),
        ({"gpus": 1}, "invalid_request", "must name a queue"),
        ({"gpus": 1, "queue": "local"}, "policy_violation", "has no GPUs"),
        ({"queue": "gpu_l4"}, "policy_violation", "is a GPU queue"),
        ({"queue": "interactive"}, "policy_violation", "unknown or not permitted"),
        ({"queue": "short", "walltime_min": 61}, "policy_violation", "exceeds queue"),
        ({"queue": "local", "walltime_min": 30000}, "policy_violation", "exceeds queue"),
        ({"gpus": 1, "queue": "gpu_l4", "gpu_mem_gb": 40}, "policy_violation", "mixed-model"),
    ],
)
def test_resource_rejections(make_spec, policy, ctx, fields, code, match):
    reject(code, match, make_spec(**fields), policy, ctx)


def test_gmem_on_gpu_short(make_spec, policy, ctx):
    job = resolve(make_spec(gpus=1, queue="gpu_short", gpu_mem_gb=40), policy, ctx)
    assert "num=1:gmem=40G" in job.bsub_args("/state")


def test_cost(make_spec, policy, ctx, make_policy):
    job = resolve(make_spec(cpus=4, mem_mb=100 * GB, walltime_min=60), policy, ctx)
    assert job.estimated_max_cost_usd == pytest.approx(7 * 0.05)
    job = resolve(make_spec(cpus=2, gpus=1, queue="gpu_h100", walltime_min=120), policy, ctx)
    assert job.estimated_max_cost_usd == pytest.approx(2 * 2 * 0.05 + 1 * 2 * 0.65)
    assert job.to_response()["estimated_max_cost_usd"] == 1.5
    job = resolve(make_spec(walltime_min=5), policy, ctx)
    assert (
        job.to_response()["estimated_max_cost_usd"] == 0.01
    )  # 1 slot x 5 min = $0.004, shown as a maximum
    assert ceil_cents(0.35) == 0.35 and ceil_cents(0.0) == 0.0 and ceil_cents(0.004) == 0.01
    big = make_spec(cpus=128, queue="local", walltime_min=20160)
    reject("policy_violation", r"cost \$2150.40 .* cap of \$200.00", big, policy, ctx)
    unlimited = make_policy(limits={"max_estimated_cost_usd": 0})
    assert resolve(big, unlimited, ctx).estimated_max_cost_usd == pytest.approx(2150.4)


def test_policy_may_lower_max_slots(make_spec, make_policy, ctx):
    p = make_policy(limits={"max_slots": 8})
    reject("policy_violation", "maximum is 8", make_spec(cpus=9), p, ctx)


# --- mounts and cwd -------------------------------------------------------------------


def test_mounts_resolved_and_readonly_downgrade(make_spec, policy, ctx, project_dir, roots):
    spec = make_spec(
        mounts=[{"path": str(project_dir), "mode": "rw"}, {"path": str(roots.raw_ro), "mode": "rw"}]
    )
    job = resolve(spec, policy, ctx)
    assert [(m.path, m.mode) for m in job.mounts] == [
        (str(project_dir), "rw"),
        (str(roots.raw_ro), "ro"),
    ]


def test_mount_rejections(make_spec, policy, ctx, project_dir, roots, tmp_home, tmp_path):
    pd = str(project_dir)
    reject("invalid_request", "no mounts", make_spec(mounts=[]), policy, ctx)
    reject(
        "policy_violation",
        "outside the allowed roots",
        make_spec(mounts=[{"path": str(roots.outside)}]),
        policy,
        ctx,
    )
    reject(
        "policy_violation",
        "denied root",
        make_spec(mounts=[{"path": str(roots.denied)}]),
        policy,
        ctx,
    )
    reject(
        "policy_violation",
        "home directory",
        make_spec(mounts=[{"path": str(tmp_home)}]),
        policy,
        ctx,
    )
    reject(
        "policy_violation",
        "home directory",
        make_spec(mounts=[{"path": str(tmp_home / ".config")}]),
        policy,
        ctx,
    )
    reject(
        "policy_violation",
        "home directory",
        make_spec(mounts=[{"path": str(tmp_path.resolve())}]),
        policy,
        ctx,
    )
    reject(
        "invalid_request",
        "not a directory",
        make_spec(mounts=[{"path": pd + "/missing"}]),
        policy,
        ctx,
    )
    reject(
        "invalid_request",
        "overlap",
        make_spec(mounts=[{"path": pd}, {"path": str(roots.allowed)}]),
        policy,
        ctx,
    )
    sub = project_dir / "sub"
    sub.mkdir()
    reject(
        "invalid_request",
        "overlap",
        make_spec(mounts=[{"path": pd}, {"path": str(sub)}]),
        policy,
        ctx,
    )


def test_symlinked_mounts_are_rejected(make_spec, policy, ctx, project_dir, roots):
    escape = roots.allowed / "escape"
    escape.symlink_to(roots.outside)
    reject(
        "policy_violation", "not canonical", make_spec(mounts=[{"path": str(escape)}]), policy, ctx
    )
    alias = roots.allowed / "alias"
    alias.symlink_to(project_dir)
    reject(
        "policy_violation", "not canonical", make_spec(mounts=[{"path": str(alias)}]), policy, ctx
    )


def test_protected_dirs(make_spec, policy, ctx, project_dir):
    inside = project_dir / "broker-state"
    inside.mkdir()
    bad_ctx = replace(ctx, protected_dirs=ctx.protected_dirs + (str(inside),))
    reject("broker_misconfigured", "broker-owned path", make_spec(), policy, bad_ctx)
    sub = inside / "x"
    sub.mkdir()
    bad_ctx = replace(ctx, protected_dirs=(str(inside),))
    spec = make_spec(
        mounts=[{"path": str(project_dir)}, {"path": str(sub)}]
    )  # overlapping, but protected check first
    reject("broker_misconfigured", "broker-owned path", spec, policy, bad_ctx)
    spec = make_spec(mounts=[{"path": str(sub)}], cwd=str(sub))
    reject("policy_violation", "inside broker-owned", spec, policy, bad_ctx)


def test_cwd_rules(make_spec, policy, ctx, project_dir, roots):
    reject("invalid_request", "cwd is required", make_spec(cwd=None), policy, ctx)
    reject(
        "invalid_request", "not a directory", make_spec(cwd=str(project_dir / "nope")), policy, ctx
    )
    reject("policy_violation", "read-only mount", make_spec(cwd=str(roots.nrs)), policy, ctx)
    reject(
        "policy_violation",
        "not inside any read-write",
        make_spec(cwd=str(roots.outside)),
        policy,
        ctx,
    )
    sub = project_dir / "deep"
    sub.mkdir()
    link = project_dir / "link"
    link.symlink_to(sub)
    assert resolve(make_spec(cwd=str(link)), policy, ctx).cwd == str(sub)  # cwd is realpath'd


# --- image, env, hosts, scratch -------------------------------------------------------


def test_bwrap_takes_no_image(make_spec, make_policy, ctx):
    p = make_policy(broker={"sandbox": "bwrap"})
    job = resolve(make_spec(), p, ctx)
    assert job.sandbox == "bwrap" and job.image == ""
    assert job.to_response()["sandbox"] == "bwrap" and job.to_response()["image"] == ""
    # The default image is a no-op (a client.toml `image` must not break CPU jobs)...
    assert resolve(make_spec(image="ghcr.io/test/agent:latest"), p, ctx).image == ""
    # ...but asking for anything else would silently run on the host toolchain: refuse.
    reject("policy_violation", "bwrap", make_spec(image="ghcr.io/test/agent:v2"), p, ctx)
    # bwrap cannot pass GPUs through: a GPU job is refused rather than quietly run elsewhere.
    reject("policy_violation", "GPU jobs need podman", make_spec(gpus=1, queue="gpu_short"), p, ctx)


def test_image(make_spec, policy, ctx):
    assert resolve(make_spec(), policy, ctx).image == "ghcr.io/test/agent:latest"
    assert (
        resolve(make_spec(image="ghcr.io/test/agent:v2"), policy, ctx).image
        == "ghcr.io/test/agent:v2"
    )
    assert (
        resolve(make_spec(image="ghcr.io/test/other-gpu:1"), policy, ctx).image
        == "ghcr.io/test/other-gpu:1"
    )
    reject("policy_violation", "localhost", make_spec(image="localhost/agent:latest"), policy, ctx)
    reject("invalid_request", "registry-qualified", make_spec(image="ubuntu:22.04"), policy, ctx)
    reject(
        "policy_violation",
        "not in the allowed",
        make_spec(image="docker.io/library/ubuntu:22.04"),
        policy,
        ctx,
    )
    reject(
        "policy_violation",
        "not in the allowed",
        make_spec(image="ghcr.io/test/other-a/b:1"),
        policy,
        ctx,
    )


def test_env(make_spec, policy, ctx):
    job = resolve(
        make_spec(env={"OMP_NUM_THREADS": "4", "MY_PROJECT_X": "y", "CUDA_LAUNCH_BLOCKING": "1"}),
        policy,
        ctx,
    )
    assert job.env == {"CUDA_LAUNCH_BLOCKING": "1", "MY_PROJECT_X": "y", "OMP_NUM_THREADS": "4"}
    for key in ("PATH", "HOME", "LSB_JOBID", "CSUB_SOCKET", "CUDA_VISIBLE_DEVICES"):
        reject("invalid_request", "reserved", make_spec(env={key: "x"}), policy, ctx)
    reject("policy_violation", "not permitted", make_spec(env={"FOO": "x"}), policy, ctx)
    reject(
        "invalid_request",
        "#BSUB",
        make_spec(env={"MY_PROJECT_X": "a\n#BSUB -m host\n"}),
        policy,
        ctx,
    )


def test_hosts(make_spec, policy, ctx):
    job = resolve(make_spec(allow_hosts=["pypi.org", "foo.janelia.org"]), policy, ctx)
    assert job.allow_hosts == ("foo.janelia.org", "pypi.org")
    reject(
        "policy_violation",
        "host evil.example",
        make_spec(allow_hosts=["evil.example"]),
        policy,
        ctx,
    )


def test_scratch(make_spec, policy, ctx, make_policy):
    assert resolve(make_spec(scratch=True), policy, ctx).scratch is True
    reject(
        "policy_violation",
        "scratch",
        make_spec(scratch=True),
        make_policy(broker={"scratch": False}),
        ctx,
    )


# --- names ------------------------------------------------------------------------------


def test_sanitize_name():
    assert sanitize_name("my job!", ("x",), max_len=64) == "my_job"
    assert sanitize_name(None, ("/usr/bin/python3", "train.py"), max_len=64) == "python3"
    assert sanitize_name(None, "echo hi", max_len=64) == "script"
    assert sanitize_name("...", ("x",), max_len=64) == "job"
    assert len(sanitize_name("a" * 100, ("x",), max_len=10)) == 10


def test_names(make_spec, policy, ctx):
    job = resolve(make_spec(name="my job!"), policy, ctx)
    assert (job.name, job.job_name) == ("my_job", "csub-my_job")
    assert resolve(make_spec(), policy, ctx).name == "sh"
    assert resolve(make_spec(name="print-stats"), policy, ctx).name == "print-stats"
    assert (
        resolve(make_spec(name="integration.pointcloud"), policy, ctx).name
        == "integration.pointcloud"
    )
    for bad in ("my-int-job", "Spark_run", "janelia", "x.master", f"run-{TEST_USER}"):
        reject("policy_violation", "job name", make_spec(name=bad), policy, ctx)


# --- #BSUB injection, dependencies, lsf_extra ----------------------------------------------


def test_bsub_directive_injection(make_spec, policy, ctx):
    reject(
        "invalid_request",
        "#BSUB",
        make_spec(command="echo a\n#BSUB -q gpu_h100\necho b", shell=True),
        policy,
        ctx,
    )
    reject("invalid_request", "#BSUB", make_spec(command=["echo", "#BSUB -m host"]), policy, ctx)
    assert resolve(make_spec(command=["echo", "x #BSUB y"]), policy, ctx)


def test_depends_on(make_spec, policy, ctx):
    reject("not_found", "dependency 12", make_spec(depends_on=["12"]), policy, ctx)
    known = {
        "12": (TEST_USER, "testsess"),
        "13": (TEST_USER, "testsess"),
        "14": (TEST_USER, "other"),
    }
    ok_ctx = replace(ctx, lookup_job=known.get)
    job = resolve(make_spec(depends_on=["12", {"job_id": "13", "when": "ended"}]), policy, ok_ctx)
    args = job.bsub_args("/state")
    assert args[args.index("-w") + 1] == "done(12) && ended(13)"
    reject("not_found", "dependency 14", make_spec(depends_on=["14"]), policy, ok_ctx)


def test_lsf_extra(make_spec, policy, ctx, make_policy):
    reject("policy_violation", "not permitted", make_spec(lsf_extra=["-P proj"]), policy, ctx)
    p = make_policy(broker={"lsf_extra_allow": [r"-P \w+", r"-R select\[\w+\]", ".*-x.*"]})
    job = resolve(make_spec(lsf_extra=["-P proj", "-R select[avx2]"]), p, ctx)
    assert job.lsf_extra_argv == ("-P", "proj", "-R", "select[avx2]")
    assert job.bsub_args("/s")[-4:] == ["-P", "proj", "-R", "select[avx2]"]
    reject("policy_violation", "does not match", make_spec(lsf_extra=["-m host"]), p, ctx)
    p2 = make_policy(broker={"lsf_extra_allow": [".*"]})
    reject("policy_violation", "set by the broker", make_spec(lsf_extra=["-q gpu_h100"]), p2, ctx)
    reject("invalid_request", "not supported", make_spec(lsf_extra=["-Is"]), p2, ctx)
    reject("invalid_request", "must start with a flag", make_spec(lsf_extra=["proj"]), p2, ctx)
    reject("invalid_request", "quotation", make_spec(lsf_extra=["-P 'unterminated"]), p2, ctx)


# --- the resolved job -----------------------------------------------------------------


def test_bsub_args_and_response(make_spec, policy, ctx, project_dir, roots):
    job = resolve(make_spec(cpus=2, walltime_min=30, name="demo"), policy, ctx)
    assert job.bsub_args("/home/u/.csub") == [
        "-q", "short", "-n", "2", "-W", "30",
        "-J", "csub-demo", "-g", f"/csub/{TEST_USER}/testsess",
        "-o", "/home/u/.csub/jobs/%J/lsf.out", "-env", "none",
    ]  # fmt: skip
    r = job.to_response()
    assert r["queue"] == "short" and r["slots"] == 2 and r["cwd"] == str(project_dir)
    assert r["mounts"] == [
        {"path": str(project_dir), "mode": "rw"},
        {"path": str(roots.nrs), "mode": "ro"},
    ]
    assert r["job_group"] == f"/csub/{TEST_USER}/testsess"
    assert job.keep_id is True and job.scratch_root == policy.scratch_root


def test_session_required(make_spec, policy, ctx):
    reject("invalid_request", "session", make_spec(session=None), policy, ctx)


def test_ctx_defaults_use_real_filesystem(tmp_home):
    c = ResolveContext(user="u", home=str(tmp_home))
    assert c.isdir(str(tmp_home)) and c.realpath(str(tmp_home)) == os.path.realpath(str(tmp_home))
    assert c.lookup_job("1") is None


def test_policy_project(make_spec, ctx, make_policy):
    p = make_policy(lsf={"project": "proj"}, broker={"lsf_extra_allow": [r"-R .*"]})
    job = resolve(make_spec(lsf_extra=["-R select[avx2]"]), p, ctx)
    assert job.lsf_extra_argv == ("-P", "proj", "-R", "select[avx2]")
    assert job.bsub_args("/state")[-4:] == ["-P", "proj", "-R", "select[avx2]"]
    # with a project in the policy, a request may not supply its own -P (LSF would take the last)
    p = make_policy(lsf={"project": "proj"}, broker={"lsf_extra_allow": [".*"]})
    reject("policy_violation", "set by the policy", make_spec(lsf_extra=["-P other"]), p, ctx)
    reject("policy_violation", "set by the policy", make_spec(lsf_extra=["-Pother"]), p, ctx)
    # without one, -P stays available through lsf_extra as before
    p = make_policy(broker={"lsf_extra_allow": [".*"]})
    assert resolve(make_spec(lsf_extra=["-P other"]), p, ctx).lsf_extra_argv == ("-P", "other")


def test_claude_jobs(make_spec, policy, ctx, make_policy):
    from csub.broker.resolve import CLAUDE_HOSTS

    reject("policy_violation", "claude is not permitted", make_spec(claude=True), policy, ctx)
    p = make_policy(broker={"allow_claude": True, "allowed_hosts": ["pypi.org", *CLAUDE_HOSTS]})
    job = resolve(make_spec(claude=True, allow_hosts=["pypi.org"]), p, ctx)
    assert (
        job.claude and set(CLAUDE_HOSTS) <= set(job.allow_hosts) and "pypi.org" in job.allow_hosts
    )
    assert not resolve(make_spec(), p, ctx).claude
    # allowed_claude without the hosts in the policy: the host check says what is missing
    p2 = make_policy(broker={"allow_claude": True, "allowed_hosts": ["pypi.org"]})
    reject(
        "policy_violation", "api.anthropic.com is not permitted", make_spec(claude=True), p2, ctx
    )
