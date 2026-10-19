"""Raw protocol against a real csub-broker process backed by the fakes."""

import json
import os
import stat
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytestmark = pytest.mark.integration


def test_probe(broker, tmp_home):
    p = broker.ok({"protocol": 1, "op": "probe"})
    assert p["user"] and p["queues"]["gpu_a100"]["mem_per_slot_mb"] == 40960
    assert "state_dir" not in p and p["allowed_hosts"][-1] == "api.anthropic.com"


def test_submit_hello(
    broker, submit, project_dir, roots, tmp_home, fake_lsf, fake_sandbox, job_dir
):
    r = submit(["sh", "-c", "echo hello; echo err >&2"], name="hello")
    jid = r["job_id"]
    assert r["queue"] == "short" and r["slots"] == 1 and r["walltime_min"] == 60
    assert r["billing_group"] == "testlab" and r["estimated_max_cost_usd"] == 0.05
    assert r["job_dir"] == str(job_dir(jid)) and r["name"] == "hello"
    assert r["mounts"] == [
        {"path": str(project_dir), "mode": "rw"},
        {"path": str(roots.nrs), "mode": "ro"},
    ]

    e = broker.wait(jid)
    assert e["state"] == "DONE" and e["exit_code"] == 0 and e["source"] == "bjobs"
    assert e["cwd"] == str(project_dir) and e["session"] == "testsess" and e["name"] == "hello"
    assert (job_dir(jid) / "stdout").read_text() == "hello\n"
    assert (job_dir(jid) / "stderr").read_text() == "err\n"

    bdir = tmp_home / ".csub" / "jobs" / jid
    for name in ("request.json", "wrapper.sh", "meta.json"):
        assert stat.S_IMODE((bdir / name).stat().st_mode) == 0o600
    assert (bdir / "exit_code").read_text() == "0\n"
    assert "Successfully completed" in (bdir / "lsf.out").read_text()
    assert not list((tmp_home / ".csub" / "jobs" / ".pending").iterdir())
    req = json.loads((bdir / "request.json").read_text())
    assert (
        req["spec"]["command"] == ["sh", "-c", "echo hello; echo err >&2"]
        and req["resolved"]["slots"] == 1
    )

    user = broker.ok({"protocol": 1, "op": "probe"})["user"]
    lsf_job = fake_lsf.job(jid)
    assert lsf_job["group"] == f"/csub/{user}/testsess"
    assert (
        lsf_job["env_mode"] == "none" and lsf_job["name"] == "csub-hello" and lsf_job["slots"] == 1
    )
    assert lsf_job["out"] == str(bdir / "lsf.out")
    # nothing agent-controlled on the bsub command line (only the sanitized -J name)
    assert "echo hello" not in " ".join(lsf_job["argv"])

    rec = fake_sandbox.for_job(jid)
    assert rec["image"] == "ghcr.io/test/agent:latest" and rec["keep_id"] and not rec["gpu"]
    import csub as _csub

    client_src = os.path.realpath(os.path.dirname(_csub.__file__))
    assert rec["rw"] == [str(project_dir)]
    assert rec["ro"] == [str(roots.nrs), str(bdir / "csub.sock"), client_src]
    assert rec["allow"] == [] and rec["cmd"][:2] == ["/bin/sh", "-c"]


def test_exit_code_and_sandbox_failure(broker, submit, job_dir, monkeypatch):
    jid = submit(["sh", "-c", "exit 3"])["job_id"]
    e = broker.wait(jid)
    assert e["state"] == "EXIT" and e["exit_code"] == 3
    broker.env["CSUB_FAKE_SANDBOX_FAIL_RC"] = "125"
    jid = submit(["true"])["job_id"]
    e = broker.wait(jid)
    assert e["state"] == "EXIT" and e["exit_code"] == 125
    assert not job_dir(jid).exists()


def test_env_forwarding_and_hygiene(broker, submit, job_dir, project_dir):
    jid = submit(["env"], env={"OMP_NUM_THREADS": "4"}, gpus=0)["job_id"]
    broker.wait(jid)
    env = dict(
        line.split("=", 1)
        for line in (job_dir(jid) / "stdout").read_text().splitlines()
        if "=" in line
    )
    assert env["OMP_NUM_THREADS"] == "4" and env["LSB_JOBID"] == jid and env["CSUB_JOB_ID"] == jid
    assert env["HOME"] == f"{project_dir}/.csub/home" and env["CSUB_TRANSPORT"] == "unix"
    assert env["CSUB_SESSION"] == "testsess" and env["LSB_DJOB_NUMPROC"] == "1"
    for leaked in (
        "CSUB_FAKE_LSF_STATE",
        "CSUB_FAKE_SANDBOX_LOG",
        "PYTEST_CURRENT_TEST",
        "LSB_JOBGROUP",
    ):
        assert leaked not in env
    err = submit(["true"], env={"PATH": "/x"}, expect_ok=False)["error"]
    assert err["code"] == "invalid_request"


def test_gpu_job(broker, submit, fake_lsf, fake_sandbox, job_dir):
    r = submit(
        ["sh", "-c", "echo $CUDA_VISIBLE_DEVICES"], gpus=1, queue="gpu_short", gpu_mem_gb=40, cpus=2
    )
    jid = r["job_id"]
    assert r["walltime_min"] == 60 and r["estimated_max_cost_usd"] == pytest.approx(2 * 0.05 + 0.10)
    broker.wait(jid)
    assert (job_dir(jid) / "stdout").read_text() == "0\n"
    assert fake_lsf.job(jid)["gpus"] == 1 and fake_lsf.job(jid)["gmem_gb"] == 40
    assert fake_sandbox.for_job(jid)["gpu"] is True


def test_scratch(broker, submit, job_dir, roots, fake_sandbox):
    jid = submit(["sh", "-c", 'echo "$TMPDIR"; touch "$TMPDIR/x"'], scratch=True)["job_id"]
    broker.wait(jid)
    tmpdir = (job_dir(jid) / "stdout").read_text().strip()
    assert tmpdir == str(
        roots.scratch_root / broker.ok({"protocol": 1, "op": "probe"})["user"] / "csub" / jid
    )
    assert not (roots.scratch_root / "x").exists() and not list(roots.scratch_root.rglob("x"))
    assert tmpdir in fake_sandbox.for_job(jid)["rw"]


def test_allow_hosts(broker, submit, job_dir, fake_sandbox, fake_lsf):
    jid = submit(["sh", "-c", 'echo "$CSUB_FAKE_ALLOW_HOSTS"'], allow_hosts=["pypi.org"])["job_id"]
    broker.wait(jid)
    assert (job_dir(jid) / "stdout").read_text() == "pypi.org\n"
    assert fake_sandbox.for_job(jid)["allow"] == ["pypi.org"]
    before = len(fake_lsf.jobs())
    err = submit(["true"], allow_hosts=["evil.example"], expect_ok=False)["error"]
    assert err["code"] == "policy_violation" and len(fake_lsf.jobs()) == before


def test_status_and_kill_scoping(broker, submit, fake_lsf):
    jid = submit(["sleep", "30"])["job_id"]
    e = broker.wait(jid, states=("RUN",))
    assert e["exec_host"] == "fakehost01"
    assert jid in broker.status(session="testsess") and broker.status(session="othersess") == {}
    broker.error({"protocol": 1, "op": "status", "job_ids": ["999999"]}, "not_found")
    broker.error({"protocol": 1, "op": "kill", "job_ids": ["999999"]}, "not_found")
    # a job the user submitted by other means is invisible even if we guess its id
    r = fake_lsf.run("bsub", "-g", "/other/group", stdin="sleep 30\n")
    foreign = r.stdout.split("<")[1].split(">")[0]
    broker.error({"protocol": 1, "op": "status", "job_ids": [foreign]}, "not_found")
    assert foreign not in broker.status(session="testsess")
    k = broker.ok({"protocol": 1, "op": "kill", "job_ids": [jid]})
    assert k["killed"] == [jid] and k["already_finished"] == []
    e = broker.wait(jid)
    assert e["state"] == "EXIT"
    k = broker.ok({"protocol": 1, "op": "kill", "job_ids": [jid]})
    assert k["killed"] == [] and k["already_finished"] == [jid]
    fake_lsf.run("bkill", foreign)


def test_child_jobs_stay_inside_the_parent(
    broker, submit, project_dir, roots, int_policy_dict, policy_path
):
    """A per-job broker (--parent JOB_DIR) confines every request to what the parent got."""
    from tests.integration.conftest import Broker
    from tests.support.policyfile import write_policy

    parent = submit(["sleep", "30"], allow_hosts=["pypi.org", "lab.janelia.org"])
    other = submit(["sleep", "30"], session="othersess")["job_id"]
    child = Broker(broker.cmd + ["--parent", parent["broker_job_dir"]], broker.env)
    sub = project_dir / "sub"
    sub.mkdir()
    elsewhere = roots.allowed / "elsewhere"
    elsewhere.mkdir()
    rw = {"path": str(project_dir), "mode": "rw"}

    def req(**fields):
        job = {
            "command": ["true"],
            "cwd": str(project_dir),
            "mounts": [
                {"path": str(project_dir), "mode": "rw"},
                {"path": str(roots.nrs), "mode": "ro"},
            ],
            "session": "testsess",
            "allow_hosts": ["pypi.org", "api.lab.janelia.org"],
            **fields,
        }
        return {"protocol": 1, "op": "submit", "job": job}

    probe = child.ok({"protocol": 1, "op": "probe"})
    assert probe["allowed_hosts"] == ["lab.janelia.org", "pypi.org"]
    assert probe["allowed_images"] == [parent["image"]]
    assert probe["allowed_roots"] == [str(project_dir), str(roots.nrs)]
    assert probe["readonly_roots"] == [str(roots.nrs)]
    # policy tightened after the parent started: probe shows the intersection
    int_policy_dict["broker"]["allowed_hosts"].remove("*.janelia.org")
    int_policy_dict["broker"]["allowed_roots"].remove(str(roots.nrs))
    write_policy(policy_path, int_policy_dict)
    probe = child.ok({"protocol": 1, "op": "probe"})
    assert probe["allowed_hosts"] == ["pypi.org"] and probe["allowed_roots"] == [str(project_dir)]
    int_policy_dict["broker"]["allowed_hosts"].append("*.janelia.org")
    int_policy_dict["broker"]["allowed_roots"].append(str(roots.nrs))
    write_policy(policy_path, int_policy_dict)
    ok = child.ok(req(cwd=str(sub), mounts=[{"path": str(sub), "mode": "rw"}]))
    assert ok["session"] == "testsess"
    pv = "policy_violation"
    wider = req(mounts=[rw, {"path": str(elsewhere)}])
    assert "outside the parent" in child.error(wider, pv)["message"]
    upgraded = req(mounts=[rw, {"path": str(roots.nrs), "mode": "rw"}])
    assert "read-only in the parent" in child.error(upgraded, pv)["message"]
    assert (
        "not allowed in the parent" in child.error(req(allow_hosts=["github.com"]), pv)["message"]
    )
    assert "parent's image" in child.error(req(image="ghcr.io/test/other-x:1"), pv)["message"]
    assert "parent's session" in child.error(req(session="othersess"), pv)["message"]
    child.error({"protocol": 1, "op": "status", "session": "othersess"}, pv)
    child.error({"protocol": 1, "op": "kill", "job_ids": [other]}, "not_found")
    assert set(child.status(session="testsess")) >= {parent["job_id"], ok["job_id"]}
    broker.ok({"protocol": 1, "op": "kill", "job_ids": [parent["job_id"], other]})


def test_top_level_envelope(broker, project_dir, roots):
    """--envelope-* bounds a top-level broker to the client sandbox's mounts and hosts."""
    from tests.integration.conftest import Broker

    env = Broker(
        broker.cmd
        + ["--envelope-mount", f"{project_dir}:rw", "--envelope-mount", f"{roots.nrs}:ro"]
        + ["--envelope-host", "pypi.org"],
        broker.env,
    )
    elsewhere = roots.allowed / "elsewhere"
    elsewhere.mkdir()
    sub = project_dir / "sub"
    sub.mkdir()

    def req(**fields):
        job = {
            "command": ["true"],
            "cwd": str(project_dir),
            "mounts": [
                {"path": str(project_dir), "mode": "rw"},
                {"path": str(roots.nrs), "mode": "ro"},
            ],
            "session": "anysess",
            "allow_hosts": ["pypi.org"],
            **fields,
        }
        return {"protocol": 1, "op": "submit", "job": job}

    pv = "policy_violation"
    probe = env.ok({"protocol": 1, "op": "probe"})
    assert probe["allowed_roots"] == [str(project_dir), str(roots.nrs)]
    assert probe["readonly_roots"] == [str(roots.nrs)]
    assert probe["allowed_hosts"] == ["pypi.org"]
    assert "default_image" in probe  # not narrowed: a top-level envelope has no image
    assert (
        env.ok(req(cwd=str(sub), mounts=[{"path": str(sub), "mode": "rw"}]))["session"] == "anysess"
    )
    wider = req(mounts=[{"path": str(project_dir), "mode": "rw"}, {"path": str(elsewhere)}])
    assert "outside the parent" in env.error(wider, pv)["message"]
    rw = {"path": str(project_dir), "mode": "rw"}
    upgraded = req(mounts=[rw, {"path": str(roots.nrs), "mode": "rw"}])
    assert "read-only in the parent" in env.error(upgraded, pv)["message"]
    assert "not allowed in the parent" in env.error(req(allow_hosts=["github.com"]), pv)["message"]
    # the same request through the unbounded broker is fine: the envelope is what refused it
    assert broker.ok(wider)


def test_kill_session(broker, submit):
    ids = [submit(["sleep", "30"])["job_id"] for _ in range(2)]
    for jid in ids:
        broker.wait(jid, states=("RUN",))
    k = broker.ok({"protocol": 1, "op": "kill", "session": "testsess"})
    assert k["killed"] == ids
    for jid in ids:
        assert broker.wait(jid)["state"] == "EXIT"


def test_dependencies(broker, submit, fake_lsf):
    a = submit(["sleep", "1"])["job_id"]
    b = submit(["true"], depends_on=[a])["job_id"]
    assert fake_lsf.job(b)["deps"] == [["done", a]]
    assert broker.status(b)[b]["state"] == "PEND"
    assert broker.wait(b, timeout=15)["state"] == "DONE"
    f = submit(["sh", "-c", "exit 1"])["job_id"]
    c = submit(["true"], depends_on=[{"job_id": f, "when": "ended"}])["job_id"]
    assert fake_lsf.job(c)["deps"] == [["ended", f]]
    assert broker.wait(c, timeout=15)["state"] == "DONE"
    assert submit(["true"], depends_on=["999999"], expect_ok=False)["error"]["code"] == "not_found"


def test_history_merge(broker, submit, fake_lsf):
    jid = submit(["sh", "-c", "exit 2"])["job_id"]
    broker.wait(jid)
    broker.env["CSUB_FAKE_LSF_CLEAN_PERIOD_S"] = "0"
    time.sleep(0.2)
    e = broker.status(jid)[jid]
    assert e["state"] == "EXIT" and e["exit_code"] == 2 and e["source"] == "exit_code"
    assert broker.status(session="testsess")[jid]["source"] == "exit_code"


def test_bsub_failure_leaves_no_state(broker, submit, tmp_home):
    broker.env["CSUB_FAKE_LSF_BSUB_FAIL"] = "Bad queue name. Job not submitted."
    err = submit(["true"], expect_ok=False)["error"]
    assert err["code"] == "lsf_error" and "Bad queue name" in err["message"]
    assert not list((tmp_home / ".csub" / "jobs" / ".pending").iterdir())
    assert not [p for p in (tmp_home / ".csub" / "jobs").iterdir() if p.name.isdigit()]


def test_rejections_through_the_wire(broker, submit, tmp_home, fake_lsf):
    assert (
        submit(["true"], image="localhost/x:1", expect_ok=False)["error"]["code"]
        == "policy_violation"
    )
    assert (
        submit(["true"], mounts=[{"path": str(tmp_home)}], expect_ok=False)["error"]["code"]
        == "policy_violation"
    )
    assert (
        submit("echo a\n#BSUB -q gpu_h100\n", shell=True, expect_ok=False)["error"]["code"]
        == "invalid_request"
    )
    assert (
        submit(["true"], name="my-int-job", expect_ok=False)["error"]["code"] == "policy_violation"
    )
    assert (
        submit(["true"], cpus=128, queue="local", walltime_min=20160, expect_ok=False)["error"][
            "code"
        ]
        == "policy_violation"
    )
    assert fake_lsf.jobs() == {}


def test_malformed_requests(broker):
    resp, rc, _ = broker.raw("not json")
    assert rc == 1 and resp["error"]["code"] == "invalid_request"
    resp, rc, _ = broker.raw('{"protocol": 2, "op": "probe"}')
    assert resp["error"]["code"] == "invalid_request"
    resp, rc, _ = broker.raw('{"protocol": 1, "op": "status"}')
    assert "session" in resp["error"]["message"]


def test_broken_policy_is_reported(broker, policy_path):
    policy_path.write_text("[broker\n")
    resp, rc, stderr = broker.raw('{"protocol": 1, "op": "probe"}')
    assert rc == 1 and resp["error"]["code"] == "broker_misconfigured" and "Traceback" not in stderr


def test_pending_cap(broker, submit, int_policy_dict, policy_path, fake_lsf):
    from tests.support.policyfile import write_policy

    int_policy_dict["limits"]["max_pending_per_session"] = 1
    write_policy(policy_path, int_policy_dict)
    a = submit(["sleep", "30"])["job_id"]
    broker.wait(a, states=("RUN",))
    b = submit(["true"], depends_on=[a])["job_id"]  # stays PEND behind a
    assert broker.status(b)[b]["state"] == "PEND"
    err = submit(["true"], expect_ok=False)["error"]
    assert err["code"] == "policy_violation" and "pending" in err["message"]
    broker.ok({"protocol": 1, "op": "kill", "session": "testsess"})


def test_parallel_submits(broker, submit):
    with ThreadPoolExecutor(5) as ex:
        ids = list(ex.map(lambda _: submit(["true"])["job_id"], range(5)))
    assert len(set(ids)) == 5
    for jid in ids:
        assert broker.wait(jid)["state"] == "DONE"
