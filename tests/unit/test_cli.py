import io
import json

import pytest

from csub.cli import EXIT_CODES, main, parse_env, parse_mem, parse_walltime
from csub.client.api import Logs, WaitEntry, WaitResult
from csub.protocol import CsubError, JobStatus, KillResult, ProbeResult, SubmitResult


class FakeClient:
    def __init__(self):
        self.calls = []
        self.statuses = []
        self.wait_result = None
        self.error = None

    def submit(self, spec):
        self.calls.append(("submit", spec))
        if self.error:
            raise self.error
        return SubmitResult(
            job_id="42",
            queue="short",
            slots=2,
            walltime_min=30,
            estimated_max_cost_usd=0.05,
            billing_group="lab",
        )

    def status(self, job_ids=None):
        self.calls.append(("status", job_ids))
        return self.statuses

    def wait(self, job_ids, *, timeout_s=None, tail_lines=50):
        self.calls.append(("wait", list(job_ids), timeout_s, tail_lines))
        return self.wait_result

    def kill(self, job_ids=None):
        self.calls.append(("kill", job_ids))
        return KillResult(killed=list(job_ids or ["7"]), already_finished=[])

    def logs(self, job_id, *, stream="both", tail=None, cwd=None):
        self.calls.append(("logs", job_id, stream, tail, cwd))
        return Logs(
            job_id=job_id,
            job_dir="/d",
            stdout="out\n" if stream != "stderr" else None,
            stderr="err\n" if stream != "stdout" else None,
        )

    def probe(self):
        return ProbeResult(
            broker_version="0.1.0",
            user="u",
            queues={"short": {"mem_per_slot_mb": 15360, "max_walltime_min": 60}},
            limits={"max_slots": 128},
            default_image="ghcr.io/x/y:1",
            allowed_roots=["/g"],
            allowed_hosts=["pypi.org"],
        )


def run(client, *argv, stdin=""):
    out, err = io.StringIO(), io.StringIO()
    rc = main(
        list(argv), client_factory=lambda: client, stdout=out, stderr=err, stdin=io.StringIO(stdin)
    )
    return rc, out.getvalue(), err.getvalue()


def test_parsers():
    assert (
        parse_mem("100G") == 102400
        and parse_mem("500") == 500
        and parse_mem("1.5g") == 1536
        and parse_mem("2048K") == 2
    )
    assert (
        parse_walltime("90") == 90
        and parse_walltime("01:30") == 90
        and parse_walltime("1:05") == 65
    )
    for bad in ("abc", "1:99", "10/host", "-5"):
        with pytest.raises(CsubError):
            parse_walltime(bad)
    with pytest.raises(CsubError):
        parse_mem("lots")


def test_submit_maps_flags(tmp_path):
    c = FakeClient()
    rc, out, _ = run(
        c, "submit", "--name", "n", "--cpus", "4", "--mem", "100G", "--gpus", "1",
        "--gpu-mem-gb", "40", "--walltime", "01:30", "-q", "gpu_short", "--image", "ghcr.io/a/b:1",
        "--env", "A=1", "--env", "B=x=y", "--allow", "pypi.org", "--scratch", "--agent", "claude",
        "--depends-on", "12", "--depends-on", "13:ended", "--cwd", str(tmp_path),
        "--session", "s", "--lsf-extra", "-P proj", "--", "python", "train.py", "--lr", "0.1",
    )  # fmt: skip
    assert rc == 0 and out.startswith(
        "Job 42 submitted to short (2 slots, 30 min, est. max $0.05) billed to lab"
    )
    spec = c.calls[0][1]
    assert (
        spec.command == ("python", "train.py", "--lr", "0.1")
        and spec.name == "n"
        and spec.cpus == 4
    )
    assert (
        spec.mem_mb == 102400
        and spec.gpus == 1
        and spec.gpu_mem_gb == 40
        and spec.walltime_min == 90
    )
    assert (
        spec.queue == "gpu_short"
        and spec.image == "ghcr.io/a/b:1"
        and spec.env == {"A": "1", "B": "x=y"}
    )
    assert (
        spec.allow_hosts == ("pypi.org",) and spec.scratch and spec.cwd == str(tmp_path.resolve())
    )
    assert spec.agent == "claude"
    assert [d.to_dict() for d in spec.depends_on] == [
        {"job_id": "12", "when": "done"},
        {"job_id": "13", "when": "ended"},
    ]
    assert spec.session == "s" and spec.lsf_extra == ("-P proj",)


def test_submit_shell_variants(tmp_path):
    c = FakeClient()
    assert run(c, "submit", "--shell", "--script", "-", stdin="#!/bin/bash\necho hi\n")[0] == 0
    assert c.calls[-1][1].command == "#!/bin/bash\necho hi\n" and c.calls[-1][1].shell
    f = tmp_path / "s.sh"
    f.write_text("echo file\n")
    assert (
        run(c, "submit", "--shell", "--script", str(f))[0] == 0
        and c.calls[-1][1].command == "echo file\n"
    )
    assert (
        run(c, "submit", "--shell", "--", "echo", "a;", "ls")[0] == 0
        and c.calls[-1][1].command == "echo a; ls\n"
    )
    rc, _, err = run(c, "submit", "--shell")
    assert rc == EXIT_CODES["invalid_request"] and "--shell needs" in err
    rc, _, err = run(c, "submit", "--script", str(f), "--", "x")
    assert rc == 2 and "requires --shell" in err
    rc, _, err = run(c, "submit")
    assert rc == 2 and "no command" in err
    rc, _, err = run(c, "submit", "--env", "CSUB_TEST_UNSET_VAR", "--", "x")
    assert rc == 2 and "not set in the environment" in err


def test_env_without_value_comes_from_the_environment():
    assert parse_env(["TOKEN", "A=1"], {"TOKEN": "s3cret"}) == {"TOKEN": "s3cret", "A": "1"}
    assert parse_env(["EMPTY"], {"EMPTY": ""}) == {"EMPTY": ""}
    with pytest.raises(CsubError, match="not set"):
        parse_env(["MISSING"], {})


def test_submit_json_and_errors():
    c = FakeClient()
    rc, out, _ = run(c, "--json", "submit", "--", "x")
    assert rc == 0 and json.loads(out)["job_id"] == "42"
    c.error = CsubError("policy_violation", "nope")
    rc, out, err = run(c, "--json", "submit", "--", "x")
    assert (
        rc == 3
        and json.loads(out)["error"]["code"] == "policy_violation"
        and "csub: policy_violation: nope" in err
    )
    c.error = CsubError("transport_error", "down")
    assert run(c, "submit", "--", "x")[0] == 4


def test_submit_wait():
    c = FakeClient()
    c.wait_result = WaitResult(
        jobs=[
            WaitEntry(
                JobStatus("42", "EXIT", exit_code=3),
                stdout_tail="o\n",
                stderr_tail="e\n",
                stdout_path="/p/stdout",
                stderr_path="/p/stderr",
            )
        ],
        timed_out=False,
        elapsed_s=1.0,
    )
    rc, out, _ = run(c, "submit", "--wait", "--timeout", "30", "--tail", "5", "--", "x")
    assert rc == 3 and "Job 42: EXIT (exit 3)" in out and "--- stdout (/p/stdout) ---\no\n" in out
    assert c.calls[-1] == ("wait", ["42"], 30.0, 5)
    rc, out, _ = run(c, "--json", "submit", "--wait", "--", "x")
    data = json.loads(out)
    assert data["submit"]["job_id"] == "42" and data["wait"]["jobs"][0]["state"] == "EXIT"


def test_status_table_and_empty():
    c = FakeClient()
    c.statuses = [
        JobStatus("1", "RUN", queue="short", exec_host="h1", name="a"),
        JobStatus("2", "EXIT", exit_code=130, name="b"),
    ]
    rc, out, _ = run(c, "status", "1", "2")
    lines = out.splitlines()
    assert rc == 0 and lines[0].split() == ["JOBID", "STATE", "EXIT", "QUEUE", "HOST", "NAME"]
    assert lines[1].split() == ["1", "RUN", "-", "short", "h1", "a"] and lines[2].split() == [
        "2",
        "EXIT",
        "130",
        "-",
        "-",
        "b",
    ]
    assert c.calls[-1] == ("status", ["1", "2"])
    c.statuses = []
    assert run(c, "status")[1] == "no jobs\n" and c.calls[-1] == ("status", None)
    c.statuses = [JobStatus("1", "RUN")]
    assert json.loads(run(c, "--json", "status")[1])[0]["state"] == "RUN"


def test_wait_exit_codes():
    c = FakeClient()
    done = JobStatus("1", "DONE", exit_code=0)
    c.wait_result = WaitResult(jobs=[WaitEntry(done)], timed_out=False, elapsed_s=0)
    assert run(c, "wait", "1")[0] == 0
    c.wait_result = WaitResult(
        jobs=[WaitEntry(done), WaitEntry(JobStatus("2", "EXIT", exit_code=7))],
        timed_out=False,
        elapsed_s=0,
    )
    assert run(c, "wait", "1", "2")[0] == 1
    c.wait_result = WaitResult(
        jobs=[WaitEntry(JobStatus("2", "EXIT", exit_code=140))], timed_out=False, elapsed_s=0
    )
    assert run(c, "wait", "2")[0] == 1
    c.wait_result = WaitResult(
        jobs=[WaitEntry(JobStatus("1", "RUN"))], timed_out=True, elapsed_s=10
    )
    rc, out, _ = run(c, "wait", "1", "--timeout", "10")
    assert rc == 124 and "Timed out" in out


def test_kill():
    c = FakeClient()
    rc, out, _ = run(c, "kill", "5")
    assert rc == 0 and out == "Job 5 is being terminated\n" and c.calls[-1] == ("kill", ["5"])
    assert run(c, "kill", "--all")[0] == 0 and c.calls[-1] == ("kill", None)
    rc, _, err = run(c, "kill")
    assert rc == 2 and "--all" in err
    assert json.loads(run(c, "--json", "kill", "1")[1])["killed"] == ["1"]


def test_logs():
    c = FakeClient()
    rc, out, _ = run(c, "logs", "3")
    assert (
        rc == 0
        and out == "out\n--- stderr ---\nerr\n"
        and c.calls[-1] == ("logs", "3", "both", None, None)
    )
    assert run(c, "logs", "3", "--stderr", "--tail", "5", "--cwd", "/w")[1] == "err\n"
    assert c.calls[-1] == ("logs", "3", "stderr", 5, "/w")
    assert json.loads(run(c, "--json", "logs", "3")[1])["stdout"] == "out\n"


def test_logs_follow():
    c = FakeClient()
    c.statuses = [JobStatus("3", "RUN", cwd="/w")]
    texts = iter(["a\n", "a\nb\n", "a\nb\n"])
    states = iter(["RUN", "RUN", "DONE"])

    def logs(job_id, *, stream="both", tail=None, cwd=None):
        return Logs(job_id=job_id, job_dir="/d", stdout=next(texts), stderr="")

    def status(job_ids=None):
        return [JobStatus("3", next(states), cwd="/w")]

    c.logs, c.status = logs, status
    from csub import cli

    out = io.StringIO()
    ns = cli.build_parser().parse_args(["logs", "3", "--follow"])
    rc = cli.cmd_logs(ns, c, out, sleep=lambda s: None)
    assert rc == 0 and out.getvalue() == "a\nb\n"


def test_probe():
    c = FakeClient()
    rc, out, _ = run(c, "probe")
    assert rc == 0 and "short" in out and "ghcr.io/x/y:1" in out and "pypi.org" in out
    assert json.loads(run(c, "--json", "probe")[1])["user"] == "u"


def test_version_and_help():
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0


def test_relay_refuses_its_own_socket(tmp_path):
    from csub.client.config import ClientConfig

    sock = str(tmp_path / "s.sock")
    c = FakeClient()
    c.config = ClientConfig(transport="unix", socket=sock)
    rc, _, err = run(c, "relay", sock)
    assert rc == EXIT_CODES["invalid_request"] and "own socket" in err
