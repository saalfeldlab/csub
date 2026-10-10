"""The csub CLI through the real broker and fakes."""

import json
import shlex
import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def cli(broker_cmd, broker_env, project_dir, roots):
    env = {
        **broker_env,
        "CSUB_TRANSPORT": "local",
        "CSUB_BROKER_CMD": shlex.join(broker_cmd),
        "CSUB_MOUNTS": f"{project_dir}:rw,{roots.nrs}:ro",
        "CSUB_SESSION": "clisess",
        "CSUB_WAIT_MAX_POLL_S": "0.5",
    }

    def _run(*args, check=True):
        cp = subprocess.run(
            [sys.executable, "-m", "csub.cli", *args],
            capture_output=True,
            text=True,
            env=env,
            cwd=project_dir,
            timeout=120,
        )
        if check:
            assert cp.returncode == 0, cp.stderr
        return cp

    return _run


def test_cli_roundtrip(cli, project_dir):
    cp = cli(
        "submit", "--name", "hello", "--walltime", "10", "--", "sh", "-c", "echo hi; echo oops >&2"
    )
    assert (
        cp.stdout.startswith("Job ")
        and "submitted to short via podman (1 slot, 10 min, est. max $0.01) billed to testlab"
        in cp.stdout
    )
    jid = cp.stdout.split()[1]
    cp = cli("wait", jid, "--timeout", "60")
    assert f"Job {jid}: DONE (exit 0)" in cp.stdout and "hi\n" in cp.stdout and "oops" in cp.stdout
    cp = cli("logs", jid)
    assert cp.stdout == "hi\n--- stderr ---\noops\n"
    cp = cli("status")
    assert jid in cp.stdout and "DONE" in cp.stdout
    cp = cli("--json", "status", jid)
    assert json.loads(cp.stdout)[0]["state"] == "DONE"
    cp = cli("--json", "probe")
    assert len(json.loads(cp.stdout)["queues"]) == 11
    assert (project_dir / ".csub" / "jobs" / jid / "stdout").read_text() == "hi\n"


def test_cli_failure_exit_codes(cli):
    cp = cli("submit", "--wait", "--timeout", "60", "--", "sh", "-c", "exit 3", check=False)
    assert cp.returncode == 3 and "EXIT (exit 3)" in cp.stdout
    cp = cli("submit", "--image", "localhost/x:1", "--", "true", check=False)
    assert cp.returncode == 3 and "csub: policy_violation" in cp.stderr
    cp = cli("status", "999999", check=False)
    assert cp.returncode == 6 and "not_found" in cp.stderr
    cp = cli(
        "submit", "--", "true", "--cpus", "x", check=False
    )  # after --, everything is the command
    assert cp.returncode == 0


def test_cli_kill(cli):
    jid = cli("submit", "--", "sleep", "30").stdout.split()[1]
    cp = cli("kill", "--all")
    assert f"Job {jid} is being terminated" in cp.stdout
    cp = cli("wait", jid, "--timeout", "30", check=False)
    assert cp.returncode in (1, 130, 137, 143) and "EXIT" in cp.stdout
