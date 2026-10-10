"""LSF command construction and output parsing with canned subprocess results."""

import subprocess

import pytest

from csub.broker.lsf import LsfError, LsfRunner
from csub.broker.policy import LsfConfig
from csub.protocol import CsubError

JANELIA_BSUB_OUT = "This job will be billed to flyem\nJob <1234> is submitted to queue <short>.\n"


class Runner:
    """Records argv and returns the next canned CompletedProcess."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, argv, **kw):
        self.calls.append((argv, kw))
        rc, out, err = self.results.pop(0)
        return subprocess.CompletedProcess(argv, rc, out, err)


def runner(*results, profile=""):
    r = Runner(*results)
    return LsfRunner(
        LsfConfig(profile=profile, bsub="/x/bsub", bjobs="/x/bjobs", bkill="/x/bkill"), run=r
    ), r


def test_bsub_parses_janelia_output():
    lsf, r = runner((0, JANELIA_BSUB_OUT, "Warning: something\n"))
    res = lsf.bsub(["-q", "short"], "#!/bin/sh\n")
    assert (res.job_id, res.queue, res.billing_group) == ("1234", "short", "flyem")
    argv, kw = r.calls[0]
    assert argv == ["/x/bsub", "-q", "short"] and kw["input"] == "#!/bin/sh\n"


def test_bsub_default_queue_and_no_billing():
    lsf, _ = runner((0, "Job <7> is submitted to default queue <local>.\n", ""))
    res = lsf.bsub([], "x")
    assert (res.job_id, res.queue, res.billing_group) == ("7", "local", None)


def test_bsub_failures():
    lsf, _ = runner((255, "", "gpu_h100: Bad queue name. Job not submitted.\n"))
    with pytest.raises(LsfError, match="Bad queue name") as e:
        lsf.bsub([], "x")
    assert e.value.code == "lsf_error"
    lsf, _ = runner((0, "something unexpected\n", ""))
    with pytest.raises(LsfError, match="something unexpected"):
        lsf.bsub([], "x")


def test_bsub_timeout_and_missing():
    def boom(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 1)

    lsf = LsfRunner(LsfConfig(profile="", bsub="/x/bsub"), run=boom)
    with pytest.raises(LsfError, match="did not finish"):
        lsf.bsub([], "x")

    def missing(argv, **kw):
        raise FileNotFoundError("no bsub")

    lsf = LsfRunner(LsfConfig(profile="", bsub="/x/bsub"), run=missing)
    with pytest.raises(LsfError, match="not found"):
        lsf.bsub([], "x")


def test_submit_host_runs_lsf_commands_remotely(monkeypatch):
    """Every command becomes `ssh host <quoted local argv>`: the profile is sourced on the far
    side and the bsub script still travels on stdin."""
    monkeypatch.delenv("LSB_JOBID", raising=False)
    r = Runner((0, JANELIA_BSUB_OUT, ""))
    cfg = LsfConfig(profile="/no/such/profile.lsf", bsub="/x/bsub", submit_host="submit")
    res = LsfRunner(cfg, run=r).bsub(["-q", "short"], "#!/bin/sh\n")
    assert res.job_id == "1234"
    argv, kw = r.calls[0]
    assert argv[0] == "ssh" and argv[-2] == "submit" and "BatchMode=yes" in argv
    assert argv[-1].startswith("bash -c ") and "/no/such/profile.lsf" in argv[-1]
    assert argv[-1].endswith(" /x/bsub -q short")
    assert kw["input"] == "#!/bin/sh\n"


@pytest.mark.parametrize("where", ["inside_job", "on_submit_host"])
def test_submit_host_ignored_where_lsf_is_local(monkeypatch, where):
    """The per-job broker on a compute node and a broker on the submit host read the same
    policy; both must call bsub directly. LSF files being visible proves nothing (shared NFS)."""
    monkeypatch.delenv("LSB_JOBID", raising=False)
    if where == "inside_job":
        monkeypatch.setenv("LSB_JOBID", "123")
    else:
        monkeypatch.setattr("csub.broker.lsf.is_this_host", lambda name: name == "submit")
    r = Runner((0, JANELIA_BSUB_OUT, ""))
    cfg = LsfConfig(profile="/etc/profile.d/lsf.sh", bsub="/x/bsub", submit_host="submit")
    LsfRunner(cfg, run=r).bsub([], "x")
    argv, _ = r.calls[0]
    assert argv[0] == "bash" and "ssh" not in argv and "submit" not in argv


def test_is_this_host():
    import socket

    from csub.broker.lsf import is_this_host

    assert is_this_host(socket.gethostname())
    assert not is_this_host("no-such-host.invalid")


def test_bjobs_rows():
    out = (
        "1|RUN|-|short|4*h01:4*h02|csub-a\n2|EXIT|3|gpu_l4|h03|csub-b\n"
        "3|DONE|-|local|h04|-\n4|PSUSP|-|local|-|x\nJOBID junk\n"
    )
    lsf, r = runner((0, out, ""))
    rows = lsf.bjobs("/csub/u/s", ["1", "2"])
    argv, _ = r.calls[0]
    assert argv[:4] == ["/x/bjobs", "-a", "-noheader", "-o"]
    assert argv[4] == 'jobid stat exit_code queue exec_host job_name delimiter="|"'
    assert argv[5:] == ["-g", "/csub/u/s", "1", "2"]
    assert [(x.job_id, x.state, x.exit_code) for x in rows] == [
        ("1", "RUN", None),
        ("2", "EXIT", 3),
        ("3", "DONE", None),
        ("4", "SUSP", None),
    ]
    assert (
        rows[0].exec_host == "4*h01:4*h02"
        and rows[2].job_name is None
        and rows[3].exec_host is None
    )


def test_bjobs_empty_variants():
    for err in ("No unfinished job found\n", "Job <5> is not found\n", "No job group found\n"):
        lsf, _ = runner((255, "", err))
        assert lsf.bjobs("/g") == []
    lsf, _ = runner((255, "", "LSF daemon (LIM) not responding\n"))
    with pytest.raises(LsfError, match="LIM"):
        lsf.bjobs("/g")


def test_bkill_parsing():
    out = "Job <1> is being terminated\nJob <2> is being terminated\n"
    err = "Job <3>: Job has already finished\nJob <4>: No matching job found\n"
    lsf, r = runner((255, out, err))
    assert lsf.bkill("/g", ["1", "2", "3", "4"]) == {
        "1": "killed",
        "2": "killed",
        "3": "finished",
        "4": "not_found",
    }
    assert r.calls[0][0] == ["/x/bkill", "-g", "/g", "1", "2", "3", "4"]
    lsf, r = runner((0, "Job <9> is being terminated\n", ""))
    assert lsf.bkill("/g", None) == {"9": "killed"}
    assert r.calls[0][0] == ["/x/bkill", "-g", "/g", "0"]
    lsf, _ = runner((255, "", "No matching job found\n"))
    assert lsf.bkill("/g", None) == {}
    lsf, _ = runner((1, "", "bkill: cannot connect\n"))
    with pytest.raises(LsfError, match="cannot connect"):
        lsf.bkill("/g", ["1"])


def test_self_shadow_guard(tmp_path):
    prefix = tmp_path / "venv"
    (prefix / "bin").mkdir(parents=True)
    shim = prefix / "bin" / "bsub"
    shim.write_text("#!/bin/sh\n")
    r = Runner((0, f"{shim}\n", ""))
    lsf = LsfRunner(LsfConfig(profile="", bsub="bsub"), run=r, own_prefix=str(prefix))
    with pytest.raises(CsubError, match="shadowed") as e:
        lsf.bsub([], "x")
    assert e.value.code == "broker_misconfigured"
    # a bsub outside our prefix is fine, and the check runs only once
    r = Runner((0, "/usr/bin/bsub\n", ""), (0, JANELIA_BSUB_OUT, ""), (0, JANELIA_BSUB_OUT, ""))
    lsf = LsfRunner(LsfConfig(profile="", bsub="bsub"), run=r, own_prefix=str(prefix))
    lsf.bsub([], "x")
    lsf.bsub([], "x")
    assert len(r.calls) == 3
