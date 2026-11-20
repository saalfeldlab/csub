"""LSF command construction and output parsing with canned subprocess results."""

import subprocess

import pytest

from csub.broker.lsf import LsfError, LsfRunner
from csub.broker.policy import LsfConfig

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


def test_bjobs_rows():
    out = (
        "1|RUN|-|short|4*h01:4*h02|csub-a|-|311 Mbytes|213 Mbytes|20 second(s)|15 G\n"
        "2|EXIT|3|gpu_l4|h03|csub-b|1131.1 second(s)|6.8 Gbytes|3.5 Gbytes|62 second(s)|-\n"
        "3|DONE|-|local|h04|-|0.7 second(s)|311 Mbytes|227 Mbytes|33 second(s)|15 G\n"
        "4|PSUSP|-|local|-|x|-|-|-|0 second(s)|720 G\nJOBID junk\n"
    )
    lsf, r = runner((0, out, ""))
    rows = lsf.bjobs("/csub/u/s", ["1", "2"])
    argv, _ = r.calls[0]
    assert argv[:4] == ["/x/bjobs", "-a", "-noheader", "-o"]
    assert argv[4] == (
        "jobid stat exit_code queue exec_host job_name "
        'cpu_used max_mem avg_mem run_time memlimit delimiter="|"'
    )
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


def test_bjobs_usage():
    out = (
        "1|RUN|-|short|h01|a|-|311 Mbytes|213 Mbytes|20 second(s)|15 G\n"
        "2|EXIT|3|gpu_l4|h03|b|1131.1 second(s)|6.8 Gbytes|3.5 Gbytes|62 second(s)|-\n"
        "3|PEND|-|local|-|c|-|-|-|0 second(s)|720 G\n"
    )
    lsf, _ = runner((0, out, ""))
    running, failed, pending = lsf.bjobs("/g")
    assert running.usage() == {
        "cpu_time_s": None,
        "max_mem_mb": 311.0,
        "avg_mem_mb": 213.0,
        "run_time_s": 20.0,
        "mem_limit_mb": 15360.0,
    }
    assert failed.cpu_time_s == 1131.1 and failed.max_mem_mb == 6963.2
    assert failed.run_time_s == 62.0 and failed.mem_limit_mb is None
    assert pending.max_mem_mb is None and pending.run_time_s == 0.0
    assert pending.mem_limit_mb == 720 * 1024


def test_parse_amounts():
    from csub.broker.lsf import parse_mb, parse_seconds

    assert parse_seconds("33 second(s)") == 33.0 and parse_seconds("1.09 sec.") == 1.09
    assert parse_seconds("-") is None and parse_seconds("") is None and parse_seconds(None) is None
    assert parse_seconds("12 Mbytes") is None
    assert parse_mb("47 MB") == 47.0 and parse_mb("1.23 MB") == 1.2
    assert parse_mb("2 Gbytes") == 2048.0 and parse_mb("15 G") == 15360.0
    assert parse_mb("512 Kbytes") == 0.5 and parse_mb("1 Tbytes") == 1024.0 * 1024
    assert parse_mb("-") is None and parse_mb("12") is None and parse_mb("3 parsecs") is None


LSF_OUT = """inside ok
Resource usage summary: not this one, the job printed it

------------------------------------------------------------
Successfully completed.

Resource usage summary:

    CPU time :                                   1.09 sec.
    Max Memory :                                 47 MB
    Average Memory :                             1.23 MB
    Total Requested Memory :                     15360.00 MB
    Delta Memory :                               15313.00 MB
    Max Swap :                                   -
    Max Processes :                              9
    Max Threads :                                23
    Run time :                                   1208 sec.
    Turnaround time :                            1209 sec.

The output (if any) is above this job summary.
"""


def test_parse_usage_summary():
    from csub.broker.lsf import parse_usage_summary

    assert parse_usage_summary(LSF_OUT) == {
        "cpu_time_s": 1.09,
        "max_mem_mb": 47.0,
        "avg_mem_mb": 1.2,
        "run_time_s": 1208.0,
        "mem_limit_mb": 15360.0,
    }
    nothing = dict.fromkeys(parse_usage_summary(LSF_OUT))
    assert parse_usage_summary("") == nothing
    assert parse_usage_summary("CPU time : 5 sec.\nMax Memory : 9 MB\n") == nothing
    # a job that never ran has a summary with no figures
    killed = "Resource usage summary:\n\n    CPU time :   -\n    Max Memory :   -\n"
    assert parse_usage_summary(killed) == nothing
