"""Status entries built from bjobs rows and the broker's records."""

from types import SimpleNamespace

from csub.broker.lsf import BjobsRow
from csub.broker.ops import _entry
from csub.broker.state import JobRecord, StateDir


def row(index, stat, exit_code=None, name="csub-sweep"):
    return BjobsRow(
        job_id="7",
        stat=stat,
        exit_code=exit_code,
        queue="short",
        exec_host=None if stat == "PEND" else "h01",
        job_name=f"{name}[{index}]" if index else name,
        index=index,
    )


def ctx(tmp_path):
    return SimpleNamespace(state=StateDir(tmp_path / "state"))


def record(array_size):
    return JobRecord("7", "u", "s", "sweep", "short", "/w", "img", 0.0, array_size)


def test_array_entry_from_record_and_rows(tmp_path):
    rows = [row(1, "DONE"), row(2, "EXIT", 3), row(3, "RUN"), row(4, "PEND")]
    e = _entry(record(4), rows, ctx(tmp_path), "7")
    assert e["state"] == "RUN" and e["exit_code"] is None and e["name"] == "sweep"
    assert e["array"] == {
        "size": 4,
        "counts": {"DONE": 1, "EXIT": 1, "RUN": 1, "PEND": 1},
        "failed": [2],
    }


def test_array_without_a_record_yet_is_still_one_array(tmp_path):
    # bsub has accepted the array, the broker has not written meta.json yet: the session
    # listing must not show one element's state as the whole job.
    rows = [row(1, "RUN"), row(2, "PEND"), row(3, "PEND")]
    e = _entry(None, rows, ctx(tmp_path), "7")
    assert e["state"] == "RUN" and e["array"] == {
        "size": 3,
        "counts": {"RUN": 1, "PEND": 2},
        "failed": [],
    }
    assert e["name"] == "csub-sweep" and e["cwd"] is None and e["queue"] == "short"


def test_ordinary_job_without_a_record_is_unchanged(tmp_path):
    e = _entry(None, [row(0, "RUN")], ctx(tmp_path), "7")
    assert e["state"] == "RUN" and "array" not in e and e["name"] == "csub-sweep"


def test_array_elements_bjobs_forgot_are_read_from_exit_codes(tmp_path):
    c = ctx(tmp_path)
    d = c.state.job_dir("7")
    for index, code in ((1, 0), (2, 5)):
        (d / str(index)).mkdir(parents=True)
        (d / str(index) / "exit_code").write_text(f"{code}\n")
    e = _entry(record(3), [], c, "7")
    assert e["state"] == "EXIT" and e["exit_code"] == 5 and e["source"] == "exit_code"
    assert e["array"] == {"size": 3, "counts": {"DONE": 1, "EXIT": 1, "UNKNOWN": 1}, "failed": [2]}
