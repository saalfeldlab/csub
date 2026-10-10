import os
import re
from pathlib import Path

import pytest

from csub.broker.policy import PolicyError, load_policy, parse_policy
from tests.support.policyfile import JANELIA_POLICY, write_policy


def test_janelia_policy_parses(tmp_home):
    p = load_policy(JANELIA_POLICY, home=str(tmp_home), check_fs=False)
    assert set(p.queues) >= {
        "short",
        "local",
        "gpu_short",
        "gpu_l4",
        "gpu_a100",
        "gpu_h100",
        "gpu_h200",
        "gpu_b300",
    }
    assert p.queues["gpu_a100"].mem_per_slot_mb == 40960
    assert p.queues["gpu_rtx6000"].max_gpus == 1
    assert p.limits.cpu_short_queue == "short"
    assert p.state_dir == str(tmp_home / ".csub")  # ~ expanded against the injected home


def test_fixture_policy_loads_and_probe_summary(policy):
    s = policy.probe_summary()
    assert s["queues"]["short"]["mem_per_slot_mb"] == 15360
    assert "name" not in s["queues"]["short"]
    assert s["limits"]["max_slots"] == 128
    assert s["lsf_extra_enabled"] is False
    assert "state_dir" not in s and "sandbox_scripts_dir" not in s


def test_load_from_file(policy_dict, tmp_home):
    path = write_policy(tmp_home / ".config" / "csub" / "broker.toml", policy_dict)
    p = load_policy(path, home=str(tmp_home))
    assert p.source == str(path)
    with pytest.raises(PolicyError, match="not found"):
        load_policy(tmp_home / "nope.toml", home=str(tmp_home))
    (tmp_home / "bad.toml").write_text("[broker\n")
    with pytest.raises(PolicyError, match="invalid TOML"):
        load_policy(tmp_home / "bad.toml", home=str(tmp_home))


def _reject(make_policy, match, **sections):
    with pytest.raises(PolicyError, match=match) as e:
        make_policy(**sections)
    assert e.value.code == "broker_misconfigured"


def test_unknown_keys_and_tables(make_policy, policy_dict, tmp_home):
    _reject(make_policy, "unknown key", broker={"typo": 1})
    _reject(make_policy, "unknown key", limits={"max_slotz": 1})
    _reject(make_policy, "unknown key", lsf={"bsubb": "x"})
    d = dict(policy_dict)
    d["extra"] = {}
    with pytest.raises(PolicyError, match="unknown table"):
        parse_policy(d, home=str(tmp_home))


def test_types(make_policy):
    _reject(make_policy, "expected int", limits={"max_slots": "128"})
    _reject(make_policy, "expected int", limits={"max_slots": 12.5})
    _reject(make_policy, "expected bool", broker={"scratch": "yes"})
    _reject(make_policy, "expected list", broker={"allowed_roots": "/groups"})
    _reject(make_policy, "expected str", broker={"default_image": 1})


def test_required(policy_dict, tmp_home):
    del policy_dict["broker"]["default_image"]
    with pytest.raises(PolicyError, match="default_image is required"):
        parse_policy(policy_dict, home=str(tmp_home))


def test_image_rules(make_policy):
    _reject(make_policy, "registry-qualified", broker={"default_image": "ubuntu:22.04"})
    _reject(make_policy, "localhost", broker={"default_image": "localhost/agent:latest"})
    _reject(make_policy, "invalid pattern", broker={"allowed_images": ["a b"]})


def test_roots_rules(make_policy):
    _reject(make_policy, "must not be empty", broker={"allowed_roots": []})
    _reject(make_policy, "absolute", broker={"allowed_roots": ["groups/lab"]})


def test_misc_rules(make_policy):
    _reject(make_policy, "hostname", broker={"allowed_hosts": ["PyPI.org"]})
    _reject(make_policy, "env_allow", broker={"env_allow": ["BAD-NAME"]})
    _reject(make_policy, "invalid regex", broker={"lsf_extra_allow": ["("]})
    _reject(make_policy, "forbidden_name_tokens", broker={"forbidden_name_tokens": ["in t"]})
    _reject(make_policy, "max_name_len", broker={"max_name_len": 3})
    _reject(make_policy, ">= 0", limits={"max_estimated_cost_usd": -1})
    _reject(make_policy, ">= 1", limits={"max_slots": 0})
    _reject(make_policy, "must not be empty", lsf={"bsub": ""})
    _reject(make_policy, "timeout_s", lsf={"timeout_s": 0})


def test_queue_rules(make_policy, policy_dict):
    _reject(make_policy, "not defined", limits={"cpu_short_queue": "nope"})
    _reject(make_policy, "CPU queue", limits={"cpu_default_queue": "gpu_l4"})
    _reject(
        make_policy,
        "slots_per_gpu: required",
        queues={"gpu_x": {"mem_per_slot_mb": 1, "max_walltime_min": 1, "gpu": True}},
    )
    _reject(
        make_policy,
        "only valid for GPU",
        queues={"cpu_x": {"mem_per_slot_mb": 1, "max_walltime_min": 1, "slots_per_gpu": 2}},
    )
    _reject(
        make_policy,
        "parallel",
        queues={"gpu_l4_parallel": {"mem_per_slot_mb": 1, "max_walltime_min": 1}},
    )
    _reject(make_policy, "mem_per_slot_mb is required", queues={"q": {"max_walltime_min": 1}})
    _reject(
        make_policy,
        "invalid queue name",
        queues={"bad queue": {"mem_per_slot_mb": 1, "max_walltime_min": 1}},
    )
    policy_dict["queues"] = {}
    with pytest.raises(PolicyError, match="queues"):
        parse_policy(
            policy_dict, home=str(policy_dict["broker"]["state_dir"]).rsplit("/.csub", 1)[0]
        )


def test_broker_dirs_must_not_be_agent_writable(make_policy, roots, scripts_dir, tmp_home):
    # Under $HOME is fine (the fixture default); under an allowed root but outside $HOME is not.
    exposed = roots.allowed / "scripts"
    exposed.mkdir()
    s = exposed / "podman-run.sh"
    s.write_text("#!/bin/sh\n")
    os.chmod(s, 0o755)
    _reject(
        make_policy,
        "sandbox_scripts_dir.*allowed root",
        broker={"sandbox_scripts_dir": str(exposed)},
    )
    _reject(make_policy, "state_dir.*allowed root", broker={"state_dir": str(roots.nrs / "state")})
    _reject(
        make_policy,
        "scratch_root.*allowed root",
        broker={"scratch_root": str(roots.allowed / "scratch")},
    )
    # Outside every allowed root and outside home is also fine.
    p = make_policy(broker={"state_dir": str(roots.outside / "state")})
    assert p.state_dir == str(roots.outside / "state")


def test_home_inside_allowed_root_is_ok(tmp_path: Path, policy_dict):
    """Janelia homes are /groups/<lab>/home/<user>: $HOME under the allowed root must work."""
    base = tmp_path.resolve()
    root = base / "groups" / "lab2"
    home = root / "home" / "u"
    scripts = home / ".local" / "share" / "csub" / "agentic-sandbox" / "scripts"
    scripts.mkdir(parents=True)
    s = scripts / "podman-run.sh"
    s.write_text("#!/bin/sh\n")
    os.chmod(s, 0o755)
    policy_dict["broker"].update(
        {
            "state_dir": "~/.csub",
            "sandbox_scripts_dir": "~/.local/share/csub/agentic-sandbox/scripts",
            "allowed_roots": [str(root)],
        }
    )
    p = parse_policy(policy_dict, home=str(home))
    assert p.state_dir == str(home / ".csub")


def test_fs_checks(make_policy, scripts_dir, tmp_home):
    os.chmod(scripts_dir / "podman-run.sh", 0o644)
    _reject(make_policy, "not executable")
    os.remove(scripts_dir / "podman-run.sh")
    _reject(make_policy, "missing")
    (scripts_dir / "podman-run.sh").write_text("#!/bin/sh\n")
    os.chmod(scripts_dir / "podman-run.sh", 0o755)
    _reject(
        make_policy, "profile.*not readable", lsf={"profile": str(tmp_home / "no-such-profile.sh")}
    )
    prof = tmp_home / "profile.sh"
    prof.write_text("")
    assert make_policy(lsf={"profile": str(prof)}).lsf.profile == str(prof)
    # With submit_host the profile belongs to the remote host, so it need not exist here.
    p = make_policy(lsf={"profile": str(tmp_home / "remote.sh"), "submit_host": "submit"})
    assert p.lsf.submit_host == "submit"


def test_lsf_extra_patterns_compile(make_policy):
    p = make_policy(broker={"lsf_extra_allow": [r"-P \w+"]})
    assert isinstance(p.lsf_extra_allow[0], re.Pattern)
    assert p.probe_summary()["lsf_extra_enabled"] is True


def test_sandbox_setting(make_policy, policy):
    import pytest

    from csub.broker.policy import PolicyError

    assert policy.sandbox == "podman"
    p = make_policy(broker={"sandbox": "bwrap"})
    assert p.sandbox == "bwrap"
    with pytest.raises(PolicyError, match="sandbox: expected one of"):
        make_policy(broker={"sandbox": "docker"})
