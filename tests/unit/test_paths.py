import pytest

from csub._paths import glob_to_regex, image_registry, is_under, normalize_abs


def test_normalize_abs():
    assert normalize_abs("/a/b/../c/") == "/a/c"
    with pytest.raises(ValueError):
        normalize_abs("relative/path")
    with pytest.raises(ValueError):
        normalize_abs("/a\0b")
    # a newline in a path would start a new line in the spooled job script
    for bad in ("/a\n#BSUB -E x/b", "/a\rb", "/a\tb"):
        with pytest.raises(ValueError, match="control character"):
            normalize_abs(bad)


def test_is_under():
    assert is_under("/a/b", "/a")
    assert is_under("/a", "/a")
    assert not is_under("/ab", "/a")
    assert not is_under("/a", "/a/b")
    assert is_under("/anything", "/")


@pytest.mark.parametrize(
    "pattern, image, ok",
    [
        ("ghcr.io/org/agent:*", "ghcr.io/org/agent:latest", True),
        ("ghcr.io/org/agent:*", "ghcr.io/org/agent:v1.2", True),
        ("ghcr.io/org/agent:*", "ghcr.io/org/agent-evil:latest", False),
        ("ghcr.io/org/*:*", "ghcr.io/org/agent:latest", True),
        ("ghcr.io/org/*:*", "ghcr.io/org/sub/agent:latest", False),
        ("ghcr.io/org/agentic-sandbox-*:*", "ghcr.io/org/agentic-sandbox-lite:latest", True),
    ],
)
def test_glob_to_regex(pattern, image, ok):
    assert bool(glob_to_regex(pattern).fullmatch(image)) is ok


@pytest.mark.parametrize(
    "image, registry",
    [
        ("ghcr.io/org/agent:latest", "ghcr.io"),
        ("localhost/agent:latest", "localhost"),
        ("registry:5000/agent", "registry:5000"),
        ("ubuntu:22.04", None),
        ("library/ubuntu", None),
    ],
)
def test_image_registry(image, registry):
    assert image_registry(image) == registry
