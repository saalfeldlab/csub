"""Path and pattern helpers shared by broker and client. Standard library only."""

from __future__ import annotations

import os
import re


def normalize_abs(path: str) -> str:
    """Return ``os.path.normpath(path)``; raise ``ValueError`` unless ``path`` is absolute."""
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError(f"not an absolute path: {path!r}")
    if any(ord(c) < 32 for c in path):
        # Paths are written into the wrapper script that bsub reads from stdin. A newline
        # there starts a new line that LSF reads as a #BSUB directive; a carriage return is a
        # line end to some tools and invisible in others; a NUL truncates C strings; the rest
        # (tab, escape, backspace...) have no place in a directory name and would only make a
        # log or an error message lie about what the path was. No real mount or cwd contains
        # any of them, so the whole range is refused rather than listing the dangerous ones.
        raise ValueError("path contains a control character")
    return os.path.normpath(path)


def is_under(path: str, root: str) -> bool:
    """True if ``path`` equals ``root`` or lies below it. Both must be normalized."""
    if root == "/":
        return path.startswith("/")
    return path == root or path.startswith(root.rstrip("/") + "/")


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Shell-style glob where ``*`` and ``?`` do not cross ``/``.

    ``fnmatch`` lets ``*`` match ``/``, which would make ``ghcr.io/org/*`` match
    ``ghcr.io/org/evil/image``. Image allowlists need the narrower semantics.
    """
    out: list[str] = []
    for ch in pattern:
        if ch == "*":
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(ch))
    return re.compile("".join(out))


def image_registry(image: str) -> str | None:
    """Return the registry component of an OCI image reference, or None if unqualified.

    Docker's rule: the first path component is a registry iff it contains ``.`` or ``:``
    or is ``localhost``.
    """
    first, sep, _rest = image.partition("/")
    if not sep:
        return None
    if "." in first or ":" in first or first == "localhost":
        return first
    return None
