"""``csub-broker``: one JSON request in, one JSON response out — or ``--serve SOCKET``.

The same handler serves the SSH forced command (stdin/stdout), the local transport, and
``--serve`` sockets, per job (started by the wrapper) or started by hand for a sandbox. Policy
is loaded per request so edits take effect immediately and a broken policy is reported, never
half-applied.
"""

from __future__ import annotations

import argparse
import json
import os
import pwd
import shlex
import sys
import traceback
from collections.abc import Callable
from typing import TextIO

from csub import __version__
from csub.broker.lsf import LsfRunner
from csub.broker.ops import BrokerContext, handle
from csub.broker.policy import default_policy_path, load_policy
from csub.broker.resolve import Parent
from csub.broker.state import REQUEST_NAME, StateDir
from csub.protocol import LIMITS, CsubError, Request, error_response


def _identity(home_override: str | None) -> tuple[str, str]:
    pw = pwd.getpwuid(os.getuid())
    return pw.pw_name, os.path.realpath(home_override or pw.pw_dir)


def broker_command(policy_path: str | None, home_override: str | None) -> str:
    """How the wrapper restarts this broker on the compute node (shell-quoted)."""
    argv = [sys.executable, "-m", "csub.broker"]
    if policy_path:
        argv += ["--policy", policy_path]
    if home_override:
        argv += ["--home", home_override]
    return shlex.join(argv)


def load_parent(job_dir: str) -> Parent:
    """The resolved request the broker itself wrote for the job this per-job broker serves."""
    try:
        with open(os.path.join(job_dir, REQUEST_NAME)) as f:
            return Parent.from_response(json.load(f)["resolved"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise CsubError("internal_error", f"cannot read the parent job's record: {e}") from None


def build_context(
    policy_path: str | None, home_override: str | None, parent_dir: str | None = None
) -> BrokerContext:
    user, home = _identity(home_override)
    path = policy_path or default_policy_path(home)
    policy = load_policy(path, home=home)
    state = StateDir(policy.state_dir)
    import csub

    protected = tuple(
        os.path.realpath(p)
        for p in (
            policy.state_dir,
            policy.sandbox_scripts_dir,
            os.path.dirname(os.path.abspath(path)),
            os.path.dirname(os.path.abspath(csub.__file__)),
            sys.prefix,
        )
    )
    return BrokerContext(
        policy=policy,
        user=user,
        home=home,
        state=state,
        lsf=LsfRunner(policy.lsf),
        protected_dirs=protected,
        broker_cmd=broker_command(os.path.abspath(path), home_override),
        parent=load_parent(parent_dir) if parent_dir else None,
    )


def handle_text(
    text: str | bytes, ctx_factory: Callable[[], BrokerContext], stderr: TextIO
) -> dict:
    """Parse, dispatch, and turn every failure into a protocol error response."""
    state: StateDir | None = None
    try:
        req = Request.from_json(text)
        ctx = ctx_factory()
        state = ctx.state
        return handle(req, ctx)
    except CsubError as e:
        return e.to_response()
    except Exception:
        if state is not None:
            ref = state.log_exception("request")
            where = f"see {state.log_path} ({ref})"
        else:
            stderr.write(traceback.format_exc())
            where = "see broker stderr"
        return error_response("internal_error", f"internal broker error; {where}")


def _read_request(stdin: TextIO) -> str:
    return stdin.read(LIMITS["max_request_bytes"] + 1)


def main(
    argv: list[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    parser = argparse.ArgumentParser(prog="csub-broker", description=__doc__.splitlines()[0])
    parser.add_argument("--policy", help="policy file (default ~/.config/csub/broker.toml)")
    parser.add_argument("--home", help=argparse.SUPPRESS)  # test hook: override the home directory
    parser.add_argument(
        "--serve", metavar="SOCKET", help="serve requests on a Unix socket until killed"
    )
    parser.add_argument(
        "--parent",
        metavar="JOB_DIR",
        help="broker-owned dir of the job this broker serves; its requests may not exceed it",
    )
    parser.add_argument("--version", action="version", version=f"csub-broker {__version__}")
    args = parser.parse_args(argv)
    os.umask(0o077)

    def ctx_factory() -> BrokerContext:
        return build_context(args.policy, args.home, args.parent)

    if args.serve:
        from csub.broker.serve import serve

        return serve(args.serve, lambda text: handle_text(text, ctx_factory, stderr), stderr=stderr)

    import json

    resp = handle_text(_read_request(stdin), ctx_factory, stderr)
    stdout.write(json.dumps(resp, separators=(",", ":")) + "\n")
    stdout.flush()
    return 0 if resp.get("ok") else 1
