import json
import os
import socket
import threading

import pytest

from csub.protocol import ProtocolError
from csub.transport import LocalTransport, SshTransport, TransportError, UnixTransport


def stub_broker(tmp_path, body: str, rc: int = 0, stderr: str = ""):
    (tmp_path / "body").write_text(body)
    (tmp_path / "err").write_text(stderr)
    script = tmp_path / "stub-broker"
    script.write_text(
        f"#!/bin/sh\ncat >/dev/null\ncat {tmp_path}/body\ncat {tmp_path}/err >&2\nexit {rc}\n"
    )
    script.chmod(0o755)
    return str(script)


def test_local_roundtrip(tmp_path):
    t = LocalTransport(stub_broker(tmp_path, '{"protocol": 1, "ok": true, "x": 1}'))
    assert t.call({"protocol": 1, "op": "probe"}) == {"protocol": 1, "ok": True, "x": 1}
    t = LocalTransport(
        stub_broker(
            tmp_path,
            '{"protocol": 1, "ok": false, "error": {"code": "not_found", "message": "m"}}',
            rc=1,
        )
    )
    assert t.call({"protocol": 1, "op": "probe"})["ok"] is False


def test_local_failures(tmp_path):
    with pytest.raises(TransportError, match="boom"):
        LocalTransport(stub_broker(tmp_path, "", rc=255, stderr="boom")).call({})
    with pytest.raises(ProtocolError, match="not valid JSON"):
        LocalTransport(stub_broker(tmp_path, "garbage")).call({})
    with pytest.raises(TransportError, match="not found"):
        LocalTransport("/no/such/broker").call({})


def test_local_env_isolation(tmp_path):
    script = tmp_path / "envbroker"
    script.write_text(
        '#!/bin/sh\ncat >/dev/null\nprintf \'{"protocol": 1, "ok": true, "home": "%s"}\' "$HOME"\n'
    )
    script.chmod(0o755)
    t = LocalTransport([str(script)], env={"HOME": "/injected", "PATH": os.environ["PATH"]})
    assert t.call({})["home"] == "/injected"


def test_unix_roundtrip_and_errors(tmp_path):
    path = str(tmp_path / "s.sock")
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(path)
    srv.listen(1)
    received = []

    def serve():
        conn, _ = srv.accept()
        with conn:
            data = b""
            while chunk := conn.recv(4096):
                data += chunk
            received.append(json.loads(data))
            conn.sendall(b'{"protocol": 1, "ok": true, "echo": true}\n')

    th = threading.Thread(target=serve)
    th.start()
    resp = UnixTransport(path).call({"protocol": 1, "op": "probe"})
    th.join()
    assert resp["echo"] is True and received == [{"protocol": 1, "op": "probe"}]
    srv.close()
    with pytest.raises(TransportError, match="cannot reach"):
        UnixTransport(str(tmp_path / "missing.sock")).call({})


def test_unix_empty_response(tmp_path):
    path = str(tmp_path / "s.sock")
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(path)
    srv.listen(1)

    def serve():
        conn, _ = srv.accept()
        conn.close()

    th = threading.Thread(target=serve)
    th.start()
    with pytest.raises(TransportError, match="without a response|cannot reach"):
        UnixTransport(path).call({})
    th.join()
    srv.close()


def test_ssh_argv(tmp_path):
    t = SshTransport(
        "h",
        user="u",
        key="~/k",
        port=2200,
        opts=("-vv", "Foo=bar"),
        control_dir=str(tmp_path / "cm"),
    )
    argv = t.argv
    assert argv[:4] == ["ssh", "-T", "-p", "2200"]
    assert argv[argv.index("-i") + 1] == os.path.expanduser("~/k")
    assert f"ControlPath={tmp_path}/cm/cm-%C" in argv and "-vv" in argv and "Foo=bar" in argv
    assert argv[-3:] == ["u", "h", "csub-broker"] and argv[argv.index("-l") + 1] == "u"
    t = SshTransport("h")
    assert "-l" not in t.argv and "-i" not in t.argv


def test_ssh_silent_failure_is_diagnosed(tmp_path, monkeypatch):
    """exit 255 with no stderr -> rerun with -v and no multiplexing, report that output."""
    fake = tmp_path / "ssh"
    fake.write_text(
        "#!/bin/sh\n"
        'case " $* " in *" -v "*) echo "debug1: proxy said no" >&2; exit 255 ;; esac\n'
        "exit 255\n"
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    t = SshTransport("h", control_dir=str(tmp_path / "cm"))
    with pytest.raises(TransportError, match="proxy said no"):
        t.call({"protocol": 1, "op": "probe"})
