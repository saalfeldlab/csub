import io
import signal
import socket
import threading

from csub.broker.serve import serve


def test_serve_refuses_a_live_socket(tmp_path):
    path = str(tmp_path / "sock")
    first = socket.socket(socket.AF_UNIX)
    first.bind(path)
    first.listen(1)
    err = io.StringIO()
    assert serve(path, lambda _: {}, stderr=err) == 1
    assert "already serving" in err.getvalue()
    first.close()


def test_serve_replaces_a_stale_socket(tmp_path):
    path = str(tmp_path / "sock")
    socket.socket(socket.AF_UNIX).bind(path)  # file left behind, nobody listening
    ready = threading.Event()
    got = []

    def client():
        ready.wait(5)
        with socket.socket(socket.AF_UNIX) as c:
            c.connect(path)
            c.sendall(b"{}")
            c.shutdown(socket.SHUT_WR)
            got.append(c.recv(4096))
        signal.raise_signal(signal.SIGTERM)

    th = threading.Thread(target=client)
    th.start()
    assert serve(path, lambda _: {"ok": True}, stderr=io.StringIO(), ready=ready.set) == 0
    th.join()
    assert got == [b'{"ok":true}\n']
