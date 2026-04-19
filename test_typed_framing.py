"""
Round-trip test: local TCP server sends typed frames, client receives with recv_typed_message.
Run: python test_typed_framing.py
"""
import json
import socket
import threading

from tcp_framing import MessageType, recv_typed_message, send_typed_message


def echo_server(sock: socket.socket, results: list) -> None:
    """Send a few typed messages (what a full server would stream in order)."""
    try:
        send_typed_message(
            sock, MessageType.EMOTION, "emotion:happiness".encode("utf-8")
        )
        timings = [{"s": 0.1, "e": 0.3, "w": "hello"}]
        send_typed_message(
            sock, MessageType.TIMING_DATA, json.dumps(timings).encode("utf-8")
        )
        send_typed_message(sock, MessageType.AUDIO_CHUNK, b"\x00\x01" * 120)  # 240 B fake PCM
        send_typed_message(sock, MessageType.END_OF_RESPONSE, b"")
        send_typed_message(sock, MessageType.SHOOT, b"False")
    finally:
        sock.close()


def main() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect(("127.0.0.1", port))

    conn, _ = listener.accept()
    listener.close()
    threading.Thread(target=echo_server, args=(conn, []), daemon=True).start()

    expected = [
        (MessageType.EMOTION, "emotion:happiness".encode("utf-8")),
        (
            MessageType.TIMING_DATA,
            json.dumps([{"s": 0.1, "e": 0.3, "w": "hello"}]).encode("utf-8"),
        ),
        (MessageType.AUDIO_CHUNK, b"\x00\x01" * 120),
        (MessageType.END_OF_RESPONSE, b""),
        (MessageType.SHOOT, b"False"),
    ]
    for want_t, want_body in expected:
        msg = recv_typed_message(client_sock)
        assert msg is not None, "expected message, got disconnect"
        got_t, got_body = msg
        assert got_t == int(want_t), (got_t, want_t)
        assert got_body == want_body, (got_body, want_body)

    assert recv_typed_message(client_sock) is None
    client_sock.close()
    print("test_typed_framing: ok")


if __name__ == "__main__":
    main()
