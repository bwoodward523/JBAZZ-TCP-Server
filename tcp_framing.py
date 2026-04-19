"""TCP length-prefix framing with optional 1-byte message type tag.

Wire format for typed messages:
  [4 bytes BE u32 length][1 byte type][N bytes payload]

Untyped helpers send_message/recv_message operate on raw payloads only;
typed helpers prepend the type byte inside the length-delimited blob.
"""

from __future__ import annotations

import struct
from enum import IntEnum


class MessageType(IntEnum):
    EMOTION = 0x00
    AUDIO_CHUNK = 0x01
    TIMING_DATA = 0x02
    END_OF_RESPONSE = 0x03
    SHOOT = 0x04


def recv_exact(sock, n: int) -> bytes | None:
    buffer = b""
    while len(buffer) < n:
        chunk = sock.recv(n - len(buffer))
        if not chunk:
            return None
        buffer += chunk
    return buffer


def send_message(sock, payload: bytes) -> None:
    header = struct.pack("!I", len(payload))
    sock.sendall(header)
    sock.sendall(payload)


def recv_message(sock) -> bytes | None:
    header = recv_exact(sock, 4)
    if header is None:
        return None
    length = struct.unpack("!I", header)[0]
    return recv_exact(sock, length)


def send_typed_message(sock, msg_type: int | MessageType, payload: bytes) -> None:
    """Send a type-tagged message using existing length-prefix framing."""
    if isinstance(msg_type, MessageType):
        msg_type = int(msg_type)
    typed_payload = bytes([msg_type]) + payload
    header = struct.pack("!I", len(typed_payload))
    sock.sendall(header)
    sock.sendall(typed_payload)


def recv_typed_message(sock) -> tuple[int, bytes] | None:
    """Receive one typed message; returns (type_byte, payload_without_type) or None."""
    raw = recv_message(sock)
    if raw is None or len(raw) == 0:
        return None
    return raw[0], raw[1:]
