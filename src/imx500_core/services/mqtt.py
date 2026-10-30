"""Minimal MQTT 3.1.1 publisher (QoS 0) on the stdlib, in a worker thread.

Only what the app needs: CONNECT (with optional user/password), PUBLISH at
QoS 0, PINGREQ to keep the session alive and DISCONNECT. No third-party
dependency on the Pi Zero. `publish()` never blocks the caller: messages go
to a bounded queue and are dropped (with a warning) if the broker is down for
long; the worker reconnects with exponential backoff.
"""

import json
import logging
import queue
import socket
import threading
import time
from collections.abc import Callable

from ..config import MqttConfig

log = logging.getLogger(__name__)

CONNECT, CONNACK, PUBLISH = 0x10, 0x20, 0x30
PINGREQ, PINGRESP, DISCONNECT = 0xC0, 0xD0, 0xE0
QUEUE_SIZE = 100
MAX_BACKOFF_S = 60.0

CONNACK_ERRORS = {
    1: "unacceptable protocol version",
    2: "identifier rejected",
    3: "server unavailable",
    4: "bad user name or password",
    5: "not authorized",
}


def _remaining_length(n: int) -> bytes:
    out = bytearray()
    while True:
        byte, n = n % 128, n // 128
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _string(value: str | bytes) -> bytes:
    data = value.encode() if isinstance(value, str) else value
    return len(data).to_bytes(2, "big") + data


def _packet(header: int, body: bytes = b"") -> bytes:
    return bytes([header]) + _remaining_length(len(body)) + body


def connect_packet(
    client_id: str,
    keepalive_s: int,
    username: str | None = None,
    password: str | None = None,
) -> bytes:
    flags = 0x02  # clean session
    payload = _string(client_id)
    if username is not None:
        flags |= 0x80
        payload += _string(username)
        if password is not None:
            flags |= 0x40
            payload += _string(password)
    header = _string("MQTT") + bytes([4, flags]) + keepalive_s.to_bytes(2, "big")
    return _packet(CONNECT, header + payload)


def publish_packet(topic: str, payload: bytes, retain: bool = False) -> bytes:
    return _packet(PUBLISH | (0x01 if retain else 0), _string(topic) + payload)


class MqttPublisher:
    def __init__(
        self,
        cfg: MqttConfig,
        connect: Callable[[tuple[str, int], float], socket.socket] | None = None,
    ) -> None:
        self._cfg = cfg
        self._connect = connect or socket.create_connection
        self._queue: queue.Queue[tuple[str, bytes] | None] = queue.Queue(QUEUE_SIZE)
        self._sock: socket.socket | None = None
        self._last_sent = 0.0
        self._thread = threading.Thread(target=self._run, name="mqtt", daemon=True)
        self._thread.start()

    def publish(self, payload: dict, subtopic: str | None = None) -> None:
        """Queues a JSON message for `<topic>[/<subtopic>]` (never blocks)."""
        topic = self._cfg.topic + (f"/{subtopic}" if subtopic else "")
        try:
            self._queue.put_nowait((topic, json.dumps(payload).encode()))
        except queue.Full:
            log.warning("MQTT queue full, dropping message for %s", topic)

    def close(self, timeout_s: float = 3.0) -> None:
        """Sends what is queued (if connected) and disconnects."""
        self._queue.put(None)
        self._thread.join(timeout_s)

    # --- worker thread -------------------------------------------------------

    def _run(self) -> None:
        backoff = 1.0
        ping_every = self._cfg.keepalive_s / 2
        while True:
            try:
                item = self._queue.get(timeout=ping_every)
            except queue.Empty:
                item = "ping"
            if item is None:
                self._disconnect()
                return
            try:
                self._ensure_connected()
                if item == "ping":
                    self._ping()
                else:
                    self._send(publish_packet(*item))
                backoff = 1.0
            except (OSError, ConnectionError) as e:
                cfg = self._cfg
                log.warning("MQTT %s:%s unavailable: %s", cfg.host, cfg.port, e)
                self._drop_socket()
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF_S)

    def _ensure_connected(self) -> None:
        if self._sock is not None:
            return
        cfg = self._cfg
        sock = self._connect((cfg.host, cfg.port), 5.0)
        sock.settimeout(5.0)
        password = cfg.password.get_secret_value() if cfg.password else None
        sock.sendall(
            connect_packet(cfg.client_id, cfg.keepalive_s, cfg.username, password)
        )
        ack = self._recv_exact(sock, 4)
        if ack[0] != CONNACK or ack[3] != 0:
            sock.close()
            reason = CONNACK_ERRORS.get(ack[3], f"code {ack[3]}")
            raise ConnectionError(f"connection refused ({reason})")
        self._sock = sock
        log.info("MQTT connected to %s:%s", cfg.host, cfg.port)

    def _send(self, data: bytes) -> None:
        assert self._sock is not None
        self._sock.sendall(data)
        self._last_sent = time.monotonic()

    def _ping(self) -> None:
        self._send(_packet(PINGREQ))
        resp = self._recv_exact(self._sock, 2)
        if resp[0] != PINGRESP:
            raise ConnectionError(f"unexpected reply to PINGREQ: {resp!r}")

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("connection closed by the broker")
            data += chunk
        return data

    def _disconnect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.sendall(_packet(DISCONNECT))
            except OSError:
                pass
        self._drop_socket()

    def _drop_socket(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None
