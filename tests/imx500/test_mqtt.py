"""Stdlib MQTT publisher against a tiny in-process broker."""

import json
import socket
import threading

from imx500_core.config import MqttConfig
from imx500_core.services import mqtt


def test_remaining_length_encoding():
    assert mqtt._remaining_length(0) == b"\x00"
    assert mqtt._remaining_length(127) == b"\x7f"
    assert mqtt._remaining_length(128) == b"\x80\x01"
    assert mqtt._remaining_length(16_383) == b"\xff\x7f"


def test_connect_packet_with_credentials():
    pkt = mqtt.connect_packet("cam", 60, "porch", "secret")
    assert pkt[0] == mqtt.CONNECT
    body = pkt[2:]
    assert body[:6] == b"\x00\x04MQTT" and body[6] == 4
    assert body[7] == 0x80 | 0x40 | 0x02  # user, password, clean session
    assert body.endswith(b"\x00\x03cam\x00\x05porch\x00\x06secret")


def _read_packet(conn):
    header = conn.recv(1)
    if not header:
        return None, b""
    length, shift = 0, 0
    while True:
        byte = conn.recv(1)[0]
        length |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            break
    body = b""
    while len(body) < length:
        body += conn.recv(length - len(body))
    return header[0], body


class _Broker:
    """Accepts one client, acks CONNECT, records PUBLISH (topic, payload)."""

    def __init__(self, return_code=0):
        self.server = socket.create_server(("127.0.0.1", 0))
        self.port = self.server.getsockname()[1]
        self.return_code = return_code
        self.messages = []
        self.got = threading.Event()
        self.disconnected = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        conn, _ = self.server.accept()
        with conn:
            while True:
                kind, body = _read_packet(conn)
                if kind is None:
                    return
                if kind == mqtt.CONNECT:
                    conn.sendall(bytes([mqtt.CONNACK, 2, 0, self.return_code]))
                elif kind & 0xF0 == mqtt.PUBLISH:
                    n = int.from_bytes(body[:2], "big")
                    self.messages.append((body[2 : 2 + n].decode(), body[2 + n :]))
                    self.got.set()
                elif kind == mqtt.PINGREQ:
                    conn.sendall(bytes([mqtt.PINGRESP, 0]))
                elif kind == mqtt.DISCONNECT:
                    self.disconnected.set()
                    return


def test_publish_reaches_the_broker_as_json():
    broker = _Broker()
    pub = mqtt.MqttPublisher(
        MqttConfig(enabled=True, port=broker.port, topic="porch/events")
    )
    pub.publish({"name": "carrying_package", "track_id": 7})
    assert broker.got.wait(3)
    pub.close()

    ((topic, payload),) = broker.messages
    assert topic == "porch/events"
    assert json.loads(payload) == {"name": "carrying_package", "track_id": 7}
    assert broker.disconnected.wait(3)


def test_refused_connection_is_logged_and_retried(caplog):
    broker = _Broker(return_code=4)
    pub = mqtt.MqttPublisher(MqttConfig(enabled=True, port=broker.port))
    pub.publish({"x": 1})
    pub.close(timeout_s=0.5)
    assert "bad user name or password" in caplog.text
