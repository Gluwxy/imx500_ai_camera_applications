"""Telegram notifier with a fake HTTP sender (no network)."""

import json

import pytest

from imx500_core.config import TelegramConfig
from imx500_core.services.telegram import TelegramNotifier, multipart


def _notifier(results=None):
    calls = []

    def sender(url, body, content_type):
        calls.append((url, body, content_type))
        return (results or {"ok": True}) if not callable(results) else results()

    cfg = TelegramConfig(chat_id=42, token="123:abc")
    return TelegramNotifier(cfg, sender=sender), calls


def test_video_is_sent_as_multipart_with_caption(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00mp4data")
    notifier, calls = _notifier()
    notifier.send_video(clip, "carrying_package: person #7")
    notifier.close()

    ((url, body, content_type),) = calls
    assert url == "https://api.telegram.org/bot123:abc/sendVideo"
    assert content_type.startswith("multipart/form-data; boundary=")
    assert b'name="chat_id"\r\n\r\n42' in body
    assert b"carrying_package: person #7" in body
    assert b'filename="clip.mp4"' in body and b"\x00mp4data" in body


def test_message_is_sent_as_json():
    notifier, calls = _notifier()
    notifier.send_message("hola")
    notifier.close()
    ((url, body, content_type),) = calls
    assert url.endswith("/sendMessage") and content_type == "application/json"
    assert json.loads(body) == {"chat_id": "42", "text": "hola"}


def test_api_errors_are_logged_not_raised(caplog):
    notifier, _ = _notifier({"ok": False, "description": "chat not found"})
    notifier.send_message("x")
    notifier.close()
    assert "chat not found" in caplog.text


def test_token_is_required():
    with pytest.raises(ValueError, match="token"):
        TelegramNotifier(TelegramConfig(enabled=False, chat_id=1))


def test_multipart_fields(tmp_path):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"v")
    body, content_type = multipart({"a": "1"}, "video", f)
    boundary = content_type.split("boundary=")[1]
    assert body.endswith(f"--{boundary}--\r\n".encode())
