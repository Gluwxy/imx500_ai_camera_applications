"""Telegram Bot API notifier on the stdlib (urllib), in a worker thread.

`send_video()` / `send_message()` only queue the request: uploads (a 10 s clip
is a few MB) never block the camera thread or the clip recorder.
"""

import json
import logging
import queue
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from urllib import request

from ..config import TelegramConfig

log = logging.getLogger(__name__)

API_URL = "https://api.telegram.org/bot{token}/{method}"
TIMEOUT_S = 60
QUEUE_SIZE = 20

# (url, body, content type) -> parsed JSON response
Sender = Callable[[str, bytes, str], dict]


def multipart(fields: dict[str, str], file_field: str, path: Path) -> tuple[bytes, str]:
    """multipart/form-data body with text fields and one file."""
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
        f'filename="{path.name}"\r\nContent-Type: video/mp4\r\n\r\n'.encode()
        + path.read_bytes()
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _http_post(url: str, body: bytes, content_type: str) -> dict:
    req = request.Request(url, data=body, headers={"Content-Type": content_type})
    with request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return json.loads(resp.read())


class TelegramNotifier:
    def __init__(self, cfg: TelegramConfig, sender: Sender | None = None) -> None:
        if cfg.token is None:
            raise ValueError("Telegram token missing (TELEGRAM_TOKEN in .env)")
        self._cfg = cfg
        self._token = cfg.token.get_secret_value()
        self._sender = sender or _http_post
        self._queue: queue.Queue[tuple | None] = queue.Queue(QUEUE_SIZE)
        self._thread = threading.Thread(target=self._run, name="telegram", daemon=True)
        self._thread.start()

    def send_message(self, text: str) -> None:
        self._enqueue(("sendMessage", text, None))

    def send_video(self, path: Path, caption: str) -> None:
        self._enqueue(("sendVideo", caption, path))

    def close(self, timeout_s: float = 10.0) -> None:
        """Lets queued notifications go out (bounded wait) and stops."""
        self._queue.put(None)
        self._thread.join(timeout_s)

    def _enqueue(self, item: tuple) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            log.warning("Telegram queue full, dropping %s", item[0])

    def _run(self) -> None:
        while (item := self._queue.get()) is not None:
            method, text, path = item
            try:
                self._call(method, text, path)
            except Exception as e:  # network down, API error: never crash the app
                log.warning("Telegram %s failed: %s", method, e)

    def _call(self, method: str, text: str, path: Path | None) -> None:
        url = API_URL.format(token=self._token, method=method)
        chat = str(self._cfg.chat_id)
        if path is None:
            body = json.dumps({"chat_id": chat, "text": text}).encode()
            result = self._sender(url, body, "application/json")
        else:
            fields = {"chat_id": chat, "caption": text, "supports_streaming": "true"}
            body, content_type = multipart(fields, "video", path)
            result = self._sender(url, body, content_type)
        if not result.get("ok"):
            raise RuntimeError(result.get("description", "unknown error"))
        log.info("Telegram %s sent", method)
