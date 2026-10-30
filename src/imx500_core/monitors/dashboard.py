"""Tiny HTTP server (stdlib): the dashboard, its status API and the alert clips.

- `GET /` and static files: `monitors/static/` (Catppuccin page, WebRTC player).
- `GET /api/status`: hardware metrics, watchdog, stream sources, alert clips.
- `POST /api/overlay`: `{"show_detections": bool}` toggles the boxes drawn
  on the stream (they are burned into the video, so every viewer sees it).
- `GET /clips/<name>`: alert clips from `data/clips`, with HTTP Range support.
"""

import functools
import json
import logging
import mimetypes
import re
import shutil
import threading
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..camera.stream import CLIP_SUFFIX, PARTIAL_SUFFIX

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATUS_PATH = "/api/status"
OVERLAY_PATH = "/api/overlay"
CLIPS_PREFIX = "/clips/"
CHUNK_SIZE = 64 * 1024
MAX_BODY = 1024
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
# Clip stems from services/alerts.clip_stem: 20261029-214305_<event>_<track>
CLIP_RE = re.compile(r"^(\d{8}-\d{6})_(.+)_(\d+)$")
# MediaMTX control API (bound to localhost in deploy/mediamtx/mediamtx.yml)
MEDIAMTX_PATHS_URL = "http://127.0.0.1:9997/v3/paths/list"


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Parses a single `bytes=start-end` Range header into an inclusive span.

    Returns None when there is no (usable) header, raises ValueError when the
    range cannot be satisfied for a file of `size` bytes.
    """
    if not header:
        return None
    match = RANGE_RE.fullmatch(header.strip())
    if match is None or match.groups() == ("", ""):
        return None  # unsupported form (e.g. multiple ranges): send the whole file

    first, last = match.groups()
    if first == "":  # suffix range: the last N bytes
        start, end = max(0, size - int(last)), size - 1
    else:
        start = int(first)
        end = min(int(last), size - 1) if last else size - 1

    if start >= size or start > end:
        raise ValueError("range not satisfiable")
    return start, end


def list_clips(clips_dir: Path) -> list[dict]:
    """Alert clips, newest first; `recording` while CircularOutput2 writes it."""
    if not clips_dir.is_dir():
        return []
    clips = []
    for path in clips_dir.glob(f"*{CLIP_SUFFIX}"):
        partial = path.name.endswith(PARTIAL_SUFFIX)
        stem = path.name.removesuffix(PARTIAL_SUFFIX if partial else CLIP_SUFFIX)
        name = stem + CLIP_SUFFIX
        clip = {
            "name": name,
            "url": CLIPS_PREFIX + urllib.parse.quote(name),
            "state": "recording" if partial else "ready",
            "time": None,
            "event": stem,
            "track_id": None,
            "size_mb": round(path.stat().st_size / 1e6, 2),
        }
        if match := CLIP_RE.match(stem):
            when = datetime.strptime(match[1], "%Y%m%d-%H%M%S").astimezone()
            clip.update(time=when.isoformat(), event=match[2], track_id=int(match[3]))
        clips.append(clip)
    return sorted(clips, key=lambda c: c["name"], reverse=True)


def mediamtx_paths(default: str, timeout_s: float = 0.5) -> list[str]:
    """Stream paths MediaMTX is serving (its local API), default first."""
    paths = [default]
    try:
        with urllib.request.urlopen(MEDIAMTX_PATHS_URL, timeout=timeout_s) as resp:
            items = json.loads(resp.read()).get("items", [])
        paths += sorted(i["name"] for i in items if i.get("name") not in paths)
    except (OSError, ValueError, KeyError, TypeError):
        pass  # API disabled or MediaMTX down: only the configured path
    return paths


def start_dashboard(
    get_status: Callable[[], dict],
    set_overlay: Callable[[dict], dict],
    clips_dir: Path,
    host: str,
    port: int,
    static_dir: Path = STATIC_DIR,
) -> ThreadingHTTPServer:
    """Starts the server in a daemon thread; call .shutdown() to stop it."""

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == STATUS_PATH:
                return self._send_json(get_status())
            if self.path.startswith(CLIPS_PREFIX):
                return self._send_clip(with_body=True)
            return super().do_GET()

        def do_HEAD(self) -> None:
            if self.path.startswith(CLIPS_PREFIX):
                return self._send_clip(with_body=False)
            return super().do_HEAD()

        def do_POST(self) -> None:
            if self.path != OVERLAY_PATH:
                return self.send_error(HTTPStatus.NOT_FOUND)
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("expected a JSON object")
                return self._send_json(set_overlay(body))
            except (ValueError, TypeError) as e:
                return self.send_error(HTTPStatus.BAD_REQUEST, str(e))

        def _send_json(self, data: dict) -> None:
            body = json.dumps(data).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_clip(self, with_body: bool) -> None:
            """Serves a clip with HTTP Range support (browsers seek with it)."""
            name = Path(urllib.parse.unquote(self.path.split("?", 1)[0])).name
            path = clips_dir / name  # .name above blocks path traversal
            if not name.endswith(CLIP_SUFFIX) or not path.is_file():
                return self.send_error(HTTPStatus.NOT_FOUND)

            size = path.stat().st_size
            try:
                span = parse_range(self.headers.get("Range"), size)
            except ValueError:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return

            start, end = span if span else (0, size - 1)
            length = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT if span else HTTPStatus.OK)
            content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if span:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()

            if not with_body:
                return
            try:
                with path.open("rb") as f:
                    f.seek(start)
                    shutil.copyfileobj(_Limited(f, length), self.wfile, CHUNK_SIZE)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser cancelled (normal when seeking)

        def log_message(self, *args) -> None:  # silence per-request logs
            pass

    handler = functools.partial(Handler, directory=str(static_dir))
    server = ThreadingHTTPServer((host, port), handler)
    threading.Thread(target=server.serve_forever, name="dashboard", daemon=True).start()
    log.info("Dashboard available on http://%s:%s", host, port)
    return server


class _Limited:
    """File-like wrapper that reads at most `remaining` bytes."""

    def __init__(self, f, remaining: int) -> None:
        self._f = f
        self._remaining = remaining

    def read(self, size: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        size = self._remaining if size < 0 else min(size, self._remaining)
        data = self._f.read(size)
        self._remaining -= len(data)
        return data
