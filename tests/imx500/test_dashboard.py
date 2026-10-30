"""Dashboard HTTP server on an ephemeral port (status, overlay toggle, clips)."""

import json
import urllib.error
import urllib.request

import pytest

from imx500_core.monitors import dashboard as db


@pytest.fixture
def server(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "20261029-214305_carrying_package_7.mp4").write_bytes(b"0123456789")
    (clips / "20261029-220000_carrying_package_9.part.mp4").write_bytes(b"x")
    (clips / "notes.txt").write_text("not a clip")
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<h1>dashboard</h1>")

    state = {"show": True}

    def set_overlay(body):
        if not isinstance(body.get("show_detections"), bool):
            raise ValueError("bad")
        state["show"] = body["show_detections"]
        return {"show_detections": state["show"]}

    srv = db.start_dashboard(
        get_status=lambda: {"overlay": {"show_detections": state["show"]}},
        set_overlay=set_overlay,
        clips_dir=clips,
        host="127.0.0.1",
        port=0,
        static_dir=static,
    )
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base, clips
    srv.shutdown()


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req) as r:
        return r.status, dict(r.headers), r.read()


def _post(url, data: bytes):
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_static_page_and_status(server):
    base, _ = server
    assert _get(base + "/")[2] == b"<h1>dashboard</h1>"
    status, headers, body = _get(base + "/api/status")
    assert headers["Cache-Control"] == "no-store"
    assert json.loads(body) == {"overlay": {"show_detections": True}}


def test_overlay_toggle(server):
    base, _ = server
    assert _post(base + "/api/overlay", b'{"show_detections": false}') == {
        "show_detections": False
    }
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base + "/api/overlay", b'{"show_detections": "x"}')
    assert e.value.code == 400


def test_clip_with_range(server):
    base, _ = server
    url = base + "/clips/20261029-214305_carrying_package_7.mp4"
    status, headers, body = _get(url)
    assert status == 200 and body == b"0123456789"
    assert headers["Content-Type"] == "video/mp4"
    status, headers, body = _get(url, {"Range": "bytes=2-4"})
    assert status == 206 and body == b"234"
    assert headers["Content-Range"] == "bytes 2-4/10"


def test_only_mp4_clips_are_served(server):
    base, _ = server
    for path in ("/clips/notes.txt", "/clips/..%2Fstatic%2Findex.html", "/clips/x.mp4"):
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(base + path)
        assert e.value.code == 404


def test_list_clips(server):
    _, clips = server
    listed = db.list_clips(clips)
    assert [(c["name"], c["state"]) for c in listed] == [
        ("20261029-220000_carrying_package_9.mp4", "recording"),
        ("20261029-214305_carrying_package_7.mp4", "ready"),
    ]
    ready = listed[1]
    assert ready["event"] == "carrying_package" and ready["track_id"] == 7
    assert ready["time"].startswith("2026-10-29T21:43:05")
    assert ready["url"] == "/clips/20261029-214305_carrying_package_7.mp4"
    assert db.list_clips(clips / "missing") == []


def test_parse_range():
    assert db.parse_range(None, 10) is None
    assert db.parse_range("bytes=-3", 10) == (7, 9)
    assert db.parse_range("bytes=5-", 10) == (5, 9)
    with pytest.raises(ValueError):
        db.parse_range("bytes=20-", 10)


def test_mediamtx_paths_falls_back_to_the_default(monkeypatch):
    monkeypatch.setattr(db, "MEDIAMTX_PATHS_URL", "http://127.0.0.1:9/v3/paths/list")
    assert db.mediamtx_paths("porch", timeout_s=0.2) == ["porch"]
