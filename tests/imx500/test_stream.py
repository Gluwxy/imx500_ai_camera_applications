"""ClipRecorder on a fake CircularOutput2 (no camera, no encoder)."""

import threading

import pytest

from imx500_core.camera.stream import ClipRecorder, partial_path


class _FakeCircular:
    """Writes the opened output when closed, like a finished clip."""

    def __init__(self):
        self.output = None
        self.opened = threading.Event()

    def open_output(self, output):
        self.output = output
        self.opened.set()

    def close_output(self):
        self.output.write_bytes(b"mp4")
        self.output = None


@pytest.fixture
def recorder(tmp_path):
    circular = _FakeCircular()
    rec = ClipRecorder(
        circular,
        clips_dir=tmp_path / "clips",
        seconds=0.05,
        max_clips=2,
        output_factory=lambda path: path,  # the "output" is the file path
    )
    return rec, circular, tmp_path / "clips"


def _wait(rec):
    rec._thread.join(2)
    assert not rec.recording


def test_clip_is_written_and_callback_gets_its_path(recorder):
    rec, circular, clips = recorder
    done = []
    name = rec.trigger("20261029-120000_carrying_7", on_done=done.append)

    assert name == "20261029-120000_carrying_7.mp4"
    circular.opened.wait(1)
    # recorded under the partial name while open
    assert circular.output == partial_path(clips / name)
    _wait(rec)
    assert done == [clips / name]
    assert (clips / name).read_bytes() == b"mp4"
    assert not partial_path(clips / name).exists()


def test_events_during_a_clip_share_it(recorder):
    rec, circular, clips = recorder
    rec._record_s = 0.3
    done = []
    first = rec.trigger("a", on_done=done.append)
    second = rec.trigger("b", on_done=done.append)
    assert first == second == "a.mp4"
    _wait(rec)
    assert done == [clips / "a.mp4", clips / "a.mp4"]


def test_old_clips_are_pruned(recorder):
    rec, _, clips = recorder
    for stem in ("1", "2", "3"):
        rec.trigger(stem)
        _wait(rec)
    assert sorted(p.name for p in clips.iterdir()) == ["2.mp4", "3.mp4"]


def test_close_finalizes_a_clip_early(recorder):
    rec, _, clips = recorder
    rec._record_s = 60
    rec.trigger("long")
    rec.close(timeout_s=2)
    assert (clips / "long.mp4").exists()


def test_partials_from_a_crash_are_discarded(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "x.part.mp4").write_bytes(b"half")
    ClipRecorder(_FakeCircular(), clips, seconds=1)
    assert not (clips / "x.part.mp4").exists()


def test_failed_clip_skips_callbacks(recorder, caplog):
    rec, circular, clips = recorder
    circular.close_output = lambda: (_ for _ in ()).throw(OSError("disk full"))
    done = []
    rec.trigger("bad", on_done=done.append)
    _wait(rec)
    assert done == []
    assert "Failed to record clip" in caplog.text
