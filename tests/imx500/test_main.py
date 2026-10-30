"""Entry point wiring: arguments, startup order and cleanup (hardware mocked)."""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("modlib")
pytest.importorskip("lap")
pytest.importorskip("cv2")

from imx500_core import main as app  # noqa: E402


def test_project_from_argument_or_environment(monkeypatch):
    monkeypatch.setenv("PROJECT", "from_env")
    assert app.parse_args(["porch_piracy"]).project == "porch_piracy"
    assert app.parse_args([]).project == "from_env"


def test_missing_project_is_an_error(monkeypatch):
    monkeypatch.delenv("PROJECT", raising=False)
    with pytest.raises(SystemExit):
        app.parse_args([])


@pytest.fixture
def wired(monkeypatch, project_dir, base_file):
    """main() with the real config loader on the fake project, camera mocked."""
    monkeypatch.setenv("TELEGRAM_TOKEN", "t")
    real_load = app.load_config
    monkeypatch.setattr(app, "load_config", lambda _: real_load(project_dir, base_file))

    picam2 = MagicMock()
    picam2.camera_configuration.return_value = {
        "main": {"size": (640, 640), "format": "XBGR8888"}
    }
    camera = MagicMock(picam2=picam2)
    camera.imx500.get_input_size.return_value = (640, 640)
    setup = MagicMock(return_value=camera)
    stream = MagicMock(return_value=(MagicMock(), MagicMock()))
    monkeypatch.setattr(app, "setup_camera", setup)
    monkeypatch.setattr(app, "start_streaming", stream)
    monkeypatch.setattr(app.signal, "signal", MagicMock())
    return picam2, setup, stream


def test_runs_and_releases_the_camera_on_shutdown(wired, monkeypatch):
    picam2, setup, stream = wired
    # The heartbeat wait returns "stop requested" at once
    monkeypatch.setattr(app.stop_event, "wait", lambda timeout: True)

    assert app.main(["demo"]) == 0

    setup.assert_called_once()
    stream.assert_called_once()
    assert picam2.pre_callback is not None
    _, recorder = stream.return_value
    recorder.close.assert_called_once()  # clip finalized before the encoder stops
    picam2.stop_recording.assert_called_once()
    picam2.close.assert_called_once()


def test_frozen_camera_exits_with_code_1(wired, monkeypatch):
    picam2, _, _ = wired
    monkeypatch.setattr(app.stop_event, "wait", lambda timeout: False)
    monkeypatch.setattr(app, "WATCHDOG_TIMEOUT_S", -1.0)

    assert app.main(["demo"]) == 1
    picam2.close.assert_called_once()


def test_stream_failure_still_releases_the_camera(wired):
    picam2, _, stream = wired
    stream.side_effect = RuntimeError("no MediaMTX")

    assert app.main(["demo"]) == 1
    picam2.close.assert_called_once()
