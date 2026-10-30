"""EventDispatcher: MQTT for every event, clip + Telegram for alerts."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("modlib")
pytest.importorskip("lap")

from imx500_core.processing.analysis import SceneEvent  # noqa: E402
from imx500_core.services.alerts import (  # noqa: E402
    EventDispatcher,
    alert_caption,
    clip_stem,
)

T0 = datetime(2026, 10, 29, 21, 43, 5)


def _event(name="carrying_package", active=True, kind="relation", track_id=7):
    return SceneEvent(kind, name, "person", track_id, active, time=T0)


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _dispatcher(**kw):
    recorder = MagicMock()
    recorder.trigger.side_effect = lambda stem, on_done: f"{stem}.mp4"
    telegram, mqtt, clock = MagicMock(), MagicMock(), _Clock()
    d = EventDispatcher(
        project="porch_piracy",
        alerts=["carrying_package"],
        cooldown_s=60,
        recorder=recorder,
        telegram=telegram,
        mqtt=mqtt,
        clock=clock,
        **kw,
    )
    return d, recorder, telegram, mqtt, clock


def test_caption_and_stem():
    assert clip_stem(_event()) == "20261029-214305_carrying_package_7"
    assert alert_caption(_event(), "porch_piracy") == (
        "carrying_package: person #7\n2026-10-29 21:43:05 · porch_piracy"
    )


def test_alert_records_a_clip_and_sends_it_when_saved():
    d, recorder, telegram, mqtt, _ = _dispatcher()
    d(_event())

    stem = "20261029-214305_carrying_package_7"
    recorder.trigger.assert_called_once()
    assert recorder.trigger.call_args.args[0] == stem
    telegram.send_video.assert_not_called()  # only once the clip is saved
    on_done = recorder.trigger.call_args.kwargs["on_done"]
    on_done(Path(f"/clips/{stem}.mp4"))
    telegram.send_video.assert_called_once_with(
        Path(f"/clips/{stem}.mp4"), alert_caption(_event(), "porch_piracy")
    )
    (payload,), _ = mqtt.publish.call_args
    assert payload["clip"] == f"{stem}.mp4" and payload["project"] == "porch_piracy"


def test_every_event_goes_to_mqtt_but_only_alerts_record():
    d, recorder, _, mqtt, _ = _dispatcher()
    d(_event(name="person", kind="detection"))
    d(_event(active=False))
    recorder.trigger.assert_not_called()
    assert mqtt.publish.call_count == 2
    assert all(c.args[0]["clip"] is None for c in mqtt.publish.call_args_list)


def test_cooldown_per_alert_name():
    d, recorder, _, _, clock = _dispatcher()
    d(_event(track_id=1))
    clock.now = 30
    d(_event(track_id=2))  # within 60 s: not notified
    clock.now = 61
    d(_event(track_id=3))
    assert recorder.trigger.call_count == 2


def test_without_recorder_telegram_gets_a_text_alert():
    d, _, telegram, _, _ = _dispatcher()
    d._recorder = None
    d(_event())
    telegram.send_message.assert_called_once()
