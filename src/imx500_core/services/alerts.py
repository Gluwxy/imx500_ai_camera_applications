"""Routes analyzer events: every event to MQTT, alerts to a clip + Telegram.

`EventDispatcher` is the analyzer's `on_event` callback, so it runs on the
camera thread: it only queues work (MQTT/Telegram workers, clip recorder
thread) and returns at once.
"""

import logging
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from ..camera.stream import ClipRecorder
from ..processing.analysis import SceneEvent
from .mqtt import MqttPublisher
from .telegram import TelegramNotifier

log = logging.getLogger(__name__)


def clip_stem(event: SceneEvent) -> str:
    """`20261029-214305_carrying_package_7`: sortable by time."""
    return f"{event.time:%Y%m%d-%H%M%S}_{event.name}_{event.track_id}"


def alert_caption(event: SceneEvent, project: str) -> str:
    """Relation (or class), track and time, e.g. for Telegram."""
    return (
        f"{event.name}: {event.subject} #{event.track_id}\n"
        f"{event.time:%Y-%m-%d %H:%M:%S} · {project}"
    )


class EventDispatcher:
    def __init__(
        self,
        project: str,
        alerts: Iterable[str],
        cooldown_s: float,
        recorder: ClipRecorder | None = None,
        telegram: TelegramNotifier | None = None,
        mqtt: MqttPublisher | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._project = project
        self._alerts = set(alerts)
        self._cooldown_s = cooldown_s
        self._recorder = recorder
        self._telegram = telegram
        self._mqtt = mqtt
        self._clock = clock
        self._last_alert: dict[str, float] = {}

    def __call__(self, event: SceneEvent) -> None:
        clip = self._alert(event) if self._is_alert(event) else None
        if self._mqtt is not None:
            self._mqtt.publish({**event.json(), "project": self._project, "clip": clip})

    def _is_alert(self, event: SceneEvent) -> bool:
        if not event.active or event.name not in self._alerts:
            return False
        now = self._clock()
        last = self._last_alert.get(event.name)
        if last is not None and now - last < self._cooldown_s:
            log.info("Alert %s within cooldown, not notified", event.describe())
            return False
        self._last_alert[event.name] = now
        return True

    def _alert(self, event: SceneEvent) -> str | None:
        """Starts the clip; Telegram gets it (with the caption) once saved."""
        caption = alert_caption(event, self._project)
        log.warning("ALERT %s", event.describe())
        if self._recorder is None:
            if self._telegram is not None:
                self._telegram.send_message(caption)
            return None

        def on_clip(path: Path) -> None:
            if self._telegram is not None:
                self._telegram.send_video(path, caption)

        return self._recorder.trigger(clip_stem(event), on_done=on_clip)
