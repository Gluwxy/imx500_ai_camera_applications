"""H.264 encoder → RTSP stream + a circular buffer for alert clips.

One hardware encoder feeds two outputs: `PyavOutput` (RTSP to MediaMTX) and a
`CircularOutput2` that keeps the last `clips.seconds`. `ClipRecorder` cuts
clips from it when an alert fires.
"""

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput2, PyavOutput

from ..config import ClipConfig, StreamConfig

log = logging.getLogger(__name__)

CLIP_SUFFIX = ".mp4"
PARTIAL_SUFFIX = ".part.mp4"  # renamed to .mp4 once the clip is complete
MP4_OPTIONS = {"movflags": "+faststart"}  # index first: players start at once

ClipCallback = Callable[[Path], None]


def partial_path(final: Path) -> Path:
    return final.with_name(final.name.removesuffix(CLIP_SUFFIX) + PARTIAL_SUFFIX)


class ClipRecorder:
    """Exports clips from a CircularOutput2: `seconds` before + after the event.

    CircularOutput2 does not dump its buffer when an output is opened: it writes
    each frame once it ages out of the buffer window. So a clip covering
    [event - seconds, event + seconds] is complete only after `2 * seconds`.
    Closing earlier would keep just the pre-event part.

    `trigger()` is called from the camera thread: it only starts a worker thread
    and returns at once. One clip is recorded at a time; events fired while a
    clip is being written share that clip and their callbacks run when it ends.
    """

    def __init__(
        self,
        circular: CircularOutput2,
        clips_dir: Path,
        seconds: float,
        max_clips: int = 20,
        output_factory: Callable[[Path], object] | None = None,
    ) -> None:
        self._circular = circular
        self._clips_dir = clips_dir
        self._record_s = 2 * seconds
        self._max_clips = max_clips
        self._output_factory = output_factory or (
            lambda path: PyavOutput(str(path), format="mp4", options=MP4_OPTIONS)
        )
        self._lock = threading.Lock()
        self._current: str | None = None
        self._callbacks: list[ClipCallback] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        clips_dir.mkdir(parents=True, exist_ok=True)
        self._discard_partials()

    @property
    def recording(self) -> bool:
        with self._lock:
            return self._current is not None

    def trigger(self, stem: str, on_done: ClipCallback | None = None) -> str:
        """Starts (or joins) a clip and returns its final file name.

        `on_done(path)` runs on the worker thread once the clip is saved.
        """
        with self._lock:
            if on_done is not None:
                self._callbacks.append(on_done)
            if self._current is not None:
                return self._current
            self._current = name = f"{stem}{CLIP_SUFFIX}"

        self._thread = threading.Thread(
            target=self._record, args=(name,), name="clip-recorder", daemon=True
        )
        self._thread.start()
        return name

    def close(self, timeout_s: float = 5.0) -> None:
        """Cuts the wait short and finalizes a clip in progress (on shutdown)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)

    def _record(self, name: str) -> None:
        final = self._clips_dir / name
        partial = partial_path(final)
        saved = False
        try:
            self._circular.open_output(self._output_factory(partial))
            self._stop.wait(self._record_s)
            self._circular.close_output()
            partial.rename(final)
            saved = True
            log.info("Saved alert clip %s", final.name)
            self._prune()
        except Exception:
            log.exception("Failed to record clip %s", name)
            partial.unlink(missing_ok=True)
        finally:
            with self._lock:
                self._current = None
                callbacks, self._callbacks = self._callbacks, []

        for callback in callbacks if saved else []:
            try:
                callback(final)
            except Exception:
                log.exception("Clip callback failed for %s", final.name)

    def _discard_partials(self) -> None:
        """Removes clips left half-written by a crash or power loss (unplayable)."""
        for leftover in self._clips_dir.glob(f"*{PARTIAL_SUFFIX}"):
            log.warning("Discarding incomplete clip %s", leftover.name)
            leftover.unlink(missing_ok=True)

    def _prune(self) -> None:
        """Keeps only the newest `max_clips` clips to protect the microSD."""
        clips = sorted(
            c
            for c in self._clips_dir.glob(f"*{CLIP_SUFFIX}")
            if not c.name.endswith(PARTIAL_SUFFIX)
        )
        for old in clips[: -self._max_clips]:
            old.unlink(missing_ok=True)


def start_streaming(
    picam2: Picamera2, stream_cfg: StreamConfig, clip_cfg: ClipConfig
) -> tuple[PyavOutput, ClipRecorder]:
    """Starts one H.264 encoder feeding the RTSP stream and the clip buffer."""
    # repeat=True re-sends SPS/PPS with every keyframe, so clips cut from the
    # middle of the stream are decodable on their own
    encoder = H264Encoder(
        bitrate=stream_cfg.bitrate, iperiod=stream_cfg.iperiod, repeat=True
    )
    rtsp = PyavOutput(
        stream_cfg.rtsp_url, format="rtsp", options={"rtsp_transport": "tcp"}
    )
    circular = CircularOutput2(buffer_duration_ms=int(clip_cfg.seconds * 1000))

    picam2.start_recording(encoder, [rtsp, circular])
    log.info("Streaming to %s", stream_cfg.rtsp_url)

    recorder = ClipRecorder(
        circular,
        clips_dir=clip_cfg.dir,
        seconds=clip_cfg.seconds,
        max_clips=clip_cfg.max_clips,
    )
    return rtsp, recorder
