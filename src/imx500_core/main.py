"""Entry point: `python -m imx500_core <project>` (or PROJECT in the env).

config → analyzer (before the camera: config errors fail fast) → camera →
overlay (pre_callback) → H.264/RTSP stream → watchdog loop. Everything after
opening the camera lives inside the `try`, so `finally` always releases it.
"""

import argparse
import logging
import os
import signal
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from threading import Event

from .camera.camera import setup_camera
from .camera.stream import start_streaming
from .config import load_config
from .processing.analysis import SceneAnalyzer
from .processing.overlay import DetectionOverlay, load_labels

log = logging.getLogger(__name__)

WATCHDOG_TIMEOUT_S = 5.0
HEARTBEAT_INTERVAL_S = 5.0

stop_event = Event()


def _handle_shutdown(signum, frame) -> None:
    """Ctrl+C (SIGINT) or systemd stop (SIGTERM): stop gracefully."""
    log.info("Shutdown signal received. Stopping the system gracefully...")
    stop_event.set()


def _safe_cleanup(step: Callable[[], None], name: str) -> None:
    """Runs a cleanup step without letting its failure skip the following ones."""
    try:
        step()
    except Exception as e:
        log.warning("Cleanup step '%s' failed: %s", name, e)


def _rss_mb() -> float | None:
    """Resident memory of this process (MB)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="imx500_core", description="Run an IMX500 AI Camera project."
    )
    parser.add_argument(
        "project",
        nargs="?",
        default=os.environ.get("PROJECT"),
        help="project folder name under projects/<task>/ (default: $PROJECT)",
    )
    args = parser.parse_args(argv)
    if not args.project:
        parser.error("no project given (argument or PROJECT environment variable)")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)
    stop_event.clear()

    log.info("Loading configuration for project '%s'...", args.project)
    config = load_config(args.project)
    labels = load_labels(config.model.labels_file)
    analysis = config.analysis
    analyzer = SceneAnalyzer(labels, analysis) if analysis.enabled else None

    log.info("Initializing Raspberry Pi AI Camera (IMX500)...")
    camera_state = setup_camera(cfg=config.camera, model_cfg=config.model)
    picam2 = camera_state.picam2

    exit_code = 0
    try:
        overlay = DetectionOverlay(
            picam2=picam2,
            imx500=camera_state.imx500,
            labels=labels,
            threshold=config.model.threshold,
            analyzer=analyzer,
            min_score=analysis.tracker.min_score if analyzer else None,
            motion_scale=analysis.motion.scale,
            show_motion=analysis.motion.show,
        )
        picam2.pre_callback = overlay

        log.info("Starting RTSP video streaming...")
        start_streaming(picam2, config.stream)
        log.info("System online. Streaming to %s", config.stream.rtsp_url)

        while not stop_event.wait(timeout=HEARTBEAT_INTERVAL_S):
            rss = _rss_mb()
            log.info(
                "IMX500 Sensor Temperature: %s °C | RSS: %s MB | detections: %d",
                overlay.sensor_temp,
                f"{rss:.1f}" if rss is not None else "n/a",
                len(overlay.detections),
            )
            if analyzer is not None:
                log.info("Scene %s", analyzer.summary())

            stalled = time.monotonic() - overlay.last_frame_time
            if stalled > WATCHDOG_TIMEOUT_S:
                log.error(
                    "No frames for %.1f s (camera frozen). Exiting for restart.",
                    stalled,
                )
                exit_code = 1
                break

    except Exception:
        log.exception("Fatal error. Shutting down.")
        exit_code = 1

    finally:
        log.info("Releasing hardware and closing connections...")
        _safe_cleanup(picam2.stop_recording, "stop_recording")
        _safe_cleanup(picam2.stop_preview, "stop_preview")
        _safe_cleanup(picam2.close, "close")
        log.info("Shutdown completed.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
