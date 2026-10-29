import logging
from dataclasses import dataclass
from pathlib import Path

from picamera2 import Picamera2, Preview
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

from ..config import IMX500Config, ModelConfig

log = logging.getLogger(__name__)
PREVIEWS = {"null": Preview.NULL, "qt": Preview.QT, "drm": Preview.DRM}


@dataclass(frozen=True, slots=True)
class Camera:
    """Handles needed during runtime."""

    picam2: Picamera2
    imx500: IMX500
    intrinsics: NetworkIntrinsics


def _load_intrinsics(
    imx500: IMX500, labels_path: Path, expected_task: str
) -> NetworkIntrinsics:
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()

    if intrinsics.task and intrinsics.task != expected_task:
        raise ValueError(
            f"Model task is '{intrinsics.task}', expected '{expected_task}'"
        )

    intrinsics.task = expected_task
    intrinsics.labels = labels_path.read_text().splitlines()
    intrinsics.update_with_defaults()
    return intrinsics


def setup_camera(cfg: IMX500Config, model_cfg: ModelConfig) -> Camera:
    """Initializes the IMX500 AI module and configures the Picamera2 instance."""
    imx500 = IMX500(str(model_cfg.network_file))
    intrinsics = _load_intrinsics(imx500, model_cfg.labels_file, model_cfg.task)
    picam2 = Picamera2(imx500.camera_num)

    try:
        inference_rate = intrinsics.inference_rate or cfg.framerate
        fps = min(cfg.framerate, inference_rate)

        if fps < cfg.framerate:
            log.warning("FrameRate %s > inference_rate; using %s", cfg.framerate, fps)

        make_config = {
            "preview": picam2.create_preview_configuration,
            "video": picam2.create_video_configuration,
        }[cfg.config_type]

        picam2.configure(
            make_config(
                main={"size": cfg.resolution},
                controls={"FrameRate": fps},
                buffer_count=cfg.buffer_count,
            )
        )

        imx500.show_network_fw_progress_bar()
        picam2.start_preview(PREVIEWS[cfg.preview_type])

        if intrinsics.preserve_aspect_ratio:
            imx500.set_auto_aspect_ratio()

    except Exception:
        picam2.close()
        raise

    return Camera(picam2=picam2, imx500=imx500, intrinsics=intrinsics)
