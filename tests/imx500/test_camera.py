from unittest.mock import patch

import pytest

from imx500_core.camera.camera import Camera, setup_camera
from imx500_core.config import IMX500Config, ModelConfig


@pytest.fixture
def mock_configs(model_files):
    network, labels = model_files
    cfg = IMX500Config(
        config_type="preview",
        resolution=(640, 640),
        framerate=30,
        buffer_count=2,
        preview_type="null",
    )
    model_cfg = ModelConfig(
        network_file=network, labels_file=labels, task="object detection"
    )
    return cfg, model_cfg


@patch("imx500_core.camera.camera.IMX500")
@patch("imx500_core.camera.camera.Picamera2")
def test_setup_camera_success(mock_picamera2_cls, mock_imx500_cls, mock_configs):
    """Test successful IMX500 camera initialization."""
    cfg, model_cfg = mock_configs

    mock_imx500 = mock_imx500_cls.return_value
    mock_imx500.camera_num = 0

    mock_intrinsics = mock_imx500.network_intrinsics
    mock_intrinsics.task = "object detection"
    mock_intrinsics.inference_rate = 30
    mock_intrinsics.preserve_aspect_ratio = True

    camera_obj = setup_camera(cfg, model_cfg)

    assert isinstance(camera_obj, Camera)
    assert camera_obj.picam2 == mock_picamera2_cls.return_value
    assert camera_obj.imx500 == mock_imx500
    assert camera_obj.intrinsics.labels == ["package", "person"]

    mock_picamera2_cls.return_value.configure.assert_called_once()
    mock_picamera2_cls.return_value.start_preview.assert_called_once()
    mock_imx500.set_auto_aspect_ratio.assert_called_once()


@patch("imx500_core.camera.camera.IMX500")
@patch("imx500_core.camera.camera.Picamera2")
def test_setup_camera_exception_closes_camera(
    mock_picamera2_cls, mock_imx500_cls, mock_configs
):
    """Test that the camera closes safely if configuration fails."""
    cfg, model_cfg = mock_configs

    mock_imx500 = mock_imx500_cls.return_value
    mock_imx500.network_intrinsics.task = "object detection"
    mock_imx500.network_intrinsics.inference_rate = 30
    mock_picam2 = mock_picamera2_cls.return_value
    mock_picam2.configure.side_effect = Exception("Config error")

    with pytest.raises(Exception, match="Config error"):
        setup_camera(cfg, model_cfg)

    mock_picam2.close.assert_called_once()
