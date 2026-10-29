"""Hybrid overlay (Picamera2 + modlib Annotator / pp). Camera hardware is mocked."""

from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("modlib")
pytest.importorskip("cv2")

from imx500_core.processing import overlay as ov  # noqa: E402


class _FakeMappedArray:
    def __init__(self, request, stream):
        self.array = request.array

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


SENSOR_W, SENSOR_H = ov.SENSOR_SIZE
# 640x640 output: libcamera crops a centered square of the 4:3 sensor
SQUARE_CROP = ((SENSOR_W - SENSOR_H) // 2, 0, SENSOR_H, SENSOR_H)


@dataclass
class _Rect:
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0


class _FakeIMX500:
    def __init__(self, outputs, roi=None):
        self.outputs = outputs
        self.config = {"roi": roi or _Rect()}

    def get_input_size(self):
        return (640, 640)

    def get_outputs(self, metadata, add_batch=False):
        return self.outputs


class _FakePicam2:
    def __init__(self, fmt="XBGR8888", size=(640, 640)):
        self.fmt, self.size = fmt, size

    def camera_configuration(self):
        return {"main": {"size": self.size, "format": self.fmt}}


class _FakeRequest:
    def __init__(self, metadata, size=(640, 640)):
        self.metadata = metadata
        self.array = np.zeros((size[1], size[0], 4), dtype=np.uint8)

    def get_metadata(self):
        return self.metadata


@pytest.fixture(autouse=True)
def _fake_mapped_array(monkeypatch):
    monkeypatch.setattr(ov, "MappedArray", _FakeMappedArray)


def _outputs(boxes_px, scores, classes, with_count=True):
    """Ultralytics IMX500 outputs (no batch dim), padded like the sensor does."""
    n, pad = len(scores), 300
    boxes = np.zeros((pad, 4), np.float32)
    boxes[:n] = boxes_px
    s = np.zeros(pad, np.float32)
    s[:n] = scores
    c = np.zeros(pad, np.float32)
    c[:n] = classes
    out = [boxes, s, c]
    return [*out, np.array([n], np.float32)] if with_count else out


def _expected_px(box_norm, crop, out=(640, 640)):
    """Picamera2's convert_inference_coords math, for an xyxy box."""
    x0, y0, x1, y1 = box_norm
    cx, cy, cw, ch = crop
    to_x = lambda v: (np.clip(v * SENSOR_W, cx, cx + cw) - cx) * out[0] / cw  # noqa: E731
    to_y = lambda v: (np.clip(v * SENSOR_H, cy, cy + ch) - cy) * out[1] / ch  # noqa: E731
    return to_x(x0), to_y(y0), to_x(x1), to_y(y1)


@pytest.mark.parametrize("with_count", [True, False])
def test_parse_outputs_uses_modlib_pp(with_count):
    outs = _outputs([[64, 128, 320, 640]], [0.9], [1], with_count=with_count)
    dets = ov.parse_outputs(outs, 640)
    dets = dets[dets.confidence > 0.5]
    assert len(dets) == 1
    np.testing.assert_allclose(dets.bbox[0], [0.1, 0.2, 0.5, 1.0])
    assert int(dets.class_id[0]) == 1


def test_parse_outputs_with_too_few_tensors_is_empty():
    assert len(ov.parse_outputs([np.zeros((1, 4))], 640)) == 0


def test_roi_full_sensor_into_square_crop():
    roi = ov.input_tensor_roi(SQUARE_CROP)
    assert roi.left == pytest.approx(-508 / 3040)
    assert roi.top == 0
    assert roi.width == pytest.approx(4056 / 3040)
    assert roi.height == 1


def test_roi_matches_picamera2_mapping():
    """modlib's compensate_for_roi + scaling == Picamera2's convert_inference_coords."""
    box = (0.5, 0.5, 0.75, 0.75)
    dets = ov.Detections(
        bbox=np.array([box], np.float32),
        confidence=np.array([0.9], np.float32),
        class_id=np.array([0], np.uint16),
    )
    dets.compensate_for_roi(ov.input_tensor_roi(SQUARE_CROP))
    got = dets.bbox[0] * 640
    np.testing.assert_allclose(got, _expected_px(box, SQUARE_CROP), atol=1)


def test_callback_draws_boxes_in_place_and_filters_threshold():
    # person at the center of the input tensor, plus a low-score package
    outs = _outputs([[256, 256, 384, 384], [0, 0, 64, 64]], [0.88, 0.27], [1, 0])
    overlay = ov.DetectionOverlay(
        _FakePicam2(), _FakeIMX500(outs), ["package", "person"], threshold=0.55
    )
    req = _FakeRequest({"ScalerCrop": SQUARE_CROP, "SensorTemperature": 42.0})

    overlay(req)

    assert len(overlay.detections) == 1
    assert overlay.labels_for(overlay.detections) == ["person 0.88"]
    assert overlay.sensor_temp == 42.0
    x0, y0, x1, y1 = (
        int(round(v)) for v in _expected_px((0.4, 0.4, 0.6, 0.6), SQUARE_CROP)
    )
    # box edges were drawn into the request buffer (no copy)
    assert req.array[(y0 + y1) // 2, x0 - 1 : x0 + 2].any()
    assert req.array[(y0 + y1) // 2, x1 - 1 : x1 + 2].any()
    # interior stays untouched (no alpha fill)
    assert not req.array[(y0 + y1) // 2, (x0 + x1) // 2].any()


def test_keeps_last_detections_on_frames_without_outputs():
    outs = _outputs([[256, 256, 384, 384]], [0.88], [1])
    imx = _FakeIMX500(outs)
    overlay = ov.DetectionOverlay(_FakePicam2(), imx, ["package", "person"])
    overlay(_FakeRequest({"ScalerCrop": SQUARE_CROP}))
    imx.outputs = None
    req = _FakeRequest({"ScalerCrop": SQUARE_CROP})
    overlay(req)
    assert len(overlay.detections) == 1
    assert req.array.any()


def test_inference_roi_is_used_when_set():
    roi = _Rect(508, 0, 3040, 3040)  # set_auto_aspect_ratio for a square model
    overlay = ov.DetectionOverlay(_FakePicam2(), _FakeIMX500(None, roi), ["a"])
    assert overlay._inference_roi() == (508, 0, 3040, 3040)
    assert tuple(ov.input_tensor_roi(SQUARE_CROP, overlay._inference_roi())) == (
        0,
        0,
        1,
        1,
    )


@pytest.mark.parametrize(
    ("fmt", "swapped"), [("XBGR8888", True), ("BGR888", True), ("XRGB8888", False)]
)
def test_palette_matches_buffer_channel_order(fmt, swapped):
    ref = ov.ColorPalette.default().colors[0]
    c = ov.palette_for_format(fmt).colors[0]
    # what modlib writes into the buffer (as_bgr) must read back as the real color
    written = c.as_bgr()
    real_in_buffer = ref.as_rgb() if swapped else ref.as_bgr()
    assert written == real_in_buffer


def test_tracks_drawn_with_alert_color_when_holder_is_active():
    pytest.importorskip("lap")
    from imx500_core.config import AnalysisConfig, RelationConfig
    from imx500_core.processing.analysis import SceneAnalyzer

    relation = RelationConfig(name="carrying", subject="person", objects=("package",))
    analyzer = SceneAnalyzer(
        ["package", "person"], AnalysisConfig(relations=(relation,))
    )
    # person at the center of the input tensor holding a package
    outs = _outputs([[256, 200, 384, 500], [290, 320, 350, 400]], [0.88, 0.62], [1, 0])
    overlay = ov.DetectionOverlay(
        _FakePicam2("XBGR8888"),
        _FakeIMX500(outs),
        ["package", "person"],
        threshold=0.55,
        analyzer=analyzer,
        min_score=0.25,
    )
    for _ in range(25):
        imx_outs = [o.copy() for o in outs]  # modlib's pp divides in place
        overlay._imx500.outputs = imx_outs
        req = _FakeRequest({"ScalerCrop": SQUARE_CROP})
        overlay(req)

    assert len(analyzer.active_ids) == 1
    assert len(overlay.detections) == 2
    x0, y0, x1, y1 = (
        int(round(v))
        for v in _expected_px((0.4, 200 / 640, 0.6, 500 / 640), SQUARE_CROP)
    )
    # XBGR8888 is [R, G, B, X] in numpy: pure red on the person's left edge
    edge = req.array[(y0 + y1) // 2, x0 - 1 : x0 + 2, :3]
    assert any(tuple(px) == (255, 0, 0) for px in edge)


def test_motion_frame_is_small_three_channel_copy():
    overlay = ov.DetectionOverlay(
        _FakePicam2(), _FakeIMX500(None), ["a"], motion_scale=0.5
    )
    image = np.zeros((640, 640, 4), np.uint8)
    frame = overlay._motion_frame(image)
    assert frame.image.shape == (320, 320, 3)
    assert (frame.width, frame.height) == (320, 320)
    assert not np.shares_memory(frame.image, image)
