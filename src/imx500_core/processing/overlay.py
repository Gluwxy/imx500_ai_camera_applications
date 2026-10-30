"""Hybrid overlay: Picamera2 owns the camera, modlib parses and draws.

modlib's `AiCamera` allocates its own frame buffers (fixed size) and does not
fit in the Pi Zero 2W's RAM. Here Picamera2 keeps the camera and buffers, and
only modlib's pure-numpy pieces are used inside the `pre_callback`:

- `pp_od_yolo_ultralytics` turns the IMX500 output tensors into `Detections`
  (normalized [x1, y1, x2, y2] relative to the input tensor).
- `Annotator.annotate_boxes` draws them directly on the `MappedArray`
  (wrapped in a modlib `Frame`, no copy), so the boxes reach the encoder.

The input tensor -> image mapping is expressed as a modlib `ROI` and applied
once per inference (`compensate_for_roi`), so every box (detections, tracks,
motion) is normalized to the output image. With a `SceneAnalyzer` the tracks
are drawn instead of the raw detections. The label of each track is its
legend: class, conditions met (`+package`, `moving`) and score; its color is
its state (`annotator.colors`, Catppuccin by default): the class color, then
`moving` once Motion sees it move, then `alert` while it is the subject of an
active alert relation. Motion boxes are thin unlabeled outlines (optional),
and an optional chip legend sits in the top-left corner.
"""

import logging
import time
from pathlib import Path

import cv2
import numpy as np
from modlib.apps import Annotator, Color, ColorPalette
from modlib.devices import IMAGE_TYPE, Frame
from modlib.models import ROI, Detections
from modlib.models.post_processors import pp_od_yolo_ultralytics
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500

from ..config import AnnotatorConfig, ColorsConfig
from .analysis import SceneAnalyzer

log = logging.getLogger(__name__)

# Fixed by Picamera2 (`IMX500.__get_full_sensor_resolution`)
SENSOR_SIZE = (4056, 3040)

# Formats whose numpy layout is [R, G, B(, X)] (see the Picamera2 manual).
# modlib draws with `Color.as_bgr()`, so their palette must be swapped.
RGB_ORDER_FORMATS = {"XBGR8888", "BGR888"}

LEGEND_MARGIN = 6  # px from the image corner and between chips

_EMPTY = Detections(
    bbox=np.empty((0, 4), dtype=np.float32),
    confidence=np.empty((0,), dtype=np.float32),
    class_id=np.empty((0,), dtype=np.uint16),
)


def load_labels(path: str | Path) -> list[str]:
    return [
        line.strip() for line in Path(path).read_text().splitlines() if line.strip()
    ]


def buffer_color(color: Color, pixel_format: str) -> Color:
    """`color` as it must be passed to modlib to look right in this buffer."""
    if pixel_format in RGB_ORDER_FORMATS:
        return Color(color.b, color.g, color.r)
    return color


def class_palette(
    labels: list[str], colors: ColorsConfig, pixel_format: str
) -> ColorPalette:
    """One color per class index (`colors.classes`, else `colors.default`)."""
    return ColorPalette(
        [
            buffer_color(
                Color.from_hex(colors.classes.get(name, colors.default)), pixel_format
            )
            for name in labels
        ]
    )


def palette_for_format(pixel_format: str) -> ColorPalette:
    """modlib's default palette in the channel order of the stream buffer."""
    palette = ColorPalette.default()
    if pixel_format in RGB_ORDER_FORMATS:
        palette = ColorPalette([buffer_color(c, pixel_format) for c in palette.colors])
    return palette


def input_tensor_roi(
    scaler_crop: tuple[int, int, int, int],
    inference_roi: tuple[int, int, int, int] | None = None,
) -> ROI:
    """Where the input tensor lies inside the output image, as a modlib `ROI`.

    Both rectangles are (x, y, w, h) in full sensor pixels: `ScalerCrop` is the
    part of the sensor the ISP outputs, `inference_roi` the part the network
    sees (full sensor unless an inference ROI / aspect ratio was set).
    Values outside [0, 1] are fine: modlib clips the boxes to the image.
    """
    ix, iy, iw, ih = inference_roi or (0, 0, *SENSOR_SIZE)
    cx, cy, cw, ch = scaler_crop
    return ROI((ix - cx) / cw, (iy - cy) / ch, iw / cw, ih / ch)


def parse_outputs(outputs: list[np.ndarray], input_size: int) -> Detections:
    """IMX500 outputs (no batch dim) -> modlib `Detections` via modlib's own pp.

    Ultralytics exports give boxes, scores, classes and the number of
    detections. If the count tensor is missing, every row is kept (the
    padding rows have score 0 and are dropped by the threshold).
    """
    if len(outputs) < 3:
        return _EMPTY
    boxes, scores, classes = outputs[:3]
    # The count arrives as shape (1,); modlib does int() on it, which newer
    # numpy only allows on scalars
    count = outputs[3].reshape(-1)[0] if len(outputs) > 3 else len(scores)
    return pp_od_yolo_ultralytics(
        [boxes, scores, classes, count], input_tensor_sz=input_size
    )


def to_image_space(dets: Detections, roi: ROI) -> Detections:
    """Input-tensor boxes -> boxes normalized to the output image (in place).

    Done once per inference so tracker, `Matcher` and `Motion` share one space
    (motion boxes come from the image). Boxes left with no area after clipping
    (outside the image crop) are dropped.
    """
    dets.compensate_for_roi(roi)
    b = dets.bbox
    return dets[(b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])]


class DetectionOverlay:
    """pre_callback: parses the IMX500 outputs, runs the analysis, draws boxes."""

    def __init__(
        self,
        picam2: Picamera2,
        imx500: IMX500,
        labels: list[str],
        threshold: float = 0.5,
        stream: str = "main",
        annotator: Annotator | None = None,
        analyzer: SceneAnalyzer | None = None,
        min_score: float | None = None,
        motion_scale: float = 0.5,
        annotator_cfg: AnnotatorConfig | None = None,
    ) -> None:
        self._picam2 = picam2
        self._imx500 = imx500
        self._labels = labels
        self._threshold = threshold
        self._stream = stream
        self._input_size = int(imx500.get_input_size()[0])
        self._analyzer = analyzer
        # With a tracker, low scores are kept: ByteTrack uses them
        self._min_score = threshold if min_score is None else min_score
        self._cfg = annotator_cfg or AnnotatorConfig()

        stream_cfg = picam2.camera_configuration()[stream]
        self._width, self._height = stream_cfg["size"]
        self._format = stream_cfg["format"]
        self._motion_size = (
            max(1, round(self._width * motion_scale)),
            max(1, round(self._height * motion_scale)),
        )
        colors = self._cfg.colors
        self._palette = class_palette(labels, colors, self._format)
        self._annotator = annotator or Annotator(
            color=self._palette,
            thickness=2,
            text_scale=0.5,
            text_thickness=1,
            text_padding=4,
        )
        self._alert_color = buffer_color(Color.from_hex(colors.alert), self._format)
        self._motion_color = buffer_color(Color.from_hex(colors.motion), self._format)
        self._moving_color = buffer_color(Color.from_hex(colors.moving), self._format)

        self.detections: Detections = _EMPTY
        self.roi = ROI(0.0, 0.0, 1.0, 1.0)
        self.last_frame_time = time.monotonic()
        self.sensor_temp: float | None = None

    @property
    def analyzer(self) -> SceneAnalyzer | None:
        return self._analyzer

    def __call__(self, request) -> None:
        try:
            metadata = request.get_metadata()
            self.last_frame_time = time.monotonic()
            self.sensor_temp = metadata.get("SensorTemperature")

            if (crop := metadata.get("ScalerCrop")) is not None:
                self.roi = input_tensor_roi(crop, self._inference_roi())

            outputs = self._imx500.get_outputs(metadata)
            with MappedArray(request, self._stream) as m:
                # Frames without outputs keep the last detections/tracks
                if outputs is not None:
                    self._update(parse_outputs(outputs, self._input_size), m.array)
                self._draw(m.array)

        except Exception:
            log.exception("Error in detection overlay")

    def _update(self, dets: Detections, image: np.ndarray) -> None:
        dets = to_image_space(dets[dets.confidence >= self._min_score], self.roi)
        self.detections = dets[dets.confidence >= self._threshold]
        if self._analyzer is not None:
            # Before drawing: boxes on the image would count as motion
            a = self._analyzer
            a.update(dets, self._motion_frame(image) if a.needs_motion else None)

    def _motion_frame(self, image: np.ndarray) -> Frame:
        """Small 3-channel copy for modlib `Motion` (it converts BGR -> gray)."""
        small = cv2.resize(image, self._motion_size, interpolation=cv2.INTER_AREA)
        if small.ndim == 3 and small.shape[2] == 4:
            small = cv2.cvtColor(small, cv2.COLOR_BGRA2BGR)
        return self._frame(small, *self._motion_size)

    def _inference_roi(self) -> tuple[int, int, int, int] | None:
        roi = self._imx500.config.get("roi")
        if roi is None or not (roi.width and roi.height):
            return None
        return (roi.x, roi.y, roi.width, roi.height)

    def labels_for(self, detections: Detections) -> list[str]:
        return [
            f"{self._labels[c] if c < len(self._labels) else c} {s:.2f}"
            for _, s, c, _ in detections
        ]

    @staticmethod
    def _frame(image: np.ndarray, width: int, height: int) -> Frame:
        # Boxes are already in image space: full-frame ROI, no compensation
        return Frame(
            timestamp="",
            image=image,
            image_type=IMAGE_TYPE.VGA,
            width=width,
            height=height,
            channels=image.shape[2] if image.ndim == 3 else 1,
            detections=None,
            new_detection=True,
            fps=0.0,
            dps=0.0,
            roi=ROI(0.0, 0.0, 1.0, 1.0),
        )

    def _draw(self, image: np.ndarray) -> None:
        # `image` is a view of the request buffer: modlib draws in place
        frame = self._frame(image, self._width, self._height)
        a, ann = self._analyzer, self._annotator

        if a is None:
            if len(self.detections):
                ann.annotate_boxes(
                    frame, self.detections, labels=self.labels_for(self.detections)
                )
            return

        if self._cfg.show_motion and len(a.motion_boxes):
            thickness, ann.thickness = ann.thickness, 1
            ann.annotate_boxes(
                frame, a.motion_boxes, skip_label=True, color=self._motion_color
            )
            ann.thickness = thickness
        tracks = a.tracks
        if len(tracks):
            ids = np.asarray(tracks.tracker_id)
            alert = np.isin(ids, list(a.alert_ids))
            moving = np.isin(ids, list(a.moving_ids)) & ~alert
            groups = (
                (~alert & ~moving, None),  # class color (palette)
                (moving, self._moving_color),
                (alert, self._alert_color),
            )
            for mask, color in groups:
                sel = tracks[mask]
                if len(sel):
                    ann.annotate_boxes(
                        frame, sel, labels=a.labels_for(sel), color=color
                    )
        if self._cfg.legend:
            self._draw_legend(image, a)

    def legend_entries(self, a: SceneAnalyzer) -> list[tuple[str, Color]]:
        """Chips of the legend: classes, `moving`, alert relations, `motion`."""
        entries = [(name, self._palette.colors[i]) for i, name in enumerate(a.labels)]
        if a.needs_motion:
            entries.append(("moving", self._moving_color))
        entries += [(name, self._alert_color) for name in a.alert_names]
        if a.needs_motion and len(a.motion_boxes):
            entries.append(("motion", self._motion_color))
        return entries

    def _draw_legend(self, image: np.ndarray, a: SceneAnalyzer) -> None:
        ann = self._annotator
        (_, text_h), _ = cv2.getTextSize(
            "Ag", cv2.FONT_HERSHEY_SIMPLEX, ann.text_scale, ann.text_thickness
        )
        step = text_h + 2 * ann.text_padding + LEGEND_MARGIN
        for i, (text, color) in enumerate(self.legend_entries(a)):
            ann.set_label(
                image,
                x=LEGEND_MARGIN,
                y=LEGEND_MARGIN + i * step,
                color=color.as_bgr(),
                label=text,
            )
