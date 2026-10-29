import logging

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import PyavOutput

from ..config import StreamConfig

log = logging.getLogger(__name__)


def start_streaming(picam2: Picamera2, stream_cfg: StreamConfig) -> PyavOutput:
    """Start an H.264 encoder streaming directly to RTSP via PyavOutput."""
    encoder = H264Encoder(bitrate=stream_cfg.bitrate, iperiod=stream_cfg.iperiod)

    rtsp = PyavOutput(
        stream_cfg.rtsp_url, format="rtsp", options={"rtsp_transport": "tcp"}
    )

    picam2.start_recording(encoder, rtsp)
    log.info("Streaming to %s", stream_cfg.rtsp_url)
    return rtsp
