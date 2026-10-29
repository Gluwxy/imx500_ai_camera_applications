"""Shared test setup. Picamera2 only exists on the Pi: it is mocked elsewhere."""

import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

try:
    import picamera2  # noqa: F401
except (ImportError, RuntimeError):
    picamera2_mock = MagicMock()
    sys.modules["picamera2"] = picamera2_mock
    sys.modules["picamera2.devices"] = picamera2_mock.devices
    sys.modules["picamera2.devices.imx500"] = picamera2_mock.devices.imx500


@pytest.fixture
def model_files(tmp_path: Path) -> tuple[Path, Path]:
    """Fake model files: `FilePath` needs them to exist and git ignores `*.rpk`."""
    network = tmp_path / "dummy_network.rpk"
    labels = tmp_path / "dummy_labels.txt"
    network.write_bytes(b"")
    labels.write_text("package\nperson\n")
    return network, labels


@pytest.fixture
def config_file(model_files: tuple[Path, Path]) -> Path:
    """`fixtures/config_test.yaml` copied next to the fake model files."""
    path = model_files[0].parent / "config_test.yaml"
    shutil.copy(FIXTURES_DIR / "config_test.yaml", path)
    return path
