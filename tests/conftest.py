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
def project_dir(tmp_path: Path) -> Path:
    """Fake project: `fixtures/project/config.yaml` plus fake model files.

    `FilePath` needs the model files to exist and git ignores `*.rpk`, so they
    are created here instead of being stored in the repo.
    """
    project = tmp_path / "projects" / "object_detection" / "demo"
    (project / "models").mkdir(parents=True)
    shutil.copy(FIXTURES_DIR / "project" / "config.yaml", project / "config.yaml")
    (project / "models" / "dummy_network.rpk").write_bytes(b"")
    (project / "dummy_labels.txt").write_text("package\nperson\n")
    return project


@pytest.fixture
def base_file(tmp_path: Path) -> Path:
    """Shared root config, outside the project folder (like the real one)."""
    path = tmp_path / "config.yaml"
    shutil.copy(FIXTURES_DIR / "base_config.yaml", path)
    return path


@pytest.fixture
def model_files(project_dir: Path) -> tuple[Path, Path]:
    network = project_dir / "models" / "dummy_network.rpk"
    return network, project_dir / "dummy_labels.txt"
