import shutil

import pytest
from pydantic import ValidationError

from imx500_core.config import find_project, load_config, project_root


@pytest.fixture(autouse=True)
def _secrets(monkeypatch):
    """Isolated from the developer's environment."""
    monkeypatch.setenv("TELEGRAM_TOKEN", "mock_token_abc123")
    monkeypatch.delenv("MQTT_PASSWORD", raising=False)


def test_project_config_is_merged_over_the_base(project_dir, base_file):
    config = load_config(project_dir, base_file)

    # from the project
    assert config.model.task == "object detection"
    assert config.model.threshold == 0.60
    assert config.camera.framerate == 10
    assert config.mqtt.topic == "project/events"
    # from the base, inside sections the project also sets
    assert config.camera.resolution == (640, 640)
    assert config.telegram.chat_id == 987654321
    # secrets
    assert config.telegram.token.get_secret_value() == "mock_token_abc123"
    assert config.mqtt.password is None


def test_model_paths_are_relative_to_the_project(project_dir, base_file, model_files):
    network, labels = model_files
    config = load_config(project_dir, base_file)

    assert config.model.network_file == network.resolve()
    assert config.model.labels_file == labels.resolve()


def test_project_found_by_name(project_dir):
    projects = project_dir.parents[1]
    assert find_project("demo", projects) == project_dir
    with pytest.raises(FileNotFoundError, match="nope"):
        find_project("nope", projects)


def test_missing_project_config_fails(project_dir, base_file):
    (project_dir / "config.yaml").unlink()
    with pytest.raises(FileNotFoundError, match="Missing project config"):
        load_config(project_dir, base_file)


def test_missing_model_file_fails(project_dir, base_file, model_files):
    model_files[0].unlink()
    with pytest.raises(ValidationError, match="network_file"):
        load_config(project_dir, base_file)


def test_missing_telegram_token_fails(project_dir, base_file, monkeypatch):
    monkeypatch.delenv("TELEGRAM_TOKEN")
    with pytest.raises(ValidationError, match="telegram_token"):
        load_config(project_dir, base_file)


def test_real_configs_are_valid(tmp_path):
    """The repo's config.yaml + porch_piracy merge and validate (model faked)."""
    project = find_project("porch_piracy")
    fake = tmp_path / "porch_piracy"
    shutil.copytree(project, fake)
    (fake / "models").mkdir(exist_ok=True)
    (fake / "models" / "network.rpk").write_bytes(b"")

    config = load_config(fake, project_root / "config.yaml")
    assert config.model.labels_file.read_text().split() == ["package", "person"]
    assert config.stream.rtsp_url.endswith("/porch")
    assert [r.name for r in config.analysis.relations] == ["carrying_package"]
    assert config.analysis.needs_motion
    assert config.analysis.alerts == ("carrying_package",)
    assert config.annotator.colors.classes["person"] == "#cba6f7"
    assert config.clips.dir == project_root / "data" / "clips"
    assert config.mqtt.enabled


def test_telegram_token_only_needed_when_enabled(project_dir, base_file, monkeypatch):
    monkeypatch.delenv("TELEGRAM_TOKEN")
    project_file = project_dir / "config.yaml"
    project_file.write_text(project_file.read_text() + "telegram:\n  enabled: false\n")
    assert load_config(project_dir, base_file).telegram.token is None


def test_colors_must_be_hex(project_dir, base_file):
    project_file = project_dir / "config.yaml"
    project_file.write_text(
        project_file.read_text() + "annotator:\n  colors:\n    alert: red\n"
    )
    with pytest.raises(ValidationError, match="#rrggbb"):
        load_config(project_dir, base_file)
