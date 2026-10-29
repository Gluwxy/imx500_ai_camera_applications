import pytest
from pydantic import ValidationError

from imx500_core.config import load_config


@pytest.fixture(autouse=True)
def _secrets(monkeypatch):
    """Isolated from the developer's environment."""
    monkeypatch.setenv("TELEGRAM_TOKEN", "mock_token_abc123")
    monkeypatch.delenv("MQTT_PASSWORD", raising=False)


def test_load_config_from_fixture(config_file, model_files):
    config = load_config(config_file)

    assert config.model.task == "object detection"
    assert config.model.threshold == 0.60
    assert config.telegram.chat_id == 987654321
    assert config.telegram.token.get_secret_value() == "mock_token_abc123"
    assert config.mqtt.password is None


def test_model_paths_resolve_next_to_the_yaml(config_file, model_files):
    network, labels = model_files
    config = load_config(config_file)

    assert config.model.network_file == network.resolve()
    assert config.model.labels_file == labels.resolve()


def test_missing_model_file_fails(config_file, model_files):
    model_files[0].unlink()
    with pytest.raises(ValidationError, match="network_file"):
        load_config(config_file)


def test_missing_telegram_token_fails(config_file, monkeypatch):
    monkeypatch.delenv("TELEGRAM_TOKEN")
    with pytest.raises(ValidationError, match="telegram_token"):
        load_config(config_file)
