from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, FilePath, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

project_root = Path(__file__).resolve().parents[2]


class Base(BaseModel):
    """Common base class: immutable and forbids extra fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelConfig(Base):
    task: str
    network_file: FilePath
    labels_file: FilePath
    threshold: float = Field(default=0.55, ge=0.0, le=1.0)


class IMX500Config(Base):
    preview_type: Literal["null", "qt", "drm"] = "null"
    config_type: Literal["preview", "video"] = "video"
    framerate: int = Field(default=30, ge=1, le=60)
    resolution: tuple[int, int] = Field(default=(640, 640))
    buffer_count: int = Field(default=6, ge=2, le=12)


class StreamConfig(Base):
    rtsp_url: str = "rtsp://127.0.0.1:8554/porch"
    bitrate: int = Field(default=2_000_000, ge=100_000)
    iperiod: int = Field(default=15, ge=1)


class DashboardConfig(Base):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)


class MqttConfig(Base):
    host: str = "localhost"
    port: int = Field(default=1883, ge=1, le=65535)
    topic: str = "porch/events"
    username: str | None = None
    password: SecretStr | None = None  # from .env


class TelegramConfig(Base):
    chat_id: int
    cooldown_s: int = Field(default=60, ge=0)
    token: SecretStr


class AppConfig(Base):
    model: ModelConfig
    camera: IMX500Config = IMX500Config()
    stream: StreamConfig = StreamConfig()
    dashboard: DashboardConfig = DashboardConfig()
    mqtt: MqttConfig = MqttConfig()
    telegram: TelegramConfig


class Secrets(BaseSettings):
    """Reads environment variables and values from the `.env` file"""

    model_config = SettingsConfigDict(extra="ignore")

    telegram_token: SecretStr
    mqtt_password: SecretStr | None = None


def load_config(yaml_file: str | Path = project_root / "config.yaml") -> AppConfig:
    yaml_file = Path(yaml_file).resolve()
    data = yaml.safe_load(yaml_file.read_text()) or {}

    base_dir = yaml_file.parent

    if "model" in data:
        if "network_file" in data["model"]:
            data["model"]["network_file"] = base_dir / data["model"]["network_file"]
        if "labels_file" in data["model"]:
            data["model"]["labels_file"] = base_dir / data["model"]["labels_file"]

    # .env secrets
    secrets = Secrets(_env_file=base_dir / ".env")
    data.setdefault("telegram", {})["token"] = secrets.telegram_token
    data.setdefault("mqtt", {})["password"] = secrets.mqtt_password

    return AppConfig.model_validate(data)
