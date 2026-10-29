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


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _merge(base: dict, override: dict) -> dict:
    """Recursive merge: `override` wins, nested sections are merged key by key."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def find_project(name: str, projects_dir: Path = project_root / "projects") -> Path:
    """`porch_piracy` -> projects/<task>/porch_piracy (task folder found)."""
    matches = sorted(p for p in projects_dir.glob(f"*/{name}") if p.is_dir())
    if not matches:
        raise FileNotFoundError(f"Project '{name}' not found in {projects_dir}/*/")
    if len(matches) > 1:
        raise ValueError(f"Project '{name}' is ambiguous: {matches}")
    return matches[0]


def load_config(
    project: str | Path, base_file: str | Path = project_root / "config.yaml"
) -> AppConfig:
    """Root config.yaml (shared) merged with the project's config.yaml.

    `project` is a project name (`porch_piracy`) or its folder. Model paths in
    the project file are relative to the project folder; secrets come from the
    environment or the `.env` next to the root config.
    """
    project_dir = Path(project)
    if not project_dir.is_dir():
        project_dir = find_project(str(project))
    project_dir = project_dir.resolve()
    base_file = Path(base_file).resolve()

    project_file = project_dir / "config.yaml"
    if not project_file.is_file():
        raise FileNotFoundError(f"Missing project config: {project_file}")

    data = _merge(_read_yaml(base_file), _read_yaml(project_file))

    model = data.get("model", {})
    for key in ("network_file", "labels_file"):
        if key in model:
            model[key] = project_dir / model[key]

    # .env secrets (shared by all projects)
    secrets = Secrets(_env_file=base_file.parent / ".env")
    data.setdefault("telegram", {})["token"] = secrets.telegram_token
    data.setdefault("mqtt", {})["password"] = secrets.mqtt_password

    return AppConfig.model_validate(data)
