from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    SecretStr,
    model_validator,
)
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


class ClipConfig(Base):
    """Alert clips cut from a CircularOutput2: `seconds` before + after the event."""

    seconds: float = Field(default=5.0, gt=0.0, le=30.0)
    dir: Path = Path("data/clips")  # relative to the repo root
    max_clips: int = Field(default=20, ge=1)  # oldest are deleted (microSD)


def _hex_color(value: str) -> str:
    v = value.lstrip("#")
    if len(v) != 6 or any(c not in "0123456789abcdefABCDEF" for c in v):
        raise ValueError(f"'{value}' is not a #rrggbb color")
    return f"#{v.lower()}"


HexColor = Annotated[str, AfterValidator(_hex_color)]


class ColorsConfig(Base):
    """Box colors (Catppuccin Mocha by default). Classes not listed: `default`."""

    default: HexColor = "#a6adc8"  # subtext0
    alert: HexColor = "#f38ba8"  # red: subject of an active alert relation
    moving: HexColor = "#a6e3a1"  # green: subject moving (relation with `moving`)
    motion: HexColor = "#b4befe"  # lavender: Motion boxes
    classes: dict[str, HexColor] = {}


class AnnotatorConfig(Base):
    colors: ColorsConfig = ColorsConfig()
    legend: bool = False  # chips with class/state colors, top-left corner
    show_motion: bool = False  # draw the Motion boxes (debug)


class DashboardConfig(Base):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)


class MqttConfig(Base):
    enabled: bool = False
    host: str = "localhost"
    port: int = Field(default=1883, ge=1, le=65535)
    topic: str = "porch/events"
    username: str | None = None
    password: SecretStr | None = None  # from .env
    client_id: str = "imx500"
    keepalive_s: int = Field(default=60, ge=5)


class TelegramConfig(Base):
    enabled: bool = True
    chat_id: int
    cooldown_s: int = Field(default=60, ge=0)  # per alert name
    token: SecretStr | None = None  # from .env

    @model_validator(mode="after")
    def _token_when_enabled(self) -> "TelegramConfig":
        if self.enabled and self.token is None:
            raise ValueError("telegram_token is required when telegram is enabled")
        return self


class TrackerConfig(Base):
    """modlib BYTETracker, one instance per class. Buffers count inferences."""

    min_score: float = Field(default=0.25, ge=0.0, le=1.0)  # fed to the tracker
    # High score: a new track needs > track_thresh + 0.1
    track_thresh: float = Field(default=0.45, ge=0.0, le=1.0)
    track_buffer: int = Field(default=30, ge=1)
    match_thresh: float = Field(default=0.8, ge=0.0, le=1.0)
    mot20: bool = True  # False fuses the score into the cost: nested boxes drop
    # Inferences a track may live on low scores only before it is ignored
    max_low_streak: int = Field(default=15, ge=0)


class MatcherConfig(Base):
    """modlib Matcher (overlap = intersection / area of the matched box)."""

    min_overlap_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    hysteresis: float = Field(default=0.4, ge=0.0, le=1.0)
    max_missing_overlap: int = Field(default=60, ge=1)
    max_missing_tracker: int = Field(default=30, ge=1)


class MotionConfig(Base):
    """modlib Motion on a downscaled copy of the frame (only run if needed)."""

    scale: float = Field(default=0.5, gt=0.0, le=1.0)
    size_threshold: int = Field(default=300, ge=1)  # px² at full resolution


class RelationConfig(Base):
    """A `subject` track is active while every condition holds.

    - `objects`: each class must overlap the subject (one modlib Matcher each).
    - `moving`: motion must overlap the subject (modlib Motion + Matcher).

    Class names come from the project's labels.txt.
    """

    name: str
    subject: str
    objects: tuple[str, ...] = ()
    moving: bool = False
    matcher: MatcherConfig = MatcherConfig()
    motion_matcher: MatcherConfig = MatcherConfig(min_overlap_threshold=0.3)

    @model_validator(mode="after")
    def _has_a_condition(self) -> "RelationConfig":
        if not self.objects and not self.moving:
            raise ValueError(f"relation '{self.name}' needs `objects` or `moving`")
        if self.subject in self.objects:
            raise ValueError(f"relation '{self.name}': subject is also an object")
        return self


class AnalysisConfig(Base):
    """What the analyzer reports as events.

    - `detections`: classes whose tracks are events (event name = class name):
      started when a track appears, ended when it is gone for `track_buffer`.
    - `relations`: see `RelationConfig` (event name = relation name).
    - `alerts`: event names whose start records a clip and notifies Telegram.
    Every event is published to MQTT.
    """

    enabled: bool = True
    tracker: TrackerConfig = TrackerConfig()
    motion: MotionConfig = MotionConfig()
    detections: tuple[str, ...] = ()
    relations: tuple[RelationConfig, ...] = ()
    alerts: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _event_names(self) -> "AnalysisConfig":
        names = [r.name for r in self.relations]
        if len(names) != len(set(names)):
            raise ValueError(f"relation names must be unique: {names}")
        if clash := set(names) & set(self.detections):
            raise ValueError(f"relation names clash with detections: {clash}")
        if unknown := set(self.alerts) - set(names) - set(self.detections):
            raise ValueError(f"alerts not in detections/relations: {unknown}")
        return self

    @property
    def needs_motion(self) -> bool:
        return any(r.moving for r in self.relations)


class AppConfig(Base):
    model: ModelConfig
    camera: IMX500Config = IMX500Config()
    analysis: AnalysisConfig = AnalysisConfig()
    stream: StreamConfig = StreamConfig()
    clips: ClipConfig = ClipConfig()
    annotator: AnnotatorConfig = AnnotatorConfig()
    dashboard: DashboardConfig = DashboardConfig()
    mqtt: MqttConfig = MqttConfig()
    telegram: TelegramConfig


class Secrets(BaseSettings):
    """Reads environment variables and values from the `.env` file"""

    model_config = SettingsConfigDict(extra="ignore")

    telegram_token: SecretStr | None = None
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

    clips = data.setdefault("clips", {})
    clips["dir"] = base_file.parent / clips.get("dir", ClipConfig().dir)

    # .env secrets (shared by all projects)
    secrets = Secrets(_env_file=base_file.parent / ".env")
    data.setdefault("telegram", {})["token"] = secrets.telegram_token
    data.setdefault("mqtt", {})["password"] = secrets.mqtt_password

    return AppConfig.model_validate(data)
