import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


class ConfigError(Exception):
    """Raised when the model config file is missing or malformed."""


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ModelConfig:
    """Per-model settings, the single source of truth shared by model loading,
    routing, request defaults, ``/v1/models`` metadata, and metrics labels."""

    name: str  # identity used as the metric ``model`` label and routing key
    task: str  # "detect" | "segment"
    model_path: str
    default_conf: float
    half: bool
    device: str


_VALID_TASKS = {"detect", "segment"}
_DEFAULT_MODELS_PATH = Path(__file__).resolve().parent / "models.yaml"


def _models_config_path() -> Path:
    raw = os.getenv("MODELS_CONFIG_PATH")
    return Path(raw).expanduser() if raw else _DEFAULT_MODELS_PATH


def _coerce_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return _to_bool(str(value), default)


def _parse_model(name: str, raw: object) -> ModelConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"Model '{name}' must be a mapping of settings.")
    missing = [key for key in ("task", "model_path") if not raw.get(key)]
    if missing:
        raise ConfigError(f"Model '{name}' is missing required field(s): {', '.join(missing)}.")
    task = str(raw["task"]).strip().lower()
    if task not in _VALID_TASKS:
        raise ConfigError(f"Model '{name}' has invalid task '{task}'; must be one of {sorted(_VALID_TASKS)}.")
    conf = raw.get("default_conf", 0.25)
    return ModelConfig(
        name=name,
        task=task,
        model_path=str(raw["model_path"]),
        default_conf=float(conf if conf is not None else 0.25),
        half=_coerce_bool(raw.get("half"), True),
        device=str(raw.get("device") or "auto"),
    )


def _build_models() -> dict[str, ModelConfig]:
    path = _models_config_path()
    if not path.exists():
        raise ConfigError(
            f"Model config file not found: {path}. "
            "Create it or set MODELS_CONFIG_PATH to a valid YAML file."
        )
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse model config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Model config root must be a mapping: {path}.")
    models_raw = data.get("models")
    if not isinstance(models_raw, dict) or not models_raw:
        raise ConfigError(f"Model config must define a non-empty 'models' mapping: {path}.")
    return {name: _parse_model(name, spec) for name, spec in models_raw.items()}


@dataclass
class Config:
    detect_default_model: str = field(default_factory=lambda: os.getenv("DETECT_DEFAULT_MODEL", "yolo26n"))
    segment_default_model: str = field(default_factory=lambda: os.getenv("SEGMENT_DEFAULT_MODEL", "sam3"))
    models: dict[str, ModelConfig] = field(default_factory=_build_models)
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "20")))
    max_image_pixels: int = field(default_factory=lambda: int(os.getenv("MAX_IMAGE_PIXELS", "30000000")))
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", "change-me"))
    rate_limit_per_minute: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")))
    output_dir: str = field(default_factory=lambda: os.getenv("OUTPUT_DIR", "outputs"))
    output_url_prefix: str = field(default_factory=lambda: os.getenv("OUTPUT_URL_PREFIX", "/outputs"))
    output_retention_hours: int = field(default_factory=lambda: int(os.getenv("OUTPUT_RETENTION_HOURS", "24")))
    cleanup_interval_seconds: int = field(default_factory=lambda: int(os.getenv("CLEANUP_INTERVAL_SECONDS", "3600")))
    skip_model_load: bool = field(default_factory=lambda: _to_bool(os.getenv("SAM3_SKIP_MODEL_LOAD"), False))
    max_batch_images: int = field(default_factory=lambda: int(os.getenv("MAX_BATCH_IMAGES", "8")))
    job_worker_count: int = field(default_factory=lambda: int(os.getenv("JOB_WORKER_COUNT", "1")))
    job_retention_hours: int = field(default_factory=lambda: int(os.getenv("JOB_RETENTION_HOURS", "24")))
    job_db_path: str = field(default_factory=lambda: os.getenv("JOB_DB_PATH", "data/jobs.sqlite3"))
    enable_auto_queue: bool = field(default_factory=lambda: _to_bool(os.getenv("ENABLE_AUTO_QUEUE"), True))
    auto_queue_max_size: int = field(default_factory=lambda: int(os.getenv("AUTO_QUEUE_MAX_SIZE", "200")))
    cache_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL_SECONDS", "3600")))
    webhook_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("WEBHOOK_TIMEOUT_SECONDS", "5")))
    webhook_max_retries: int = field(default_factory=lambda: int(os.getenv("WEBHOOK_MAX_RETRIES", "3")))
    webhook_retry_base_seconds: float = field(default_factory=lambda: float(os.getenv("WEBHOOK_RETRY_BASE_SECONDS", "1.0")))
    allowed_extensions: set[str] = field(default_factory=lambda: {"jpg", "jpeg", "png", "webp"})
    min_conf: float = 0.0
    max_conf: float = 1.0

    def __post_init__(self):
        self._validate_default_model("DETECT_DEFAULT_MODEL", self.detect_default_model, "detect")
        self._validate_default_model("SEGMENT_DEFAULT_MODEL", self.segment_default_model, "segment")

    def _validate_default_model(self, env_name: str, model_name: str, task: str) -> None:
        if model_name not in self.models:
            raise ConfigError(
                f"{env_name} '{model_name}' is not in the model config "
                f"(available: {sorted(self.models)})."
            )
        model_task = self.models[model_name].task
        if model_task != task:
            raise ConfigError(
                f"{env_name} '{model_name}' points to a '{model_task}' model; expected task '{task}'."
            )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024
