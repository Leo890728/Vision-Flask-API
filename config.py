import os
from dataclasses import dataclass, field


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    model_path: str = field(default_factory=lambda: os.getenv("SAM3_MODEL_PATH", "models/sam3.1_multiplex.pt"))
    model_half: bool = field(default_factory=lambda: _to_bool(os.getenv("SAM3_HALF"), True))
    model_default_conf: float = field(default_factory=lambda: float(os.getenv("SAM3_DEFAULT_CONF", "0.25")))
    model_device: str = field(default_factory=lambda: os.getenv("SAM3_DEVICE", "auto"))
    yolo_model_path: str = field(default_factory=lambda: os.getenv("YOLO_MODEL_PATH", "models/yolo11n.pt"))
    yolo_half: bool = field(default_factory=lambda: _to_bool(os.getenv("YOLO_HALF"), True))
    yolo_default_conf: float = field(default_factory=lambda: float(os.getenv("YOLO_DEFAULT_CONF", "0.25")))
    yolo_device: str = field(default_factory=lambda: os.getenv("YOLO_DEVICE", "auto"))
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

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024
