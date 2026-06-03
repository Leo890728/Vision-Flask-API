from __future__ import annotations

from config import Config
from errors import APIError

_TASK_VALUES = {"segment", "detect"}


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_task(value: str | None) -> str:
    if not value:
        return "segment"
    parsed = value.strip().lower()
    if parsed not in _TASK_VALUES:
        raise APIError("INVALID_TASK", f"task must be one of: {', '.join(sorted(_TASK_VALUES))}.", 400)
    return parsed


def validate_upload_filename(file_name: str, config: Config) -> None:
    if not file_name or "." not in file_name:
        raise APIError("INVALID_IMAGE", "image must include a valid file name.", 400)
    ext = file_name.rsplit(".", 1)[1].lower()
    if ext not in config.allowed_extensions:
        raise APIError(
            "INVALID_IMAGE",
            f"Unsupported image extension: .{ext}. Allowed: {sorted(config.allowed_extensions)}",
            400,
        )
