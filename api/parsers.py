from __future__ import annotations

import json
from typing import Any, Mapping

from config import Config
from errors import APIError

_OVERLAY_VALUES = {"none", "bbox", "mask", "both"}


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_overlay(value: str | None) -> str:
    if not value:
        return "none"
    parsed = value.strip().lower()
    if parsed not in _OVERLAY_VALUES:
        raise APIError("INVALID_OVERLAY", f"overlay must be one of: {', '.join(sorted(_OVERLAY_VALUES))}.", 400)
    return parsed


def validate_conf(value: str | None, config: Config) -> float:
    if value is None or value == "":
        return config.model_default_conf
    try:
        conf = float(value)
    except ValueError as exc:
        raise APIError("INVALID_CONF", "conf must be a number.", 400) from exc
    if conf < config.min_conf or conf > config.max_conf:
        raise APIError("INVALID_CONF", f"conf must be between {config.min_conf} and {config.max_conf}.", 400)
    return conf


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


def _parse_json_field(raw: str | None, field_name: str):
    if raw is None or raw == "":
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise APIError("INVALID_PROMPT_DATA", f"{field_name} must be valid JSON.", 400) from exc


def parse_prompt_inputs(form_data: Mapping[str, Any], require_input: bool = True) -> dict[str, Any]:
    prompt = (form_data.get("prompt") or "").strip()
    points = _parse_json_field(form_data.get("points"), "points")
    point_labels = _parse_json_field(form_data.get("point_labels"), "point_labels")
    boxes = _parse_json_field(form_data.get("boxes"), "boxes")

    if points is not None and (not isinstance(points, list) or any(not isinstance(p, list) or len(p) != 2 for p in points)):
        raise APIError("INVALID_PROMPT_DATA", "points must be [[x,y], ...].", 400)

    if boxes is not None:
        if isinstance(boxes, list) and len(boxes) == 4 and all(isinstance(v, (int, float)) for v in boxes):
            boxes = [boxes]
        if not isinstance(boxes, list) or any(not isinstance(b, list) or len(b) != 4 for b in boxes):
            raise APIError("INVALID_PROMPT_DATA", "boxes must be [x1,y1,x2,y2] or [[x1,y1,x2,y2], ...].", 400)

    if point_labels is not None and (not isinstance(point_labels, list) or any(not isinstance(v, int) for v in point_labels)):
        raise APIError("INVALID_PROMPT_DATA", "point_labels must be [0|1, ...].", 400)

    if points is not None:
        if point_labels is None:
            point_labels = [1] * len(points)
        if len(point_labels) != len(points):
            raise APIError("INVALID_PROMPT_DATA", "point_labels count must match points count.", 400)

    has_any_prompt = bool(prompt) or bool(points) or bool(boxes)
    if require_input and not has_any_prompt:
        raise APIError("MISSING_PROMPT", "Provide prompt, points, or boxes.", 400)

    return {
        "prompt": prompt or None,
        "points": points,
        "point_labels": point_labels,
        "boxes": boxes,
    }


def parse_output_formats(form_data: Mapping[str, Any]) -> set[str]:
    raw = (form_data.get("output_formats") or "").strip()
    if not raw:
        return {"mask_png"}
    parsed = _parse_json_field(raw, "output_formats")
    if isinstance(parsed, str):
        tokens = [t.strip() for t in parsed.split(",") if t.strip()]
    elif isinstance(parsed, list):
        tokens = [str(t).strip() for t in parsed if str(t).strip()]
    else:
        raise APIError("INVALID_OUTPUT_FORMAT", "output_formats must be JSON list or comma-separated string.", 400)

    allowed = {"mask_png", "rle", "polygon", "alpha_matte"}
    selected = set(tokens)
    unsupported = sorted(selected - allowed)
    if unsupported:
        raise APIError("INVALID_OUTPUT_FORMAT", f"Unsupported output formats: {unsupported}", 400)
    if not selected:
        selected = {"mask_png"}
    return selected
