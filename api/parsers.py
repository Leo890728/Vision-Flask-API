from __future__ import annotations

import json
from typing import Any, Mapping

from config import Config
from errors import APIError

_OVERLAY_VALUES = {"none", "bbox", "mask", "both"}
_DETECT_OVERLAY_VALUES = {"none", "bbox"}
_TASK_VALUES = {"segment", "detect"}


def _invalid_classes_error(message: str, raw: Any = None) -> APIError:
    details = {
        "received": repr(raw),
        "accepted_formats": [
            "[0,\"person\"]",
            "[\"person\"]",
            "[0,person]",
            "person",
            "0,2",
        ],
    }
    return APIError("INVALID_CLASSES", message, 400, details)


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


def parse_detect_overlay(value: str | None) -> str:
    parsed = parse_overlay(value)
    if parsed not in _DETECT_OVERLAY_VALUES:
        raise APIError(
            "INVALID_OVERLAY",
            f"overlay must be one of: {', '.join(sorted(_DETECT_OVERLAY_VALUES))} for detect task.",
            400,
        )
    return parsed


def parse_task(value: str | None) -> str:
    if not value:
        return "segment"
    parsed = value.strip().lower()
    if parsed not in _TASK_VALUES:
        raise APIError("INVALID_TASK", f"task must be one of: {', '.join(sorted(_TASK_VALUES))}.", 400)
    return parsed


def validate_conf(value: str | None, config: Config, default_conf: float | None = None) -> float:
    if default_conf is None:
        default_conf = config.model_default_conf
    if value is None or value == "":
        return default_conf
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


def parse_classes(form_data: Mapping[str, Any]) -> list[int | str] | None:
    if hasattr(form_data, "getlist"):
        try:
            raw_list = form_data.getlist("classes")
        except Exception:
            raw_list = []
        if raw_list:
            if len(raw_list) > 1:
                return _validate_classes_list(raw_list)
            raw = raw_list[0]
        else:
            raw = form_data.get("classes")
    else:
        raw = form_data.get("classes")

    if raw is None or raw == "":
        return None
    parsed: Any = None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
    else:
        parsed = raw

    if isinstance(parsed, list):
        return _validate_classes_list(parsed)
    if isinstance(parsed, dict):
        raise _invalid_classes_error("classes must be a list, not an object.", raw=raw)
    if isinstance(parsed, str):
        nested = _try_parse_nested_classes_json(parsed)
        if isinstance(nested, list):
            return _validate_classes_list(nested)

    if isinstance(raw, str):
        direct = _parse_classes_direct(raw)
        if direct is not None:
            return _validate_classes_list(direct)

    if isinstance(raw, str):
        nested_raw = _try_parse_nested_classes_json(raw)
        if isinstance(nested_raw, list):
            return _validate_classes_list(nested_raw)

    if isinstance(parsed, list):
        return _validate_classes_list(parsed)

    # Be tolerant for multipart form inputs where quotes are often stripped,
    # e.g. [0,person] sent from shell/UI tooling.
    if isinstance(raw, str):
        parsed = _parse_classes_loose(raw)
        if parsed is not None:
            return _validate_classes_list(parsed)

    raise _invalid_classes_error("classes format is invalid.", raw=raw)


def _validate_classes_list(parsed: Any) -> list[int | str]:
    if not isinstance(parsed, list):
        raise _invalid_classes_error("classes must be a list.", raw=parsed)

    values: list[int | str] = []
    for item in parsed:
        if isinstance(item, bool):
            raise _invalid_classes_error("classes values must be int or string.", raw=parsed)
        if isinstance(item, int):
            values.append(item)
        elif isinstance(item, str) and item.strip():
            values.append(item.strip())
        else:
            raise _invalid_classes_error("classes values must be int or non-empty string.", raw=parsed)
    return values


def _parse_classes_loose(raw: str) -> list[int | str] | None:
    text = raw.strip()
    text = _strip_outer_quotes(text)
    if not (text.startswith("[") and text.endswith("]")):
        return None

    inner = text[1:-1].strip()
    if inner == "":
        return []

    # normalize common smart quotes copied from docs/chat tools
    inner = inner.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
    tokens = [token.strip() for token in inner.split(",")]
    out: list[int | str] = []
    for token in tokens:
        if not token:
            raise _invalid_classes_error("classes values must be int or non-empty string.", raw=raw)
        if (token.startswith("\"") and token.endswith("\"")) or (token.startswith("'") and token.endswith("'")):
            value = token[1:-1].strip()
            if not value:
                raise _invalid_classes_error("classes values must be int or non-empty string.", raw=raw)
            out.append(value)
            continue
        try:
            out.append(int(token))
            continue
        except ValueError:
            # accept unquoted class names from form clients, e.g. [0,person]
            out.append(token)
    return out


def _parse_classes_direct(raw: str) -> list[int | str] | None:
    text = _strip_outer_quotes(raw.strip())
    if text == "":
        return None
    if text.startswith("[") and text.endswith("]"):
        return None

    # Accept plain single class token or CSV list, e.g. person or person,car or 0,2
    tokens = [token.strip() for token in text.split(",") if token.strip()]
    if not tokens:
        return None
    out: list[int | str] = []
    for token in tokens:
        try:
            out.append(int(token))
        except ValueError:
            out.append(token)
    return out


def _try_parse_nested_classes_json(value: str) -> list[int | str] | None:
    text = _strip_outer_quotes(value.strip())
    if not (text.startswith("[") and text.endswith("]")):
        return None
    text = text.replace("\\\"", "\"")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, list):
        return parsed
    return None


def _strip_outer_quotes(text: str) -> str:
    if len(text) >= 2 and (
        (text.startswith("\"") and text.endswith("\""))
        or (text.startswith("'") and text.endswith("'"))
    ):
        return text[1:-1].strip()
    return text
