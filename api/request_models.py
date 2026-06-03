from __future__ import annotations

import json
from typing import Any, Literal, Mapping

from pydantic import BaseModel, Field

from config import Config
from errors import APIError


SegmentOverlay = Literal["none", "bbox", "mask", "both"]
DetectOverlay = Literal["none", "bbox"]
OutputFormat = Literal["mask_png", "rle", "polygon", "alpha_matte"]


class SegmentParams(BaseModel):
    segment_model: str = "sam3"
    prompt: str | None = None
    classes: list[int | str] | None = None
    points: list[list[float]] | None = None
    point_labels: list[int] | None = None
    boxes: list[list[float]] | None = None
    conf: float
    overlay: SegmentOverlay = "none"
    output_formats: set[OutputFormat] = Field(default_factory=lambda: {"mask_png"})

    @classmethod
    def from_form(cls, form_data: Mapping[str, Any], config: Config, require_input: bool = True) -> "SegmentParams":
        segment_model = _parse_segment_model(form_data.get("segment_model"), config)
        prompt = _clean_optional_str(form_data.get("prompt"))
        classes = _parse_classes(form_data)
        points = _parse_json_field(form_data.get("points"), "points", "INVALID_PROMPT_DATA")
        point_labels = _parse_json_field(form_data.get("point_labels"), "point_labels", "INVALID_PROMPT_DATA")
        boxes = _parse_json_field(form_data.get("boxes"), "boxes", "INVALID_PROMPT_DATA")

        if points is not None and (
            not isinstance(points, list)
            or any(not isinstance(point, list) or len(point) != 2 for point in points)
        ):
            raise APIError("INVALID_PROMPT_DATA", "points must be [[x,y], ...].", 400)

        if boxes is not None:
            if isinstance(boxes, list) and len(boxes) == 4 and all(isinstance(value, (int, float)) for value in boxes):
                boxes = [boxes]
            if not isinstance(boxes, list) or any(not isinstance(box, list) or len(box) != 4 for box in boxes):
                raise APIError("INVALID_PROMPT_DATA", "boxes must be [x1,y1,x2,y2] or [[x1,y1,x2,y2], ...].", 400)

        if point_labels is not None and (
            not isinstance(point_labels, list)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in point_labels)
        ):
            raise APIError("INVALID_PROMPT_DATA", "point_labels must be [0|1, ...].", 400)

        if points is not None:
            if point_labels is None:
                point_labels = [1] * len(points)
            if len(point_labels) != len(points):
                raise APIError("INVALID_PROMPT_DATA", "point_labels count must match points count.", 400)

        modes = set(config.models[segment_model].input_modes)
        visual_modes = modes & {"prompt", "points", "boxes"}

        if points and "points" not in modes:
            raise APIError(
                "UNSUPPORTED_SEGMENT_PROMPT",
                f"{segment_model} does not support point prompts.",
                400,
            )
        if boxes and "boxes" not in modes:
            raise APIError(
                "UNSUPPORTED_SEGMENT_PROMPT",
                f"{segment_model} does not support box prompts.",
                400,
            )
        if prompt and "prompt" not in modes and "classes" not in modes:
            raise APIError(
                "UNSUPPORTED_SEGMENT_PROMPT",
                f"{segment_model} does not support text prompts.",
                400,
            )
        if classes is not None and "classes" not in modes:
            raise APIError(
                "UNSUPPORTED_SEGMENT_PROMPT",
                f"{segment_model} does not support class filters.",
                400,
            )
        if visual_modes and require_input and not (
            (prompt and "prompt" in modes)
            or (points and "points" in modes)
            or (boxes and "boxes" in modes)
        ):
            raise APIError("MISSING_PROMPT", "Provide prompt, points, or boxes.", 400)
        if "classes" in modes and classes is None and prompt and "prompt" not in modes:
            classes = [prompt]

        return cls(
            segment_model=segment_model,
            prompt=prompt,
            classes=classes,
            points=points,
            point_labels=point_labels,
            boxes=boxes,
            conf=_parse_conf(form_data.get("conf"), config, _default_segment_conf(config, segment_model)),
            overlay=_parse_segment_overlay(form_data.get("overlay")),
            output_formats=_parse_output_formats(form_data),
        )

    def to_prompt_inputs(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "classes": self.classes,
            "points": self.points,
            "point_labels": self.point_labels,
            "boxes": self.boxes,
        }


class DetectParams(BaseModel):
    detect_model: str
    classes: list[int | str] | None = None
    conf: float
    overlay: DetectOverlay = "none"

    @classmethod
    def from_form(cls, form_data: Mapping[str, Any], config: Config) -> "DetectParams":
        detect_model = _parse_detect_model(form_data.get("detect_model"), config)
        return cls(
            detect_model=detect_model,
            classes=_parse_classes(form_data),
            conf=_parse_conf(form_data.get("conf"), config, config.models[detect_model].default_conf),
            overlay=_parse_detect_overlay(form_data.get("overlay")),
        )


def _clean_optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _parse_segment_model(raw: Any, config: Config) -> str:
    value = str(raw or config.segment_default_model or "sam3").strip()
    if value in config.models and config.models[value].task == "segment":
        return value

    aliases = {
        "sam": "sam3",
        "yoloseg": "yolo_seg",
        "yolo-seg": "yolo_seg",
        "yolo_segment": "yolo_seg",
        "yolo_segmentation": "yolo_seg",
    }
    alias = aliases.get(value.lower())
    if alias in config.models and config.models[alias].task == "segment":
        return alias

    segment_models = sorted(name for name, model in config.models.items() if model.task == "segment")
    raise APIError(
        "INVALID_SEGMENT_MODEL",
        f"segment_model must be one of: {', '.join(segment_models)}.",
        400,
        {"available": segment_models},
    )


def _parse_detect_model(raw: Any, config: Config) -> str:
    value = str(raw or config.detect_default_model).strip()
    if value in config.models and config.models[value].task == "detect":
        return value
    detect_models = sorted(name for name, model in config.models.items() if model.task == "detect")
    raise APIError(
        "INVALID_DETECT_MODEL",
        f"detect_model must be one of: {', '.join(detect_models)}.",
        400,
        {"available": detect_models},
    )


def _default_segment_conf(config: Config, segment_model: str) -> float:
    return config.models[segment_model].default_conf


def _parse_json_field(raw: Any, field_name: str, error_code: str) -> Any:
    if raw is None or raw == "":
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise APIError(error_code, f"{field_name} must be valid JSON.", 400) from exc


def _parse_conf(raw: Any, config: Config, default_conf: float) -> float:
    if raw is None or raw == "":
        return default_conf
    try:
        conf = float(raw)
    except ValueError as exc:
        raise APIError("INVALID_CONF", "conf must be a number.", 400) from exc
    if conf < config.min_conf or conf > config.max_conf:
        raise APIError("INVALID_CONF", f"conf must be between {config.min_conf} and {config.max_conf}.", 400)
    return conf


def _parse_segment_overlay(raw: Any) -> SegmentOverlay:
    if not raw:
        return "none"
    parsed = str(raw).strip().lower()
    allowed = {"none", "bbox", "mask", "both"}
    if parsed not in allowed:
        raise APIError("INVALID_OVERLAY", f"overlay must be one of: {', '.join(sorted(allowed))}.", 400)
    return parsed  # type: ignore[return-value]


def _parse_detect_overlay(raw: Any) -> DetectOverlay:
    parsed = _parse_segment_overlay(raw)
    allowed = {"none", "bbox"}
    if parsed not in allowed:
        raise APIError(
            "INVALID_OVERLAY",
            f"overlay must be one of: {', '.join(sorted(allowed))} for detect task.",
            400,
        )
    return parsed  # type: ignore[return-value]


def _parse_output_formats(form_data: Mapping[str, Any]) -> set[OutputFormat]:
    raw = str(form_data.get("output_formats") or "").strip()
    if not raw:
        return {"mask_png"}
    parsed = _parse_json_field(raw, "output_formats", "INVALID_OUTPUT_FORMAT")
    if isinstance(parsed, str):
        tokens = [token.strip() for token in parsed.split(",") if token.strip()]
    elif isinstance(parsed, list):
        tokens = [str(token).strip() for token in parsed if str(token).strip()]
    else:
        raise APIError("INVALID_OUTPUT_FORMAT", "output_formats must be JSON list or comma-separated string.", 400)

    allowed = {"mask_png", "rle", "polygon", "alpha_matte"}
    selected = set(tokens)
    unsupported = sorted(selected - allowed)
    if unsupported:
        raise APIError("INVALID_OUTPUT_FORMAT", f"Unsupported output formats: {unsupported}", 400)
    return selected or {"mask_png"}  # type: ignore[return-value]


def _parse_classes(form_data: Mapping[str, Any]) -> list[int | str] | None:
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

    if isinstance(raw, str):
        loose = _parse_classes_loose(raw)
        if loose is not None:
            return _validate_classes_list(loose)

    raise _invalid_classes_error("classes format is invalid.", raw=raw)


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
        except ValueError:
            out.append(token)
    return out


def _parse_classes_direct(raw: str) -> list[int | str] | None:
    text = _strip_outer_quotes(raw.strip())
    if text == "":
        return None
    if text.startswith("[") and text.endswith("]"):
        return None

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
