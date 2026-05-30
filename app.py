from __future__ import annotations

import logging
import os
import time
import uuid
import json
import hashlib
import base64
import hmac
import io
import zipfile
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, g, has_request_context, request, send_file, send_from_directory
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

from config import Config
from errors import APIError
from middlewares.auth import require_api_key
from middlewares.rate_limit import InMemoryRateLimiter, apply_rate_limit
from services.cache_service import CacheService
from services.job_service import JobService
from services.metrics_service import MetricsService
from services.sam3_service import SAM3Service
from services.storage_service import StorageService
from services.webhook_retry_service import WebhookRetryService

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency fallback
    cv2 = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency fallback
    torch = None


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _error_response(code: str, message: str, status_code: int, details: dict | None = None):
    payload = {
        "code": code,
        "message": message,
        "details": details or {},
        "request_id": getattr(g, "request_id", None),
    }
    return jsonify(payload), status_code


def _validate_conf(value: str | None, config: Config) -> float:
    if value is None or value == "":
        return config.model_default_conf
    try:
        conf = float(value)
    except ValueError as exc:
        raise APIError("INVALID_CONF", "conf must be a number.", 400) from exc
    if conf < config.min_conf or conf > config.max_conf:
        raise APIError("INVALID_CONF", f"conf must be between {config.min_conf} and {config.max_conf}.", 400)
    return conf


def _validate_upload(file_name: str, config: Config) -> None:
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


def _parse_prompt_inputs(form_data: dict, require_input: bool = True) -> dict[str, Any]:
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


def _parse_output_formats(form_data: dict) -> set[str]:
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


def _mask_to_rle(mask: Any) -> dict[str, Any]:
    # Uncompressed COCO-style RLE using column-major flattening.
    flat = mask.astype("uint8").T.flatten()
    counts: list[int] = []
    count = 0
    prev = 0
    for px in flat:
        val = int(px > 0)
        if val == prev:
            count += 1
        else:
            counts.append(count)
            count = 1
            prev = val
    counts.append(count)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def _mask_to_polygon(mask: Any) -> list[list[float]]:
    if cv2 is not None:
        contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for cnt in contours:
            if len(cnt) < 3:
                continue
            poly = cnt.squeeze(axis=1).astype(float).flatten().tolist()
            if len(poly) >= 6:
                polygons.append(poly)
        if polygons:
            return polygons

    ys, xs = (mask > 0).nonzero()
    if len(xs) == 0:
        return []
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [[float(x1), float(y1), float(x2), float(y1), float(x2), float(y2), float(x1), float(y2)]]


def _cache_key(image_sha256: str, prompt_inputs: dict[str, Any], conf: float, return_overlay: bool, output_formats: set[str]) -> str:
    key_obj = {
        "image_sha256": image_sha256,
        "prompt_inputs": prompt_inputs,
        "conf": conf,
        "return_overlay": return_overlay,
        "output_formats": sorted(output_formats),
    }
    raw = json.dumps(key_obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _build_webhook_signature(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def _render_segmentation_response(
    request_id: str,
    seg_result: dict[str, Any],
    storage_service: StorageService,
    return_overlay: bool,
    output_formats: set[str],
) -> tuple[list[dict[str, Any]], str | None]:
    masks = seg_result["masks"]
    boxes = seg_result["boxes"]
    scores = seg_result["scores"]

    detections = []
    for idx in range(int(masks.shape[0])):
        bbox = boxes[idx] if idx < len(boxes) else [0.0, 0.0, 0.0, 0.0]
        score = float(scores[idx]) if idx < len(scores) else 1.0
        item = {
            "id": idx,
            "score": score,
            "bbox": [float(v) for v in bbox],
        }
        if "mask_png" in output_formats:
            _rel_path, mask_url = storage_service.save_mask(request_id, idx, masks[idx])
            item["mask_url"] = mask_url
        if "alpha_matte" in output_formats:
            _rel_path, alpha_url = storage_service.save_alpha_matte(request_id, idx, masks[idx])
            item["alpha_url"] = alpha_url
        if "rle" in output_formats:
            item["rle"] = _mask_to_rle(masks[idx])
        if "polygon" in output_formats:
            item["polygon"] = _mask_to_polygon(masks[idx])
        detections.append(item)

    overlay_url = None
    if return_overlay and seg_result.get("overlay_bgr") is not None:
        _rel_path, overlay_url = storage_service.save_overlay(request_id, seg_result["overlay_bgr"])

    return detections, overlay_url


def _openapi_spec(config: Config) -> dict:
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "SAM3 Flask API",
            "version": "1.0.0",
            "description": "SAM3 semantic segmentation API with text prompt input.",
        },
        "servers": [{"url": "/"}],
        "components": {
            "securitySchemes": {
                "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
            },
            "schemas": {
                "ErrorResponse": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "message": {"type": "string"},
                        "details": {"type": "object"},
                        "request_id": {"type": "string", "nullable": True},
                    },
                    "required": ["code", "message", "details", "request_id"],
                }
            },
        },
        "paths": {
            "/healthz": {
                "get": {
                    "summary": "Liveness check",
                    "responses": {"200": {"description": "Service is alive"}},
                }
            },
            "/readyz": {
                "get": {
                    "summary": "Readiness check",
                    "responses": {
                        "200": {"description": "Model is ready"},
                        "503": {"description": "Model is not ready"},
                    },
                }
            },
            "/v1/models": {
                "get": {
                    "summary": "Get model metadata",
                    "security": [{"ApiKeyAuth": []}],
                    "responses": {
                        "200": {"description": "Model metadata"},
                        "401": {"description": "Unauthorized"},
                    },
                }
            },
            "/metrics": {
                "get": {
                    "summary": "Prometheus-style metrics",
                    "security": [{"ApiKeyAuth": []}],
                    "responses": {"200": {"description": "Metrics text"}},
                }
            },
            "/v1/segment": {
                "post": {
                    "summary": "Segment single image by text or visual prompts",
                    "security": [{"ApiKeyAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "required": ["image"],
                                    "properties": {
                                        "image": {"type": "string", "format": "binary"},
                                        "prompt": {"type": "string"},
                                        "points": {"type": "string", "description": "JSON: [[x,y], ...]"},
                                        "point_labels": {"type": "string", "description": "JSON: [1,0,...]"},
                                        "boxes": {"type": "string", "description": "JSON: [x1,y1,x2,y2] or [[...],...]"},
                                        "output_formats": {
                                            "type": "string",
                                            "description": "JSON list or csv. Options: mask_png,rle,polygon,alpha_matte",
                                        },
                                        "conf": {"type": "number", "default": config.model_default_conf},
                                        "return_overlay": {"type": "boolean", "default": False},
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Segmentation result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "request_id": {"type": "string"},
                                            "prompt": {"type": "string"},
                                            "image_meta": {
                                                "type": "object",
                                                "properties": {
                                                    "width": {"type": "integer"},
                                                    "height": {"type": "integer"},
                                                },
                                            },
                                            "detections": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "id": {"type": "integer"},
                                                        "score": {"type": "number"},
                                                        "bbox": {
                                                            "type": "array",
                                                            "items": {"type": "number"},
                                                            "minItems": 4,
                                                            "maxItems": 4,
                                                        },
                                                        "mask_url": {"type": "string"},
                                                    },
                                                },
                                            },
                                            "overlay_url": {"type": "string", "nullable": True},
                                            "timing_ms": {
                                                "type": "object",
                                                "properties": {
                                                    "decode": {"type": "number"},
                                                    "infer": {"type": "number"},
                                                    "postprocess": {"type": "number"},
                                                    "total": {"type": "number"},
                                                },
                                            },
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"description": "Bad request"},
                        "401": {"description": "Unauthorized"},
                        "202": {"description": "Auto queued as async job"},
                        "429": {"description": "Rate limit exceeded"},
                        "503": {"description": "Queue full or model not ready"},
                        "500": {"description": "Inference failed"},
                    },
                }
            },
            "/v1/segment/batch": {
                "post": {
                    "summary": "Batch segment multiple images in one request",
                    "security": [{"ApiKeyAuth": []}],
                    "responses": {"200": {"description": "Batch segmentation result"}},
                }
            },
            "/v1/jobs": {
                "post": {
                    "summary": "Submit async segmentation job",
                    "security": [{"ApiKeyAuth": []}],
                    "responses": {"202": {"description": "Job accepted"}},
                }
            },
            "/v1/jobs/{job_id}": {
                "get": {
                    "summary": "Get async job status/result",
                    "security": [{"ApiKeyAuth": []}],
                    "parameters": [
                        {
                            "name": "job_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {"200": {"description": "Job status"}},
                },
                "delete": {
                    "summary": "Cancel queued/running async job",
                    "security": [{"ApiKeyAuth": []}],
                    "parameters": [
                        {
                            "name": "job_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {"200": {"description": "Job canceled"}},
                }
            },
            "/v1/jobs/{job_id}/retry": {
                "post": {
                    "summary": "Retry a failed/canceled async job",
                    "security": [{"ApiKeyAuth": []}],
                    "parameters": [
                        {
                            "name": "job_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "202": {"description": "Retry job accepted"},
                        "404": {"description": "Job not found"},
                        "409": {"description": "Job state does not allow retry"},
                    },
                }
            },
            "/v1/jobs/{job_id}/export": {
                "get": {
                    "summary": "Export job outputs and result JSON as zip",
                    "security": [{"ApiKeyAuth": []}],
                    "parameters": [
                        {
                            "name": "job_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "200": {"description": "Zip file download"},
                        "404": {"description": "Job/output not found"},
                        "409": {"description": "Job not completed yet"},
                    },
                }
            },
            f"{config.output_url_prefix}/{{filename}}": {
                "get": {
                    "summary": "Get generated output file",
                    "parameters": [
                        {
                            "name": "filename",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {"200": {"description": "File content"}},
                }
            },
        },
    }


def create_app(sam3_service: SAM3Service | None = None, storage_service: StorageService | None = None) -> Flask:
    config = Config()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.max_upload_bytes

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(request_id)s] %(message)s",
    )

    sam3_service = sam3_service or SAM3Service(config)
    storage_service = storage_service or StorageService(config)
    rate_limiter = InMemoryRateLimiter(config.rate_limit_per_minute)
    job_service = JobService(
        db_path=config.job_db_path,
        worker_count=config.job_worker_count,
        retention_hours=config.job_retention_hours,
    )
    cache_service = CacheService(ttl_seconds=config.cache_ttl_seconds)
    metrics_service = MetricsService()
    webhook_retry_service = WebhookRetryService(
        max_retries=config.webhook_max_retries,
        base_delay_seconds=config.webhook_retry_base_seconds,
    )
    app.extensions["sam3_services"] = {
        "sam3_service": sam3_service,
        "storage_service": storage_service,
        "job_service": job_service,
        "cache_service": cache_service,
        "metrics_service": metrics_service,
        "webhook_retry_service": webhook_retry_service,
    }

    class RequestIDFilter(logging.Filter):
        def filter(self, record):
            record.request_id = getattr(g, "request_id", "-") if has_request_context() else "-"
            return True

    for handler in logging.getLogger().handlers:
        handler.addFilter(RequestIDFilter())

    @app.before_request
    def setup_request_context():
        g.request_id = str(uuid.uuid4())
        g.request_start = time.perf_counter()
        g.prompt = None
        metrics_service.inc_request()

    @app.after_request
    def log_request(response):
        elapsed_ms = (time.perf_counter() - g.request_start) * 1000
        logging.info(
            "method=%s path=%s status=%s latency_ms=%.1f prompt=%s",
            request.method,
            request.path,
            response.status_code,
            elapsed_ms,
            getattr(g, "prompt", None),
        )
        if response.status_code >= 400:
            metrics_service.inc_error()
        response.headers["X-Request-ID"] = g.request_id
        return response

    @app.errorhandler(APIError)
    def handle_api_error(err: APIError):
        return _error_response(err.code, err.message, err.status_code, err.details)

    @app.errorhandler(HTTPException)
    def handle_http_exception(err: HTTPException):
        return _error_response("HTTP_ERROR", err.description, err.code or 500)

    @app.errorhandler(Exception)
    def handle_unexpected_error(err: Exception):
        logging.exception("Unhandled error: %s", err)
        return _error_response("INTERNAL_ERROR", "Internal server error.", 500)

    @app.errorhandler(413)
    def handle_file_too_large(_err):
        return _error_response("FILE_TOO_LARGE", "Uploaded file exceeds size limit.", 413)

    def _save_and_validate_upload(upload, request_id: str):
        _validate_upload(upload.filename or "", config)
        decode_start = time.perf_counter()
        source_path = storage_service.save_uploaded_image(request_id, upload)
        try:
            with Image.open(source_path) as img:
                width, height = img.size
        except UnidentifiedImageError as exc:
            raise APIError("INVALID_IMAGE", "Uploaded file is not a valid image.", 400) from exc

        if width * height > config.max_image_pixels:
            raise APIError(
                "IMAGE_TOO_LARGE",
                "Image pixel count exceeds limit.",
                400,
                {"max_pixels": config.max_image_pixels, "actual_pixels": width * height},
            )

        decode_ms = (time.perf_counter() - decode_start) * 1000
        return source_path, width, height, decode_ms

    def _segment_from_saved(
        source_path: str,
        request_id: str,
        prompt_inputs: dict[str, Any],
        conf: float,
        return_overlay: bool,
        output_formats: set[str],
        decode_ms: float = 0.0,
    ) -> dict[str, Any]:
        infer_start = time.perf_counter()
        seg_result = sam3_service.segment(
            source_path,
            prompt=prompt_inputs["prompt"],
            conf=conf,
            points=prompt_inputs["points"],
            point_labels=prompt_inputs["point_labels"],
            boxes=prompt_inputs["boxes"],
        )
        infer_ms = (time.perf_counter() - infer_start) * 1000
        metrics_service.observe_inference_latency(infer_ms)

        post_start = time.perf_counter()
        detections, overlay_url = _render_segmentation_response(
            request_id,
            seg_result,
            storage_service,
            return_overlay,
            output_formats,
        )
        post_ms = (time.perf_counter() - post_start) * 1000

        total_ms = decode_ms + infer_ms + post_ms
        return {
            "request_id": request_id,
            "prompt": prompt_inputs["prompt"],
            "cached": False,
            "output_formats": sorted(output_formats),
            "prompt_inputs": {
                "points": prompt_inputs["points"],
                "boxes": prompt_inputs["boxes"],
            },
            "image_meta": {
                "width": seg_result["width"],
                "height": seg_result["height"],
            },
            "detections": detections,
            "overlay_url": overlay_url,
            "timing_ms": {
                "decode": round(decode_ms, 2),
                "infer": round(infer_ms, 2),
                "postprocess": round(post_ms, 2),
                "total": round(total_ms, 2),
            },
        }

    def _job_handler(payload: dict[str, Any]) -> dict[str, Any]:
        return _segment_from_saved(
            source_path=payload["source_path"],
            request_id=payload["request_id"],
            prompt_inputs=payload["prompt_inputs"],
            conf=payload["conf"],
            return_overlay=payload["return_overlay"],
            output_formats=set(payload.get("output_formats", ["mask_png"])),
            decode_ms=payload.get("decode_ms", 0.0),
        )

    def _post_webhook(url: str, body: dict[str, Any], secret: str | None = None) -> None:
        body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url=url, data=body_bytes, method="POST")
        req.add_header("Content-Type", "application/json")
        if secret:
            req.add_header("X-Webhook-Signature", _build_webhook_signature(secret, body_bytes))
        with urllib.request.urlopen(req, timeout=config.webhook_timeout_seconds):
            return

    def _job_completion_hook(record) -> None:
        if not record.webhook_url:
            return
        payload = {
            "job_id": record.job_id,
            "status": record.status,
            "result": record.result,
            "error": record.error,
            "updated_at": record.updated_at,
        }
        secret = None
        if record.payload:
            secret = record.payload.get("webhook_secret")
        webhook_retry_service.submit(record.job_id, record.webhook_url, payload, secret=secret)

    def _render_metrics_text() -> str:
        snap = metrics_service.snapshot()
        job_stats = job_service.stats()
        webhook_stats = webhook_retry_service.stats()
        cache_size = cache_service.size()
        gpu_mem_used = 0
        gpu_mem_total = 0
        if torch is not None and torch.cuda.is_available():
            try:
                gpu_mem_used = int(torch.cuda.memory_allocated())
                gpu_mem_total = int(torch.cuda.get_device_properties(0).total_memory)
            except Exception:
                gpu_mem_used = 0
                gpu_mem_total = 0
        lines = [
            f"sam3_requests_total {snap['requests_total']}",
            f"sam3_errors_total {snap['errors_total']}",
            f"sam3_error_rate {snap['error_rate']}",
            f"sam3_qps {snap['qps']}",
            f"sam3_inference_count {snap['inference_count']}",
            f"sam3_inference_latency_sum_ms {snap['inference_latency_sum_ms']}",
            f"sam3_inference_latency_avg_ms {snap['inference_latency_avg_ms']}",
            f"sam3_auto_queued_total {snap['auto_queued_total']}",
            f"sam3_jobs_created_total {snap['jobs_created_total']}",
            f"sam3_job_queue_size {job_stats['queue_size']}",
            f"sam3_job_queued_jobs {job_stats['queued_jobs']}",
            f"sam3_job_running_jobs {job_stats['running_jobs']}",
            f"sam3_job_canceling_jobs {job_stats['canceling_jobs']}",
            f"sam3_webhook_pending {webhook_stats['pending']}",
            f"sam3_webhook_delivered_total {webhook_stats['delivered_total']}",
            f"sam3_webhook_retried_total {webhook_stats['retried_total']}",
            f"sam3_webhook_failed_total {webhook_stats['failed_total']}",
            f"sam3_cache_hits_total {cache_service.hits}",
            f"sam3_cache_misses_total {cache_service.misses}",
            f"sam3_cache_entries {cache_size}",
            f"sam3_gpu_memory_bytes {gpu_mem_used}",
            f"sam3_gpu_total_memory_bytes {gpu_mem_total}",
        ]
        return "\n".join(lines) + "\n"

    @app.get("/openapi.json")
    def openapi_json():
        return jsonify(_openapi_spec(config)), 200

    @app.get("/metrics")
    @require_api_key(config.api_key)
    def metrics():
        return _render_metrics_text(), 200, {"Content-Type": "text/plain; version=0.0.4; charset=utf-8"}

    @app.get("/docs")
    def swagger_docs():
        html = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>SAM3 API Docs</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
  </head>
  <body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script>
      window.ui = SwaggerUIBundle({
        url: "/openapi.json",
        dom_id: "#swagger-ui",
        deepLinking: true
      });
    </script>
  </body>
</html>"""
        return html, 200, {"Content-Type": "text/html; charset=utf-8"}

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok"}), 200

    @app.get("/readyz")
    def readyz():
        if sam3_service.is_ready:
            return jsonify({"status": "ready"}), 200
        return jsonify({"status": "not_ready", "reason": sam3_service.last_error}), 503

    @app.get("/v1/models")
    @require_api_key(config.api_key)
    def model_metadata():
        return jsonify(sam3_service.metadata()), 200

    @app.get(f"{config.output_url_prefix}/<path:filename>")
    def serve_output_file(filename: str):
        return send_from_directory(config.output_dir, filename)

    @app.post("/v1/segment")
    @require_api_key(config.api_key)
    @apply_rate_limit(rate_limiter)
    def segment():
        upload = request.files.get("image")
        if upload is None:
            raise APIError("MISSING_IMAGE", "image is required.", 400)

        prompt_inputs = _parse_prompt_inputs(request.form, require_input=True)
        output_formats = _parse_output_formats(request.form)
        return_overlay = _parse_bool(request.form.get("return_overlay"), False)
        conf = _validate_conf(request.form.get("conf"), config)
        g.prompt = prompt_inputs["prompt"]

        source_path, _w, _h, decode_ms = _save_and_validate_upload(upload, g.request_id)
        image_sha256 = storage_service.file_sha256(source_path)
        cache_key = _cache_key(image_sha256, prompt_inputs, conf, return_overlay, output_formats)
        cached = cache_service.get(cache_key)
        if cached is not None:
            payload = dict(cached)
            payload["request_id"] = g.request_id
            payload["cached"] = True
            payload["cache_key"] = cache_key
            return jsonify(payload), 200

        if config.enable_auto_queue and sam3_service.is_busy:
            qsize = job_service.stats()["queue_size"]
            if qsize >= config.auto_queue_max_size:
                raise APIError("QUEUE_FULL", "Server is overloaded, queue is full.", 503)
            job_payload = {
                "request_id": g.request_id,
                "source_path": str(source_path),
                "prompt_inputs": prompt_inputs,
                "conf": conf,
                "return_overlay": return_overlay,
                "output_formats": sorted(output_formats),
                "decode_ms": decode_ms,
            }
            job_id = job_service.submit(job_payload)
            metrics_service.inc_auto_queued()
            metrics_service.inc_jobs_created()
            return (
                jsonify(
                    {
                        "request_id": g.request_id,
                        "status": "queued",
                        "mode": "auto_queued",
                        "job_id": job_id,
                        "status_url": f"/v1/jobs/{job_id}",
                    }
                ),
                202,
            )

        payload = _segment_from_saved(
            source_path=str(source_path),
            request_id=g.request_id,
            prompt_inputs=prompt_inputs,
            conf=conf,
            return_overlay=return_overlay,
            output_formats=output_formats,
            decode_ms=decode_ms,
        )
        payload["cache_key"] = cache_key
        cache_service.set(cache_key, payload)
        return jsonify(payload), 200

    @app.post("/v1/segment/batch")
    @require_api_key(config.api_key)
    @apply_rate_limit(rate_limiter)
    def segment_batch():
        uploads = request.files.getlist("images")
        if not uploads:
            raise APIError("MISSING_IMAGE", "images is required.", 400)
        if len(uploads) > config.max_batch_images:
            raise APIError("BATCH_TOO_LARGE", f"Maximum {config.max_batch_images} images per batch.", 400)

        prompt_inputs = _parse_prompt_inputs(request.form, require_input=True)
        output_formats = _parse_output_formats(request.form)
        return_overlay = _parse_bool(request.form.get("return_overlay"), False)
        conf = _validate_conf(request.form.get("conf"), config)
        g.prompt = prompt_inputs["prompt"]

        batch_start = time.perf_counter()
        items = []
        for idx, upload in enumerate(uploads):
            item_request_id = f"{g.request_id}_{idx:03d}"
            try:
                source_path, _w, _h, decode_ms = _save_and_validate_upload(upload, item_request_id)
                image_sha256 = storage_service.file_sha256(source_path)
                cache_key = _cache_key(image_sha256, prompt_inputs, conf, return_overlay, output_formats)
                cached = cache_service.get(cache_key)
                if cached is not None:
                    result = dict(cached)
                    result["request_id"] = item_request_id
                    result["cached"] = True
                    result["cache_key"] = cache_key
                else:
                    result = _segment_from_saved(
                        source_path=str(source_path),
                        request_id=item_request_id,
                        prompt_inputs=prompt_inputs,
                        conf=conf,
                        return_overlay=return_overlay,
                        output_formats=output_formats,
                        decode_ms=decode_ms,
                    )
                    result["cache_key"] = cache_key
                    cache_service.set(cache_key, result)
                items.append(
                    {
                        "index": idx,
                        "filename": upload.filename,
                        "status": "ok",
                        "result": result,
                    }
                )
            except APIError as err:
                items.append(
                    {
                        "index": idx,
                        "filename": upload.filename,
                        "status": "error",
                        "error": {"code": err.code, "message": err.message, "details": err.details},
                    }
                )

        total_ms = (time.perf_counter() - batch_start) * 1000
        return (
            jsonify(
                {
                    "request_id": g.request_id,
                    "batch_size": len(uploads),
                    "timing_ms": {"total": round(total_ms, 2)},
                    "items": items,
                }
            ),
            200,
        )

    @app.post("/v1/jobs")
    @require_api_key(config.api_key)
    @apply_rate_limit(rate_limiter)
    def create_job():
        upload = request.files.get("image")
        if upload is None:
            raise APIError("MISSING_IMAGE", "image is required.", 400)

        prompt_inputs = _parse_prompt_inputs(request.form, require_input=True)
        output_formats = _parse_output_formats(request.form)
        return_overlay = _parse_bool(request.form.get("return_overlay"), False)
        conf = _validate_conf(request.form.get("conf"), config)
        webhook_url = (request.form.get("webhook_url") or "").strip() or None
        webhook_secret = (request.form.get("webhook_secret") or "").strip() or None
        g.prompt = prompt_inputs["prompt"]

        source_path, _w, _h, decode_ms = _save_and_validate_upload(upload, g.request_id)
        if job_service.stats()["queue_size"] >= config.auto_queue_max_size:
            raise APIError("QUEUE_FULL", "Server queue is full.", 503)
        job_payload = {
            "request_id": g.request_id,
            "source_path": str(source_path),
            "prompt_inputs": prompt_inputs,
            "conf": conf,
            "return_overlay": return_overlay,
            "output_formats": sorted(output_formats),
            "decode_ms": decode_ms,
            "webhook_secret": webhook_secret,
        }
        job_id = job_service.submit(job_payload, webhook_url=webhook_url)
        metrics_service.inc_jobs_created()
        return (
            jsonify(
                {
                    "job_id": job_id,
                    "status": "queued",
                    "status_url": f"/v1/jobs/{job_id}",
                    "webhook_url": webhook_url,
                }
            ),
            202,
        )

    @app.get("/v1/jobs/<job_id>")
    @require_api_key(config.api_key)
    def get_job(job_id: str):
        record = job_service.get(job_id)
        if record is None:
            raise APIError("JOB_NOT_FOUND", "Job not found.", 404)

        payload = {
            "job_id": record.job_id,
            "status": record.status,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }
        if record.result is not None:
            payload["result"] = record.result
        if record.error is not None:
            payload["error"] = record.error
        return jsonify(payload), 200

    @app.delete("/v1/jobs/<job_id>")
    @require_api_key(config.api_key)
    def cancel_job(job_id: str):
        record = job_service.cancel(job_id)
        if record is None:
            raise APIError("JOB_NOT_FOUND", "Job not found.", 404)
        return (
            jsonify(
                {
                    "job_id": record.job_id,
                    "status": record.status,
                    "cancel_requested": record.cancel_requested,
                }
            ),
            200,
        )

    @app.post("/v1/jobs/<job_id>/retry")
    @require_api_key(config.api_key)
    def retry_job(job_id: str):
        record = job_service.get(job_id)
        if record is None:
            raise APIError("JOB_NOT_FOUND", "Job not found.", 404)

        if record.status not in {"failed", "canceled"}:
            raise APIError(
                "INVALID_JOB_STATE",
                "Retry is only allowed for failed or canceled jobs.",
                409,
                {"status": record.status},
            )

        if not record.payload or not record.payload.get("source_path"):
            raise APIError("INVALID_JOB_PAYLOAD", "Job payload is incomplete and cannot be retried.", 409)

        source_path = Path(record.payload["source_path"])
        if not source_path.exists():
            raise APIError("SOURCE_NOT_FOUND", "Source image for retry no longer exists.", 404)

        retry_payload = json.loads(json.dumps(record.payload, ensure_ascii=False))
        retry_payload["request_id"] = str(uuid.uuid4())
        retry_payload["retry_of"] = record.job_id

        if job_service.stats()["queue_size"] >= config.auto_queue_max_size:
            raise APIError("QUEUE_FULL", "Server queue is full.", 503)

        retry_job_id = job_service.submit(retry_payload, webhook_url=record.webhook_url)
        metrics_service.inc_jobs_created()
        return (
            jsonify(
                {
                    "job_id": retry_job_id,
                    "status": "queued",
                    "retry_of": record.job_id,
                    "status_url": f"/v1/jobs/{retry_job_id}",
                }
            ),
            202,
        )

    @app.get("/v1/jobs/<job_id>/export")
    @require_api_key(config.api_key)
    def export_job(job_id: str):
        record = job_service.get(job_id)
        if record is None:
            raise APIError("JOB_NOT_FOUND", "Job not found.", 404)
        if record.status != "done" or record.result is None:
            raise APIError("JOB_NOT_READY", "Job result is not ready for export.", 409, {"status": record.status})

        request_id = record.result.get("request_id") or record.payload.get("request_id")
        if not request_id:
            raise APIError("EXPORT_NOT_FOUND", "Cannot resolve output directory for this job.", 404)

        output_dir = Path(config.output_dir) / request_id
        if not output_dir.exists() or not output_dir.is_dir():
            raise APIError("EXPORT_NOT_FOUND", "Output directory not found.", 404)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            files = sorted([p for p in output_dir.rglob("*") if p.is_file()])
            if not files:
                raise APIError("EXPORT_NOT_FOUND", "No files available to export.", 404)
            for path in files:
                arcname = f"{request_id}/{path.relative_to(output_dir).as_posix()}"
                zf.write(path, arcname=arcname)
            zf.writestr(
                f"{request_id}/result.json",
                json.dumps(
                    {
                        "job_id": record.job_id,
                        "status": record.status,
                        "created_at": record.created_at,
                        "updated_at": record.updated_at,
                        "result": record.result,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )

        zip_buffer.seek(0)
        return send_file(
            zip_buffer,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"{job_id}.zip",
        )

    with app.app_context():
        try:
            if not config.skip_model_load:
                sam3_service.load()
                sample_image = "image (2).jpg"
                if Path(sample_image).exists():
                    sam3_service.warmup(sample_image_path=sample_image)
            storage_service.start_cleanup_thread()
            webhook_retry_service.start(_post_webhook)
            job_service.start(_job_handler, completion_hook=_job_completion_hook)
            logging.info("SAM3 service initialized.")
        except Exception as exc:
            logging.exception("Failed to initialize SAM3 service: %s", exc)

    return app


if _parse_bool(os.getenv("SAM3_AUTOSTART_APP"), True):
    app = create_app()
else:
    app = Flask(__name__)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
