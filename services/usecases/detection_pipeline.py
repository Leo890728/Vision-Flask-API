from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from config import Config
from errors import APIError
from services.infra.metrics_service import MetricsService
from services.infra.storage_service import StorageService
from services.infra.upload_service import UploadService


class DetectionPipeline:
    def __init__(
        self,
        config: Config,
        detection_service: Any,
        storage_service: StorageService,
        metrics_service: MetricsService,
        upload_service: UploadService,
    ):
        self.config = config
        self.detection_service = detection_service
        self.storage_service = storage_service
        self.metrics_service = metrics_service
        self.upload_service = upload_service

    def save_and_validate_upload(self, upload, request_id: str):
        return self.upload_service.save_and_validate_upload(upload, request_id)

    def build_cache_key(
        self,
        image_sha256: str,
        detect_model: str,
        conf: float,
        overlay: str,
        classes: list[int | str] | None,
    ) -> str:
        key_obj = {
            "task": "detect",
            "detect_model": detect_model,
            "image_sha256": image_sha256,
            "conf": conf,
            "overlay": overlay,
            "classes": classes or [],
        }
        raw = json.dumps(key_obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def detect_from_saved(
        self,
        source_path: str,
        request_id: str,
        detect_model: str,
        conf: float,
        overlay: str,
        classes: list[int | str] | None,
        decode_ms: float = 0.0,
    ) -> dict[str, Any]:
        infer_start = time.perf_counter()
        try:
            class_ids = self.detection_service.resolve_class_ids(classes, detect_model=detect_model)
        except APIError:
            raise
        except Exception as exc:
            raise APIError("INVALID_CLASSES", str(exc), 400) from exc
        detect_result = self.detection_service.detect(
            image_path=source_path,
            detect_model=detect_model,
            conf=conf,
            class_ids=class_ids,
            overlay=overlay,
        )
        infer_ms = (time.perf_counter() - infer_start) * 1000
        self.metrics_service.observe_inference_latency(infer_ms, task="detect", model=detect_model)

        post_start = time.perf_counter()
        detections, overlay_url = self._render_detection_response(
            request_id=request_id,
            detect_result=detect_result,
            overlay=overlay,
        )
        post_ms = (time.perf_counter() - post_start) * 1000
        total_ms = decode_ms + infer_ms + post_ms
        return {
            "request_id": request_id,
            "task": "detect",
            "detect_model": detect_model,
            "cached": False,
            "classes": classes or [],
            "image_meta": {
                "width": detect_result["width"],
                "height": detect_result["height"],
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

    def job_handler(self, payload: dict[str, Any]) -> dict[str, Any]:
        overlay = payload.get("overlay", "none")
        return self.detect_from_saved(
            source_path=payload["source_path"],
            request_id=payload["request_id"],
            detect_model=payload.get("detect_model", self.config.detect_default_model),
            conf=payload["conf"],
            overlay=overlay,
            classes=payload.get("classes"),
            decode_ms=payload.get("decode_ms", 0.0),
        )

    def _render_detection_response(
        self,
        request_id: str,
        detect_result: dict[str, Any],
        overlay: str,
    ) -> tuple[list[dict[str, Any]], str | None]:
        detections = list(detect_result.get("detections", []))
        overlay_url = None
        if overlay == "bbox" and detect_result.get("overlay_bgr") is not None:
            _rel_path, overlay_url = self.storage_service.save_overlay(request_id, detect_result["overlay_bgr"])
        return detections, overlay_url
