from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from config import Config
from services.metrics_service import MetricsService
from services.segmentation_service import SegmentationService
from services.storage_service import StorageService
from services.upload_service import UploadService

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency fallback
    cv2 = None


class SegmentationPipeline:
    def __init__(
        self,
        config: Config,
        segmentation_service: SegmentationService,
        storage_service: StorageService,
        metrics_service: MetricsService,
        upload_service: UploadService,
    ):
        self.config = config
        self.segmentation_service = segmentation_service
        self.storage_service = storage_service
        self.metrics_service = metrics_service
        self.upload_service = upload_service

    def save_and_validate_upload(self, upload, request_id: str):
        return self.upload_service.save_and_validate_upload(upload, request_id)

    def build_cache_key(
        self,
        image_sha256: str,
        prompt_inputs: dict[str, Any],
        conf: float,
        overlay: str,
        output_formats: set[str],
        segment_model: str = "sam3",
        classes: list[int | str] | None = None,
    ) -> str:
        key_obj = {
            "task": "segment",
            "segment_model": segment_model,
            "image_sha256": image_sha256,
            "prompt_inputs": prompt_inputs,
            "classes": classes or [],
            "conf": conf,
            "overlay": overlay,
            "output_formats": sorted(output_formats),
        }
        raw = json.dumps(key_obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def segment_from_saved(
        self,
        source_path: str,
        request_id: str,
        prompt_inputs: dict[str, Any],
        conf: float,
        overlay: str,
        output_formats: set[str],
        decode_ms: float = 0.0,
        segment_model: str = "sam3",
        classes: list[int | str] | None = None,
    ) -> dict[str, Any]:
        infer_start = time.perf_counter()
        seg_result = self.segmentation_service.segment(
            image_path=source_path,
            prompt_inputs=prompt_inputs,
            conf=conf,
            overlay=overlay,
            segment_model=segment_model,
            classes=classes,
        )
        infer_ms = (time.perf_counter() - infer_start) * 1000
        self.metrics_service.observe_inference_latency(infer_ms)

        post_start = time.perf_counter()
        detections, overlay_url = self._render_segmentation_response(
            request_id=request_id,
            seg_result=seg_result,
            overlay=overlay,
            output_formats=output_formats,
        )
        post_ms = (time.perf_counter() - post_start) * 1000

        total_ms = decode_ms + infer_ms + post_ms
        return {
            "request_id": request_id,
            "segment_model": segment_model,
            "prompt": prompt_inputs["prompt"],
            "cached": False,
            "output_formats": sorted(output_formats),
            "classes": classes or [],
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

    def job_handler(self, payload: dict[str, Any]) -> dict[str, Any]:
        overlay = payload.get("overlay", "none")
        segment_model = payload.get("segment_model", "sam3")
        return self.segment_from_saved(
            source_path=payload["source_path"],
            request_id=payload["request_id"],
            prompt_inputs=payload["prompt_inputs"],
            conf=payload["conf"],
            overlay=overlay,
            output_formats=set(payload.get("output_formats", ["mask_png"])),
            decode_ms=payload.get("decode_ms", 0.0),
            segment_model=segment_model,
            classes=payload.get("classes"),
        )

    def _render_segmentation_response(
        self,
        request_id: str,
        seg_result: dict[str, Any],
        overlay: str,
        output_formats: set[str],
    ) -> tuple[list[dict[str, Any]], str | None]:
        masks = seg_result["masks"]
        boxes = seg_result["boxes"]
        scores = seg_result["scores"]

        detections = []
        count = int(masks.shape[0]) if hasattr(masks, "shape") else len(masks)
        for idx in range(count):
            mask = masks[idx]
            bbox = boxes[idx] if idx < len(boxes) else [0.0, 0.0, 0.0, 0.0]
            score = float(scores[idx]) if idx < len(scores) else 1.0
            item = {
                "id": idx,
                "score": score,
                "bbox": [float(v) for v in bbox],
            }
            if "mask_png" in output_formats:
                _rel_path, mask_url = self.storage_service.save_mask(request_id, idx, mask)
                item["mask_url"] = mask_url
            if "alpha_matte" in output_formats:
                _rel_path, alpha_url = self.storage_service.save_alpha_matte(request_id, idx, mask)
                item["alpha_url"] = alpha_url
            if "rle" in output_formats:
                item["rle"] = self._mask_to_rle(mask)
            if "polygon" in output_formats:
                item["polygon"] = self._mask_to_polygon(mask)
            detections.append(item)

        overlay_url = None
        if overlay != "none" and seg_result.get("overlay_bgr") is not None:
            _rel_path, overlay_url = self.storage_service.save_overlay(request_id, seg_result["overlay_bgr"])
        return detections, overlay_url

    @staticmethod
    def _mask_to_rle(mask: Any) -> dict[str, Any]:
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

    @staticmethod
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
