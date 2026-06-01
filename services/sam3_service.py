from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics.models.sam.predict import SAM3Predictor

from config import Config
from errors import APIError


class SAM3Service:
    def __init__(self, config: Config):
        self.config = config
        self._predictor: SAM3SemanticPredictor | None = None
        self._visual_predictor: SAM3Predictor | None = None
        self._lock = threading.Lock()
        self._active_inference = 0
        self._ready = False
        self._last_error: str | None = None

    def load(self) -> None:
        overrides = {
            "conf": self.config.model_default_conf,
            "task": "segment",
            "mode": "predict",
            "model": self.config.model_path,
            "half": self.config.model_half,
            "save": False,
            "verbose": False,
        }
        if self.config.model_device != "auto":
            overrides["device"] = self.config.model_device

        self._predictor = SAM3SemanticPredictor(overrides=overrides)
        self._visual_predictor = SAM3Predictor(overrides=overrides)
        self._ready = True
        self._last_error = None

    def warmup(self, sample_image_path: str | None = None) -> None:
        if self._predictor is None:
            raise RuntimeError("Predictor is not initialized.")
        if sample_image_path is None:
            return

        try:
            start = time.perf_counter()
            with self._lock:
                self._predictor.set_image(sample_image_path)
                self._predictor(text=["object"])
            elapsed_ms = (time.perf_counter() - start) * 1000
            logging.info("SAM3 warmup completed in %.1f ms", elapsed_ms)
        except Exception as exc:
            # Warmup failures should not block startup if model itself loaded.
            logging.warning("SAM3 warmup failed: %s", exc)

    @property
    def is_ready(self) -> bool:
        return self._ready and self._predictor is not None and self._visual_predictor is not None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def metadata(self) -> dict[str, Any]:
        return {
            "model_path": self.config.model_path,
            "default_conf": self.config.model_default_conf,
            "half": self.config.model_half,
            "device": self.config.model_device,
            "ready": self.is_ready,
            "last_error": self._last_error,
            "busy": self.is_busy,
        }

    @property
    def is_busy(self) -> bool:
        return self._active_inference > 0

    def segment(
        self,
        image_path: str,
        prompt: str | None = None,
        conf: float | None = None,
        points: list[list[float]] | None = None,
        point_labels: list[int] | None = None,
        boxes: list[list[float]] | None = None,
        overlay: str = "none",
    ) -> dict[str, Any]:
        if not self.is_ready or self._predictor is None or self._visual_predictor is None:
            raise APIError("MODEL_NOT_READY", "Model is not ready.", 503)

        if conf is None:
            conf = self.config.model_default_conf

        if not prompt and not points and not boxes:
            raise APIError("MISSING_PROMPT", "Provide prompt, points, or boxes.", 400)

        try:
            with self._lock:
                self._active_inference += 1
                self._predictor.args.conf = conf
                self._visual_predictor.args.conf = conf
                if points:
                    self._visual_predictor.set_image(image_path)
                    results = self._visual_predictor(points=points, labels=point_labels)
                elif boxes and not prompt:
                    self._visual_predictor.set_image(image_path)
                    results = self._visual_predictor(bboxes=boxes)
                else:
                    self._predictor.set_image(image_path)
                    kwargs: dict[str, Any] = {}
                    if boxes:
                        kwargs["bboxes"] = boxes
                        kwargs["labels"] = [1] * len(boxes)
                    results = self._predictor(text=[prompt] if prompt else None, **kwargs)
        except Exception as exc:
            self._last_error = str(exc)
            raise APIError("INFERENCE_FAILED", "Model inference failed.", 500, {"reason": str(exc)}) from exc
        finally:
            with self._lock:
                self._active_inference = max(0, self._active_inference - 1)

        if not results:
            return {"width": 0, "height": 0, "masks": [], "boxes": [], "scores": [], "overlay_bgr": None}

        result = results[0]
        height, width = result.orig_shape
        masks_tensor = result.masks.data if result.masks is not None else None
        boxes_tensor = result.boxes.xyxy if result.boxes is not None else None
        conf_tensor = result.boxes.conf if result.boxes is not None else None

        masks = masks_tensor.detach().cpu().numpy().astype(np.uint8) if masks_tensor is not None else np.zeros((0, height, width), dtype=np.uint8)
        boxes = boxes_tensor.detach().cpu().numpy().tolist() if boxes_tensor is not None else []
        scores = conf_tensor.detach().cpu().numpy().tolist() if conf_tensor is not None else []

        if overlay == "none":
            overlay_bgr = None
        else:
            draw_boxes = overlay in {"bbox", "both"}
            draw_masks = overlay in {"mask", "both"}
            overlay_bgr = result.plot(boxes=draw_boxes, masks=draw_masks)

        return {
            "width": int(width),
            "height": int(height),
            "masks": masks,
            "boxes": boxes,
            "scores": scores,
            "overlay_bgr": overlay_bgr,
        }
