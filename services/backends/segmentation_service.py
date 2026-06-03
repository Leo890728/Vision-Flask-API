from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from ultralytics import YOLO
from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics.models.sam.predict import SAM3Predictor

from config import Config
from errors import APIError
from services.backends.model_backend import BaseModelBackend, resolve_yolo_class_ids

if TYPE_CHECKING:
    from services.infra.model_pool import ModelPool


def _build_segmentation_payload(results: Any, overlay: str) -> dict[str, Any]:
    """Normalize an ultralytics segmentation result into the response dict."""
    if not results:
        return {
            "width": 0,
            "height": 0,
            "masks": np.zeros((0, 0, 0), dtype=np.uint8),
            "boxes": [],
            "scores": [],
            "overlay_bgr": None,
        }

    result = results[0]
    height, width = result.orig_shape
    masks_tensor = result.masks.data if result.masks is not None else None
    boxes_tensor = result.boxes.xyxy if result.boxes is not None else None
    conf_tensor = result.boxes.conf if result.boxes is not None else None

    masks = (
        masks_tensor.detach().cpu().numpy().astype(np.uint8)
        if masks_tensor is not None
        else np.zeros((0, height, width), dtype=np.uint8)
    )
    boxes = boxes_tensor.detach().cpu().numpy().tolist() if boxes_tensor is not None else []
    scores = conf_tensor.detach().cpu().numpy().tolist() if conf_tensor is not None else []

    overlay_bgr = None
    if overlay != "none":
        overlay_bgr = result.plot(boxes=overlay in {"bbox", "both"}, masks=overlay in {"mask", "both"})

    return {
        "width": int(width),
        "height": int(height),
        "masks": masks,
        "boxes": boxes,
        "scores": scores,
        "overlay_bgr": overlay_bgr,
    }


class _SAM3Backend(BaseModelBackend):
    def __init__(self, config: Config):
        super().__init__(config, config.models["sam3"])
        self._predictor: SAM3SemanticPredictor | None = None
        self._visual_predictor: SAM3Predictor | None = None

    def _models_loaded(self) -> bool:
        return self._predictor is not None and self._visual_predictor is not None

    def load(self) -> None:
        overrides = {
            "conf": self.model_cfg.default_conf,
            "task": self.model_cfg.task,
            "mode": "predict",
            "model": self.model_cfg.model_path,
            "half": self.model_cfg.half,
            "save": False,
            "verbose": False,
        }
        if self.model_cfg.device != "auto":
            overrides["device"] = self.model_cfg.device

        self._predictor = SAM3SemanticPredictor(overrides=overrides)
        self._visual_predictor = SAM3Predictor(overrides=overrides)
        self._ready = True
        self._last_error = None

    def unload(self) -> None:
        del self._predictor
        del self._visual_predictor
        self._predictor = None
        self._visual_predictor = None
        self._ready = False

    def warmup(self, sample_image_path: str | None = None) -> None:
        if self._predictor is None:
            raise RuntimeError("Predictor is not initialized.")
        if sample_image_path is None:
            return

        try:
            start = time.perf_counter()
            with self.track_inference():
                self._predictor.set_image(sample_image_path)
                self._predictor(text=["object"])
            elapsed_ms = (time.perf_counter() - start) * 1000
            logging.info("SAM3 warmup completed in %.1f ms", elapsed_ms)
        except Exception as exc:
            logging.warning("SAM3 warmup failed: %s", exc)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.model_cfg.name,
            "task": self.model_cfg.task,
            "model_path": self.model_cfg.model_path,
            "default_conf": self.model_cfg.default_conf,
            "half": self.model_cfg.half,
            "device": self.model_cfg.device,
            "ready": self.is_ready,
            "last_error": self._last_error,
            "busy": self.is_busy,
        }

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
            raise APIError("MODEL_NOT_READY", "SAM3 model is not ready.", 503)
        if conf is None:
            conf = self.model_cfg.default_conf
        if not prompt and not points and not boxes:
            raise APIError("MISSING_PROMPT", "Provide prompt, points, or boxes.", 400)

        try:
            with self.track_inference():
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
            raise APIError("INFERENCE_FAILED", "SAM3 inference failed.", 500, {"reason": str(exc)}) from exc

        return _build_segmentation_payload(results, overlay)


class _YOLOSegBackend(BaseModelBackend):
    def __init__(self, config: Config):
        super().__init__(config, config.models["yolo_seg"])
        self._model: YOLO | None = None

    def _models_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        self._model = YOLO(self.model_cfg.model_path)
        self._ready = True
        self._last_error = None

    def unload(self) -> None:
        del self._model
        self._model = None
        self._ready = False

    def warmup(self, sample_image_path: str | None = None) -> None:
        if self._model is None or sample_image_path is None:
            return
        try:
            with self.track_inference():
                self._model.predict(
                    source=sample_image_path,
                    conf=self.model_cfg.default_conf,
                    device=None if self.model_cfg.device == "auto" else self.model_cfg.device,
                    half=self.model_cfg.half,
                    verbose=False,
                )
        except Exception as exc:
            self._last_error = str(exc)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.model_cfg.name,
            "task": self.model_cfg.task,
            "model_path": self.model_cfg.model_path,
            "default_conf": self.model_cfg.default_conf,
            "half": self.model_cfg.half,
            "device": self.model_cfg.device,
            "ready": self.is_ready,
            "last_error": self._last_error,
            "busy": self.is_busy,
        }

    def resolve_class_ids(self, classes: list[int | str] | None) -> list[int] | None:
        if not classes:
            return None
        if not self.is_ready or self._model is None:
            raise APIError("MODEL_NOT_READY", "YOLO segmentation model is not ready.", 503)
        return resolve_yolo_class_ids(self._model, classes)

    def segment(
        self,
        image_path: str,
        conf: float | None = None,
        classes: list[int | str] | None = None,
        overlay: str = "none",
    ) -> dict[str, Any]:
        if not self.is_ready or self._model is None:
            raise APIError("MODEL_NOT_READY", "YOLO segmentation model is not ready.", 503)
        if conf is None:
            conf = self.model_cfg.default_conf

        class_ids = self.resolve_class_ids(classes)
        predict_kwargs: dict[str, Any] = {
            "source": image_path,
            "conf": conf,
            "device": None if self.model_cfg.device == "auto" else self.model_cfg.device,
            "half": self.model_cfg.half,
            "verbose": False,
        }
        if class_ids:
            predict_kwargs["classes"] = class_ids

        try:
            with self.track_inference():
                results = self._model.predict(**predict_kwargs)
        except Exception as exc:
            self._last_error = str(exc)
            raise APIError("INFERENCE_FAILED", "YOLO segmentation inference failed.", 500, {"reason": str(exc)}) from exc

        return _build_segmentation_payload(results, overlay)


class SegmentationService:
    def __init__(
        self,
        config: Config,
        sam3_backend: Any | None = None,
        yolo_seg_backend: Any | None = None,
        pool: ModelPool | None = None,
    ):
        self.config = config
        self._pool = pool
        self.sam3_backend = sam3_backend or _SAM3Backend(config)
        self.yolo_seg_backend = yolo_seg_backend or _YOLOSegBackend(config)

    def load(self) -> None:
        if self._pool:
            # With a pool, only pre-load the default backend so all models don't
            # occupy GPU simultaneously at startup.
            backend = self._backend_for(self.config.segment_default_model)
            try:
                self._pool.ensure_loaded(backend)
            except Exception as exc:
                backend._ready = False
                backend._last_error = str(exc)
                logging.warning("Failed to load segmentation model %s: %s", backend.name, exc)
        else:
            self.sam3_backend.load()
            self.yolo_seg_backend.load()

    def warmup(self, sample_image_path: str | None = None) -> None:
        for backend in (self.sam3_backend, self.yolo_seg_backend):
            if backend.is_ready:
                backend.warmup(sample_image_path=sample_image_path)

    @property
    def is_ready(self) -> bool:
        return bool(self._backend_for(self.config.segment_default_model).is_ready)

    @property
    def last_error(self) -> str | None:
        return self._backend_for(self.config.segment_default_model).last_error

    @property
    def is_busy(self) -> bool:
        return self.is_busy_for(self.config.segment_default_model)

    def is_busy_for(self, segment_model: str) -> bool:
        return bool(self._backend_for(segment_model).is_busy)

    def metadata(self) -> dict[str, Any]:
        return {
            "default_model": self.config.segment_default_model,
            "backends": {
                "sam3": self.sam3_backend.metadata(),
                "yolo_seg": self.yolo_seg_backend.metadata(),
            },
        }

    def segment(
        self,
        image_path: str,
        segment_model: str,
        prompt_inputs: dict[str, Any],
        conf: float,
        classes: list[int | str] | None = None,
        overlay: str = "none",
    ) -> dict[str, Any]:
        backend = self._backend_for(segment_model)
        if self._pool:
            self._pool.ensure_loaded(backend)
        if segment_model == "yolo_seg":
            return self.yolo_seg_backend.segment(
                image_path=image_path,
                conf=conf,
                classes=classes,
                overlay=overlay,
            )
        if segment_model == "sam3":
            return self.sam3_backend.segment(
                image_path,
                prompt=prompt_inputs["prompt"],
                conf=conf,
                points=prompt_inputs["points"],
                point_labels=prompt_inputs["point_labels"],
                boxes=prompt_inputs["boxes"],
                overlay=overlay,
            )
        raise APIError("INVALID_SEGMENT_MODEL", f"Unsupported segment_model: {segment_model}", 400)

    def _backend_for(self, segment_model: str):
        normalized = str(segment_model or "sam3").strip().lower().replace("-", "_")
        if normalized == "sam3":
            return self.sam3_backend
        if normalized == "yolo_seg":
            return self.yolo_seg_backend
        raise APIError("INVALID_SEGMENT_MODEL", f"Unsupported segment_model: {segment_model}", 400)
