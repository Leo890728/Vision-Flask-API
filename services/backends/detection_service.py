from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ultralytics import YOLO

from config import Config, ModelConfig
from errors import APIError
from services.backends.model_backend import BaseModelBackend, resolve_yolo_class_ids

if TYPE_CHECKING:
    from services.infra.model_pool import ModelPool


def _extract_class_names(model: Any) -> list[str] | None:
    if model is None:
        return None
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names.keys())]
    if isinstance(names, list):
        return [str(n) for n in names]
    return None


class _YOLODetectionBackend(BaseModelBackend):
    def __init__(self, config: Config, model_cfg: ModelConfig):
        super().__init__(config, model_cfg)
        self._model: YOLO | None = None
        self._class_names: list[str] | None = None

    @property
    def class_names(self) -> list[str] | None:
        return self._class_names

    def _models_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        self._model = YOLO(self.model_cfg.model_path, task=self.model_cfg.task)
        self._class_names = _extract_class_names(self._model)
        self._ready = True
        self._last_error = None

    def unload(self) -> None:
        del self._model
        self._model = None
        self._ready = False
        # _class_names intentionally kept — survives LRU eviction

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
            "class_names": self._class_names,
        }

    def resolve_class_ids(self, classes: list[int | str] | None) -> list[int] | None:
        if not classes:
            return None
        if not self.is_ready or self._model is None:
            raise APIError("MODEL_NOT_READY", "YOLO detection model is not ready.", 503)
        return resolve_yolo_class_ids(self._model, classes)

    def detect(
        self,
        image_path: str,
        conf: float | None = None,
        class_ids: list[int] | None = None,
        overlay: str = "none",
    ) -> dict[str, Any]:
        if not self.is_ready or self._model is None:
            raise APIError("MODEL_NOT_READY", "YOLO detection model is not ready.", 503)
        if conf is None:
            conf = self.model_cfg.default_conf

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
            raise APIError(
                "INFERENCE_FAILED",
                "YOLO detection inference failed.",
                500,
                {"reason": str(exc), "detect_model": self.model_cfg.name},
            ) from exc

        if not results:
            return {"width": 0, "height": 0, "detections": [], "overlay_bgr": None}

        result = results[0]
        height, width = result.orig_shape
        names_map = result.names or {}
        detections: list[dict[str, Any]] = []
        boxes = result.boxes
        if boxes is not None and boxes.xyxy is not None and boxes.conf is not None and boxes.cls is not None:
            xyxy = boxes.xyxy.detach().cpu().numpy().tolist()
            confs = boxes.conf.detach().cpu().numpy().tolist()
            class_arr = boxes.cls.detach().cpu().numpy().tolist()
            for idx, bbox in enumerate(xyxy):
                class_id = int(class_arr[idx]) if idx < len(class_arr) else -1
                detections.append(
                    {
                        "id": idx,
                        "score": float(confs[idx]) if idx < len(confs) else 0.0,
                        "bbox": [float(value) for value in bbox],
                        "class_id": class_id,
                        "class_name": str(names_map.get(class_id, str(class_id))),
                    }
                )

        overlay_bgr = None
        if overlay == "bbox":
            overlay_bgr = result.plot(boxes=True, labels=True, conf=True)

        return {
            "width": int(width),
            "height": int(height),
            "detections": detections,
            "overlay_bgr": overlay_bgr,
        }


class DetectionService:
    def __init__(
        self,
        config: Config,
        backends: dict[str, Any] | None = None,
        pool: ModelPool | None = None,
    ):
        self.config = config
        self._pool = pool
        if backends is None:
            backends = {
                name: _YOLODetectionBackend(config, model_cfg)
                for name, model_cfg in config.models.items()
                if model_cfg.task == "detect"
            }
        self.backends = dict(backends)

    @property
    def name(self) -> str:
        return self.config.detect_default_model

    @property
    def task(self) -> str:
        return "detect"

    @property
    def is_ready(self) -> bool:
        return bool(self._backend_for(self.config.detect_default_model).is_ready)

    @property
    def last_error(self) -> str | None:
        return self._backend_for(self.config.detect_default_model).last_error

    @property
    def is_busy(self) -> bool:
        return self.is_busy_for(self.config.detect_default_model)

    def load(self) -> None:
        if self._pool:
            # With a pool, only pre-load the default backend so all models don't
            # occupy GPU simultaneously at startup.
            backend = self._backend_for(self.config.detect_default_model)
            try:
                self._pool.ensure_loaded(backend)
            except Exception as exc:
                backend._ready = False
                backend._last_error = str(exc)
                logging.warning("Failed to load detection model %s: %s", backend.name, exc)
        else:
            for backend in self.backends.values():
                try:
                    backend.load()
                except Exception as exc:
                    if hasattr(backend, "_ready"):
                        backend._ready = False
                    if hasattr(backend, "_last_error"):
                        backend._last_error = str(exc)
                    logging.warning("Failed to load detection model %s: %s", backend.name, exc)

    def warmup(self, sample_image_path: str | None = None) -> None:
        for backend in self.backends.values():
            if backend.is_ready:
                backend.warmup(sample_image_path=sample_image_path)

    def metadata(self) -> dict[str, Any]:
        return {
            "default_model": self.config.detect_default_model,
            "backends": {name: backend.metadata() for name, backend in self.backends.items()},
        }

    def is_busy_for(self, detect_model: str | None = None) -> bool:
        return bool(self._backend_for(detect_model).is_busy)

    def get_class_names(self, detect_model: str | None = None) -> list[str] | None:
        backend = self._backend_for(detect_model)
        if backend.class_names is None:
            if self._pool:
                self._pool.ensure_loaded(backend)
            else:
                backend.load()
        return backend.class_names

    def resolve_class_ids(
        self,
        classes: list[int | str] | None,
        detect_model: str | None = None,
    ) -> list[int] | None:
        backend = self._backend_for(detect_model)
        if self._pool:
            self._pool.ensure_loaded(backend)
        return backend.resolve_class_ids(classes)

    def detect(
        self,
        image_path: str,
        detect_model: str | None = None,
        conf: float | None = None,
        class_ids: list[int] | None = None,
        overlay: str = "none",
    ) -> dict[str, Any]:
        backend = self._backend_for(detect_model)
        if self._pool:
            self._pool.ensure_loaded(backend)
        return backend.detect(
            image_path=image_path,
            conf=conf,
            class_ids=class_ids,
            overlay=overlay,
        )

    def _backend_for(self, detect_model: str | None):
        name = str(detect_model or self.config.detect_default_model).strip()
        backend = self.backends.get(name)
        if backend is not None:
            return backend

        detect_models = sorted(name for name, cfg in self.config.models.items() if cfg.task == "detect")
        raise APIError(
            "INVALID_DETECT_MODEL",
            f"detect_model must be one of: {', '.join(detect_models)}.",
            400,
            {"available": detect_models},
        )
