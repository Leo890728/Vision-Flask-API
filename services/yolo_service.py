from __future__ import annotations

import threading
from typing import Any

from ultralytics import YOLO

from config import Config
from errors import APIError


class YOLOService:
    def __init__(self, config: Config):
        self.config = config
        self._model: YOLO | None = None
        self._lock = threading.Lock()
        self._active_inference = 0
        self._ready = False
        self._last_error: str | None = None

    def load(self) -> None:
        self._model = YOLO(self.config.yolo_model_path)
        self._ready = True
        self._last_error = None

    def warmup(self, sample_image_path: str | None = None) -> None:
        if self._model is None or sample_image_path is None:
            return
        try:
            with self._lock:
                self._model.predict(
                    source=sample_image_path,
                    conf=self.config.yolo_default_conf,
                    device=None if self.config.yolo_device == "auto" else self.config.yolo_device,
                    half=self.config.yolo_half,
                    verbose=False,
                )
        except Exception as exc:
            self._last_error = str(exc)

    @property
    def is_ready(self) -> bool:
        return self._ready and self._model is not None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def is_busy(self) -> bool:
        return self._active_inference > 0

    def metadata(self) -> dict[str, Any]:
        return {
            "model_path": self.config.yolo_model_path,
            "default_conf": self.config.yolo_default_conf,
            "half": self.config.yolo_half,
            "device": self.config.yolo_device,
            "ready": self.is_ready,
            "last_error": self._last_error,
            "busy": self.is_busy,
        }

    def resolve_class_ids(self, classes: list[int | str] | None) -> list[int] | None:
        if not classes:
            return None
        if not self.is_ready or self._model is None:
            raise APIError("MODEL_NOT_READY", "YOLO model is not ready.", 503)

        names_obj = self._model.names
        names_map: dict[int, str] = {}
        if isinstance(names_obj, dict):
            names_map = {int(k): str(v) for k, v in names_obj.items()}
        elif isinstance(names_obj, list):
            names_map = {idx: str(v) for idx, v in enumerate(names_obj)}

        reverse = {name.lower(): idx for idx, name in names_map.items()}
        out: list[int] = []
        for item in classes:
            if isinstance(item, int):
                out.append(item)
                continue
            try:
                out.append(int(item))
                continue
            except Exception:
                pass
            key = str(item).strip().lower()
            if key not in reverse:
                raise APIError("INVALID_CLASSES", f"Unknown class name: {item}", 400)
            out.append(reverse[key])

        # keep order while removing duplicates
        seen: set[int] = set()
        deduped: list[int] = []
        for cid in out:
            if cid in seen:
                continue
            seen.add(cid)
            deduped.append(cid)
        return deduped

    def detect(
        self,
        image_path: str,
        conf: float | None = None,
        class_ids: list[int] | None = None,
        overlay: str = "none",
    ) -> dict[str, Any]:
        if not self.is_ready or self._model is None:
            raise APIError("MODEL_NOT_READY", "YOLO model is not ready.", 503)
        if conf is None:
            conf = self.config.yolo_default_conf

        predict_kwargs: dict[str, Any] = {
            "source": image_path,
            "conf": conf,
            "device": None if self.config.yolo_device == "auto" else self.config.yolo_device,
            "half": self.config.yolo_half,
            "verbose": False,
        }
        if class_ids:
            predict_kwargs["classes"] = class_ids

        try:
            with self._lock:
                self._active_inference += 1
                results = self._model.predict(**predict_kwargs)
        except Exception as exc:
            self._last_error = str(exc)
            raise APIError("INFERENCE_FAILED", "YOLO inference failed.", 500, {"reason": str(exc)}) from exc
        finally:
            with self._lock:
                self._active_inference = max(0, self._active_inference - 1)

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
                        "bbox": [float(v) for v in bbox],
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
