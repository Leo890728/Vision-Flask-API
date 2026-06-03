from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator

from config import Config
from errors import APIError


class BaseModelBackend:
    """Shared state and locking for inference backends.

    Subclasses provide ``load``/``warmup`` and the concrete inference call,
    wrapping the model invocation in :meth:`track_inference` so that the model
    is accessed under a lock and ``is_busy`` reflects an in-flight request.
    """

    def __init__(self, config: Config):
        self.config = config
        self._lock = threading.Lock()
        self._active_inference = 0
        self._ready = False
        self._last_error: str | None = None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def is_busy(self) -> bool:
        return self._active_inference > 0

    @property
    def is_ready(self) -> bool:
        return self._ready and self._models_loaded()

    def _models_loaded(self) -> bool:
        raise NotImplementedError

    @contextmanager
    def track_inference(self) -> Iterator[None]:
        """Hold the model lock for the duration of one inference call.

        The lock serializes access to the underlying model while the active
        counter records that a request is in flight.
        """
        with self._lock:
            self._active_inference += 1
            try:
                yield
            finally:
                self._active_inference = max(0, self._active_inference - 1)


def resolve_yolo_class_ids(model: Any, classes: list[int | str] | None) -> list[int] | None:
    """Map class names/ids to deduped int ids using a YOLO model's ``names``."""
    if not classes:
        return None

    names_obj = model.names
    names_map: dict[int, str] = {}
    if isinstance(names_obj, dict):
        names_map = {int(k): str(value) for k, value in names_obj.items()}
    elif isinstance(names_obj, list):
        names_map = {idx: str(value) for idx, value in enumerate(names_obj)}

    reverse = {name.lower(): idx for idx, name in names_map.items()}
    out: list[int] = []
    for item in classes:
        if isinstance(item, int):
            out.append(item)
            continue
        try:
            out.append(int(item))
            continue
        except (TypeError, ValueError):
            pass
        key = str(item).strip().lower()
        if key not in reverse:
            raise APIError("INVALID_CLASSES", f"Unknown class name: {item}", 400)
        out.append(reverse[key])

    return list(dict.fromkeys(out))
