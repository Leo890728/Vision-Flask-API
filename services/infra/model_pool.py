from __future__ import annotations

import gc
import logging
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.backends.model_backend import BaseModelBackend

logger = logging.getLogger(__name__)


class ModelPool:
    """LRU GPU model pool — at most ``max_loaded`` backends live on the GPU at once.

    When a new backend needs to be loaded and the pool is full, the
    least-recently-used non-busy backend is evicted first.  If every resident
    backend is currently running inference the new model is loaded anyway
    (temporary overflow) to avoid starving the request.
    """

    def __init__(self, max_loaded: int = 1) -> None:
        self._max_loaded = max_loaded
        self._loaded: OrderedDict[str, BaseModelBackend] = OrderedDict()
        self._lock = threading.Lock()

    def ensure_loaded(self, backend: BaseModelBackend) -> None:
        """Guarantee *backend* is in GPU memory, evicting LRU entries as needed."""
        with self._lock:
            name = backend.name
            if name in self._loaded:
                self._loaded.move_to_end(name)
                return

            while len(self._loaded) >= self._max_loaded:
                if not self._try_evict_one():
                    logger.warning(
                        "All %d loaded model(s) are busy; loading '%s' over the pool limit.",
                        self._max_loaded,
                        name,
                    )
                    break

            self._flush_cuda_cache()
            logger.info("Loading model '%s' into GPU.", name)
            backend.load()
            self._loaded[name] = backend

    def unload_all(self) -> None:
        with self._lock:
            for backend in list(self._loaded.values()):
                try:
                    backend.unload()
                except Exception:
                    logger.exception("Error unloading model '%s'.", backend.name)
            self._loaded.clear()
            self._flush_cuda_cache()

    def _try_evict_one(self) -> bool:
        for name in list(self._loaded.keys()):
            backend = self._loaded[name]
            if not backend.is_busy:
                del self._loaded[name]
                logger.info("Evicting model '%s' from GPU to free memory.", name)
                try:
                    backend.unload()
                except Exception:
                    logger.exception("Error while evicting model '%s'.", name)
                return True
        return False

    @staticmethod
    def _flush_cuda_cache() -> None:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
