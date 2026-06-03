from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    expires_at: float
    value: dict[str, Any]


class CacheService:
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = max(1, ttl_seconds)
        self._entries: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> dict[str, Any] | None:
        now = time.time()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            if entry.expires_at < now:
                self._entries.pop(key, None)
                self.misses += 1
                return None
            self.hits += 1
            return entry.value

    def set(self, key: str, value: dict[str, Any]) -> None:
        expires_at = time.time() + self.ttl_seconds
        with self._lock:
            self._entries[key] = CacheEntry(expires_at=expires_at, value=value)

    def cleanup_expired(self) -> int:
        now = time.time()
        removed = 0
        with self._lock:
            for key in list(self._entries.keys()):
                if self._entries[key].expires_at < now:
                    self._entries.pop(key, None)
                    removed += 1
        return removed

    def size(self) -> int:
        with self._lock:
            return len(self._entries)
