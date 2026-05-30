from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from functools import wraps

from flask import Request, request

from errors import APIError


class InMemoryRateLimiter:
    def __init__(self, limit_per_minute: int):
        self.limit_per_minute = max(1, limit_per_minute)
        self._records: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        start = now - 60.0
        with self._lock:
            q = self._records[key]
            while q and q[0] < start:
                q.popleft()
            if len(q) >= self.limit_per_minute:
                return False
            q.append(now)
            return True


def _client_key(req: Request) -> str:
    forwarded_for = req.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return req.remote_addr or "unknown"


def apply_rate_limit(limiter: InMemoryRateLimiter):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = _client_key(request)
            if not limiter.allow(key):
                raise APIError("RATE_LIMIT_EXCEEDED", "Rate limit exceeded.", 429)
            return func(*args, **kwargs)

        return wrapper

    return decorator
