from __future__ import annotations

import threading
import time


class MetricsService:
    def __init__(self):
        self.started_at = time.time()
        self._lock = threading.Lock()
        self.requests_total = 0
        self.errors_total = 0
        self.inference_count = 0
        self.inference_latency_sum_ms = 0.0
        self.auto_queued_total = 0
        self.jobs_created_total = 0

    def inc_request(self):
        with self._lock:
            self.requests_total += 1

    def inc_error(self):
        with self._lock:
            self.errors_total += 1

    def observe_inference_latency(self, latency_ms: float):
        with self._lock:
            self.inference_count += 1
            self.inference_latency_sum_ms += float(latency_ms)

    def inc_auto_queued(self):
        with self._lock:
            self.auto_queued_total += 1

    def inc_jobs_created(self):
        with self._lock:
            self.jobs_created_total += 1

    def snapshot(self) -> dict:
        with self._lock:
            uptime = max(1e-6, time.time() - self.started_at)
            return {
                "uptime_seconds": uptime,
                "requests_total": self.requests_total,
                "errors_total": self.errors_total,
                "error_rate": self.errors_total / self.requests_total if self.requests_total else 0.0,
                "qps": self.requests_total / uptime,
                "inference_count": self.inference_count,
                "inference_latency_sum_ms": self.inference_latency_sum_ms,
                "inference_latency_avg_ms": (
                    self.inference_latency_sum_ms / self.inference_count if self.inference_count else 0.0
                ),
                "auto_queued_total": self.auto_queued_total,
                "jobs_created_total": self.jobs_created_total,
            }
