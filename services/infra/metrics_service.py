from __future__ import annotations

import threading
import time
from collections import defaultdict


class MetricsService:
    def __init__(self):
        self.started_at = time.time()
        self._lock = threading.Lock()
        self.requests_total = 0
        self.errors_total = 0
        self.auto_queued_total = 0
        self.jobs_created_total = 0
        # Inference counters are keyed by (task, model) so detection and
        # segmentation (and each concrete model) are tracked independently
        # instead of being merged into a single "sam3" bucket.
        self._inference_count: dict[tuple[str, str], int] = defaultdict(int)
        self._inference_latency_sum_ms: dict[tuple[str, str], float] = defaultdict(float)

    def inc_request(self):
        with self._lock:
            self.requests_total += 1

    def inc_error(self):
        with self._lock:
            self.errors_total += 1

    def observe_inference_latency(self, latency_ms: float, *, task: str, model: str):
        with self._lock:
            key = (task, model)
            self._inference_count[key] += 1
            self._inference_latency_sum_ms[key] += float(latency_ms)

    def inc_auto_queued(self):
        with self._lock:
            self.auto_queued_total += 1

    def inc_jobs_created(self):
        with self._lock:
            self.jobs_created_total += 1

    def snapshot(self) -> dict:
        with self._lock:
            uptime = max(1e-6, time.time() - self.started_at)
            inference = [
                {
                    "task": task,
                    "model": model,
                    "count": count,
                    "latency_sum_ms": self._inference_latency_sum_ms[(task, model)],
                    "latency_avg_ms": (
                        self._inference_latency_sum_ms[(task, model)] / count if count else 0.0
                    ),
                }
                for (task, model), count in sorted(self._inference_count.items())
            ]
            return {
                "uptime_seconds": uptime,
                "requests_total": self.requests_total,
                "errors_total": self.errors_total,
                "error_rate": self.errors_total / self.requests_total if self.requests_total else 0.0,
                "qps": self.requests_total / uptime,
                "inference": inference,
                "auto_queued_total": self.auto_queued_total,
                "jobs_created_total": self.jobs_created_total,
            }
