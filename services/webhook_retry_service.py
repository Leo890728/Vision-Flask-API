from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass(order=True)
class _WebhookTask:
    due_at: float
    seq: int
    job_id: str = field(compare=False)
    url: str = field(compare=False)
    payload: dict = field(compare=False)
    secret: str | None = field(compare=False, default=None)
    attempt: int = field(compare=False, default=1)


class WebhookRetryService:
    def __init__(self, max_retries: int = 3, base_delay_seconds: float = 1.0):
        self.max_retries = max(1, max_retries)
        self.base_delay_seconds = max(0.01, float(base_delay_seconds))
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._heap: list[_WebhookTask] = []
        self._seq = 0
        self._stop_event = threading.Event()
        self._sender: Callable[[str, dict, str | None], None] | None = None
        self._thread: threading.Thread | None = None

        self.delivered_total = 0
        self.failed_total = 0
        self.retried_total = 0

    def start(self, sender: Callable[[str, dict, str | None], None]) -> None:
        self._sender = sender
        self._thread = threading.Thread(target=self._run, daemon=True, name="webhook-retry-worker")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._cv:
            self._cv.notify_all()

    def submit(self, job_id: str, url: str, payload: dict, secret: str | None = None, delay_seconds: float = 0.0) -> None:
        with self._cv:
            self._seq += 1
            task = _WebhookTask(
                due_at=time.time() + max(0.0, delay_seconds),
                seq=self._seq,
                job_id=job_id,
                url=url,
                payload=payload,
                secret=secret,
                attempt=1,
            )
            heapq.heappush(self._heap, task)
            self._cv.notify()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "pending": len(self._heap),
                "delivered_total": self.delivered_total,
                "failed_total": self.failed_total,
                "retried_total": self.retried_total,
            }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            task: _WebhookTask | None = None
            with self._cv:
                while not self._heap and not self._stop_event.is_set():
                    self._cv.wait(timeout=1.0)
                if self._stop_event.is_set():
                    return
                now = time.time()
                top = self._heap[0]
                if top.due_at > now:
                    self._cv.wait(timeout=min(1.0, top.due_at - now))
                    continue
                task = heapq.heappop(self._heap)

            if task is None or self._sender is None:
                continue

            try:
                self._sender(task.url, task.payload, task.secret)
            except Exception:
                if task.attempt < self.max_retries:
                    next_attempt = task.attempt + 1
                    delay = self.base_delay_seconds * (2 ** (task.attempt - 1))
                    with self._cv:
                        self._seq += 1
                        heapq.heappush(
                            self._heap,
                            _WebhookTask(
                                due_at=time.time() + delay,
                                seq=self._seq,
                                job_id=task.job_id,
                                url=task.url,
                                payload=task.payload,
                                secret=task.secret,
                                attempt=next_attempt,
                            ),
                        )
                        self.retried_total += 1
                        self._cv.notify()
                else:
                    with self._lock:
                        self.failed_total += 1
            else:
                with self._lock:
                    self.delivered_total += 1
