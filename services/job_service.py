from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class JobRecord:
    job_id: str
    status: str
    created_at: float
    updated_at: float
    payload: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class JobService:
    def __init__(self, worker_count: int = 1, retention_hours: int = 24):
        self.worker_count = max(1, worker_count)
        self.retention_seconds = max(1, retention_hours) * 3600
        self._queue: queue.Queue[str] = queue.Queue()
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._workers: list[threading.Thread] = []

    def start(self, handler: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        for idx in range(self.worker_count):
            t = threading.Thread(target=self._worker, args=(handler,), daemon=True, name=f"job-worker-{idx}")
            t.start()
            self._workers.append(t)

    def stop(self) -> None:
        self._stop_event.set()

    def submit(self, payload: dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        now = time.time()
        record = JobRecord(job_id=job_id, status="queued", created_at=now, updated_at=now, payload=payload)
        with self._lock:
            self._jobs[job_id] = record
        self._queue.put(job_id)
        return job_id

    def get(self, job_id: str) -> JobRecord | None:
        self.cleanup_expired()
        with self._lock:
            return self._jobs.get(job_id)

    def cleanup_expired(self) -> int:
        threshold = time.time() - self.retention_seconds
        removed = 0
        with self._lock:
            to_delete = [job_id for job_id, rec in self._jobs.items() if rec.updated_at < threshold and rec.status in {"done", "failed"}]
            for job_id in to_delete:
                self._jobs.pop(job_id, None)
                removed += 1
        return removed

    def _worker(self, handler: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        while not self._stop_event.is_set():
            try:
                job_id = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            with self._lock:
                record = self._jobs.get(job_id)
                if record is None:
                    self._queue.task_done()
                    continue
                record.status = "running"
                record.updated_at = time.time()

            try:
                result = handler(record.payload)
            except Exception as exc:
                with self._lock:
                    current = self._jobs.get(job_id)
                    if current is not None:
                        current.status = "failed"
                        current.updated_at = time.time()
                        current.error = {"message": str(exc)}
            else:
                with self._lock:
                    current = self._jobs.get(job_id)
                    if current is not None:
                        current.status = "done"
                        current.updated_at = time.time()
                        current.result = result
            finally:
                self._queue.task_done()
