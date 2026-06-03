from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


TERMINAL_STATUSES = {"done", "failed", "canceled"}


@dataclass
class JobRecord:
    job_id: str
    status: str
    created_at: float
    updated_at: float
    payload: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    cancel_requested: bool = False
    webhook_url: str | None = None


class JobService:
    def __init__(self, db_path: str, worker_count: int = 1, retention_hours: int = 24):
        self.db_path = db_path
        self.worker_count = max(1, worker_count)
        self.retention_seconds = max(1, retention_hours) * 3600
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._workers: list[threading.Thread] = []
        self._handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None
        self._completion_hook: Callable[[JobRecord], None] | None = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _conn(self):
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    result_json TEXT,
                    error_json TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    webhook_url TEXT
                )
                """
            )
            conn.commit()

    def start(
        self,
        handler: Callable[[dict[str, Any]], dict[str, Any]],
        completion_hook: Callable[[JobRecord], None] | None = None,
    ) -> None:
        self._handler = handler
        self._completion_hook = completion_hook
        self._init_db()
        self._requeue_unfinished_jobs()
        for idx in range(self.worker_count):
            t = threading.Thread(target=self._worker, daemon=True, name=f"job-worker-{idx}")
            t.start()
            self._workers.append(t)

    def stop(self) -> None:
        self._stop_event.set()

    def submit(self, payload: dict[str, Any], webhook_url: str | None = None) -> str:
        job_id = str(uuid.uuid4())
        now = time.time()
        payload_json = json.dumps(payload, ensure_ascii=False)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, status, created_at, updated_at, payload_json, webhook_url)
                VALUES (?, 'queued', ?, ?, ?, ?)
                """,
                (job_id, now, now, payload_json, webhook_url),
            )
            conn.commit()
        self._queue.put(job_id)
        return job_id

    def get(self, job_id: str) -> JobRecord | None:
        self.cleanup_expired()
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def cancel(self, job_id: str) -> JobRecord | None:
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            status = row["status"]
            if status in TERMINAL_STATUSES:
                return self._row_to_record(row)

            now = time.time()
            if status == "queued":
                new_status = "canceled"
            else:
                new_status = "canceling"

            conn.execute(
                "UPDATE jobs SET status = ?, cancel_requested = 1, updated_at = ? WHERE job_id = ?",
                (new_status, now, job_id),
            )
            conn.commit()
            updated_row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            return self._row_to_record(updated_row)

    def stats(self) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as c FROM jobs GROUP BY status"
            ).fetchall()
        by_status = {row["status"]: int(row["c"]) for row in rows}
        return {
            "queue_size": self._queue.qsize(),
            "queued_jobs": by_status.get("queued", 0),
            "running_jobs": by_status.get("running", 0),
            "canceling_jobs": by_status.get("canceling", 0),
        }

    def cleanup_expired(self) -> int:
        threshold = time.time() - self.retention_seconds
        with self._conn() as conn:
            cur = conn.execute(
                """
                DELETE FROM jobs
                WHERE updated_at < ? AND status IN ('done', 'failed', 'canceled')
                """,
                (threshold,),
            )
            conn.commit()
            return cur.rowcount if cur.rowcount is not None else 0

    def _requeue_unfinished_jobs(self) -> None:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT job_id FROM jobs WHERE status IN ('queued', 'running', 'canceling') ORDER BY created_at ASC"
            ).fetchall()
            # Any running job from previous process is now queued again.
            conn.execute(
                "UPDATE jobs SET status = 'queued', updated_at = ? WHERE status IN ('running', 'canceling')",
                (time.time(),),
            )
            conn.commit()
        for row in rows:
            self._queue.put(row["job_id"])

    def _row_to_record(self, row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            payload=json.loads(row["payload_json"]) if row["payload_json"] else {},
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error=json.loads(row["error_json"]) if row["error_json"] else None,
            cancel_requested=bool(row["cancel_requested"]),
            webhook_url=row["webhook_url"],
        )

    def _set_status(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        status: str,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        now = time.time()
        conn.execute(
            """
            UPDATE jobs
            SET status = ?, updated_at = ?, result_json = ?, error_json = ?
            WHERE job_id = ?
            """,
            (
                status,
                now,
                json.dumps(result, ensure_ascii=False) if result is not None else None,
                json.dumps(error, ensure_ascii=False) if error is not None else None,
                job_id,
            ),
        )

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                job_id = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if self._handler is None:
                self._queue.task_done()
                continue

            final_record: JobRecord | None = None
            try:
                with self._lock, self._conn() as conn:
                    row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
                    if row is None:
                        self._queue.task_done()
                        continue
                    record = self._row_to_record(row)
                    if record.status in TERMINAL_STATUSES:
                        self._queue.task_done()
                        continue
                    if record.cancel_requested and record.status in {"queued", "canceling"}:
                        self._set_status(conn, job_id, "canceled", error={"message": "Job canceled."})
                        conn.commit()
                        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
                        final_record = self._row_to_record(row) if row else None
                        self._queue.task_done()
                        if final_record and self._completion_hook:
                            self._completion_hook(final_record)
                        continue

                    self._set_status(conn, job_id, "running")
                    conn.commit()
                    row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
                    running_record = self._row_to_record(row) if row else record
            except sqlite3.OperationalError:
                self._init_db()
                self._queue.task_done()
                continue

            try:
                result = self._handler(running_record.payload)
            except Exception as exc:
                try:
                    with self._lock, self._conn() as conn:
                        self._set_status(conn, job_id, "failed", error={"message": str(exc)})
                        conn.commit()
                        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
                        final_record = self._row_to_record(row) if row else None
                except sqlite3.OperationalError:
                    self._init_db()
            else:
                try:
                    with self._lock, self._conn() as conn:
                        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
                        cancel_requested = bool(row["cancel_requested"]) if row else False
                        if cancel_requested:
                            self._set_status(conn, job_id, "canceled", error={"message": "Job canceled."})
                        else:
                            self._set_status(conn, job_id, "done", result=result)
                        conn.commit()
                        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
                        final_record = self._row_to_record(row) if row else None
                except sqlite3.OperationalError:
                    self._init_db()
            finally:
                self._queue.task_done()
 
            if final_record and self._completion_hook:
                self._completion_hook(final_record)
