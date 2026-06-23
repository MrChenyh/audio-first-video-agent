from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import JobStatus


class JobStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.uploads_dir = data_dir / "uploads"
        self.jobs_dir = data_dir / "jobs"
        self.db_path = data_dir / "app.db"
        self._lock = threading.RLock()

    def init(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    current_node TEXT NOT NULL DEFAULT 'queued',
                    error TEXT,
                    result_json TEXT,
                    upload_path TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def create_job(self, job_id: str, question: str, upload_path: Path) -> None:
        now = self._now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, question, status, progress, current_node, upload_path, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, question, "queued", 0, "queued", str(upload_path), now, now),
            )
            conn.commit()

    def update_job(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        progress: int | None = None,
        current_node: str | None = None,
        error: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        fields: list[str] = []
        values: list[Any] = []
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if progress is not None:
            fields.append("progress = ?")
            values.append(max(0, min(100, int(progress))))
        if current_node is not None:
            fields.append("current_node = ?")
            values.append(current_node)
        if error is not None:
            fields.append("error = ?")
            values.append(error)
        if result is not None:
            fields.append("result_json = ?")
            values.append(json.dumps(result, ensure_ascii=False))
        fields.append("updated_at = ?")
        values.append(self._now())
        values.append(job_id)

        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?", values)
            conn.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        payload = dict(row)
        if payload.get("result_json"):
            payload["result"] = json.loads(payload["result_json"])
        payload.pop("result_json", None)
        return payload

    def get_result(self, job_id: str) -> dict[str, Any] | None:
        job = self.get_job(job_id)
        if not job:
            return None
        return job.get("result")

    def job_dir(self, job_id: str) -> Path:
        path = self.jobs_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def upload_dir(self, job_id: str) -> Path:
        path = self.uploads_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def state_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "state.json"

    def save_state(self, job_id: str, state: dict[str, Any]) -> None:
        state_path = self.state_path(job_id)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_state(self, job_id: str) -> dict[str, Any] | None:
        state_path = self.state_path(job_id)
        if not state_path.exists():
            return None
        return json.loads(state_path.read_text(encoding="utf-8"))
