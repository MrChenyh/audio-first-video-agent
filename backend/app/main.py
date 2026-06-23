from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from .config import load_settings
from .runner import JobRunner
from .storage import JobStore

settings = load_settings()
store = JobStore(settings.data_dir)
runner = JobRunner(settings, store)

app = FastAPI(title="Audio-First Video Understanding Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    store.init()


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "ok": True,
        "mock_mode": settings.use_mock_models,
        "data_dir": str(settings.data_dir),
    }


@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    question: str = Form(...),
) -> dict[str, str]:
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")

    job_id = uuid4().hex
    upload_dir = store.upload_dir(job_id)
    upload_path = upload_dir / "source.mp4"

    try:
        with upload_path.open("wb") as handle:
            shutil.copyfileobj(video.file, handle)
    finally:
        await video.close()

    store.create_job(job_id, question.strip(), upload_path)
    background_tasks.add_task(runner.run_job, job_id, str(upload_path), question.strip())
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _public_job(job)


@app.get("/api/jobs/{job_id}/result")
def get_result(job_id: str) -> dict[str, object]:
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if not job.get("result"):
        raise HTTPException(status_code=409, detail="Job result is not ready.")
    return job["result"]


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    if not store.get_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found.")

    async def stream():
        last_payload: str | None = None
        while True:
            job = store.get_job(job_id)
            if not job:
                break
            payload = json.dumps(_public_job(job), ensure_ascii=False)
            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload
            if job["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.8)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/jobs/{job_id}/frames/{filename}")
def get_frame(job_id: str, filename: str) -> FileResponse:
    frames_dir = (store.job_dir(job_id) / "frames").resolve()
    frame_path = (frames_dir / filename).resolve()
    if not str(frame_path).startswith(str(frames_dir)) or not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found.")
    return FileResponse(frame_path)


def _public_job(job: dict[str, object]) -> dict[str, object]:
    keys = ["job_id", "question", "status", "progress", "current_node", "error", "created_at", "updated_at"]
    return {key: job.get(key) for key in keys}
