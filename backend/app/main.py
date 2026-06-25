from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .ai import AIClient
from .config import load_settings
from .runner import JobRunner
from .storage import JobStore

settings = load_settings()
store = JobStore(settings.data_dir)
runner = JobRunner(settings, store)

app = FastAPI(title="Audio-First Video Understanding Agent")


class UrlJobRequest(BaseModel):
    url: str
    question: str


class AskRequest(BaseModel):
    question: str

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
        "fast_mode": settings.fast_mode,
        "vision_provider": settings.vision_provider,
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


@app.post("/api/jobs/url")
async def create_url_job(request: UrlJobRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    question = request.question.strip()
    url = request.url.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")
    if not _is_http_url(url):
        raise HTTPException(status_code=400, detail="Only http:// and https:// video URLs are supported.")

    job_id = uuid4().hex
    upload_dir = store.upload_dir(job_id)
    upload_path = upload_dir / "source.mp4"
    store.create_job(job_id, question, upload_path)
    background_tasks.add_task(_download_url_and_run_job, job_id, url, question, upload_path)
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


@app.get("/api/jobs/{job_id}/partial")
def get_partial(job_id: str) -> dict[str, object]:
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    state = store.load_state(job_id) or {}
    if job.get("result"):
        return {**job["result"], "partial": False}
    return _state_to_partial(job, state)


@app.post("/api/jobs/{job_id}/ask")
def ask_job(job_id: str, request: AskRequest) -> dict[str, object]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    state = store.load_state(job_id) or {}
    result = job.get("result") or _state_to_partial(job, state)
    if not state and not job.get("result"):
        raise HTTPException(status_code=409, detail="No analysis context is available yet.")

    answer = AIClient(settings).answer_followup(question=question, result=result)
    return {"job_id": job_id, "question": question, "answer": answer}


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


@app.get("/api/jobs/{job_id}/source")
def get_source_video(job_id: str) -> FileResponse:
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    upload_path_value = job.get("upload_path")
    if not upload_path_value:
        raise HTTPException(status_code=404, detail="Source video not found.")
    upload_dir = store.upload_dir(job_id).resolve()
    video_path = Path(str(upload_path_value)).resolve()
    try:
        video_path.relative_to(upload_dir)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Source video not found.") from exc
    if not video_path.exists():
        raise HTTPException(status_code=409, detail="Source video is still being downloaded.")
    return FileResponse(video_path, media_type="video/mp4", filename=video_path.name)


def _public_job(job: dict[str, object]) -> dict[str, object]:
    keys = ["job_id", "question", "status", "progress", "current_node", "error", "created_at", "updated_at"]
    return {key: job.get(key) for key in keys}


def _state_to_partial(job: dict[str, object], state: dict[str, object]) -> dict[str, object]:
    observations = state.get("frame_observations") if isinstance(state, dict) else []
    observation_map = {
        str(obs.get("filename")): obs
        for obs in observations or []
        if isinstance(obs, dict) and obs.get("filename")
    }
    frames = []
    for frame in state.get("extracted_frames", []) or []:
        if not isinstance(frame, dict):
            continue
        observation = observation_map.get(str(frame.get("filename")))
        frames.append({**frame, "observation": observation})

    audio_world_model = state.get("audio_world_model", {}) if isinstance(state, dict) else {}
    timeline = audio_world_model.get("timeline", []) if isinstance(audio_world_model, dict) else []
    return {
        "job_id": job.get("job_id"),
        "question": job.get("question"),
        "partial": True,
        "status": job.get("status"),
        "progress": job.get("progress"),
        "current_node": job.get("current_node"),
        "answer": state.get("answer") or None,
        "timeline": timeline,
        "transcript_segments": state.get("transcript_segments", []) or [],
        "audio_world_model": audio_world_model if isinstance(audio_world_model, dict) else {},
        "frames": frames,
        "prediction_checks": state.get("prediction_checks", []) or [],
        "evidence": [],
        "metadata": {
            "duration_seconds": state.get("duration_seconds"),
            "fps": state.get("fps"),
            "width": state.get("width"),
            "height": state.get("height"),
            "has_audio": state.get("has_audio"),
            "mock_mode": settings.use_mock_models,
            "fast_mode": settings.fast_mode,
            "vision_provider": settings.vision_provider,
            "vision_request_count": state.get("vision_request_count", 0),
        },
        "transcription_status": state.get("transcription_status", {}) or {},
    }


def _download_url_and_run_job(job_id: str, url: str, question: str, upload_path: Path) -> None:
    try:
        store.update_job(job_id, status="running", current_node="download_url", progress=2)
        _download_video_url(url, upload_path)
        runner.run_job(job_id, str(upload_path), question)
    except Exception as exc:
        store.update_job(job_id, status="failed", current_node="failed", error=str(exc))


def _download_video_url(url: str, output_path: Path) -> None:
    try:
        _download_direct_video_url(url, output_path)
        return
    except _NotDirectVideoUrl as exc:
        direct_error = str(exc)
    except Exception as exc:
        direct_error = str(exc)

    try:
        _download_with_ytdlp(url, output_path)
    except Exception as exc:
        message = str(exc)
        if "yt-dlp" not in message.lower():
            message = f"yt-dlp download failed: {message}"
        if 'direct_error' in locals() and direct_error:
            message = f"Direct URL download failed: {direct_error}. {message}"
        raise RuntimeError(message) from exc


def _download_direct_video_url(url: str, output_path: Path) -> None:
    headers = {"User-Agent": "audio-first-video-agent/0.1"}
    timeout = httpx.Timeout(connect=10.0, read=90.0, write=10.0, pool=10.0)
    with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if not _looks_like_media_response(url, content_type):
                raise _NotDirectVideoUrl(
                    "URL looks like a web page rather than a direct video file. "
                    "Install yt-dlp for Bilibili/Douyin/YouTube-style page URLs."
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = output_path.with_suffix(output_path.suffix + ".download")
            with temp_path.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            temp_path.replace(output_path)


def _download_with_ytdlp(url: str, output_path: Path) -> None:
    try:
        from yt_dlp import YoutubeDL
    except Exception as exc:
        raise RuntimeError(
            "yt-dlp is not installed, so page URLs are not available yet. "
            "Direct mp4/webm/mov URLs still work."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    options = {
        "outtmpl": str(output_path),
        "format": "bv*+ba/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "noplaylist": True,
        "retries": 2,
    }
    with YoutubeDL(options) as downloader:
        downloader.download([url])
    if not output_path.exists():
        candidates = sorted(
            output_path.parent.glob(f"{output_path.stem}.*"),
            key=lambda item: item.stat().st_size if item.exists() else 0,
            reverse=True,
        )
        for candidate in candidates:
            if candidate.name.endswith(".download") or candidate == output_path:
                continue
            candidate.replace(output_path)
            break
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("yt-dlp completed but did not create a readable video file.")


def _is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _looks_like_media_response(url: str, content_type: str) -> bool:
    path = urlparse(url).path.lower()
    if path.endswith((".mp4", ".webm", ".mov", ".mkv", ".m4v")):
        return True
    if content_type.startswith("video/"):
        return True
    if content_type in {"application/octet-stream", "binary/octet-stream"}:
        return True
    return False


class _NotDirectVideoUrl(RuntimeError):
    pass
