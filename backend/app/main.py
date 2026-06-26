from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import httpx
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .ai import AIClient
from .config import load_settings
from .live import LiveSessionManager
from .runner import JobRunner
from .storage import JobStore
from .video import VideoProcessor
from .web_search import search_web

settings = load_settings()
store = JobStore(settings.data_dir)
runner = JobRunner(settings, store)
live_manager = LiveSessionManager(settings, store)

app = FastAPI(title="Audio-First Video Understanding Agent")


class UrlJobRequest(BaseModel):
    url: str
    question: str


class AskRequest(BaseModel):
    question: str
    use_web_search: bool = False


class LiveSessionRequest(BaseModel):
    url: str
    question: str = "实时总结直播中正在发生什么，关注主播动作、商品/场景变化、屏幕文字和语音重点。"
    window_seconds: float | None = None
    max_segments: int | None = None

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


@app.post("/api/live/sessions")
def create_live_session(request: LiveSessionRequest) -> dict[str, str]:
    url = request.url.strip()
    question = request.question.strip() or "实时总结直播中正在发生什么。"
    if not _is_http_url(url):
        raise HTTPException(status_code=400, detail="Only http:// and https:// live URLs are supported.")
    session_id = uuid4().hex
    live_manager.create_session(
        session_id=session_id,
        source_url=url,
        question=question,
        max_segments=request.max_segments,
        window_seconds=request.window_seconds,
    )
    return {"session_id": session_id}


@app.get("/api/live/sessions/{session_id}")
def get_live_session(session_id: str) -> dict[str, object]:
    payload = live_manager.public_session(session_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Live session not found.")
    return payload


@app.post("/api/live/sessions/{session_id}/stop")
def stop_live_session(session_id: str) -> dict[str, object]:
    if not live_manager.stop_session(session_id):
        raise HTTPException(status_code=404, detail="Live session not found.")
    return {"ok": True, "session_id": session_id}


@app.get("/api/live/sessions/{session_id}/events")
async def live_session_events(session_id: str) -> StreamingResponse:
    if not live_manager.get_session(session_id):
        raise HTTPException(status_code=404, detail="Live session not found.")

    async def stream():
        last_payload: str | None = None
        while True:
            payload_obj = live_manager.public_session(session_id)
            if not payload_obj:
                break
            payload = json.dumps(payload_obj, ensure_ascii=False)
            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload
            if payload_obj["status"] in {"succeeded", "failed", "stopped"}:
                break
            await asyncio.sleep(0.8)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/live/{session_id}/frames/{filename}")
def get_live_frame(session_id: str, filename: str) -> FileResponse:
    session_dir = (settings.data_dir / "live" / session_id).resolve()
    frame_path = (session_dir / filename).resolve()
    try:
        frame_path.relative_to(session_dir)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Frame not found.") from exc
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found.")
    return FileResponse(frame_path)


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

    result = _augment_result_for_followup(job, state, result, question)
    web_context = search_web(_web_search_queries(question, result), settings) if request.use_web_search else None
    answer = AIClient(settings).answer_followup(question=question, result=result, web_context=web_context)
    return {"job_id": job_id, "question": question, "answer": answer, "web_context": web_context}


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


def _augment_result_for_followup(
    job: dict[str, object],
    state: dict[str, object],
    result: dict[str, object],
    question: str,
) -> dict[str, object]:
    if settings.use_mock_models:
        return result
    if not _followup_needs_visual_probe(question):
        return result
    video_path_value = state.get("video_path") or job.get("upload_path")
    if not video_path_value:
        return result
    video_path = Path(str(video_path_value))
    if not video_path.exists():
        return result

    duration = _safe_float(state.get("duration_seconds") or (result.get("metadata") or {}).get("duration_seconds"), 0.0)
    times = _followup_probe_times(question, result, duration)
    if not times:
        return result

    existing_frames = list(state.get("extracted_frames") or result.get("frames") or [])
    existing_observations = list(state.get("frame_observations") or [])
    observed_names = {str(item.get("filename")) for item in existing_observations if item.get("filename")}
    existing_by_time = {round(_safe_float(frame.get("time"), 0.0), 2): frame for frame in existing_frames if isinstance(frame, dict)}

    processor = VideoProcessor(settings)
    ai = AIClient(settings)
    frames_dir = store.job_dir(str(job["job_id"])) / "frames"
    new_frames = []
    for time in times:
        rounded = round(time, 2)
        existing = existing_by_time.get(rounded)
        if existing:
            if str(existing.get("filename")) not in observed_names:
                new_frames.append(existing)
            continue
        filename = f"followup_{abs(hash(question)) % 100000:05d}_{int(rounded * 100):08d}.jpg"
        output_path = frames_dir / filename
        try:
            processor.extract_frame(video_path, rounded, output_path)
        except Exception:
            continue
        frame = {
            "time": rounded,
            "filename": filename,
            "path": str(output_path),
            "url": f"/api/jobs/{job['job_id']}/frames/{filename}",
            "reason": f"follow-up probe for: {question}",
            "probe": {
                "type": "followup",
                "question": question,
                "expected_visuals": _followup_expected_visuals(question),
            },
        }
        existing_frames.append(frame)
        new_frames.append(frame)

    new_observations = []
    if new_frames:
        try:
            new_observations = ai.observe_frames(
                question=question,
                frames=new_frames[:3],
                audio_world_model=state.get("audio_world_model") or {"timeline": result.get("timeline") or []},
                session_id=f"followup-{job['job_id']}",
            )
        except Exception:
            new_observations = []

    if new_observations:
        merged_observations = sorted(existing_observations + new_observations, key=lambda item: _safe_float(item.get("time"), 0.0))
        state_update = {**state, "extracted_frames": existing_frames, "frame_observations": merged_observations}
        state_update["vision_request_count"] = int(state.get("vision_request_count") or 0) + ai.last_vision_request_count
        store.save_state(str(job["job_id"]), state_update)

        observation_by_filename = {str(obs.get("filename")): obs for obs in merged_observations if obs.get("filename")}
        merged_frames = []
        for frame in existing_frames:
            if not isinstance(frame, dict):
                continue
            observation = observation_by_filename.get(str(frame.get("filename"))) or frame.get("observation")
            merged_frames.append({**frame, "observation": observation})
        result = {
            **result,
            "frames": sorted(merged_frames, key=lambda item: _safe_float(item.get("time"), 0.0)),
            "supplemental_frame_observations": new_observations,
            "metadata": {
                **(result.get("metadata") or {}),
                "vision_request_count": state_update["vision_request_count"],
            },
        }
    return result


def _followup_probe_times(question: str, result: dict[str, object], duration: float, limit: int = 3) -> list[float]:
    explicit_times = _explicit_times_from_question(question, duration)
    if explicit_times:
        times: list[float] = []
        for base in explicit_times:
            for candidate in (base, max(0.0, base - 0.5), base + 0.5):
                candidate = _clamp_time(candidate, duration)
                if all(abs(candidate - existing) >= 0.35 for existing in times):
                    times.append(candidate)
                if len(times) >= limit:
                    return times
        return times

    chunks = AIClient._rank_knowledge_chunks(question, AIClient._video_knowledge_chunks(result))
    times: list[float] = []
    for chunk in chunks:
        value = chunk.get("time")
        if value is None:
            continue
        base = _clamp_time(_safe_float(value, 0.0), duration)
        for candidate in (base, base + 1.0):
            candidate = _clamp_time(candidate, duration)
            if all(abs(candidate - existing) >= 0.75 for existing in times):
                times.append(candidate)
            if len(times) >= limit:
                return times
    if not times:
        for frame in result.get("frames") or []:
            if not isinstance(frame, dict):
                continue
            candidate = _clamp_time(_safe_float(frame.get("time"), 0.0), duration)
            if all(abs(candidate - existing) >= 0.75 for existing in times):
                times.append(candidate)
            if len(times) >= limit:
                break
    return times


def _explicit_times_from_question(question: str, duration: float) -> list[float]:
    import re

    text = str(question or "")
    times: list[float] = []
    occupied_spans: list[tuple[int, int]] = []
    for match in re.finditer(r"(?<!\d)(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?!\d)", text):
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        times.append(_clamp_time(float(hours * 3600 + minutes * 60 + seconds), duration))
        occupied_spans.append(match.span())
    for match in re.finditer(r"(?<!\d)(\d{1,3})\s*分(?:钟)?\s*(\d{1,2})?\s*秒?", text):
        minutes = int(match.group(1))
        seconds = int(match.group(2) or 0)
        times.append(_clamp_time(float(minutes * 60 + seconds), duration))
        occupied_spans.append(match.span())
    for match in re.finditer(r"(?<!\d)(\d{1,5})\s*秒", text):
        if any(match.start() >= start and match.end() <= end for start, end in occupied_spans):
            continue
        times.append(_clamp_time(float(match.group(1)), duration))
    deduped: list[float] = []
    for time in times:
        if all(abs(time - existing) >= 0.35 for existing in deduped):
            deduped.append(time)
    return deduped


def _followup_expected_visuals(question: str) -> list[str]:
    text = question.lower()
    if any(token in question for token in ("画质", "成像", "动态范围", "高光", "暗部", "低光", "iso")):
        return ["样片画面", "高光暗部细节", "低光测试", "字幕或参数说明"]
    if any(token in question for token in ("长焦", "镜头", "三倍", "60")):
        return ["长焦样片", "镜头/视角切换", "拍摄距离变化", "字幕说明"]
    if any(token in question for token in ("发热", "稳定", "体验")):
        return ["机身操作", "稳定性测试", "发热测试", "使用场景"]
    if any(token in question for token in ("买", "推荐", "值得", "选", "哪一代")):
        return ["结论字幕", "样片对比", "优缺点说明", "价格或购买建议"]
    if "visual" in text:
        return ["visible evidence relevant to the question"]
    return ["与追问相关的画面细节", "字幕文字", "人物动作", "场景变化"]


def _followup_needs_visual_probe(question: str) -> bool:
    compact = "".join(str(question or "").lower().split())
    visual_terms = (
        "画面",
        "看到",
        "看见",
        "长什么样",
        "外观",
        "字幕",
        "文字",
        "截图",
        "镜头里",
        "这一帧",
        "那一帧",
        "场景",
        "动作",
        "样片",
        "展示",
        "对比画面",
        "视觉",
        "可见",
        "visible",
        "frame",
        "screenshot",
    )
    time_reference = any(char.isdigit() for char in compact) and (":" in compact or "秒" in compact or "分钟" in compact)
    if time_reference:
        return True
    if any(term in compact for term in visual_terms):
        return True
    non_visual_terms = (
        "总结",
        "概括",
        "主要内容",
        "推荐",
        "买",
        "哪一代",
        "哪款",
        "值得",
        "区别",
        "升级",
        "对比",
        "怎么选",
        "为什么",
        "结论",
    )
    if any(term in compact for term in non_visual_terms):
        return False
    return False


def _web_search_queries(question: str, result: dict[str, object]) -> list[str]:
    question_text = " ".join(str(question or "").split())
    queries: list[str] = []
    external_terms = _external_search_terms(question_text)
    for term in external_terms:
        queries.extend(
            [
                f"{term} specs dynamic range",
                f"{term} review specs",
            ]
        )
    topic_parts: list[str] = []
    answer = result.get("answer") or {}
    if isinstance(answer, dict):
        topic_parts.append(str(answer.get("direct_answer") or ""))
    for segment in (result.get("transcript_segments") or [])[:8]:
        if isinstance(segment, dict):
            topic_parts.append(str(segment.get("text") or ""))
    topic = " ".join(" ".join(topic_parts).split())[:180]
    if external_terms:
        for term in external_terms[:2]:
            queries.append(f"{term} vs DJI Pocket 4P")
    if topic:
        queries.append(f"{question_text} {topic}")
    queries.append(question_text)
    deduped = []
    for query in queries:
        clean = " ".join(query.split())
        if clean and clean not in deduped:
            deduped.append(clean)
    return deduped[:5]


def _external_search_terms(question: str) -> list[str]:
    text = str(question or "")
    terms: list[str] = []
    patterns = [
        r"(?i)(insta360\s+[a-z0-9][a-z0-9\- ]{0,32})",
        r"(?i)(dji\s+[a-z0-9][a-z0-9\- ]{0,32})",
        r"(?i)(gopro\s+[a-z0-9][a-z0-9\- ]{0,32})",
        r"(?i)(pocket\s*\d+[a-z]?)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            term = " ".join(match.group(1).split()).strip(" ，。；;?？")
            if term and term.lower() not in {item.lower() for item in terms}:
                terms.append(term)
    if "luna" in text.lower() and not any("luna" in term.lower() for term in terms):
        terms.append("Insta360 Luna")
    return terms[:3]


def _web_search_query(question: str, result: dict[str, object]) -> str:
    return _web_search_queries(question, result)[0]


def _clamp_time(value: float, duration: float) -> float:
    if duration <= 0:
        return max(0.0, value)
    return max(0.0, min(duration, value))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


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

    ytdlp_errors = []
    for candidate_url in _video_page_url_candidates(url):
        try:
            _download_with_ytdlp(candidate_url, output_path)
            return
        except Exception as exc:
            ytdlp_errors.append(f"{candidate_url}: {exc}")
            if output_path.exists():
                output_path.unlink(missing_ok=True)
    message = "; ".join(ytdlp_errors) or "yt-dlp did not run."
    if "yt-dlp" not in message.lower():
        message = f"yt-dlp download failed: {message}"
    if 'direct_error' in locals() and direct_error:
        message = f"Direct URL download failed: {direct_error}. {message}"
    raise RuntimeError(_friendly_ytdlp_error(message))


def _download_direct_video_url(url: str, output_path: Path) -> None:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        )
    }
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
        "format": "bv*[ext=mp4]+ba[ext=m4a]/bv*+ba/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "noplaylist": True,
        "retries": 2,
        "fragment_retries": 2,
        "socket_timeout": 30,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    }
    cookies_file = os.getenv("YTDLP_COOKIES_FILE") or os.getenv("DOUYIN_COOKIES_FILE")
    if cookies_file:
        options["cookiefile"] = cookies_file
    cookies_from_browser = os.getenv("YTDLP_COOKIES_FROM_BROWSER")
    if cookies_from_browser:
        options["cookiesfrombrowser"] = tuple(part.strip() for part in cookies_from_browser.split(":") if part.strip())
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


def _video_page_url_candidates(url: str) -> list[str]:
    candidates = [url]
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    query = parse_qs(parsed.query)
    if "douyin.com" in host:
        ids = []
        for key in ("modal_id", "aweme_id", "video_id", "item_id"):
            ids.extend(value for value in query.get(key, []) if value)
        path_parts = [part for part in parsed.path.split("/") if part]
        for marker in ("video", "note"):
            if marker in path_parts:
                index = path_parts.index(marker)
                if index + 1 < len(path_parts):
                    ids.append(path_parts[index + 1])
        for video_id in ids:
            if video_id.isdigit():
                candidates.extend(
                    [
                        f"https://www.douyin.com/video/{video_id}",
                        f"https://www.douyin.com/note/{video_id}",
                        f"https://www.iesdouyin.com/share/video/{video_id}/",
                    ]
                )
    deduped = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _friendly_ytdlp_error(message: str) -> str:
    if "Fresh cookies" in message or "cookies" in message.lower():
        return (
            f"{message}. Douyin requires fresh cookies for this page. "
            "Set DOUYIN_COOKIES_FILE/YTDLP_COOKIES_FILE to a Netscape cookies.txt exported from a logged-in browser, "
            "or set YTDLP_COOKIES_FROM_BROWSER=edge/chrome/firefox after closing that browser."
        )
    return message


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
