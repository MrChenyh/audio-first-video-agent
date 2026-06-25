from __future__ import annotations

import html
import json
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from .ai import AIClient
from .config import Settings
from .storage import JobStore
from .video import VideoProcessor


LIVE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Referer": "https://live.douyin.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


@dataclass
class LiveSession:
    session_id: str
    source_url: str
    question: str
    status: str = "queued"
    current_node: str = "queued"
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    resolved_url: str | None = None
    segments: list[dict[str, Any]] = field(default_factory=list)
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)


class LiveSessionManager:
    def __init__(self, settings: Settings, store: JobStore):
        self.settings = settings
        self.store = store
        self._lock = threading.RLock()
        self._sessions: dict[str, LiveSession] = {}
        self._threads: dict[str, threading.Thread] = {}

    def create_session(
        self,
        *,
        session_id: str,
        source_url: str,
        question: str,
        max_segments: int | None = None,
        window_seconds: float | None = None,
    ) -> LiveSession:
        session = LiveSession(session_id=session_id, source_url=source_url, question=question)
        with self._lock:
            self._sessions[session_id] = session
        thread = threading.Thread(
            target=self._run_session,
            args=(session_id, max_segments, window_seconds),
            daemon=True,
        )
        self._threads[session_id] = thread
        thread.start()
        return session

    def get_session(self, session_id: str) -> LiveSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def stop_session(self, session_id: str) -> bool:
        session = self.get_session(session_id)
        if not session:
            return False
        session.stop_event.set()
        self._mark(session, status="stopping", current_node="stopping")
        return True

    def public_session(self, session_id: str) -> dict[str, Any] | None:
        session = self.get_session(session_id)
        if not session:
            return None
        return self._public(session)

    def _run_session(self, session_id: str, max_segments: int | None, window_seconds: float | None) -> None:
        session = self.get_session(session_id)
        if not session:
            return
        try:
            self._mark(session, status="running", current_node="resolve_stream")
            resolved = resolve_live_stream_url(session.source_url)
            session.resolved_url = resolved
            self._mark(session, current_node="capture_loop")

            processor = VideoProcessor(self.settings)
            ffmpeg = processor.ffmpeg
            if not ffmpeg:
                raise RuntimeError("ffmpeg was not found. Install FFmpeg or set FFMPEG_PATH.")
            ai = AIClient(self.settings)
            session_dir = self.store.data_dir / "live" / session.session_id
            session_dir.mkdir(parents=True, exist_ok=True)

            segment_limit = max_segments if max_segments is not None else self.settings.live_max_segments
            window = max(1.0, window_seconds or self.settings.live_window_seconds)
            index = 0
            while not session.stop_event.is_set():
                if segment_limit and index >= segment_limit:
                    self._mark(session, status="succeeded", current_node="complete")
                    break
                started = time.time()
                segment = self._capture_and_analyze_segment(
                    session=session,
                    ai=ai,
                    processor=processor,
                    ffmpeg=ffmpeg,
                    source_url=resolved,
                    session_dir=session_dir,
                    segment_index=index,
                    window_seconds=window,
                )
                segment["elapsed_seconds"] = round(time.time() - started, 2)
                with self._lock:
                    session.segments.append(segment)
                    session.updated_at = time.time()
                    session.current_node = "streaming"
                index += 1
            if session.status not in {"succeeded", "failed"}:
                self._mark(session, status="stopped", current_node="stopped")
        except Exception as exc:
            self._mark(session, status="failed", current_node="failed", error=str(exc))

    def _capture_and_analyze_segment(
        self,
        *,
        session: LiveSession,
        ai: AIClient,
        processor: VideoProcessor,
        ffmpeg: str,
        source_url: str,
        session_dir: Path,
        segment_index: int,
        window_seconds: float,
    ) -> dict[str, Any]:
        self._mark(session, current_node=f"capture_segment {segment_index + 1}")
        capture_started = time.time()
        if self.settings.live_fast_capture:
            audio_path, frame_path, capture_info = self._capture_fast_audio_and_frame(
                ffmpeg=ffmpeg,
                source_url=source_url,
                session_dir=session_dir,
                segment_index=segment_index,
                window_seconds=window_seconds,
            )
        else:
            audio_path, frame_path, capture_info = self._capture_compat_segment(
                processor=processor,
                ffmpeg=ffmpeg,
                source_url=source_url,
                session_dir=session_dir,
                segment_index=segment_index,
                window_seconds=window_seconds,
            )
        capture_info["capture_seconds"] = round(time.time() - capture_started, 2)

        self._mark(session, current_node=f"analyze_segment {segment_index + 1}")
        analysis_started = time.time()
        transcript_started = time.time()
        transcript_segments = ai.transcribe(audio_path, window_seconds)
        transcript_seconds = round(time.time() - transcript_started, 2)
        transcript_text = " ".join(str(item.get("text", "")).strip() for item in transcript_segments if item.get("text"))
        frame_time = capture_info.get("frame_time", max(0.1, window_seconds / 2))
        frame = {
            "time": segment_index * window_seconds + frame_time,
            "filename": frame_path.name,
            "path": str(frame_path),
            "url": f"/api/live/{session.session_id}/frames/{frame_path.name}",
            "reason": "live segment midpoint evidence",
            "probe": {
                "type": "live_segment",
                "question": session.question,
                "expected_visuals": ["直播画面主体", "可见动作", "屏幕文字", "商品或场景变化"],
            },
        }
        world_model = {
            "timeline": [
                {
                    "time": frame["time"],
                    "end_time": frame["time"] + window_seconds,
                    "label": "直播窗口",
                    "evidence": transcript_text,
                    "expected_visuals": ["直播画面主体", "可见动作", "屏幕文字"],
                    "visual_question": session.question,
                }
            ]
        }
        vision_started = time.time()
        observations = ai.observe_frames(
            question=session.question,
            frames=[frame],
            audio_world_model=world_model,
            session_id=f"live-{session.session_id}",
        )
        vision_seconds = round(time.time() - vision_started, 2)
        observation = observations[0] if observations else {}
        summary = summarize_live_segment(
            index=segment_index,
            start_time=segment_index * window_seconds,
            transcript=transcript_text,
            observation=observation,
            question=session.question,
        )
        return {
            "index": segment_index,
            "start_time": round(segment_index * window_seconds, 2),
            "end_time": round((segment_index + 1) * window_seconds, 2),
            "transcript": transcript_text,
            "frame": {k: frame[k] for k in ["time", "filename", "url", "reason"]},
            "observation": observation,
            "summary": summary,
            "transcription_status": ai.last_transcription_status,
            "capture": capture_info,
            "analysis_seconds": round(time.time() - analysis_started, 2),
            "transcript_seconds": transcript_seconds,
            "vision_seconds": vision_seconds,
        }

    def _capture_fast_audio_and_frame(
        self,
        *,
        ffmpeg: str,
        source_url: str,
        session_dir: Path,
        segment_index: int,
        window_seconds: float,
    ) -> tuple[Path, Path, dict[str, Any]]:
        audio_path = session_dir / f"segment_{segment_index:05d}.wav"
        frame_path = session_dir / f"segment_{segment_index:05d}.jpg"
        frame_time = max(0.1, window_seconds / 2)
        audio_command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-rw_timeout",
            str(int(self.settings.live_segment_timeout_seconds * 1_000_000)),
            "-i",
            source_url,
            "-t",
            f"{window_seconds:.2f}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(audio_path),
        ]
        frame_command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-rw_timeout",
            str(int(self.settings.live_segment_timeout_seconds * 1_000_000)),
            "-ss",
            f"{frame_time:.2f}",
            "-i",
            source_url,
            "-frames:v",
            "1",
            "-vf",
            f"scale={self.settings.live_frame_width}:-2",
            "-q:v",
            "3",
            str(frame_path),
        ]
        timeout = self.settings.live_segment_timeout_seconds + window_seconds + 8
        audio_proc = subprocess.Popen(audio_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        frame_proc = subprocess.Popen(frame_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        audio_stdout, audio_stderr = audio_proc.communicate(timeout=timeout)
        frame_stdout, frame_stderr = frame_proc.communicate(timeout=timeout)
        _ = (audio_stdout, frame_stdout)
        if audio_proc.returncode != 0 or not audio_path.exists() or audio_path.stat().st_size == 0:
            raise RuntimeError(f"Live audio capture failed: {audio_stderr.strip()}")
        if frame_proc.returncode != 0 or not frame_path.exists() or frame_path.stat().st_size == 0:
            raise RuntimeError(f"Live frame capture failed: {frame_stderr.strip()}")
        return audio_path, frame_path, {"mode": "fast", "frame_time": frame_time}

    def _capture_compat_segment(
        self,
        *,
        processor: VideoProcessor,
        ffmpeg: str,
        source_url: str,
        session_dir: Path,
        segment_index: int,
        window_seconds: float,
    ) -> tuple[Path, Path, dict[str, Any]]:
        segment_path = session_dir / f"segment_{segment_index:05d}.mp4"
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-rw_timeout",
            str(int(self.settings.live_segment_timeout_seconds * 1_000_000)),
            "-i",
            source_url,
            "-t",
            f"{window_seconds:.2f}",
            "-c",
            "copy",
            str(segment_path),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.settings.live_segment_timeout_seconds + window_seconds + 8,
            check=False,
        )
        if completed.returncode != 0 or not segment_path.exists() or segment_path.stat().st_size == 0:
            raise RuntimeError(f"Live segment capture failed: {completed.stderr.strip()}")
        has_audio = True
        try:
            metadata = processor.probe_video(segment_path)
            has_audio = bool(metadata.get("has_audio", True))
        except Exception:
            metadata = {"duration_seconds": window_seconds, "has_audio": True}
        audio_path = session_dir / f"segment_{segment_index:05d}.wav"
        extracted_audio = processor.extract_audio(segment_path, audio_path, has_audio)
        if extracted_audio is None:
            raise RuntimeError("Live segment did not include an audio track.")
        frame_path = session_dir / f"segment_{segment_index:05d}.jpg"
        frame_time = min(max(0.1, window_seconds / 2), max(0.1, float(metadata.get("duration_seconds", window_seconds)) - 0.1))
        processor.extract_frame(segment_path, frame_time, frame_path)
        return extracted_audio, frame_path, {"mode": "compat", "frame_time": frame_time, "segment_path": str(segment_path)}

    def _mark(
        self,
        session: LiveSession,
        *,
        status: str | None = None,
        current_node: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            if status is not None:
                session.status = status
            if current_node is not None:
                session.current_node = current_node
            if error is not None:
                session.error = error
            session.updated_at = time.time()

    @staticmethod
    def _public(session: LiveSession) -> dict[str, Any]:
        return {
            "session_id": session.session_id,
            "source_url": session.source_url,
            "question": session.question,
            "status": session.status,
            "current_node": session.current_node,
            "error": session.error,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "resolved_url": session.resolved_url,
            "segments": session.segments,
        }


def resolve_live_stream_url(url: str) -> str:
    if _is_direct_stream(url):
        return url
    parsed = urlparse(url)
    if parsed.netloc.endswith("douyin.com") and "/" in parsed.path:
        return _resolve_douyin_live_url(url)
    raise RuntimeError("Unsupported live URL. Provide a direct m3u8/flv URL or a supported Douyin live room URL.")


def _resolve_douyin_live_url(url: str) -> str:
    with httpx.Client(headers=LIVE_HEADERS, follow_redirects=True, timeout=20) as client:
        response = client.get(url)
        response.raise_for_status()
    text = html.unescape(response.text).replace("\\u0026", "&").replace("\\/", "/")
    candidates = _extract_stream_urls(text, ".m3u8") + _extract_stream_urls(text, ".flv")
    if not candidates:
        raise RuntimeError("Could not find a playable Douyin stream URL in the room page. The room may be offline or require cookies.")
    return _prefer_low_latency_stream(candidates)


def _extract_stream_urls(text: str, suffix: str) -> list[str]:
    pattern = rf"https?://[^\"'<>\\ ]+?{re.escape(suffix)}[^\"'<>\\ ]*"
    urls = []
    for match in re.findall(pattern, text):
        cleaned = match.replace("\\u0026", "&").replace("&amp;", "&").rstrip("\\")
        if cleaned not in urls:
            urls.append(cleaned)
    return urls


def _prefer_low_latency_stream(urls: list[str]) -> str:
    def score(url: str) -> tuple[int, int]:
        lowered = url.lower()
        resolution = 0
        if "_ld" in lowered or "_sd" in lowered:
            resolution = 0
        elif "_hd" in lowered:
            resolution = 1
        elif "full" in lowered:
            resolution = 3
        else:
            resolution = 2
        protocol = 0 if ".m3u8" in lowered else 1
        return (protocol, resolution)

    return sorted(urls, key=score)[0]


def _is_direct_stream(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith((".m3u8", ".flv", ".mp4"))


def summarize_live_segment(
    *,
    index: int,
    start_time: float,
    transcript: str,
    observation: dict[str, Any],
    question: str,
) -> str:
    visual = str(observation.get("evidence_assessment") or observation.get("scene") or "").strip()
    transcript = " ".join(transcript.split())
    pieces = [f"第 {index + 1} 段（{start_time:.0f}s 起）"]
    if transcript:
        pieces.append(f"音频：{_clip(transcript, 90)}")
    if visual:
        pieces.append(f"画面：{_clip(visual, 120)}")
    if len(pieces) == 1:
        pieces.append(f"已按问题“{_clip(question, 40)}”观察该直播窗口，但暂未得到稳定音频或画面描述。")
    return "；".join(pieces) + "。"


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."
