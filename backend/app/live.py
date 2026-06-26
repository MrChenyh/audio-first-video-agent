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


TECHNICAL_VISION_TERMS = (
    "JoyAI",
    "endpoint failed",
    "vision endpoint",
    "returned silence",
    "could not inspect pixels",
    "configured model endpoint",
)

PROFANITY_TERMS = (
    "傻逼",
    "煞笔",
    "傻屄",
    "他妈的",
    "妈的",
    "尼玛",
    "卧槽",
    "我靠",
    "操你",
    "草你",
    "滚蛋",
    "去死",
    "fuck",
    "shit",
    "bitch",
    "asshole",
)

PROFANITY_PATTERNS = (
    re.compile(r"(?<![A-Za-z])s\s*\.?\s*b(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])f\s*\*+\s*k(?![A-Za-z])", re.IGNORECASE),
)

VISUAL_RISK_KEYWORDS: dict[str, tuple[str, ...]] = {
    "smoking": ("抽烟", "吸烟", "香烟", "烟头", "烟雾", "电子烟", "vape"),
    "sexual_suggestive": ("擦边", "性暗示", "裸露", "暴露", "低俗", "挑逗"),
    "nudity": ("裸露", "裸体", "nudity"),
    "violence": ("打斗", "殴打", "暴力", "武器"),
    "dangerous": ("危险动作", "自残", "危险行为"),
    "alcohol": ("饮酒", "酒瓶", "酒精"),
    "gambling": ("赌博", "下注", "博彩"),
    "drugs": ("毒品", "吸毒"),
}

MODERATION_KEEP_TERMS = PROFANITY_TERMS + tuple(
    keyword for keywords in VISUAL_RISK_KEYWORDS.values() for keyword in keywords
)

RISK_LEVEL_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}
CATEGORY_LABELS = {
    "profanity": "违禁词/粗口",
    "sexual_suggestive": "擦边/性暗示",
    "smoking": "抽烟",
    "nudity": "裸露",
    "violence": "暴力",
    "dangerous": "危险行为",
    "alcohol": "饮酒",
    "gambling": "赌博",
    "drugs": "毒品",
    "other": "其他风险",
}


def _initial_live_model() -> dict[str, Any]:
    return {
        "status": "warming_up",
        "program_type": "直播合规监控",
        "current_focus": "正在扫描直播音频和画面风险",
        "stable_summary": "尚未发现违禁风险。系统只会在命中风险时保留证据。",
        "confidence": 0.0,
        "evidence_count": 0,
        "audio_evidence_count": 0,
        "visual_evidence_count": 0,
        "segment_count": 0,
        "last_updated_segment": None,
        "audio_points": [],
        "visual_points": [],
        "timeline": [],
        "entities": [],
        "risk_state": {
            "status": "clear",
            "scanned_segments": 0,
            "risk_segments": 0,
            "last_alert_segment": None,
            "last_alert_time": None,
            "highest_level": "none",
            "category_counts": {},
            "recent_alerts": [],
        },
        "debug": {"vision_unavailable_count": 0},
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
    live_model: dict[str, Any] = field(default_factory=_initial_live_model)
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
                    session.live_model = update_live_world_model(session.live_model, segment, session.question)
                    if (segment.get("moderation") or {}).get("has_risk"):
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
        raw_transcript_text = " ".join(str(item.get("text", "")).strip() for item in transcript_segments if item.get("text"))
        transcript_text = "" if _is_low_quality_live_audio(raw_transcript_text) else raw_transcript_text
        frame_time = capture_info.get("frame_time", max(0.1, window_seconds / 2))
        frame = {
            "time": segment_index * window_seconds + frame_time,
            "filename": frame_path.name,
            "path": str(frame_path),
            "url": f"/api/live/{session.session_id}/frames/{frame_path.name}",
            "reason": "live moderation midpoint evidence",
            "probe": {
                "type": "live_segment",
                "question": session.question,
                "expected_visuals": ["抽烟/电子烟", "擦边或裸露", "暴力/危险动作", "赌博/毒品/酒精等可见风险"],
            },
        }
        world_model = {
            "timeline": [
                {
                    "time": frame["time"],
                    "end_time": frame["time"] + window_seconds,
                    "label": "直播合规监控窗口",
                    "evidence": transcript_text,
                    "expected_visuals": ["抽烟/电子烟", "擦边或裸露", "暴力/危险动作", "赌博/毒品/酒精等可见风险"],
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
        moderation = evaluate_live_moderation(
            index=segment_index,
            start_time=segment_index * window_seconds,
            end_time=(segment_index + 1) * window_seconds,
            transcript=transcript_text,
            raw_transcript=raw_transcript_text,
            observation=observation,
        )
        if not moderation.get("has_risk"):
            _cleanup_non_alert_files(audio_path, frame_path, capture_info)
        return {
            "index": segment_index,
            "start_time": round(segment_index * window_seconds, 2),
            "end_time": round((segment_index + 1) * window_seconds, 2),
            "transcript": transcript_text,
            "raw_transcript": raw_transcript_text if raw_transcript_text != transcript_text else "",
            "transcript_quality": "usable" if transcript_text else ("noisy" if raw_transcript_text else "empty"),
            "frame": {k: frame[k] for k in ["time", "filename", "url", "reason"]},
            "observation": observation,
            "moderation": moderation,
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
            "live_model": session.live_model,
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
    visual = "" if observation.get("vision_error") else _clean_live_text(str(observation.get("evidence_assessment") or observation.get("scene") or ""))
    transcript = " ".join(transcript.split())
    pieces = [f"第 {index + 1} 段（{start_time:.0f}s 起）"]
    if transcript:
        pieces.append(f"音频：{_clip(transcript, 90)}")
    if visual:
        pieces.append(f"画面：{_clip(visual, 120)}")
    if len(pieces) == 1:
        pieces.append(f"已按问题“{_clip(question, 40)}”观察该直播窗口，但暂未得到稳定音频或画面描述。")
    return "；".join(pieces) + "。"


def evaluate_live_moderation(
    *,
    index: int,
    start_time: float,
    end_time: float,
    transcript: str,
    raw_transcript: str,
    observation: dict[str, Any],
) -> dict[str, Any]:
    audio_text = transcript or raw_transcript
    violations = _audio_moderation_violations(audio_text)
    violations.extend(_visual_moderation_violations(observation))
    normalized = [_normalize_violation(item) for item in violations]
    normalized = [item for item in normalized if item.get("category")]
    highest = _highest_risk_level(normalized)
    return {
        "has_risk": bool(normalized),
        "risk_level": highest,
        "violations": normalized,
        "summary": _moderation_summary(normalized),
        "start_time": round(start_time, 2),
        "end_time": round(end_time, 2),
        "segment_index": index,
    }


def update_live_world_model(current: dict[str, Any] | None, segment: dict[str, Any], question: str) -> dict[str, Any]:
    model = _clone_live_model(current)
    segment_index = int(segment.get("index") or 0)
    start_time = float(segment.get("start_time") or 0.0)
    observation = segment.get("observation") or {}
    moderation = segment.get("moderation") if isinstance(segment.get("moderation"), dict) else {}
    violations = moderation.get("violations") if isinstance(moderation.get("violations"), list) else []
    has_risk = bool(moderation.get("has_risk") and violations)
    audio_violations = [item for item in violations if isinstance(item, dict) and item.get("source") == "audio"]
    visual_violations = [item for item in violations if isinstance(item, dict) and item.get("source") == "visual"]
    visual = _clean_live_text(str(observation.get("evidence_assessment") or observation.get("scene") or ""))
    vision_unavailable = bool(observation.get("vision_error")) or _is_technical_vision_text(visual)

    if vision_unavailable:
        debug = dict(model.get("debug") or {})
        debug["vision_unavailable_count"] = int(debug.get("vision_unavailable_count") or 0) + 1
        model["debug"] = debug

    if audio_violations:
        model["audio_points"] = _append_unique_point(
            model.get("audio_points") or [],
            {"time": start_time, "text": _violations_text(audio_violations)},
            limit=8,
        )
    if visual_violations:
        model["visual_points"] = _append_unique_point(
            model.get("visual_points") or [],
            {"time": start_time, "text": _violations_text(visual_violations)},
            limit=8,
        )

    if has_risk:
        model["timeline"] = _append_unique_point(
            model.get("timeline") or [],
            {
                "time": start_time,
                "end_time": segment.get("end_time"),
                "audio": _violations_text(audio_violations),
                "visual": _violations_text(visual_violations),
                "summary": moderation.get("summary") or _violations_text(violations),
            },
            limit=12,
        )

    audio_points = model.get("audio_points") or []
    visual_points = model.get("visual_points") or []
    evidence_count = len(audio_points) + len(visual_points)
    model["segment_count"] = max(int(model.get("segment_count") or 0), segment_index + 1)
    model["audio_evidence_count"] = len(audio_points)
    model["visual_evidence_count"] = len(visual_points)
    model["evidence_count"] = evidence_count
    model["last_updated_segment"] = segment_index
    model["program_type"] = "直播合规监控"
    model["entities"] = _moderation_entities(model.get("timeline") or [])
    model["risk_state"] = _update_risk_state(model.get("risk_state"), moderation, segment_index, start_time)
    model["status"] = "ready"
    model["confidence"] = _live_confidence(model)
    model["current_focus"] = _infer_current_focus(model, question)
    model["stable_summary"] = _stable_live_summary(model)
    return model


def _clone_live_model(current: dict[str, Any] | None) -> dict[str, Any]:
    model = _initial_live_model()
    if not current:
        return model
    for key, value in current.items():
        if key in {"audio_points", "visual_points", "timeline", "entities"}:
            model[key] = [dict(item) if isinstance(item, dict) else item for item in (value or [])]
        elif key == "debug":
            model[key] = dict(value or {})
        else:
            model[key] = value
    return model


def _append_unique_point(points: list[dict[str, Any]], point: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    text = _clean_live_text(str(point.get("text") or point.get("summary") or ""))
    if not text:
        return points[-limit:]
    point = dict(point)
    if "text" in point:
        point["text"] = text
    if "summary" in point:
        point["summary"] = _clean_live_text(str(point.get("summary") or ""))
    signatures = {_point_signature(item) for item in points}
    if _point_signature(point) not in signatures:
        points.append(point)
    return points[-limit:]


def _point_signature(point: dict[str, Any]) -> tuple[int, str]:
    time_key = int(round(float(point.get("time") or 0.0)))
    text = _clean_live_text(str(point.get("text") or point.get("summary") or ""))[:42]
    return time_key, text


def _contains_any(text: str, lowered: str, terms: tuple[str, ...]) -> bool:
    return any(term.lower() in lowered or term in text for term in terms)


def _infer_current_focus(model: dict[str, Any], question: str) -> str:
    risk_state = model.get("risk_state") if isinstance(model.get("risk_state"), dict) else {}
    scanned = int(risk_state.get("scanned_segments") or model.get("segment_count") or 0)
    risk_segments = int(risk_state.get("risk_segments") or 0)
    if risk_segments <= 0:
        return f"已扫描 {scanned} 个窗口，暂未发现违禁风险"
    latest = (risk_state.get("recent_alerts") or [{}])[-1]
    categories = "、".join(_category_label(item) for item in latest.get("categories") or [])
    return f"最近告警：{categories or '疑似违规'}，{_format_seconds(float(latest.get('time') or 0.0))}"


def _stable_live_summary(model: dict[str, Any]) -> str:
    risk_state = model.get("risk_state") if isinstance(model.get("risk_state"), dict) else {}
    scanned = int(risk_state.get("scanned_segments") or model.get("segment_count") or 0)
    risk_segments = int(risk_state.get("risk_segments") or 0)
    if risk_segments <= 0:
        return f"已扫描 {scanned} 个直播窗口，尚未发现违禁风险；正常窗口不会保留证据。"
    highest = str(risk_state.get("highest_level") or "low")
    counts = risk_state.get("category_counts") if isinstance(risk_state.get("category_counts"), dict) else {}
    categories = "、".join(f"{_category_label(key)} {value} 次" for key, value in counts.items())
    return f"已扫描 {scanned} 个窗口，其中 {risk_segments} 个窗口触发告警；最高风险 {highest}。{categories}。"


def _live_confidence(model: dict[str, Any]) -> float:
    risk_state = model.get("risk_state") if isinstance(model.get("risk_state"), dict) else {}
    scanned = int(risk_state.get("scanned_segments") or model.get("segment_count") or 0)
    risk_segments = int(risk_state.get("risk_segments") or 0)
    if scanned <= 0:
        return 0.0
    base = min(0.9, 0.35 + 0.04 * scanned)
    if risk_segments:
        base = min(0.96, base + 0.12)
    return round(base, 2)


def _audio_moderation_violations(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    lowered = text.lower()
    compact = _normalize_moderation_text(text)
    matches: list[str] = []
    for term in PROFANITY_TERMS:
        if term.lower() in lowered or _normalize_moderation_text(term) in compact:
            matches.append(term)
    for pattern in PROFANITY_PATTERNS:
        if pattern.search(text):
            matches.append(pattern.pattern)
    unique = []
    for item in matches:
        if item not in unique:
            unique.append(item)
    return [
        {
            "source": "audio",
            "category": "profanity",
            "severity": "medium",
            "confidence": 0.92,
            "evidence": f"音频命中疑似违禁词：{match}",
            "matched_text": match,
            "context": _clip(text, 120),
        }
        for match in unique[:6]
    ]


def _visual_moderation_violations(observation: dict[str, Any]) -> list[dict[str, Any]]:
    moderation = observation.get("live_moderation") if isinstance(observation.get("live_moderation"), dict) else None
    if moderation:
        risk_level = str(moderation.get("risk_level") or "none").lower()
        if risk_level in {"none", "unknown", ""}:
            return []
        return [item for item in moderation.get("violations") or [] if isinstance(item, dict)]
    text = _clean_live_text(str(observation.get("evidence_assessment") or observation.get("scene") or ""))
    if not text or observation.get("vision_error"):
        return []
    violations = []
    lowered = text.lower()
    for category, keywords in VISUAL_RISK_KEYWORDS.items():
        matched = [keyword for keyword in keywords if keyword.lower() in lowered or keyword in text]
        if matched:
            violations.append(
                {
                    "source": "visual",
                    "category": category,
                    "severity": "medium",
                    "confidence": 0.68,
                    "evidence": f"画面描述命中风险关键词：{'、'.join(matched[:3])}。{_clip(text, 100)}",
                    "visible_text": [],
                }
            )
    return violations


def _normalize_violation(item: dict[str, Any]) -> dict[str, Any]:
    source = str(item.get("source") or "visual").strip() or "visual"
    category = str(item.get("category") or "other").strip() or "other"
    severity = str(item.get("severity") or "low").strip().lower()
    if severity not in SEVERITY_RANK:
        severity = "low"
    confidence = _safe_confidence(item.get("confidence"))
    evidence = _clean_live_text(str(item.get("evidence") or item.get("context") or category))
    visible_text = [str(value).strip() for value in item.get("visible_text") or [] if str(value).strip()]
    payload = {
        "source": source,
        "category": category,
        "category_label": _category_label(category),
        "severity": severity,
        "confidence": confidence,
        "evidence": evidence,
        "visible_text": visible_text,
    }
    if item.get("matched_text"):
        payload["matched_text"] = str(item.get("matched_text"))
    if item.get("context"):
        payload["context"] = str(item.get("context"))
    return payload


def _highest_risk_level(violations: list[dict[str, Any]]) -> str:
    highest = "none"
    for item in violations:
        severity = str(item.get("severity") or "low").lower()
        if SEVERITY_RANK.get(severity, 0) >= 3:
            level = "high"
        elif SEVERITY_RANK.get(severity, 0) == 2 or float(item.get("confidence") or 0.0) >= 0.75:
            level = "medium"
        else:
            level = "low"
        if RISK_LEVEL_RANK[level] > RISK_LEVEL_RANK[highest]:
            highest = level
    return highest


def _moderation_summary(violations: list[dict[str, Any]]) -> str:
    if not violations:
        return "未发现违禁风险"
    parts = []
    for item in violations[:4]:
        parts.append(f"{_category_label(str(item.get('category') or 'other'))}：{_clip(str(item.get('evidence') or ''), 70)}")
    return "；".join(parts)


def _violations_text(violations: list[dict[str, Any]]) -> str:
    return "；".join(
        f"{_category_label(str(item.get('category') or 'other'))}：{_clip(str(item.get('evidence') or ''), 72)}"
        for item in violations[:4]
    )


def _update_risk_state(
    current: dict[str, Any] | None,
    moderation: dict[str, Any],
    segment_index: int,
    start_time: float,
) -> dict[str, Any]:
    state = dict(current or _initial_live_model()["risk_state"])
    state["scanned_segments"] = max(int(state.get("scanned_segments") or 0), segment_index + 1)
    violations = [item for item in moderation.get("violations") or [] if isinstance(item, dict)] if moderation else []
    if not moderation.get("has_risk") or not violations:
        state["status"] = "alert" if int(state.get("risk_segments") or 0) else "clear"
        return state
    state["status"] = "alert"
    state["risk_segments"] = int(state.get("risk_segments") or 0) + 1
    state["last_alert_segment"] = segment_index
    state["last_alert_time"] = round(start_time, 2)
    level = str(moderation.get("risk_level") or _highest_risk_level(violations))
    if RISK_LEVEL_RANK.get(level, 0) > RISK_LEVEL_RANK.get(str(state.get("highest_level") or "none"), 0):
        state["highest_level"] = level
    counts = dict(state.get("category_counts") or {})
    categories = []
    for item in violations:
        category = str(item.get("category") or "other")
        counts[category] = int(counts.get(category) or 0) + 1
        if category not in categories:
            categories.append(category)
    state["category_counts"] = counts
    alerts = list(state.get("recent_alerts") or [])
    alerts.append(
        {
            "segment_index": segment_index,
            "time": round(start_time, 2),
            "risk_level": level,
            "categories": categories,
            "summary": moderation.get("summary") or _moderation_summary(violations),
            "violations": violations,
        }
    )
    state["recent_alerts"] = alerts[-12:]
    return state


def _moderation_entities(timeline: list[dict[str, Any]]) -> list[str]:
    entities = []
    for item in timeline:
        text = str(item.get("summary") or item.get("audio") or item.get("visual") or "")
        for category, label in CATEGORY_LABELS.items():
            if category in text or label in text:
                if label not in entities:
                    entities.append(label)
    return entities[:8]


def _category_label(category: str) -> str:
    return CATEGORY_LABELS.get(category, category or "其他风险")


def _safe_confidence(value: Any) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 2)
    except Exception:
        return 0.0


def _normalize_moderation_text(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", str(text or "").lower())


def _clean_live_text(text: str) -> str:
    text = " ".join(str(text or "").split())
    if _is_technical_vision_text(text):
        return ""
    return text.strip()


def _is_low_quality_live_audio(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return True
    if _audio_moderation_violations(text):
        return False
    if _contains_any(compact, compact.lower(), MODERATION_KEEP_TERMS):
        return False
    if re.search(r"[A-Za-z]{3,}", compact):
        return False
    if re.search(r"\d", compact) and len(compact) < 8:
        return True
    if len(compact) <= 5:
        return True
    if len(compact) <= 8 and not re.search(r"[，。！？、；：]", compact):
        return True
    unique_ratio = len(set(compact)) / max(1, len(compact))
    if len(compact) <= 10 and unique_ratio > 0.85:
        return True
    return False


def _is_technical_vision_text(text: str) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in TECHNICAL_VISION_TERMS)


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _format_seconds(seconds: float) -> str:
    safe = max(0, int(round(seconds)))
    return f"{safe // 60}:{safe % 60:02d}"


def _cleanup_non_alert_files(audio_path: Path, frame_path: Path, capture_info: dict[str, Any]) -> None:
    for candidate in (audio_path, frame_path, Path(str(capture_info.get("segment_path"))) if capture_info.get("segment_path") else None):
        if candidate is None:
            continue
        try:
            candidate.unlink(missing_ok=True)
        except Exception:
            pass
