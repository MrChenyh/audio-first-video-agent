from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]


def _bool_env(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    openai_api_key: str | None
    openai_base_url: str | None
    openai_org_id: str | None
    mock_mode: str
    fast_mode: bool
    ffmpeg_path: str | None
    ffprobe_path: str | None
    transcribe_model: str
    transcribe_fallback_model: str
    audio_chat_transcribe_model: str | None
    vision_provider: str
    vision_model: str
    joyai_api_base: str
    joyai_api_key: str
    joyai_model: str
    joyai_timeout_seconds: float
    reasoning_model: str
    reasoning_effort: str | None
    min_video_seconds: float
    max_video_seconds: float
    max_refinement_rounds: int
    max_keyframes: int
    enhanced_initial_keyframes: int
    fast_max_keyframes: int
    fast_seconds_per_frame: float
    fast_max_timeline_events: int
    refinement_samples_per_window: int
    keyframe_strategy: str
    candidate_sample_fps: float
    candidate_max_per_event: int
    candidate_hash_min_distance: int
    vision_batch_size: int
    llm_timeout_seconds: float
    llm_max_retries: int
    allow_model_fallback: bool
    local_transcribe_first: bool
    local_transcribe_fallback: bool
    local_transcribe_model: str
    live_window_seconds: float
    live_max_segments: int
    live_segment_timeout_seconds: float
    cors_origins: tuple[str, ...]

    @property
    def use_mock_models(self) -> bool:
        if self.mock_mode == "auto":
            return not bool(self.openai_api_key)
        return _bool_env(self.mock_mode)


def load_settings() -> Settings:
    backend_dir = Path(__file__).resolve().parents[1]
    project_root = backend_dir.parent
    if load_dotenv is not None:
        load_dotenv(project_root / ".env", override=True)

    data_dir = Path(os.getenv("DATA_DIR", project_root / "data"))
    if not data_dir.is_absolute():
        data_dir = (project_root / data_dir).resolve()

    cors_raw = os.getenv("CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173")
    cors_origins = tuple(origin.strip() for origin in cors_raw.split(",") if origin.strip())

    return Settings(
        project_root=project_root,
        data_dir=data_dir,
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
        openai_org_id=os.getenv("OPENAI_ORG_ID") or None,
        mock_mode=os.getenv("AUDIO_FIRST_MOCK_MODE", "auto").strip().lower(),
        fast_mode=_bool_env(os.getenv("AUDIO_FIRST_FAST_MODE") or os.getenv("FAST_MODE"), False),
        ffmpeg_path=os.getenv("FFMPEG_PATH") or None,
        ffprobe_path=os.getenv("FFPROBE_PATH") or None,
        transcribe_model=os.getenv("TRANSCRIBE_MODEL", "gpt-4o-transcribe-diarize"),
        transcribe_fallback_model=os.getenv("TRANSCRIBE_FALLBACK_MODEL", "gpt-4o-transcribe"),
        audio_chat_transcribe_model=os.getenv("AUDIO_CHAT_TRANSCRIBE_MODEL") or None,
        vision_provider=os.getenv("VISION_PROVIDER", "openai").strip().lower(),
        vision_model=os.getenv("VISION_MODEL", "gpt-5.4-mini"),
        joyai_api_base=os.getenv("JOYAI_API_BASE", "http://127.0.0.1:8070/v1").rstrip("/"),
        joyai_api_key=os.getenv("JOYAI_API_KEY", "EMPTY"),
        joyai_model=os.getenv("JOYAI_MODEL", "JoyAI-VL-Interaction-Preview"),
        joyai_timeout_seconds=float(os.getenv("JOYAI_TIMEOUT_SECONDS", "30")),
        reasoning_model=os.getenv("REASONING_MODEL", "gpt-5.5"),
        reasoning_effort=os.getenv("REASONING_EFFORT") or None,
        min_video_seconds=float(os.getenv("MIN_VIDEO_SECONDS", "30")),
        max_video_seconds=float(os.getenv("MAX_VIDEO_SECONDS", "600")),
        max_refinement_rounds=int(os.getenv("MAX_REFINEMENT_ROUNDS", "1")),
        max_keyframes=int(os.getenv("MAX_KEYFRAMES", "8")),
        enhanced_initial_keyframes=max(1, int(os.getenv("ENHANCED_INITIAL_KEYFRAMES", "4"))),
        fast_max_keyframes=max(1, int(os.getenv("FAST_MAX_KEYFRAMES", "12"))),
        fast_seconds_per_frame=max(1.0, float(os.getenv("FAST_SECONDS_PER_FRAME", "120"))),
        fast_max_timeline_events=max(1, int(os.getenv("FAST_MAX_TIMELINE_EVENTS", "12"))),
        refinement_samples_per_window=max(1, int(os.getenv("REFINEMENT_SAMPLES_PER_WINDOW", "1"))),
        keyframe_strategy=os.getenv("KEYFRAME_STRATEGY", "enhanced").strip().lower(),
        candidate_sample_fps=float(os.getenv("CANDIDATE_SAMPLE_FPS", "3")),
        candidate_max_per_event=int(os.getenv("CANDIDATE_MAX_PER_EVENT", "2")),
        candidate_hash_min_distance=int(os.getenv("CANDIDATE_HASH_MIN_DISTANCE", "4")),
        vision_batch_size=max(1, int(os.getenv("VISION_BATCH_SIZE", "3"))),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "90")),
        llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "1")),
        allow_model_fallback=_bool_env(os.getenv("ALLOW_MODEL_FALLBACK"), True),
        local_transcribe_first=_bool_env(os.getenv("LOCAL_TRANSCRIBE_FIRST"), False),
        local_transcribe_fallback=_bool_env(os.getenv("LOCAL_TRANSCRIBE_FALLBACK"), True),
        local_transcribe_model=os.getenv("LOCAL_TRANSCRIBE_MODEL", "tiny"),
        live_window_seconds=max(1.0, float(os.getenv("LIVE_WINDOW_SECONDS", "4"))),
        live_max_segments=max(0, int(os.getenv("LIVE_MAX_SEGMENTS", "0"))),
        live_segment_timeout_seconds=max(5.0, float(os.getenv("LIVE_SEGMENT_TIMEOUT_SECONDS", "18"))),
        cors_origins=cors_origins,
    )
