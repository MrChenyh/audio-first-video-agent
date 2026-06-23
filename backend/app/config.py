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
    ffmpeg_path: str | None
    ffprobe_path: str | None
    transcribe_model: str
    transcribe_fallback_model: str
    audio_chat_transcribe_model: str | None
    vision_model: str
    reasoning_model: str
    reasoning_effort: str | None
    min_video_seconds: float
    max_video_seconds: float
    max_refinement_rounds: int
    max_keyframes: int
    llm_timeout_seconds: float
    llm_max_retries: int
    allow_model_fallback: bool
    local_transcribe_fallback: bool
    local_transcribe_model: str
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
        ffmpeg_path=os.getenv("FFMPEG_PATH") or None,
        ffprobe_path=os.getenv("FFPROBE_PATH") or None,
        transcribe_model=os.getenv("TRANSCRIBE_MODEL", "gpt-4o-transcribe-diarize"),
        transcribe_fallback_model=os.getenv("TRANSCRIBE_FALLBACK_MODEL", "gpt-4o-transcribe"),
        audio_chat_transcribe_model=os.getenv("AUDIO_CHAT_TRANSCRIBE_MODEL") or None,
        vision_model=os.getenv("VISION_MODEL", "gpt-5.4-mini"),
        reasoning_model=os.getenv("REASONING_MODEL", "gpt-5.5"),
        reasoning_effort=os.getenv("REASONING_EFFORT") or None,
        min_video_seconds=float(os.getenv("MIN_VIDEO_SECONDS", "30")),
        max_video_seconds=float(os.getenv("MAX_VIDEO_SECONDS", "600")),
        max_refinement_rounds=int(os.getenv("MAX_REFINEMENT_ROUNDS", "1")),
        max_keyframes=int(os.getenv("MAX_KEYFRAMES", "8")),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "90")),
        llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "1")),
        allow_model_fallback=_bool_env(os.getenv("ALLOW_MODEL_FALLBACK"), True),
        local_transcribe_fallback=_bool_env(os.getenv("LOCAL_TRANSCRIBE_FALLBACK"), True),
        local_transcribe_model=os.getenv("LOCAL_TRANSCRIBE_MODEL", "tiny"),
        cors_origins=cors_origins,
    )
