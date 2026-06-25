from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field


JobStatus = Literal["queued", "running", "succeeded", "failed"]


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str = "unknown"
    confidence: float | None = None


class AudioEvent(BaseModel):
    time: float
    end_time: float | None = None
    label: str
    evidence: str
    expected_visuals: list[str] = Field(default_factory=list)
    visual_question: str = ""


class KeyframePlanItem(BaseModel):
    time: float
    reason: str
    source: str = "audio"
    group: str | None = None
    probe: dict[str, Any] | None = None


class ExtractedFrame(BaseModel):
    time: float
    filename: str
    path: str
    url: str
    reason: str
    probe: dict[str, Any] | None = None


class FrameObservation(BaseModel):
    time: float
    filename: str
    visual_target: str = ""
    evidence_assessment: str = ""
    scene: str
    objects: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    visible_text: list[str] = Field(default_factory=list)
    audio_alignment: Literal["match", "conflict", "uncertain"] = "uncertain"
    notes: str = ""


class EventPrediction(BaseModel):
    window_start: float
    window_end: float
    hypothesis: str
    expected_evidence: list[str] = Field(default_factory=list)
    source_event: str = ""


class PredictionCheck(BaseModel):
    window_start: float
    window_end: float
    hypothesis: str
    status: Literal["match", "conflict", "uncertain"]
    conflict_score: float
    evidence: str


class JobPublic(BaseModel):
    job_id: str
    question: str
    status: JobStatus
    progress: int
    current_node: str
    error: str | None = None
    created_at: str
    updated_at: str


class VideoAgentState(TypedDict, total=False):
    job_id: str
    question: str
    video_path: str
    duration_seconds: float
    fps: float
    width: int
    height: int
    has_audio: bool
    audio_path: NotRequired[str | None]
    transcript_segments: list[dict[str, Any]]
    transcription_status: dict[str, Any]
    audio_world_model: dict[str, Any]
    frame_candidates: list[dict[str, Any]]
    vision_request_count: int
    keyframe_plan: list[dict[str, Any]]
    extracted_frames: list[dict[str, Any]]
    frame_observations: list[dict[str, Any]]
    predictions: list[dict[str, Any]]
    prediction_checks: list[dict[str, Any]]
    refinement_windows: list[dict[str, Any]]
    refinement_rounds: int
    should_refine: bool
    answer: dict[str, Any]
    result: dict[str, Any]
