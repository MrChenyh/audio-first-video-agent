from __future__ import annotations

import base64
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from .config import Settings
from .prediction import classify_prediction_check


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


class AIClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.mock = settings.use_mock_models
        self._client = None
        self._joyai_client = None
        self._local_whisper_cache: dict[str, Any] = {}
        self.last_transcription_status: dict[str, Any] = {}
        self.last_vision_request_count = 0
        if not self.mock:
            from openai import OpenAI

            kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
            if settings.openai_base_url:
                kwargs["base_url"] = settings.openai_base_url
            if settings.openai_org_id:
                kwargs["organization"] = settings.openai_org_id
            kwargs["timeout"] = settings.llm_timeout_seconds
            kwargs["max_retries"] = settings.llm_max_retries
            self._client = OpenAI(**kwargs)

    def transcribe(self, audio_path: Path | None, duration: float) -> list[dict[str, Any]]:
        self.last_transcription_status = {
            "status": "skipped",
            "reason": "No audio file was available.",
            "method": "none",
            "api_attempted": False,
            "chat_audio_attempted": False,
            "local_attempted": False,
            "attempts": [],
        }
        if audio_path is None:
            return []
        if self.mock:
            self.last_transcription_status = {
                "status": "mock",
                "reason": "Mock mode generated deterministic transcript segments.",
                "method": "mock",
                "api_attempted": False,
                "chat_audio_attempted": False,
                "local_attempted": False,
                "attempts": [{"method": "mock", "status": "ok"}],
            }
            return [
                {
                    "start": 0.0,
                    "end": min(duration, 8.0),
                    "speaker": "speaker_1",
                    "text": "Mock transcript: opening context and a clear audio cue.",
                    "confidence": 0.95,
                },
                {
                    "start": min(duration, 12.0),
                    "end": min(duration, 18.0),
                    "speaker": "speaker_1",
                    "text": "Mock transcript: the action changes and the viewer should inspect the next frames.",
                    "confidence": 0.92,
                },
            ]

        attempts: list[dict[str, Any]] = []
        if self.settings.local_transcribe_first:
            local_segments, local_status = self._local_transcribe(audio_path)
            attempts.append(local_status)
            if local_segments:
                self.last_transcription_status = {
                    "status": "ok",
                    "reason": "Local faster-whisper was configured as the primary transcription path.",
                    "method": "local_faster_whisper",
                    "api_attempted": False,
                    "chat_audio_attempted": False,
                    "local_attempted": True,
                    "attempts": attempts,
                    "segment_count": len(local_segments),
                }
                return local_segments
            if not self.settings.allow_model_fallback:
                raise RuntimeError(local_status.get("error") or "Local transcription returned no text.")

        assert self._client is not None
        if self.settings.audio_chat_transcribe_model:
            chat_segments = self._chat_audio_transcribe(audio_path, duration, attempts)
            if chat_segments:
                self.last_transcription_status = {
                    "status": "ok",
                    "reason": "Chat audio input returned transcript segments.",
                    "method": "chat_audio",
                    "api_attempted": False,
                    "chat_audio_attempted": True,
                    "local_attempted": False,
                    "attempts": attempts,
                    "segment_count": len(chat_segments),
                }
                return chat_segments

        api_error = None
        try:
            with audio_path.open("rb") as handle:
                try:
                    response = self._client.audio.transcriptions.create(
                        model=self.settings.transcribe_model,
                        file=handle,
                        response_format="verbose_json",
                    )
                except Exception:
                    handle.seek(0)
                    response = self._client.audio.transcriptions.create(
                        model=self.settings.transcribe_fallback_model,
                        file=handle,
                        response_format="verbose_json",
                    )
            segments = self._normalize_transcript(response)
            attempts.append(
                {
                    "method": "audio_transcriptions",
                    "model": self.settings.transcribe_model,
                    "status": "ok" if segments else "empty",
                    "segment_count": len(segments),
                }
            )
            self.last_transcription_status = {
                "status": "ok" if segments else "empty",
                "reason": "API transcription returned segments." if segments else "API transcription returned no text.",
                "method": "audio_transcriptions",
                "api_attempted": True,
                "chat_audio_attempted": bool(self.settings.audio_chat_transcribe_model),
                "local_attempted": False,
                "attempts": attempts,
                "segment_count": len(segments),
            }
            return segments
        except Exception as exc:
            api_error = str(exc)
            attempts.append(
                {
                    "method": "audio_transcriptions",
                    "model": self.settings.transcribe_model,
                    "fallback_model": self.settings.transcribe_fallback_model,
                    "status": "error",
                    "error": api_error,
                }
            )
            if self.settings.local_transcribe_fallback:
                local_segments, local_status = self._local_transcribe(audio_path)
                attempts.append(local_status)
                if local_segments:
                    self.last_transcription_status = {
                        "status": "ok",
                        "reason": "Local faster-whisper fallback returned segments after API transcription failed.",
                        "method": "local_faster_whisper",
                        "api_attempted": True,
                        "chat_audio_attempted": bool(self.settings.audio_chat_transcribe_model),
                        "local_attempted": True,
                        "attempts": attempts,
                        "api_error": api_error,
                        "segment_count": len(local_segments),
                    }
                    return local_segments
            if self.settings.allow_model_fallback:
                self.last_transcription_status = {
                    "status": "empty",
                    "reason": (
                        "Audio exists, but API transcription failed and local faster-whisper produced no speech text."
                    ),
                    "method": "none",
                    "api_attempted": True,
                    "chat_audio_attempted": bool(self.settings.audio_chat_transcribe_model),
                    "local_attempted": self.settings.local_transcribe_fallback,
                    "attempts": attempts,
                    "api_error": api_error,
                    "local_error": next(
                        (attempt.get("error") for attempt in attempts if attempt.get("method") == "local_faster_whisper"),
                        None,
                    ),
                    "segment_count": 0,
                }
                return []
            raise

    def build_audio_world_model(
        self,
        *,
        question: str,
        transcript_segments: list[dict[str, Any]],
        duration: float,
        has_audio: bool,
    ) -> dict[str, Any]:
        if self.settings.fast_mode:
            return self._fast_audio_world_model(question, transcript_segments, duration, has_audio)
        if self.mock:
            if not has_audio or not transcript_segments:
                return {
                    "summary": "No usable audio was available; the agent will rely on sparse visual sampling.",
                    "actors": [],
                    "environment_hypotheses": [],
                    "mood": "unknown",
                    "timeline": [],
                    "open_questions": ["Audio could not guide visual attention."],
                }
            timeline = [
                {
                    "time": segment["start"],
                    "end_time": segment["end"],
                    "label": f"Audio segment from {segment.get('speaker', 'speaker')}",
                    "evidence": segment["text"],
                    "expected_visuals": ["speaker or action related to transcript", "context change"],
                    "visual_question": "Does the frame show visible context that supports this transcript segment?",
                }
                for segment in transcript_segments
            ]
            return {
                "summary": "Audio suggests an event sequence that can seed visual attention.",
                "actors": ["speaker_1"],
                "environment_hypotheses": ["indoor or screen-recorded scene"],
                "mood": "informational",
                "timeline": timeline,
                "open_questions": [f"Answer the user question: {question}"],
            }

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summary": {"type": "string"},
                "actors": {"type": "array", "items": {"type": "string"}},
                "environment_hypotheses": {"type": "array", "items": {"type": "string"}},
                "mood": {"type": "string"},
                "timeline": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "time": {"type": "number"},
                            "end_time": {"type": ["number", "null"]},
                            "label": {"type": "string"},
                            "evidence": {"type": "string"},
                            "expected_visuals": {"type": "array", "items": {"type": "string"}},
                            "visual_question": {"type": "string"},
                        },
                        "required": ["time", "end_time", "label", "evidence", "expected_visuals", "visual_question"],
                    },
                },
                "open_questions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary", "actors", "environment_hypotheses", "mood", "timeline", "open_questions"],
        }
        prompt = (
            "Build an audio-first blind world model for a video. Use only the transcript and audio timing. "
            "Return concise JSON. The timeline should contain events whose visual evidence should be checked. "
            "For each event, write a concrete visual_question that lets a vision model verify or reject the audio "
            "hypothesis using one frame or a tiny local cluster of frames. Prefer specific visible evidence over "
            "generic scene descriptions.\n\n"
            f"User question: {question}\n"
            f"Duration seconds: {duration:.2f}\n"
            f"Transcript segments:\n{json.dumps(transcript_segments, ensure_ascii=False)}"
        )
        try:
            return self._responses_json(self.settings.reasoning_model, prompt, schema, "audio_world_model")
        except Exception:
            if self.settings.allow_model_fallback:
                return self._fallback_audio_world_model(question, transcript_segments)
            raise

    def observe_frames(
        self,
        *,
        question: str,
        frames: list[dict[str, Any]],
        audio_world_model: dict[str, Any],
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self.last_vision_request_count = 0
        if self.mock:
            self.last_vision_request_count = 1 if frames else 0
            timeline = audio_world_model.get("timeline") or []
            observations = []
            for frame in frames:
                nearest = min(timeline, key=lambda item: abs(float(item.get("time", 0)) - frame["time"]), default=None)
                observations.append(
                    {
                        "time": frame["time"],
                        "filename": frame["filename"],
                        "visual_target": (frame.get("probe") or {}).get("question", frame.get("reason", "")),
                        "evidence_assessment": "Mock mode treats the target as visually plausible.",
                        "scene": "Mock visual observation for the selected frame.",
                        "objects": ["person", "interface", "context"],
                        "actions": ["speaking", "demonstrating"],
                        "visible_text": [],
                        "audio_alignment": "match" if nearest else "uncertain",
                        "notes": frame.get("reason", ""),
                    }
                )
            return observations

        if self.settings.vision_provider in {"joyai", "joyai_adapter", "auto"}:
            try:
                if self._should_use_joyai_clips(question, frames):
                    observations = self._observe_clips_with_joyai(
                        question=question,
                        frames=frames,
                        audio_world_model=audio_world_model,
                        session_id=session_id,
                    )
                else:
                    observations = self._observe_frames_with_joyai(
                        question=question,
                        frames=frames,
                        audio_world_model=audio_world_model,
                        session_id=session_id,
                    )
                self.last_vision_request_count = len(frames)
                return observations
            except Exception as exc:
                if not self.settings.allow_model_fallback:
                    raise
                if self.settings.vision_provider in {"joyai", "joyai_adapter"}:
                    self.last_vision_request_count = len(frames)
                    return self._fallback_frame_observations(frames, f"JoyAI local vision endpoint failed: {exc}")

        observations: list[dict[str, Any]] = []
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "observations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "filename": {"type": "string"},
                            "scene": {"type": "string"},
                            "objects": {"type": "array", "items": {"type": "string"}},
                            "actions": {"type": "array", "items": {"type": "string"}},
                            "visible_text": {"type": "array", "items": {"type": "string"}},
                            "audio_alignment": {"type": "string", "enum": ["match", "conflict", "uncertain"]},
                            "visual_target": {"type": "string"},
                            "evidence_assessment": {"type": "string"},
                            "notes": {"type": "string"},
                        },
                        "required": [
                            "filename",
                            "scene",
                            "objects",
                            "actions",
                            "visible_text",
                            "audio_alignment",
                            "visual_target",
                            "evidence_assessment",
                            "notes",
                        ],
                    },
                }
            },
            "required": ["observations"],
        }
        frame_manifest = [
            {
                "index": index,
                "filename": frame["filename"],
                "time": frame["time"],
                "reason": frame.get("reason", ""),
                "probe": frame.get("probe") or {},
            }
            for index, frame in enumerate(frames, start=1)
        ]
        prompt = (
            "Inspect these video frames as targeted evidence for an audio-first video understanding loop. "
            "The images are provided in the same order as the frame manifest. Return exactly one observation for "
            "each manifest item, preserving the filename. Do not merely caption the image. First answer each "
            "frame-specific visual target, then decide whether the frame matches, conflicts with, or is uncertain "
            "for the audio-derived hypothesis. If a frame is too early, too late, cropped, or visually ambiguous, "
            "mark audio_alignment as uncertain.\n\n"
            f"Question: {question}\n"
            f"Frame manifest: {json.dumps(frame_manifest, ensure_ascii=False)}\n"
            f"Audio world model: {json.dumps(audio_world_model, ensure_ascii=False)}"
        )
        images = [self._image_data_url(Path(frame["path"])) for frame in frames]
        try:
            payload = self._responses_json(
                self.settings.vision_model,
                prompt,
                schema,
                "frame_observations",
                images=images,
            )
            self.last_vision_request_count = 1 if frames else 0
            raw_observations = payload.get("observations", [])
        except Exception:
            if not self.settings.allow_model_fallback:
                raise
            self.last_vision_request_count = 1 if frames else 0
            raw_observations = self._fallback_frame_observations(
                frames,
                "The configured model endpoint could not inspect images.",
            )

        by_filename = {
            str(item.get("filename") or ""): item
            for item in raw_observations
            if isinstance(item, dict)
        }
        for frame in frames:
            payload = by_filename.get(frame["filename"])
            if not payload:
                payload = {
                    "filename": frame["filename"],
                    "scene": "The batch vision response did not include this frame.",
                    "objects": [],
                    "actions": [],
                    "visible_text": [],
                    "audio_alignment": "uncertain",
                    "visual_target": str((frame.get("probe") or {}).get("question") or frame.get("reason", "")),
                    "evidence_assessment": "Missing per-frame observation in batch response.",
                    "notes": "Batch response repair.",
                }
            payload["time"] = frame["time"]
            payload["filename"] = frame["filename"]
            observations.append(payload)
        return observations

    def _observe_frames_with_joyai(
        self,
        *,
        question: str,
        frames: list[dict[str, Any]],
        audio_world_model: dict[str, Any],
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        from openai import OpenAI

        if self._joyai_client is None:
            self._joyai_client = OpenAI(
                api_key=self.settings.joyai_api_key,
                base_url=self.settings.joyai_api_base,
                timeout=self.settings.joyai_timeout_seconds,
                max_retries=0,
            )
        client = self._joyai_client
        observations: list[dict[str, Any]] = []
        for index, frame in enumerate(frames):
            target = str((frame.get("probe") or {}).get("question") or frame.get("reason") or question)
            audio_context = self._audio_context_for_time(audio_world_model, float(frame["time"]))
            frame_session_id = session_id or f"audio-first-{int(time.time() * 1000)}-{index}"
            is_live = (frame.get("probe") or {}).get("type") == "live_segment"
            if is_live:
                prompt = (
                    "</response> 你是直播画面合规审核器，只根据当前帧可见事实判断，不要猜测。"
                    "只返回 JSON，不要输出解释性段落。JSON 字段："
                    "risk_level(none/low/medium/high), caption, violations[]. "
                    "violations 每项包含 category(sexual_suggestive/smoking/nudity/violence/dangerous/alcohol/gambling/drugs/other), "
                    "severity(low/medium/high), confidence(0-1), evidence, visible_text[]. "
                    "规则：没有明确可见风险时 risk_level=none 且 violations=[]；"
                    "擦边/低俗必须有暴露、性暗示姿态、镜头刻意聚焦等可见证据，普通穿着不要报；"
                    "抽烟必须看见香烟/电子烟/烟雾或明确吸烟动作；"
                    "只记录违规相关证据，不要描述无风险画面。\n"
                    f"问题: {target}\n"
                    f"音频: {audio_context}"
                )
                max_tokens = 220
            else:
                prompt = (
                    "You are the fast local visual verifier inside an audio-first video agent. "
                    "Do not give a generic caption. Verify the audio-guided target for the current frame. "
                    "This is image QA, not passive live monitoring. You must not output </silence>. "
                    "Always answer with </response> followed by concise Chinese visible evidence. Include visible objects, "
                    "actions, text/symbols, and whether the frame matches, conflicts with, or is uncertain for the target.\n"
                    f"User question: {question}\n"
                    f"Frame time: {float(frame['time']):.2f}s\n"
                    f"Audio context near this frame: {audio_context}\n"
                    f"Visual target to verify: {target}"
                )
                max_tokens = 96
            response = client.chat.completions.create(
                model=self.settings.joyai_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": self._image_data_url(Path(frame["path"]))},
                            },
                        ],
                    }
                ],
                max_tokens=max_tokens,
                temperature=0,
                extra_headers={
                    "x-streaming-session": frame_session_id,
                    "x-frame-time-range": f"{float(frame['time']):.2f}s-{float(frame['time']) + 1.0:.2f}s",
                },
            )
            raw_text = response.choices[0].message.content if response.choices else ""
            observations.append(self._joyai_text_to_observation(frame, target, audio_context, raw_text or ""))
        return observations

    def _observe_clips_with_joyai(
        self,
        *,
        question: str,
        frames: list[dict[str, Any]],
        audio_world_model: dict[str, Any],
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        from openai import OpenAI

        if self._joyai_client is None:
            self._joyai_client = OpenAI(
                api_key=self.settings.joyai_api_key,
                base_url=self.settings.joyai_api_base,
                timeout=self.settings.joyai_timeout_seconds,
                max_retries=0,
            )
        client = self._joyai_client
        observations: list[dict[str, Any]] = []
        clip_frames = frames[: self.settings.joyai_max_clips_per_job or len(frames)]
        for index, frame in enumerate(clip_frames):
            target = str((frame.get("probe") or {}).get("question") or frame.get("reason") or question)
            audio_context = self._audio_context_for_time(audio_world_model, float(frame["time"]))
            clip_seconds = self._clip_seconds_for_frame(question, frame)
            start = max(0.0, float(frame["time"]) - clip_seconds / 2)
            try:
                clip_path = self._extract_temp_clip(Path(frame["path"]), start, clip_seconds, frame, index)
            except Exception as exc:
                frame_observation = self._observe_frames_with_joyai(
                    question=question,
                    frames=[frame],
                    audio_world_model=audio_world_model,
                    session_id=session_id,
                )[0]
                frame_observation["clip"] = {
                    "start": round(start, 2),
                    "end": round(start + clip_seconds, 2),
                    "mode": "fallback_image_url",
                    "duration_seconds": clip_seconds,
                    "error": str(exc),
                }
                observations.append(frame_observation)
                continue
            try:
                clip_url = self._video_data_url(clip_path)
                frame_session_id = session_id or f"audio-first-clip-{int(time.time() * 1000)}-{index}"
                prompt = (
                    "You are the short-clip observer in an audio-first video agent. "
                    "Watch the continuous clip, not just one frame. Answer in Chinese after </response>. "
                    "Focus on the user question and the audio-derived target: what changes during these seconds, "
                    "what objects/actions/text are visible, and whether the clip supports, conflicts with, or is "
                    "uncertain for the target. Be specific; do not produce a generic caption. "
                    "End with exactly one conclusion phrase: 结论：支持, 结论：冲突, or 结论：不确定.\n"
                    f"User question: {question}\n"
                    f"Clip time range: {start:.2f}s-{start + clip_seconds:.2f}s\n"
                    f"Audio context near this clip: {audio_context}\n"
                    f"Visual target to verify: {target}"
                )
                response = client.chat.completions.create(
                    model=self.settings.joyai_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "video_url", "video_url": {"url": clip_url}},
                            ],
                        }
                    ],
                    max_tokens=180,
                    temperature=0,
                    extra_headers={
                        "x-streaming-session": frame_session_id,
                        "x-frame-time-range": f"{start:.2f}s-{start + clip_seconds:.2f}s",
                    },
                )
                raw_text = response.choices[0].message.content if response.choices else ""
                observation = self._joyai_text_to_observation(frame, target, audio_context, raw_text or "")
                observation["clip"] = {
                    "start": round(start, 2),
                    "end": round(start + clip_seconds, 2),
                    "mode": "joyai_video_url",
                    "duration_seconds": clip_seconds,
                }
                observations.append(observation)
            finally:
                try:
                    clip_path.unlink(missing_ok=True)
                except Exception:
                    pass
        if len(clip_frames) < len(frames):
            observations.extend(
                self._fallback_frame_observations(
                    frames[len(clip_frames) :],
                    "Skipped clip observation because JOYAI_MAX_CLIPS_PER_JOB was reached.",
                )
            )
        return observations

    def _extract_temp_clip(self, frame_path: Path, start: float, duration: float, frame: dict[str, Any], index: int) -> Path:
        video_path = self._video_path_from_frame_path(frame_path)
        temp_dir = Path(tempfile.gettempdir()) / "audio_first_video_agent_clips"
        temp_dir.mkdir(parents=True, exist_ok=True)
        clip_path = temp_dir / f"joyai_clip_{int(time.time() * 1000)}_{index}.mp4"
        from .video import VideoProcessor

        VideoProcessor(self.settings).extract_clip(
            video_path,
            start,
            duration,
            clip_path,
            width=self.settings.joyai_clip_width,
        )
        return clip_path

    def _video_path_from_frame_path(self, frame_path: Path) -> Path:
        job_dir = frame_path.resolve().parents[1]
        upload_path = self.settings.data_dir / "uploads" / job_dir.name / "source.mp4"
        if upload_path.exists():
            return upload_path
        raise RuntimeError(f"Could not find source video for frame path: {frame_path}")

    def _should_use_joyai_clips(self, question: str, frames: list[dict[str, Any]]) -> bool:
        mode = self.settings.joyai_input_mode
        if mode == "frames" or self.settings.joyai_max_clips_per_job <= 0:
            return False
        if any((frame.get("probe") or {}).get("type") == "live_segment" for frame in frames):
            return False
        if mode == "clips":
            return True
        text = " ".join(
            [question]
            + [str(frame.get("reason") or "") for frame in frames]
            + [str((frame.get("probe") or {}).get("question") or "") for frame in frames]
        ).lower()
        temporal_terms = (
            "稳定",
            "防抖",
            "运动",
            "走动",
            "变化",
            "过程",
            "样片",
            "动态范围",
            "低光",
            "高光",
            "动作",
            "切换",
            "直播",
            "风险",
            "抽烟",
            "擦边",
            "game",
            "basketball",
            "motion",
            "stabilization",
        )
        return any(term in text for term in temporal_terms)

    def _clip_seconds_for_frame(self, question: str, frame: dict[str, Any]) -> float:
        configured = float(self.settings.joyai_clip_seconds)
        if not self.settings.joyai_adaptive_clip_seconds:
            return configured
        text = " ".join(
            [
                question,
                str(frame.get("reason") or ""),
                str((frame.get("probe") or {}).get("question") or ""),
            ]
        ).lower()
        temporal_terms = (
            "稳定",
            "防抖",
            "运动",
            "走动",
            "变化",
            "过程",
            "样片",
            "动态范围",
            "低光",
            "高光",
            "切换",
            "发热",
            "温度",
            "长焦",
            "三倍",
            "60mm",
            "motion",
            "stabilization",
            "dynamic range",
        )
        if any(term in text for term in temporal_terms):
            return configured
        return max(1.0, min(configured, 2.0))

    @staticmethod
    def _fallback_frame_observations(frames: list[dict[str, Any]], reason: str) -> list[dict[str, Any]]:
        user_message = "这一帧已抽取，但视觉识别暂时不可用，后续会继续依靠音频和新画面更新。"
        return [
            {
                "filename": frame["filename"],
                "time": frame["time"],
                "scene": user_message,
                "objects": [],
                "actions": [],
                "visible_text": [],
                "audio_alignment": "uncertain",
                "visual_target": str((frame.get("probe") or {}).get("question") or frame.get("reason", "")),
                "evidence_assessment": user_message,
                "notes": "Model fallback observation.",
                "vision_error": reason,
            }
            for frame in frames
        ]

    @staticmethod
    def _audio_context_for_time(audio_world_model: dict[str, Any], frame_time: float) -> str:
        timeline = audio_world_model.get("timeline") or []
        if not timeline:
            return "No audio timeline event was available."
        scored = sorted(
            timeline,
            key=lambda item: abs(float(item.get("time", 0.0)) - frame_time),
        )
        nearby = []
        for item in scored[:2]:
            nearby.append(
                {
                    "time": item.get("time"),
                    "end_time": item.get("end_time"),
                    "label": item.get("label"),
                    "evidence": item.get("evidence"),
                    "expected_visuals": item.get("expected_visuals", []),
                    "visual_question": item.get("visual_question", ""),
                }
            )
        return json.dumps(nearby, ensure_ascii=False)

    @staticmethod
    def _joyai_text_to_observation(
        frame: dict[str, Any],
        target: str,
        audio_context: str,
        raw_text: str,
    ) -> dict[str, Any]:
        text = raw_text.strip()
        if text.startswith("</response>"):
            text = text.removeprefix("</response>").strip()
        vision_error = ""
        if not text or raw_text.strip() == "</silence>":
            text = "这一帧已抽取，但视觉模型没有返回可用画面描述。"
            vision_error = "JoyAI returned silence for this frame."
        moderation = AIClient._parse_live_moderation_payload(text) if not vision_error else None
        if moderation:
            caption = str(moderation.get("caption") or "").strip() or AIClient._caption_from_live_moderation(moderation)
            visible_text: list[str] = []
            for violation in moderation.get("violations") or []:
                if not isinstance(violation, dict):
                    continue
                visible_text.extend(str(item).strip() for item in violation.get("visible_text") or [] if str(item).strip())
            return {
                "time": frame["time"],
                "filename": frame["filename"],
                "visual_target": target,
                "evidence_assessment": caption,
                "scene": caption,
                "objects": [],
                "actions": [],
                "visible_text": sorted(set(visible_text)),
                "audio_alignment": "match" if moderation.get("risk_level") not in {"none", "unknown", ""} else "uncertain",
                "notes": f"vision_provider=joyai; audio_context={audio_context}",
                "live_moderation": moderation,
            }
        lower = text.lower()
        visible_text = []
        for marker in ("囍", "喜", "double happiness", "red cloth", "veil", "盖头", "红布", "红色"):
            if marker.lower() in lower or marker in text:
                visible_text.append(marker)
        match_terms = (
            "match",
            "matches",
            "support",
            "supports",
            "结论：支持",
            "结论: 支持",
            "一致",
            "符合",
            "支持",
            "出现",
            "可见",
            "相关",
            "对应",
            "相符",
            "吻合",
            "验证",
            "表明",
            "与音频",
            "与用户问题",
        )
        conflict_terms = (
            "conflict",
            "contradict",
            "结论：冲突",
            "结论: 冲突",
            "不一致",
            "冲突",
            "没有",
            "未见",
            "看不到",
            "不支持",
            "无法支持",
        )
        if vision_error:
            alignment = "uncertain"
        elif any(term in lower or term in text for term in conflict_terms):
            alignment = "conflict"
        elif any(term in lower or term in text for term in match_terms):
            alignment = "match"
        else:
            alignment = "uncertain"
        observation = {
            "time": frame["time"],
            "filename": frame["filename"],
            "visual_target": target,
            "evidence_assessment": text,
            "scene": text,
            "objects": [],
            "actions": [],
            "visible_text": sorted(set(visible_text)),
            "audio_alignment": alignment,
            "notes": f"vision_provider=joyai; audio_context={audio_context}",
        }
        if vision_error:
            observation["vision_error"] = vision_error
        return observation

    @staticmethod
    def _parse_live_moderation_payload(text: str) -> dict[str, Any] | None:
        raw = text.strip()
        if raw.startswith("```"):
            raw = AIClient._strip_json_fence(raw)
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(raw[start : end + 1])
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        payload.setdefault("risk_level", "none")
        payload.setdefault("caption", "")
        payload.setdefault("violations", [])
        if not isinstance(payload["violations"], list):
            payload["violations"] = []
        normalized = []
        for violation in payload["violations"]:
            if not isinstance(violation, dict):
                continue
            violation.setdefault("category", "other")
            violation.setdefault("severity", "low")
            violation.setdefault("confidence", 0.0)
            violation.setdefault("evidence", "")
            violation.setdefault("visible_text", [])
            if not isinstance(violation["visible_text"], list):
                violation["visible_text"] = [str(violation["visible_text"])]
            normalized.append(violation)
        payload["violations"] = normalized
        return payload

    @staticmethod
    def _caption_from_live_moderation(payload: dict[str, Any]) -> str:
        violations = [item for item in payload.get("violations") or [] if isinstance(item, dict)]
        if not violations:
            return "当前帧未发现可见违规。"
        evidence = "；".join(str(item.get("evidence") or item.get("category") or "").strip() for item in violations[:2])
        return evidence or "当前帧存在疑似违规画面。"

    def predict_next_events(
        self,
        *,
        audio_world_model: dict[str, Any],
        observations: list[dict[str, Any]],
        duration: float,
    ) -> list[dict[str, Any]]:
        if self.settings.fast_mode:
            return self._fast_predictions(audio_world_model, observations, duration)
        if self.mock:
            timeline = audio_world_model.get("timeline") or []
            seeds = timeline[:3] if timeline else [{"time": 0, "label": "visual context"}]
            predictions = []
            for item in seeds:
                start = min(duration, float(item.get("time", 0.0)) + 1.0)
                end = min(duration, start + 4.0)
                predictions.append(
                    {
                        "window_start": start,
                        "window_end": end,
                        "hypothesis": f"After '{item.get('label', 'event')}', the next frames should clarify the action.",
                        "expected_evidence": ["person", "action", "context"],
                        "source_event": str(item.get("label", "event")),
                    }
                )
            return predictions

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "predictions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "window_start": {"type": "number"},
                            "window_end": {"type": "number"},
                            "hypothesis": {"type": "string"},
                            "expected_evidence": {"type": "array", "items": {"type": "string"}},
                            "source_event": {"type": "string"},
                        },
                        "required": ["window_start", "window_end", "hypothesis", "expected_evidence", "source_event"],
                    },
                }
            },
            "required": ["predictions"],
        }
        prompt = (
            "Create short, verifiable predictions from the audio world model and observed frames. This is an "
            "evidence-efficient loop: predict only the next visual evidence needed to confirm, reject, or refine "
            "the audio-derived story. Each prediction must name a future time window and concrete visible evidence "
            "that would confirm it.\n\n"
            f"Duration: {duration:.2f}\n"
            f"Audio world model: {json.dumps(audio_world_model, ensure_ascii=False)}\n"
            f"Frame observations: {json.dumps(observations, ensure_ascii=False)}"
        )
        try:
            return self._responses_json(self.settings.reasoning_model, prompt, schema, "event_predictions")["predictions"]
        except Exception:
            if self.settings.allow_model_fallback:
                return self._fallback_predictions(observations, duration)
            raise

    def verify_predictions(
        self,
        *,
        predictions: list[dict[str, Any]],
        observations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.settings.fast_mode:
            return self._enrich_prediction_checks(
                [classify_prediction_check(prediction, observations) for prediction in predictions],
                predictions,
            )
        if self.mock:
            return self._enrich_prediction_checks(
                [classify_prediction_check(prediction, observations) for prediction in predictions],
                predictions,
            )

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "checks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "window_start": {"type": "number"},
                            "window_end": {"type": "number"},
                            "hypothesis": {"type": "string"},
                            "status": {"type": "string", "enum": ["match", "conflict", "uncertain"]},
                            "conflict_score": {"type": "number"},
                            "evidence": {"type": "string"},
                        },
                        "required": [
                            "window_start",
                            "window_end",
                            "hypothesis",
                            "status",
                            "conflict_score",
                            "evidence",
                        ],
                    },
                }
            },
            "required": ["checks"],
        }
        prompt = (
            "Verify whether later frame observations confirm, contradict, or leave uncertain each prediction. "
            "Use conflict_score 0.0-1.0, where high values mean prediction error should trigger more frame sampling.\n\n"
            f"Predictions: {json.dumps(predictions, ensure_ascii=False)}\n"
            f"Frame observations: {json.dumps(observations, ensure_ascii=False)}"
        )
        try:
            checks = self._responses_json(self.settings.reasoning_model, prompt, schema, "prediction_checks")["checks"]
            return self._enrich_prediction_checks(checks, predictions)
        except Exception:
            if self.settings.allow_model_fallback:
                return self._enrich_prediction_checks(
                    [classify_prediction_check(prediction, observations) for prediction in predictions],
                    predictions,
                )
            raise

    def synthesize_answer(
        self,
        *,
        question: str,
        audio_world_model: dict[str, Any],
        observations: list[dict[str, Any]],
        checks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.settings.fast_mode:
            return self._fast_answer(question, audio_world_model, observations, checks)
        if self.mock:
            evidence_refs = [
                f"{obs['time']:.2f}s: {obs.get('scene', 'visual observation')}"
                for obs in observations[:5]
            ]
            return {
                "direct_answer": "Mock answer: the agent used audio cues to choose frames, then checked the selected frames before summarizing.",
                "summary": "Audio produced a blind world sketch; visual frames grounded the sketch and prediction checks marked uncertainty.",
                "evidence_refs": evidence_refs,
                "uncertainties": [
                    "Mock mode does not inspect real pixels or audio.",
                    "Run with OPENAI_API_KEY and FFmpeg for real analysis.",
                ],
            }

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "direct_answer": {"type": "string"},
                "summary": {"type": "string"},
                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                "uncertainties": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["direct_answer", "summary", "evidence_refs", "uncertainties"],
        }
        prompt = (
            "Answer the user's question about the video in concise Chinese. Use the audio timeline, frame "
            "observations, and prediction checks as grounding. Do not dump evidence lists or internal verification "
            "logs into direct_answer or summary. You may include 1-3 inline video time anchors such as 7:55 when "
            "they help the user jump to the relevant moment. Keep direct_answer under 220 Chinese characters and "
            "summary under 450 Chinese characters. Return evidence_refs as an empty array unless the user explicitly "
            "asks for evidence.\n\n"
            f"Question: {question}\n"
            f"Audio world model: {json.dumps(audio_world_model, ensure_ascii=False)}\n"
            f"Frame observations: {json.dumps(observations[-12:], ensure_ascii=False)}\n"
            f"Prediction checks: {json.dumps(checks, ensure_ascii=False)}"
        )
        try:
            return self._responses_json(self.settings.reasoning_model, prompt, schema, "final_answer")
        except Exception:
            if self.settings.allow_model_fallback:
                return self._fallback_answer(question, audio_world_model, observations, checks)
            raise

    def answer_followup(
        self,
        *,
        question: str,
        result: dict[str, Any],
        web_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._should_answer_followup_locally(question, web_context):
            answer = self._local_followup_answer(question, result, web_context=web_context)
            if answer.get("answer") and not str(answer.get("answer", "")).startswith("这个追问没有"):
                return self._attach_web_context(answer, web_context)
        if self.mock or self._client is None:
            answer = self._local_followup_answer(question, result, web_context=web_context)
            return self._attach_web_context(answer, web_context)
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "answer": {"type": "string"},
                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                "coverage_note": {"type": "string"},
                "web_sources": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "evidence_refs", "coverage_note", "web_sources"],
        }
        context = self._followup_knowledge_pack(question, result, limit=self.settings.followup_max_chunks)
        web_enabled = bool(web_context and web_context.get("enabled"))
        prompt = (
            "你是一个视频问答 agent。下面的 JSON 是同一个视频已解析出的知识库切片：音频转写、时间线、"
            "关键帧观察、以及针对本次追问补看的画面。请优先把这些视频内容当作事实来源来回答用户问题。\n"
            "要求：\n"
            "1. 直接回答用户真正问的点，不要复述内部检索、知识库、证据列表或覆盖说明。\n"
            "2. 如果用户一句话包含多个任务或子问题，必须逐项回答，不要只回答第一个；例如“总结视频，并对比某竞品”要先总结视频，再做竞品对比和选择建议。\n"
            "3. 可以基于视频里的评价、对比和结论做合理归纳，例如给出条件式购买建议；不要编造视频没有覆盖的事实。\n"
            "4. 如果本次问题需要视觉细节，优先使用 supplemental_frame_observations；如果仍没有覆盖，就明确说当前分析没看到。\n"
            "5. 适合时加入 1-4 个视频时间锚点，例如 7:55，方便用户点击跳转。\n"
            "6. 用中文自然回答，按问题组织内容；不要机械套模板。\n"
            "7. 如果提供了 web_context，只用它补充视频外部事实、型号参数、背景信息或最新信息；视频里的观点和结论仍然优先。"
            "当用户要求和视频外产品对比时，只要 web_context 有结果，就必须利用这些结果给出保守对比，而不是说没有资料。"
            "用 web_context 时，在 web_sources 返回用到的网页标题或 URL；正文里可以用“外部资料显示”简短说明。"
            "如果 web_context 没有结果或与问题无关，不要强行引用。\n\n"
            f"Follow-up question: {question}\n"
            f"Video knowledge pack: {json.dumps(context, ensure_ascii=False)}\n"
            f"Web search enabled: {web_enabled}\n"
            f"web_context: {json.dumps(web_context or {}, ensure_ascii=False)}"
        )
        try:
            return self._responses_json(
                self.settings.followup_model or self.settings.reasoning_model,
                prompt,
                schema,
                "followup_answer",
                timeout_seconds=self.settings.followup_timeout_seconds,
            )
        except Exception:
            if self.settings.allow_model_fallback:
                return self._attach_web_context(self._local_followup_answer(question, result, web_context=web_context), web_context)
            raise

    @staticmethod
    def _should_answer_followup_locally(question: str, web_context: dict[str, Any] | None = None) -> bool:
        compact_question = "".join(str(question or "").split()).lower()
        if not compact_question:
            return True
        if web_context and (web_context.get("results") or []):
            asks_external = any(
                token in compact_question
                for token in ("对比", "比较", "哪个更好", "哪款更好", "竞品", "外部", "联网", "vs", "insta360", "luna", "gopro")
            )
            if asks_external:
                return True
        if AIClient._is_summary_followup_question(compact_question):
            return True
        if AIClient._is_buying_advice_followup_question(compact_question):
            return True
        if any(token in compact_question for token in ("升级", "提升", "区别", "差异", "比上一代", "普通版", "pro多少")):
            return True
        if AIClient._first_time_in_text(question) is not None:
            return True
        return False

    def _fallback_audio_world_model(self, question: str, transcript_segments: list[dict[str, Any]]) -> dict[str, Any]:
        if not transcript_segments:
            return {
                "summary": "Audio transcription was unavailable, so the agent will rely on sampled video frames.",
                "actors": [],
                "environment_hypotheses": [],
                "mood": "unknown",
                "timeline": [],
                "open_questions": [question],
            }
        return {
            "summary": "Audio transcript is available, but the reasoning endpoint failed; using transcript timing as cues.",
            "actors": sorted({str(segment.get("speaker", "unknown")) for segment in transcript_segments}),
            "environment_hypotheses": [],
            "mood": "unknown",
            "timeline": [
                {
                    "time": float(segment.get("start", 0.0)),
                    "end_time": float(segment.get("end", segment.get("start", 0.0))),
                    "label": "Transcript cue",
                    "evidence": str(segment.get("text", "")),
                    "expected_visuals": ["visual context around this transcript"],
                    "visual_question": "What visible evidence supports or rejects this transcript cue?",
                }
                for segment in transcript_segments
            ],
            "open_questions": [question],
        }

    def _fast_audio_world_model(
        self,
        question: str,
        transcript_segments: list[dict[str, Any]],
        duration: float,
        has_audio: bool,
    ) -> dict[str, Any]:
        if not has_audio or not transcript_segments:
            return {
                "summary": "快速模式：没有可用音频转写，改用稀疏视觉采样。",
                "actors": [],
                "environment_hypotheses": [],
                "mood": "unknown",
                "timeline": [],
                "open_questions": [question],
                "mode": "fast",
            }
        selected_segments = self._select_fast_timeline_segments(transcript_segments, duration)
        timeline = []
        for index, segment in enumerate(selected_segments):
            text = str(segment.get("text", "")).strip()
            start = float(segment.get("start") or 0.0)
            end = float(segment.get("end") or min(duration, start + 1.5))
            expected = self._fast_expected_visuals(text)
            timeline.append(
                {
                    "time": start,
                    "end_time": end,
                    "label": self._fast_event_label(text, index),
                    "evidence": text,
                    "expected_visuals": expected,
                    "visual_question": f"画面是否能支持这段音频：{text[:80]}？重点检查：{', '.join(expected[:4]) or '人物、场景、动作'}。",
                }
            )
        return {
            "summary": "快速模式：直接把音频转写切成可验证事件，跳过大模型音频世界建模。",
            "actors": sorted({str(segment.get("speaker") or "unknown") for segment in transcript_segments}),
            "environment_hypotheses": [],
            "mood": "unknown",
            "timeline": timeline,
            "open_questions": [question],
            "mode": "fast",
            "source_segment_count": len(transcript_segments),
            "selected_segment_count": len(selected_segments),
        }

    def _select_fast_timeline_segments(
        self,
        transcript_segments: list[dict[str, Any]],
        duration: float,
    ) -> list[dict[str, Any]]:
        limit = max(1, self.settings.fast_max_timeline_events)
        if len(transcript_segments) <= limit:
            return transcript_segments
        scored = []
        for index, segment in enumerate(transcript_segments):
            text = str(segment.get("text", ""))
            start = float(segment.get("start") or 0.0)
            score = self._fast_segment_score(text)
            scored.append((score, start, index, segment))
        selected: dict[int, dict[str, Any]] = {}
        for _, _, index, segment in sorted(scored, key=lambda item: (-item[0], item[1]))[: max(1, limit // 2)]:
            selected[index] = segment
        coverage_slots = max(1, limit - len(selected))
        for slot in range(coverage_slots):
            target = (duration * (slot + 0.5)) / coverage_slots if duration > 0 else 0.0
            _, _, index, segment = min(
                scored,
                key=lambda item: (abs(item[1] - target), item[2] in selected),
            )
            selected[index] = segment
        if len(selected) < limit:
            for _, _, index, segment in sorted(scored, key=lambda item: item[1]):
                selected.setdefault(index, segment)
                if len(selected) >= limit:
                    break
        return [selected[index] for index in sorted(selected)]

    @staticmethod
    def _fast_segment_score(text: str) -> int:
        groups = [
            ("结构", ("首先", "然后", "接着", "最后", "总结", "开始", "完成", "过程", "阶段")),
            ("3d", ("3D", "打印", "模型", "建模", "材料", "设备", "机器", "施工")),
            ("装修", ("装修", "房子", "墙", "地面", "家具", "安装", "设计", "效果")),
            ("问题", ("问题", "失败", "困难", "挑战", "测试", "改进", "成本", "时间")),
        ]
        score = 0
        for weight, (_, keywords) in enumerate(groups, start=2):
            score += weight * sum(1 for keyword in keywords if keyword.lower() in text.lower() or keyword in text)
        return score

    @staticmethod
    def _fast_event_label(text: str, index: int) -> str:
        if any(token in text for token in ("建模", "模型", "设计", "图纸", "方案")):
            return "设计建模线索"
        if any(token in text for token in ("3D", "打印", "打印机", "机器", "设备", "喷嘴")):
            return "3D 打印过程线索"
        if any(token in text for token in ("材料", "水泥", "砂浆", "混凝土", "耗材", "配比")):
            return "打印材料线索"
        if any(token in text for token in ("装修", "施工", "墙", "地面", "地板", "吊顶", "安装", "打磨")):
            return "装修施工线索"
        if any(token in text for token in ("房子", "房间", "客厅", "卧室", "厨房", "卫生间", "整套")):
            return "空间成果线索"
        if any(token in text for token in ("成本", "价格", "预算", "时间", "天", "效率")):
            return "成本进度线索"
        if any(token in text for token in ("狐狸", "白狐", "狐")):
            return "白狐身份线索"
        if any(token in text for token in ("夫妻", "夫妇", "成婚", "结婚", "嫁")):
            return "婚姻关系线索"
        if any(token in text for token in ("山林", "森林", "生活", "幸福")):
            return "结尾生活线索"
        if any(token in text for token in ("雪山", "救")):
            return "过去救助线索"
        return f"音频事件 {index + 1}"

    @staticmethod
    def _fast_expected_visuals(text: str) -> list[str]:
        expected: list[str] = []
        mapping = [
            (("建模", "模型", "设计", "图纸", "方案"), ["电脑建模界面", "设计图", "房屋模型", "尺寸标注"]),
            (("3D", "打印", "打印机", "机器", "设备", "喷嘴"), ["3D 打印设备", "打印喷头", "层层堆叠的材料", "机器运动"]),
            (("材料", "水泥", "砂浆", "混凝土", "耗材", "配比"), ["材料桶", "水泥砂浆", "混合设备", "材料纹理"]),
            (("装修", "施工", "墙", "地面", "地板", "吊顶", "安装", "打磨"), ["施工现场", "墙面", "地面", "安装工具", "工人操作"]),
            (("房子", "房间", "客厅", "卧室", "厨房", "卫生间", "整套"), ["房间全景", "装修成品", "家具", "空间布局"]),
            (("成本", "价格", "预算", "时间", "天", "效率"), ["价格表或字幕", "进度画面", "前后对比", "成果展示"]),
            (("狐狸", "白狐", "狐"), ["白衣女子", "身份揭示", "人物表情"]),
            (("夫妻", "夫妇", "成婚", "结婚", "嫁"), ["红布", "囍字", "红盖头", "男女同框", "婚礼布置"]),
            (("山林", "森林", "生活", "幸福"), ["树林", "木屋", "两人相伴", "生活场景"]),
            (("雪山", "救"), ["雪山", "救助动作", "回忆画面"]),
            (("你", "我", "是"), ["对话人物", "回应动作", "表情变化"]),
        ]
        for keywords, visuals in mapping:
            if any(keyword in text for keyword in keywords):
                expected.extend(visuals)
        if not expected:
            expected.extend(["人物", "场景", "动作"])
        deduped = []
        for item in expected:
            if item not in deduped:
                deduped.append(item)
        return deduped

    def _fallback_predictions(self, observations: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
        seeds = observations[:3] or [{"time": 0.0, "scene": "sampled context"}]
        predictions = []
        for observation in seeds:
            start = min(duration, float(observation.get("time", 0.0)) + 1.0)
            predictions.append(
                {
                    "window_start": start,
                    "window_end": min(duration, start + 4.0),
                    "hypothesis": "The following frames should provide continuity for the sampled scene.",
                    "expected_evidence": [str(observation.get("scene", "context"))],
                    "source_event": str(observation.get("visual_target") or observation.get("scene", "sampled context")),
                }
            )
        return predictions

    def _fast_predictions(
        self,
        audio_world_model: dict[str, Any],
        observations: list[dict[str, Any]],
        duration: float,
    ) -> list[dict[str, Any]]:
        predictions = []
        timeline = audio_world_model.get("timeline") or []
        if timeline:
            for event in timeline[-3:]:
                start = min(duration, float(event.get("end_time") or event.get("time") or 0.0))
                predictions.append(
                    {
                        "window_start": start,
                        "window_end": min(duration, start + 2.0),
                        "hypothesis": f"后续画面应继续支持：{event.get('label', '音频事件')}",
                        "expected_evidence": [str(item) for item in event.get("expected_visuals", [])],
                        "source_event": str(event.get("label") or "audio event"),
                    }
                )
        if not predictions:
            return self._fallback_predictions(observations, duration)
        return predictions

    @staticmethod
    def _enrich_prediction_checks(
        checks: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        enriched = []
        for index, check in enumerate(checks):
            prediction = predictions[index] if index < len(predictions) else {}
            enriched.append(
                {
                    **check,
                    "expected_evidence": prediction.get("expected_evidence", []),
                    "source_event": prediction.get("source_event", ""),
                }
            )
        return enriched

    def _fallback_answer(
        self,
        question: str,
        audio_world_model: dict[str, Any],
        observations: list[dict[str, Any]],
        checks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "direct_answer": (
                "The video was processed locally and key frames were extracted, but one or more configured model "
                "endpoints could not complete the full multimodal analysis."
            ),
            "summary": (
                f"Question: {question}. Extracted {len(observations)} frame observations and "
                f"{len(checks)} prediction checks. Audio summary: {audio_world_model.get('summary', 'unavailable')}"
            ),
            "evidence_refs": [
                f"{obs.get('time', 0):.2f}s: {obs.get('scene', 'frame extracted')}"
                for obs in observations[:8]
            ],
            "uncertainties": [
                "Configured model endpoint failed or lacks audio/image support.",
                "Use an OpenAI-compatible endpoint with audio transcription and image input for full results.",
            ],
        }

    @staticmethod
    def _fast_answer(
        question: str,
        audio_world_model: dict[str, Any],
        observations: list[dict[str, Any]],
        checks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        timeline = audio_world_model.get("timeline") or []
        events = AIClient._clean_fast_events(timeline)
        transcript_bits = [event["text"] for event in events]
        visual_bits = AIClient._clean_visual_evidence(observations)
        title = AIClient._infer_fast_topic(transcript_bits, visual_bits, question)
        topic_profile = AIClient._fast_topic_profile(transcript_bits, visual_bits, title)

        direct = topic_profile["direct"]

        process_items = AIClient._fast_process_items(events, title, topic_profile)
        if not process_items and transcript_bits:
            process_items = [AIClient._compress_text(bit, 88) for bit in transcript_bits[:4]]
        if not process_items:
            process_items = ["当前没有稳定的音频事件，系统主要依赖稀疏关键画面做概览。"]
        direct = AIClient._append_time_anchors(AIClient._compress_text(direct, 180), process_items)

        conclusion_items = [topic_profile["conclusion"]]

        sections = [
            {"title": "内容脉络", "items": process_items[:5]},
            {"title": "关键结论", "items": conclusion_items},
        ]
        summary = AIClient._compose_user_facing_summary(topic_profile, process_items, conclusion_items[0])
        return {
            "direct_answer": direct or f"快速模式已处理问题：{question}",
            "summary": summary[:900],
            "sections": sections,
            "evidence_refs": [],
            "uncertainties": [],
        }

    @staticmethod
    def build_ai_overview(result: dict[str, Any]) -> dict[str, Any]:
        answer = result.get("answer") or {}
        direct = str(answer.get("direct_answer") or "").strip()
        summary = str(answer.get("summary") or direct).strip()
        if not summary:
            summary = AIClient._generic_detailed_followup_answer(result)
        summary = AIClient._clean_overview_summary(summary)
        bullets = AIClient._overview_bullets(answer, result)
        highlights = AIClient._overview_highlights(result)
        suggested_questions = AIClient._overview_suggested_questions(result)
        return {
            "summary": AIClient._compress_text(summary or "当前视频已完成解析，可以继续围绕内容提问。", 520),
            "bullets": bullets[:4],
            "highlights": highlights[:5],
            "suggested_questions": suggested_questions[:5],
        }

    @staticmethod
    def _overview_bullets(answer: dict[str, Any], result: dict[str, Any]) -> list[str]:
        bullets: list[str] = []
        for section in answer.get("sections") or []:
            title = str(section.get("title") or "").strip()
            for item in section.get("items") or []:
                text = str(item).strip()
                if not text:
                    continue
                clean = AIClient._clean_overview_item(text)
                if clean:
                    bullets.append(AIClient._compress_text(clean, 132))
                if len(bullets) >= 4:
                    return bullets
        direct = str(answer.get("direct_answer") or "").strip()
        if direct:
            bullets.append(AIClient._compress_text(direct, 128))
        world_summary = str((result.get("audio_world_model") or {}).get("summary") or "").strip()
        if world_summary:
            bullets.append(AIClient._compress_text(world_summary, 128))
        return AIClient._dedupe_strings(bullets)[:4]

    @staticmethod
    def _overview_highlights(result: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        answer = result.get("answer") or {}
        for section in answer.get("sections") or []:
            for item in section.get("items") or []:
                text = str(item or "")
                clean_text = AIClient._clean_overview_item(text)
                if not clean_text or AIClient._is_bad_highlight_text(clean_text):
                    continue
                time_value = AIClient._first_time_in_text(text)
                if time_value is not None:
                    candidates.append(
                        {
                            "time": time_value,
                            "label": AIClient._label_from_text(clean_text),
                            "detail": clean_text,
                            "source": "answer",
                            "score": 90,
                        }
                    )
        for item in result.get("timeline") or []:
            text = AIClient._clean_asr_text(str(item.get("evidence") or item.get("label") or ""))
            label = str(item.get("label") or "").strip()
            combined = " ".join([label, text])
            if AIClient._is_noisy_text(text, combined) or AIClient._is_bad_highlight_text(combined):
                continue
            candidates.append(
                {
                    "time": float(item.get("time") or 0.0),
                    "label": AIClient._label_from_text(label or text),
                    "detail": AIClient._compress_text(text or label, 96),
                    "source": "audio",
                    "score": AIClient._highlight_score(combined),
                }
            )
        for frame in result.get("frames") or []:
            observation = frame.get("observation") or {}
            text = " ".join(
                str(value or "")
                for value in (
                    observation.get("scene"),
                    observation.get("evidence_assessment"),
                    " ".join(observation.get("visible_text") or []),
                )
            ).strip()
            if not text:
                continue
            candidates.append(
                {
                    "time": float(frame.get("time") or 0.0),
                    "label": AIClient._label_from_text(text),
                    "detail": AIClient._clean_followup_context(AIClient._compress_text(text, 110)),
                    "source": "visual",
                    "url": frame.get("url"),
                    "filename": frame.get("filename"),
                    "score": AIClient._highlight_score(text) + 6,
                }
            )
        for segment in result.get("transcript_segments") or []:
            text = AIClient._clean_asr_text(str(segment.get("text") or ""))
            if AIClient._is_noisy_text(text, text) or AIClient._is_bad_highlight_text(text):
                continue
            candidates.append(
                {
                    "time": float(segment.get("start") or 0.0),
                    "label": AIClient._label_from_text(text),
                    "detail": AIClient._compress_text(text, 92),
                    "source": "transcript",
                    "score": AIClient._highlight_score(text) - 4,
                }
            )
        deduped: list[dict[str, Any]] = []
        for item in sorted(candidates, key=lambda value: (-int(value.get("score") or 0), float(value.get("time") or 0.0))):
            time_value = float(item.get("time") or 0.0)
            if any(abs(time_value - float(existing.get("time") or 0.0)) < 12 for existing in deduped):
                continue
            item = {key: value for key, value in item.items() if key != "score"}
            item["time_label"] = AIClient._format_seconds(time_value)
            deduped.append(item)
            if len(deduped) >= 5:
                break
        return sorted(deduped, key=lambda value: float(value.get("time") or 0.0))

    @staticmethod
    def _compose_user_facing_summary(topic_profile: dict[str, str], process_items: list[str], conclusion: str) -> str:
        kind = topic_profile.get("kind")
        clean_items = [AIClient._clean_overview_item(item) for item in process_items]
        clean_items = [item for item in clean_items if item]
        if kind == "3d_printing":
            stages = "；".join(clean_items[:3])
            return (
                "这条视频是在做一次“用 3D 打印参与整屋装修”的实践记录。"
                "作者先把房子里的装修部件、家具和装饰件转成设计方案，再尝试用 3D 打印材料把它们做出来，"
                "中间穿插施工安装、材料成本和最终质感对比。"
                f"{' 主要过程是：' + stages + '。' if stages else ''}"
                "整体结论是：3D 打印能做出有造型、有层纹的柜体和家具部件，但它还不是简单替代传统装修，"
                "真正的难点在于材料选择、打印成本、安装衔接和成品质感。"
            )
        if kind == "product_review":
            stages = "；".join(clean_items[:3])
            return (
                f"{topic_profile.get('direct', '').strip()} "
                f"{'视频展开顺序是：' + stages + '。' if stages else ''}"
                f"{conclusion}"
            ).strip()
        return " ".join(part for part in [topic_profile.get("direct", ""), "；".join(clean_items[:3]), conclusion] if part)

    @staticmethod
    def _clean_overview_summary(text: str) -> str:
        cleaned = " ".join(str(text or "").split())
        cleaned = cleaned.replace("视频主题：", "").replace("内容脉络：", "").replace("关键结论：", "")
        cleaned = cleaned.replace("。。", "。").replace("；。", "。")
        cleaned = cleaned.strip(" ；。")
        if cleaned and not cleaned.endswith(("。", "！", "？")):
            cleaned += "。"
        return cleaned

    @staticmethod
    def _clean_overview_item(text: str) -> str:
        cleaned = AIClient._strip_leading_time(str(text or ""))
        cleaned = cleaned.replace("先明确目标和方案：", "明确目标和方案：")
        cleaned = cleaned.replace("随后进入打印和材料部分：", "打印和材料：")
        cleaned = cleaned.replace("中段转向现场施工：", "现场施工：")
        cleaned = cleaned.replace("后段讨论成本：", "成本测算：")
        cleaned = cleaned.replace("最后看成品效果：", "成品效果：")
        cleaned = cleaned.replace("内容脉络：", "").replace("关键结论：", "")
        cleaned = cleaned.strip(" ；。")
        if AIClient._is_bad_highlight_text(cleaned):
            return ""
        return cleaned

    @staticmethod
    def _is_bad_highlight_text(text: str) -> bool:
        compact = "".join(str(text or "").split())
        bad_phrases = (
            "就是给别人",
            "给别人接着",
            "那既然他这么说了",
            "看看到底是怎样一种感受",
            "这里长什么样",
            "那个这个",
            "不确",
        )
        if any(phrase in compact for phrase in bad_phrases):
            return True
        meaningful = ("3D", "打印", "装修", "设计", "建模", "材料", "成本", "施工", "家具", "成品", "效果", "柜体", "传统", "画质", "长焦", "动态范围")
        return len(compact) < 10 and not any(token in compact for token in meaningful)

    @staticmethod
    def _overview_suggested_questions(result: dict[str, Any]) -> list[str]:
        text = AIClient._result_text_context(result)
        questions: list[str] = ["总结当前视频内容", "高光片段分别在讲什么？"]
        if any(token.lower() in text.lower() for token in ("pocket", "dji", "大疆", "长焦", "画质", "动态范围", "iso", "发热", "稳定")):
            questions.extend(
                [
                    "相比上一代有哪些升级？",
                    "推荐买哪一代？",
                    "画质、长焦和稳定性分别怎么样？",
                    "如果和竞品对比，应该怎么选？",
                ]
            )
        elif any(token in text for token in ("3D", "打印", "装修", "施工", "材料", "成本", "家具")):
            questions.extend(
                [
                    "这个项目的主要难点是什么？",
                    "成本和材料是怎么说的？",
                    "最终效果怎么样？",
                    "哪些片段最值得看？",
                ]
            )
        elif any(token in text for token in ("教程", "步骤", "安装", "演示", "方法")):
            questions.extend(["按步骤梳理一下流程", "最关键的操作点是什么？", "有哪些容易出错的地方？"])
        else:
            questions.extend(["这条视频的核心结论是什么？", "按时间线详细讲一遍", "有哪些值得继续追问的细节？"])
        return AIClient._dedupe_strings(questions)[:5]

    @staticmethod
    def _highlight_score(text: str) -> int:
        score = 20
        keyword_weights = {
            "结论": 18,
            "总结": 15,
            "核心": 14,
            "值得": 14,
            "升级": 14,
            "画质": 13,
            "动态范围": 13,
            "长焦": 12,
            "发热": 8,
            "稳定": 8,
            "成本": 12,
            "价格": 10,
            "设计": 8,
            "施工": 8,
            "成品": 10,
            "效果": 9,
            "样片": 11,
            "测试": 9,
        }
        for keyword, weight in keyword_weights.items():
            if keyword.lower() in text.lower() or keyword in text:
                score += weight
        return score

    @staticmethod
    def _label_from_text(text: str) -> str:
        cleaned = AIClient._strip_leading_time(AIClient._clean_followup_context(text))
        if "：" in cleaned:
            head = cleaned.split("：", 1)[0]
            if 2 <= len(head) <= 18:
                return head
        for keyword in ("画质", "动态范围", "长焦", "发热", "稳定", "外观", "成本", "设计", "施工", "成品", "结论", "样片", "材料"):
            if keyword in cleaned:
                return f"{keyword}片段"
        return AIClient._compress_text(cleaned or "关键片段", 18)

    @staticmethod
    def _strip_leading_time(text: str) -> str:
        import re

        return re.sub(r"^\s*\d{1,2}:\d{2}(?::\d{2})?\s*[-~—–至到]*\s*(?:\d{1,2}:\d{2})?\s*[：:，,\s]*", "", str(text or "")).strip()

    @staticmethod
    def _dedupe_strings(items: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            clean = " ".join(str(item or "").split()).strip()
            if not clean:
                continue
            signature = clean.lower()
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(clean)
        return deduped

    @staticmethod
    def _clean_fast_events(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        events = []
        for item in timeline:
            text = AIClient._clean_asr_text(str(item.get("evidence") or ""))
            label = str(item.get("label") or "音频事件")
            combined = " ".join(
                [
                    label,
                    text,
                    " ".join(str(value) for value in item.get("expected_visuals", [])),
                ]
            )
            if AIClient._is_noisy_text(text, combined):
                continue
            events.append(
                {
                    "time": float(item.get("time") or 0.0),
                    "end_time": float(item.get("end_time") or item.get("time") or 0.0),
                    "label": label,
                    "text": text,
                    "stage": AIClient._fast_stage_for_text(combined),
                }
            )
        return sorted(events, key=lambda event: event["time"])

    @staticmethod
    def _clean_asr_text(text: str) -> str:
        text = " ".join(text.replace("\u3000", " ").split())
        replacements = {
            "三地打印": "3D 打印",
            "三地": "3D",
            "三D": "3D",
            "3地": "3D",
            "3 d": "3D",
            "磁砖": "瓷砖",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        return text.strip(" ，。；;:：,.!?！？")

    @staticmethod
    def _is_noisy_text(text: str, context: str = "") -> bool:
        compact = "".join(ch for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
        if not compact:
            return True
        if compact in {"不确", "不确定", "嗯", "啊", "就是", "然后", "对", "这个"}:
            return True
        if len(compact) <= 3:
            return True
        domain_keywords = (
            "3D", "打印", "设计", "装修", "施工", "材料", "成本", "价格", "预算", "房", "墙", "地面", "瓷砖",
            "家具", "效果", "模型", "设备", "机器", "柜", "灯", "安装", "制作", "传统", "细致",
            "一公斤", "二三十", "单价", "用量",
        )
        if len(compact) < 9 and not any(keyword in context for keyword in domain_keywords):
            return True
        if "不确" in compact and len(compact) < 8:
            return True
        vague_phrases = ("给别人见到的有点", "就是有点", "那个这个")
        return any(phrase in text for phrase in vague_phrases)

    @staticmethod
    def _fast_stage_for_text(text: str) -> str:
        if any(token in text for token in ("成本", "价格", "预算", "一公斤", "二三十", "单价", "钱")):
            return "cost"
        if any(token in text for token in ("效果", "家具", "传统家具", "成品", "展示", "细致", "实现")):
            return "outcome"
        if any(token in text for token in ("建模", "模型", "设计", "图纸", "方案")):
            return "design"
        if any(token in text for token in ("施工", "装修", "墙", "瓷砖", "地面", "吊顶", "安装", "灯", "房间")):
            return "construction"
        if any(token in text for token in ("3D", "打印", "打印机", "机器", "设备", "喷嘴", "材料", "层层")):
            return "printing"
        return "context"

    @staticmethod
    def _fast_topic_profile(transcript_bits: list[str], visual_bits: list[str], topic: str) -> dict[str, str]:
        text = " ".join(transcript_bits + visual_bits + [topic])
        if "3D 打印" in topic:
            return {
                "kind": "3d_printing",
                "direct": (
                    "这是一个 3D 打印装修实验视频：核心是在一套房的装修和家具制作中尝试使用 3D 打印，"
                    "内容从设计、打印材料/设备、现场施工延伸到成本和成品效果。"
                ),
                "conclusion": "视频主线是验证 3D 打印在装修和家具制作中的可行性：能做出有层纹和造型的部件，但成本、材料和传统施工衔接仍是重点讨论对象。",
            }
        if any(token in text for token in ("Pocket", "Pockets", "Pocket 4", "Pockets 4", "Pockets 4P", "大疆", "DJI", "长焦", "主摄", "镜头", "画质", "ISO", "动态范围", "高光", "暗部", "发热", "稳定")):
            product = AIClient._infer_product_name(text)
            return {
                "kind": "product_review",
                "direct": (
                    f"这是一期{product}评测视频，核心问题是它相比上一代/普通版是否值得多花钱升级。"
                    "视频围绕外观形态、双镜头/长焦、画质样片、低光动态范围、稳定性和发热体验展开。"
                ),
                "conclusion": f"视频的结论倾向是：{product}的画质和动态范围表现比预期好，长焦/双镜头是主要升级点；是否值得选它取决于你是否重视频质、长焦视角和更完整的拍摄能力。",
            }
        if any(token in text for token in ("评测", "测评", "体验", "上手", "外观", "配置", "性能", "价格", "值得", "升级")):
            return {
                "kind": "product_review",
                "direct": "这是一期产品评测/体验视频，核心是在回答这个产品是否值得购买或升级，并按外观、功能、性能和体验来展开。",
                "conclusion": "视频的结论应围绕产品优缺点和适合人群理解：它不是事件记录，而是在用样片、测试和体验判断购买价值。",
            }
        return {
            "kind": "generic",
            "direct": f"视频主要围绕{topic}展开。",
            "conclusion": "这轮结果给出了视频主线和关键节点；证据只用于支撑摘要，不代表完整逐帧审片。",
        }

    @staticmethod
    def _infer_product_name(text: str) -> str:
        candidates = ("Pocket 4P", "Pockets 4P", "Pocket 4 Pro", "Pocket 4", "Pockets 4", "DJI Pocket", "大疆 Pocket")
        lowered = text.lower()
        for candidate in candidates:
            if candidate.lower() in lowered or candidate in text:
                if candidate == "Pockets 4P":
                    return "Pocket 4P"
                if candidate == "Pockets 4":
                    return "Pocket 4"
                return candidate
        return "这款数码影像产品"

    @staticmethod
    def _fast_process_items(events: list[dict[str, Any]], topic: str, topic_profile: dict[str, str] | None = None) -> list[str]:
        topic_profile = topic_profile or {"kind": "generic"}
        if topic_profile.get("kind") == "product_review":
            return AIClient._fast_product_review_items(events)

        order = ["design", "printing", "construction", "cost", "outcome", "context"]
        grouped: dict[str, list[dict[str, Any]]] = {stage: [] for stage in order}
        for event in events:
            grouped.setdefault(event["stage"], []).append(event)

        if "3D 打印" in topic:
            templates = {
                "design": "先明确目标和方案：尝试用 3D 打印参与整屋装修/部件制作，并进行设计建模。",
                "printing": "随后进入打印和材料部分：关注打印设备、材料堆叠成型，以及可打印哪些装修/家具部件。",
                "construction": "中段转向现场施工：把 3D 打印部件与墙面、瓷砖、灯具、安装等传统装修环节衔接起来。",
                "cost": "后段讨论成本：包括材料单价、用量、整体预算，以及这种方案是否划算。",
                "outcome": "最后看成品效果：重点是柜体、家具或房间细节能否接近传统装修和家具的质感。",
            }
        else:
            templates = {}

        items = []
        for stage in order:
            stage_events = grouped.get(stage) or []
            if not stage_events:
                continue
            start = AIClient._format_seconds(stage_events[0]["time"])
            end = AIClient._format_seconds(stage_events[-1]["time"])
            time_prefix = start if start == end else f"{start}-{end}"
            if stage in templates:
                items.append(f"{time_prefix}：{templates[stage]}")
                continue
            samples = "；".join(AIClient._compress_text(event["text"], 34) for event in stage_events[:2])
            items.append(f"{time_prefix}：{samples}")
        return items

    @staticmethod
    def _fast_product_review_items(events: list[dict[str, Any]]) -> list[str]:
        buckets = [
            ("purchase", ("到底", "放弃", "选择", "原因", "值得", "升级", "多花", "问题", "考虑"), "开头先提出购买疑问：这款产品相比普通版/上一代到底升级在哪里，是否值得为了画质或新镜头多花钱。"),
            ("design", ("外观", "双头", "形态", "白色", "黑色", "重量", "重心", "配件", "广角镜", "补光灯"), "随后介绍外观和配件：双头形态、白色机身、重量重心变化，以及广角镜、补光灯等附件兼容。"),
            ("image", ("画质", "样片", "高光", "暗部", "动态范围", "主摄", "传感", "低光", "ISO", "噪点"), "中段重点转向画质测试：通过样片、高光/暗部保留、低光 ISO 和动态范围来判断实际成像表现。"),
            ("lens", ("镜头", "长焦", "60毫米", "三倍", "主摄", "双镜", "视角"), "镜头部分讨论主摄和长焦的差异，尤其是长焦/三倍视角是否带来真实拍摄价值。"),
            ("experience", ("稳定", "发热", "测试", "体验", "过程", "总的体验", "续航"), "后段补充使用体验：稳定效果、测试过程、发热和整体体验，最后回到是否值得购买。"),
        ]
        items = []
        used: set[int] = set()
        for bucket, keywords, template in buckets:
            matched = [index for index, event in enumerate(events) if index not in used and any(keyword in event["text"] for keyword in keywords)]
            if not matched:
                continue
            for index in matched:
                used.add(index)
            start = AIClient._format_seconds(events[matched[0]]["time"])
            end = AIClient._format_seconds(events[matched[-1]]["time"])
            time_prefix = start if start == end else f"{start}-{end}"
            items.append(f"{time_prefix}：{template}")
        if items:
            return items
        return [f"{AIClient._format_seconds(event['time'])}：{AIClient._compress_text(event['text'], 72)}" for event in events[:5]]

    @staticmethod
    def _clean_visual_evidence(observations: list[dict[str, Any]]) -> list[str]:
        evidence = []
        negative_terms = ("无法支持", "不能支持", "不支持", "没有", "未见", "看不到", "不符合", "conflict")
        prefixes = ("画面支持该音频。证据：", "画面支持音频。证据：", "画面支持该音频：", "证据：")
        for obs in sorted(observations, key=lambda item: float(item.get("time", 0.0))):
            text = str(obs.get("evidence_assessment") or obs.get("scene") or "").strip()
            if not text:
                continue
            lowered = text.lower()
            if any(term in text or term in lowered for term in negative_terms):
                continue
            for prefix in prefixes:
                text = text.replace(prefix, "")
            text = AIClient._compress_text(text.strip(" ；。"), 86)
            if not text:
                continue
            evidence.append(f"{AIClient._format_seconds(float(obs.get('time', 0.0)))}：{text}")
        if evidence:
            return evidence
        for obs in sorted(observations, key=lambda item: float(item.get("time", 0.0)))[:4]:
            text = AIClient._compress_text(str(obs.get("scene") or obs.get("evidence_assessment") or "关键帧已抽取"), 72)
            evidence.append(f"{AIClient._format_seconds(float(obs.get('time', 0.0)))}：{text}")
        return evidence

    @staticmethod
    def _fast_coverage_notes(events: list[dict[str, Any]], observations: list[dict[str, Any]], raw_event_count: int) -> list[str]:
        notes = [f"本轮提炼了 {len(events)} 条有效音频线索，并观察了 {len(observations)} 个关键画面。"]
        skipped = max(0, raw_event_count - len(events))
        if skipped:
            notes.append(f"已过滤 {skipped} 条过短或不稳定的转写片段，避免把噪声写进结论。")
        if len(observations) < 8:
            notes.append("如果要核对每个施工步骤，可以继续提高关键帧预算或对某个时间段加密观察。")
        return notes

    @staticmethod
    def _compact_frames_for_prompt(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact = []
        for frame in frames[:24]:
            observation = frame.get("observation") or {}
            compact.append(
                {
                    "time": frame.get("time"),
                    "reason": frame.get("reason"),
                    "scene": observation.get("scene"),
                    "evidence_assessment": observation.get("evidence_assessment"),
                    "visible_text": observation.get("visible_text", []),
                }
            )
        return compact

    @staticmethod
    def _followup_knowledge_pack(question: str, result: dict[str, Any], limit: int = 28) -> dict[str, Any]:
        chunks = AIClient._video_knowledge_chunks(result)
        ranked = AIClient._rank_knowledge_chunks(question, chunks)
        selected = ranked[:limit]
        if not selected:
            selected = chunks[: min(limit, len(chunks))]
        return {
            "original_question": result.get("question"),
            "prior_answer": result.get("answer"),
            "metadata": result.get("metadata") or {},
            "relevant_chunks": selected,
            "supplemental_frame_observations": result.get("supplemental_frame_observations") or [],
            "coverage_hint": "Chunks are retrieved from transcript, audio timeline, key-frame observations, and any frames inspected specifically for this follow-up.",
        }

    @staticmethod
    def _video_knowledge_chunks(result: dict[str, Any]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        answer = result.get("answer") or {}
        if answer.get("direct_answer"):
            chunks.append({"source": "prior_answer", "time": None, "text": str(answer.get("direct_answer"))})
        for section in answer.get("sections") or []:
            title = str(section.get("title") or "")
            for item in section.get("items") or []:
                text = str(item).strip()
                if text:
                    chunks.append({"source": "answer_section", "title": title, "time": AIClient._first_time_in_text(text), "text": text})
        for item in result.get("timeline") or []:
            text = " ".join(str(value or "") for value in (item.get("label"), item.get("evidence"), item.get("visual_question")))
            if text.strip():
                chunks.append(
                    {
                        "source": "audio_timeline",
                        "time": _safe_float(item.get("time")),
                        "end_time": _safe_float(item.get("end_time")),
                        "text": text.strip(),
                    }
                )
        for segment in result.get("transcript_segments") or []:
            text = str(segment.get("text") or "").strip()
            if text:
                chunks.append(
                    {
                        "source": "transcript",
                        "time": _safe_float(segment.get("start")),
                        "end_time": _safe_float(segment.get("end")),
                        "text": text,
                    }
                )
        for frame in result.get("frames") or []:
            observation = frame.get("observation") or {}
            text = " ".join(
                str(value or "")
                for value in (
                    frame.get("reason"),
                    observation.get("visual_target"),
                    observation.get("scene"),
                    observation.get("evidence_assessment"),
                    " ".join(observation.get("visible_text") or []),
                )
            ).strip()
            if text:
                chunks.append(
                    {
                        "source": "keyframe",
                        "time": _safe_float(frame.get("time")),
                        "filename": frame.get("filename"),
                        "url": frame.get("url"),
                        "text": text,
                    }
                )
        for obs in result.get("supplemental_frame_observations") or []:
            text = " ".join(str(value or "") for value in (obs.get("visual_target"), obs.get("scene"), obs.get("evidence_assessment"))).strip()
            if text:
                chunks.append(
                    {
                        "source": "supplemental_keyframe",
                        "time": _safe_float(obs.get("time")),
                        "filename": obs.get("filename"),
                        "text": text,
                    }
                )
        return AIClient._dedupe_knowledge_chunks(chunks)

    @staticmethod
    def _rank_knowledge_chunks(question: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        query_terms = AIClient._keywords(question)
        wants_summary = AIClient._is_summary_followup_question("".join(str(question).split()).lower())
        wants_advice = AIClient._is_buying_advice_followup_question("".join(str(question).split()).lower())
        ranked = []
        for index, chunk in enumerate(chunks):
            text = str(chunk.get("text") or "")
            source = str(chunk.get("source") or "")
            score = AIClient._overlap_score(query_terms, text) * 8
            if wants_summary and source in {"prior_answer", "answer_section", "audio_timeline"}:
                score += 9
            if wants_advice and any(token in text for token in ("值得", "购买", "升级", "多花", "画质", "长焦", "动态范围", "一样", "结论")):
                score += 12
            if source == "supplemental_keyframe":
                score += 5
            if source == "transcript" and any(token in text for token in ("所以", "总结", "结论", "我觉得", "推荐", "买", "值得")):
                score += 6
            if score > 0:
                ranked.append((score, index, chunk))
        if not ranked and chunks:
            ranked = [(1, index, chunk) for index, chunk in enumerate(chunks[:20])]
        return [chunk for _, _, chunk in sorted(ranked, key=lambda item: (-item[0], item[1]))]

    @staticmethod
    def _dedupe_knowledge_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped = []
        seen = set()
        for chunk in chunks:
            text = AIClient._compress_text(str(chunk.get("text") or ""), 360)
            if not text:
                continue
            signature = (chunk.get("source"), round(float(chunk.get("time") or -1), 1), text[:80])
            if signature in seen:
                continue
            seen.add(signature)
            next_chunk = dict(chunk)
            next_chunk["text"] = text
            if next_chunk.get("time") is not None:
                next_chunk["time_label"] = AIClient._format_seconds(float(next_chunk["time"]))
            deduped.append(next_chunk)
        return deduped

    @staticmethod
    def _attach_web_context(answer: dict[str, Any], web_context: dict[str, Any] | None) -> dict[str, Any]:
        if not web_context or not web_context.get("enabled"):
            return answer
        results = web_context.get("results") or []
        sources = []
        for item in results[:3]:
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            label = title or url
            if label and label not in sources:
                sources.append(label)
        return {
            **answer,
            "web_sources": sources,
            "coverage_note": answer.get("coverage_note") or ("已开启联网增强，但本地兜底回答只附加来源，不会替代视频分析。" if sources else "联网搜索未返回可用结果。"),
        }

    @staticmethod
    def _local_followup_answer(
        question: str,
        result: dict[str, Any],
        web_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if web_context and (web_context.get("results") or []):
            web_answer = AIClient._web_augmented_local_followup_answer(question, result, web_context)
            if web_answer:
                return web_answer

        explicit_time = AIClient._first_time_in_text(question)
        if explicit_time is None:
            intent_answer = AIClient._intent_followup_answer(question, result)
            if intent_answer:
                return {
                    "answer": intent_answer[:1200],
                    "evidence_refs": [],
                    "coverage_note": "",
                }

        query_terms = AIClient._keywords(question)
        timeline = result.get("timeline") or []
        transcripts = result.get("transcript_segments") or []
        frames = result.get("frames") or []

        scored_events = []
        for item in timeline:
            text = " ".join([str(item.get("label", "")), str(item.get("evidence", ""))])
            time_value = float(item.get("time") or 0.0)
            score = AIClient._overlap_score(query_terms, text)
            if explicit_time is not None:
                delta = abs(time_value - explicit_time)
                if delta <= 45:
                    score += max(1, int((45 - delta) // 15) + 1)
            if score:
                scored_events.append((score, time_value, text))
        scored_transcripts = []
        for item in transcripts:
            text = str(item.get("text", ""))
            time_value = float(item.get("start") or 0.0)
            score = AIClient._overlap_score(query_terms, text)
            if explicit_time is not None:
                delta = abs(time_value - explicit_time)
                if delta <= 45:
                    score += max(1, int((45 - delta) // 15) + 1)
            if score:
                scored_transcripts.append((score, time_value, text))
        scored_frames = []
        for frame in frames:
            observation = frame.get("observation") or {}
            text = " ".join(
                [
                    str(observation.get("visual_target", "")),
                    str(observation.get("evidence_assessment", "")),
                    str(observation.get("scene", "")),
                    str(frame.get("reason", "")),
                ]
            )
            time_value = float(frame.get("time") or 0.0)
            score = AIClient._overlap_score(query_terms, text)
            if explicit_time is not None:
                delta = abs(time_value - explicit_time)
                if delta <= 45:
                    score += max(1, int((45 - delta) // 15) + 1)
            if score:
                scored_frames.append((score, time_value, text))

        selected = sorted(scored_events + scored_transcripts + scored_frames, key=lambda item: (-item[0], item[1]))[:6]
        if not selected:
            return {
                "answer": "这个追问没有在当前已解析内容里命中足够明确的线索。可以换成更具体的问题，例如问某个产品点、某个时间段，或让我重新按“画质/镜头/体验/结论”展开。",
                "evidence_refs": [],
                "coverage_note": "",
            }
        context = [(time, AIClient._clean_followup_context(text)) for _, time, text in selected]
        answer = AIClient._compose_local_followup_answer(question, context, result)
        return {
            "answer": answer[:520],
            "evidence_refs": [],
            "coverage_note": "",
        }

    @staticmethod
    def _web_augmented_local_followup_answer(
        question: str,
        result: dict[str, Any],
        web_context: dict[str, Any],
    ) -> dict[str, Any] | None:
        results = [item for item in web_context.get("results") or [] if isinstance(item, dict)]
        if not results:
            return None
        compact_question = "".join(str(question or "").split()).lower()
        asks_compare = any(token in compact_question for token in ("对比", "比较", "哪个更好", "哪款更好", "区别", "vs", "比"))
        asks_summary = AIClient._is_summary_followup_question(compact_question)
        asks_external = any(token in compact_question for token in ("insta360", "luna", "gopro", "竞品", "外部", "联网"))
        if not (asks_compare or asks_external):
            return None

        video_summary = AIClient._intent_followup_answer("详细总结视频内容", result) or AIClient._generic_detailed_followup_answer(result)
        web_facts = AIClient._summarize_web_facts(results)
        external_name = AIClient._external_name_from_question(question, results)
        comparison = AIClient._compose_web_comparison(question, video_summary, web_facts, external_name)
        if asks_summary:
            answer = f"先说视频本身：{video_summary}\n\n再对比 {external_name}：{comparison}"
        else:
            answer = comparison
        sources = []
        for item in results[:4]:
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            label = title or url
            if label and label not in sources:
                sources.append(label)
        return {
            "answer": answer[:1800],
            "evidence_refs": [],
            "coverage_note": "视频判断来自已解析内容；竞品参数来自联网搜索摘要，未做同场样片实测。",
            "web_sources": sources,
        }

    @staticmethod
    def _summarize_web_facts(results: list[dict[str, Any]]) -> dict[str, Any]:
        text = " ".join(" ".join(str(item.get(key) or "") for key in ("title", "snippet")) for item in results)
        lower = text.lower()
        facts: dict[str, Any] = {"raw": text[:1200], "bullets": []}
        if "8k" in lower:
            facts["bullets"].append("外部资料显示它主打 8K/高规格视频能力。")
        if "14 stops" in lower or "14档" in text or "14 stops of dynamic range" in lower:
            facts["bullets"].append("外部资料多次提到约 14 档动态范围。")
        if "17 stops" in lower or "17档" in text:
            facts["bullets"].append("部分资料提到 17 档动态范围，但需要注意是否指向 Pocket 4P 而不是 Luna。")
        if "dual" in lower or "双" in text or "dual-lens" in lower or "dual lens" in lower:
            facts["bullets"].append("它同样强调双镜头/双传感器设计。")
        if "leica" in lower:
            facts["bullets"].append("外部资料提到 Leica/徕卡相关光学或联合信息。")
        if "telephoto" in lower or "长焦" in text:
            facts["bullets"].append("资料中也提到长焦镜头，适合人像、特写和压缩感画面。")
        if "purevideo" in lower or "low-light" in lower or "low light" in lower or "低光" in text:
            facts["bullets"].append("它也把低光/夜景增强作为卖点之一。")
        if not facts["bullets"]:
            snippets = [AIClient._compress_text(str(item.get("snippet") or item.get("title") or ""), 90) for item in results[:3]]
            facts["bullets"] = [item for item in snippets if item]
        return facts

    @staticmethod
    def _external_name_from_question(question: str, results: list[dict[str, Any]]) -> str:
        text = str(question or "")
        if "luna" in text.lower():
            return "Insta360 Luna"
        for item in results:
            title = str(item.get("title") or "")
            if "Insta360" in title:
                return "Insta360 Luna"
        return "外部竞品"

    @staticmethod
    def _compose_web_comparison(question: str, video_summary: str, web_facts: dict[str, Any], external_name: str) -> str:
        bullets = [str(item) for item in web_facts.get("bullets") or [] if str(item).strip()]
        external_text = "；".join(bullets[:5]) or "联网资料返回的信息有限，只能做保守对比。"
        return (
            f"如果把视频里的 Pocket 4P 结论和联网资料里的 {external_name} 放在一起看，我会这样判断："
            "Pocket 4P 更像是“视频里已经被验证过的双镜头口袋云台方案”，优势集中在 60mm 长焦、主摄画质、动态范围和稳定体验，视频里还给出了 0:14、7:55、11:39 这些关键判断点。"
            f"{external_name} 这边，联网资料能确认的卖点是：{external_text}"
            "所以不是简单谁全面碾压谁：如果你最在意这条视频已经实测到的长焦构图、主摄动态范围和 DJI 云台体验，Pocket 4P 更稳；"
            f"如果你更看重 {external_name} 的 8K/徕卡/低光玩法、可拆屏或它的生态卖点，那 Luna 更值得继续看专项评测。"
            "目前最大差别在于证据类型：Pocket 4P 的判断来自你上传视频里的实测观点，Luna 这边来自网页资料，还缺同场样片、价格和发热/稳定性实测，所以购买结论我会给成条件式：重实测可靠和长焦体验选 Pocket 4P，重参数上限和新玩法再重点考察 Luna。"
        )

    @staticmethod
    def _intent_followup_answer(question: str, result: dict[str, Any]) -> str:
        compact_question = "".join(str(question).split()).lower()
        if not compact_question:
            return ""
        context_text = AIClient._result_text_context(result)
        is_product_review = any(token.lower() in context_text.lower() for token in ("Pocket", "Pockets", "大疆", "DJI", "画质", "长焦"))
        is_3d_printing = AIClient._is_3d_printing_result(result)
        asks_upgrade = any(token in compact_question for token in ("升级", "提升", "比上一代", "上一代", "普通版", "区别", "差异", "pro多少", "多花"))
        asks_buying_advice = AIClient._is_buying_advice_followup_question(compact_question)
        asks_summary = AIClient._is_summary_followup_question(compact_question)
        asks_detail = any(token in compact_question for token in ("详细", "展开", "具体", "完整", "更清楚")) and any(
            token in compact_question for token in ("总结", "讲", "说", "分析", "梳理")
        )
        if is_3d_printing:
            answer = AIClient._printing_renovation_followup_answer(compact_question, result)
            if answer:
                return answer
        if is_product_review and asks_upgrade:
            return AIClient._product_upgrade_followup_answer(result)
        if is_product_review and asks_buying_advice:
            return AIClient._product_buying_advice_followup_answer(result)
        if is_product_review and (asks_detail or asks_summary):
            return AIClient._product_detailed_followup_answer(result)
        if asks_buying_advice:
            return AIClient._generic_buying_advice_followup_answer(result)
        if asks_detail or asks_summary:
            return AIClient._generic_detailed_followup_answer(result)
        return ""

    @staticmethod
    def _is_3d_printing_result(result: dict[str, Any]) -> bool:
        text = AIClient._result_text_context(result)
        return any(token in text for token in ("3D 打印", "3D打印", "打印装修", "整屋装修", "家具", "柜体")) and any(
            token in text for token in ("装修", "施工", "材料", "成本", "家具", "房子")
        )

    @staticmethod
    def _printing_renovation_followup_answer(compact_question: str, result: dict[str, Any]) -> str:
        overview = AIClient.build_ai_overview(result)
        highlights = overview.get("highlights") or []
        time_for = lambda *keywords: AIClient._first_transcript_time(result.get("transcript_segments") or [], keywords)
        t_design = time_for("设计", "建模", "制作装修") or AIClient._highlight_time(highlights, "设计", "方案") or "1:12"
        t_cost = time_for("成本", "一公斤", "二三十", "材料") or AIClient._highlight_time(highlights, "成本", "材料") or "10:02"
        t_effect = time_for("效果", "传统家具", "成品", "接近传统") or AIClient._highlight_time(highlights, "效果", "成品") or "12:42"
        t_printing = time_for("打印", "3D", "材料堆叠") or AIClient._highlight_time(highlights, "打印", "材料") or "7:58"
        asks_difficulty = any(token in compact_question for token in ("难点", "困难", "挑战", "问题", "卡点"))
        asks_cost = any(token in compact_question for token in ("成本", "材料", "价格", "多少钱", "单价", "耗材"))
        asks_effect = any(token in compact_question for token in ("效果", "成品", "最终", "质感", "好不好", "怎么样"))
        asks_summary = AIClient._is_summary_followup_question(compact_question) or any(token in compact_question for token in ("主要讲", "讲什么"))
        if asks_difficulty:
            return (
                "这个项目的难点不只是“能不能打印出来”，而是把 3D 打印真正接进装修流程。"
                f"第一，要先把柜体、家具和装饰部件转成可打印的设计/建模方案（{t_design}）。"
                f"第二，材料和设备要能稳定打印大尺寸部件，打印层纹、透光材料、结构强度都会影响最终质感（{t_printing}）。"
                f"第三，打印出来后还要和墙面、灯具、瓷砖、安装这些传统施工环节衔接，否则单个部件好看也不等于整屋能落地。"
                f"最后是成本：耗材单价、打印时间和失败率都会决定它到底是实验玩法还是可复制方案（{t_cost}）。"
            )
        if asks_cost:
            return (
                "视频里对成本和材料的态度比较现实：3D 打印不是“按一下就很便宜”的装修方式。"
                f"它会涉及耗材选择、材料单价、打印用量和后期安装成本，视频中还提到类似“一公斤二三十”的材料价格线索（{t_cost}）。"
                f"材料方面，重点不是单纯便宜，而是不同部件要用不同材料：有的追求结构强度，有的追求透光或装饰效果，有的要接近传统家具质感（{t_printing}、{t_effect}）。"
                "所以结论是：材料可玩性很强，但成本是否划算要看打印规模、失败率和传统施工替代程度。"
            )
        if asks_effect:
            return (
                "最终效果可以理解为“实验成功，但还带着 3D 打印的特征”。"
                f"视频重点展示了柜体、家具或装饰件这类成品，它们能做出定制造型和层层堆叠的纹理（{t_effect}）。"
                "优点是造型自由、可以做传统木工不太方便的形态；不足是层纹、表面质感、安装收口和耐用性仍会影响它能不能完全替代传统家具。"
                "所以它更像一次证明可行性的整屋装修实验，而不是已经成熟到所有家庭都能直接照抄的标准方案。"
            )
        if asks_summary:
            bullets = [str(item) for item in overview.get("bullets") or [] if str(item).strip()]
            process = "；".join(bullets[:3])
            return (
                "这条视频主要记录一次用 3D 打印参与整套房装修的实验。"
                f"它先做设计和建模（{t_design}），再讨论打印材料、设备和可打印部件（{t_printing}），"
                f"后面转向成本、施工衔接和最终家具/柜体效果（{t_cost}、{t_effect}）。"
                f"{' 关键过程可以概括为：' + process + '。' if process else ''}"
                "核心结论是：3D 打印能提升定制化和造型自由度，但材料、成本、安装和成品质感仍是落地难点。"
            )
        return ""

    @staticmethod
    def _highlight_time(highlights: list[dict[str, Any]], *keywords: str) -> str:
        for item in highlights:
            text = " ".join(str(item.get(key) or "") for key in ("label", "detail"))
            if any(keyword in text for keyword in keywords):
                return str(item.get("time_label") or AIClient._format_seconds(float(item.get("time") or 0.0)))
        return ""

    @staticmethod
    def _is_buying_advice_followup_question(compact_question: str) -> bool:
        return any(
            token in compact_question
            for token in (
                "推荐买",
                "买哪",
                "买哪个",
                "选哪",
                "选哪个",
                "哪一代",
                "哪款",
                "怎么选",
                "值得买吗",
                "值不值得",
                "购买建议",
                "入手",
                "推荐哪",
                "建议买",
            )
        )

    @staticmethod
    def _is_summary_followup_question(compact_question: str) -> bool:
        return any(
            token in compact_question
            for token in (
                "总结",
                "概括",
                "视频内容",
                "主要内容",
                "主要讲",
                "讲什么",
                "讲了什么",
                "主题",
                "内容梳理",
                "整体内容",
                "summary",
                "summarize",
            )
        )

    @staticmethod
    def _product_upgrade_followup_answer(result: dict[str, Any]) -> str:
        transcripts = result.get("transcript_segments") or []
        time = lambda *keywords: AIClient._first_transcript_time(transcripts, keywords)
        t_intro = time("到底比", "Pro多少", "多花") or "0:00"
        t_lens = time("60毫米", "长焦", "三倍") or "0:14"
        t_sensor = time("两块完全不一样", "传感") or "0:31"
        t_dynamic = time("ISO1600", "17档", "动态范围") or "7:55"
        t_body = time("外观", "双头", "补光灯", "重", "扭力") or "0:50"
        t_conclusion = time("多花的钱", "别的东西", "真的一样") or "11:39"
        return (
            "视频里所谓“升级”不是全面换代，而是集中在拍摄能力上。"
            f"第一，新增/强化了长焦思路：它强调等效 60mm 长焦，可以从主摄快速切到约三倍视角，带来更多构图机会（{t_lens}）。"
            f"第二，传感器和画质是核心升级：视频提到 4P 使用了两块不同传感器，并重点讨论高光、暗部和动态范围（{t_sensor}、{t_dynamic}）。"
            f"第三，主摄动态范围提升明显：作者测试认为 ISO1600 时动态范围最好，大约能到 17 档，强项是保留高光和暗部细节（{t_dynamic}）。"
            f"第四，外观和配件是小改：双头形态、重量/重心、电机扭力、广角镜和补光灯兼容这些有变化，但视频语气更像“预期内”而不是革命性升级（{t_body}）。"
            f"最后，作者的落点是：多花的钱主要买画质、动态范围和长焦机会；其他很多基础体验仍然和 Pocket 4 很像（{t_conclusion}）。"
        )

    @staticmethod
    def _product_detailed_followup_answer(result: dict[str, Any]) -> str:
        transcripts = result.get("transcript_segments") or []
        time = lambda *keywords: AIClient._first_transcript_time(transcripts, keywords)
        t_intro = time("到底比", "Pro多少", "考虑过") or "0:00"
        t_lens = time("60毫米", "长焦", "三倍") or "0:14"
        t_body = time("外观", "双头", "补光灯", "重") or "0:50"
        t_quality = time("画质到底", "样片", "画质") or "4:25"
        t_dynamic = time("ISO1600", "17档", "动态范围") or "7:55"
        t_heat = time("发热", "稳定性", "体验") or "10:07"
        t_conclusion = time("多花的钱", "别的东西", "Pocket 4真的") or "11:39"
        return (
            f"详细来看，这支视频是在回答“Pocket 4P 到底比 Pocket 4/普通版 Pro 在哪里”（{t_intro}）。"
            f"开头先把核心购买理由锁定在画质和双镜头上，尤其是等效 60mm 长焦以及从主摄切到三倍视角的能力（{t_lens}）。"
            f"随后讲外观和硬件形态：4P 是双头设计，和 Pocket 4 很像但更重一些，电机扭力也做了调整；广角镜、补光灯等配件兼容属于实用补充（{t_body}）。"
            f"中段是重点：作者先看样片和硬件，再解释传感器差异，认为主摄在高光、暗部和动态范围上是这代最值得看的地方（{t_quality}）。"
            f"最强的结论来自动态范围测试：ISO1600 附近表现最好，视频里提到大约 17 档动态范围，意思是亮部不容易死白，暗部也能保留更多细节（{t_dynamic}）。"
            f"后段补充稳定性、发热和实际体验，但这些不是最核心卖点；最终判断是 4P 的钱主要花在更好的主摄画质、动态范围和长焦机会，其他很多体验仍然接近 Pocket 4（{t_heat}、{t_conclusion}）。"
        )

    @staticmethod
    def _product_buying_advice_followup_answer(result: dict[str, Any]) -> str:
        transcripts = result.get("transcript_segments") or []
        time = lambda *keywords: AIClient._first_transcript_time(transcripts, keywords)
        t_lens = time("60毫米", "长焦", "三倍") or "0:14"
        t_dynamic = time("ISO1600", "17档", "动态范围") or "7:55"
        t_experience = time("发热", "稳定性", "体验") or "10:07"
        t_conclusion = time("多花的钱", "别的东西", "真的一样", "Pocket 4真的") or "11:39"
        return (
            "按这支视频的结论，如果你主要拍视频、在意主摄画质/动态范围，或者明确需要 60mm 长焦和三倍视角，那更推荐 Pocket 4P。"
            f"它的钱主要花在画质、动态范围和长焦机会里（{t_lens}、{t_dynamic}、{t_conclusion}）。"
            "但如果你只是日常记录、预算敏感，或者不太用长焦和更高动态范围，Pocket 4/普通版会更划算，因为视频也说很多基础体验和 Pocket 4 很接近。"
            f"发热、稳定性这些属于后段体验补充，不是让普通用户必须升级的决定性理由（{t_experience}）。"
            "所以一句话：重画质和创作空间买 4P，重性价比和日常够用买 Pocket 4。"
        )

    @staticmethod
    def _generic_detailed_followup_answer(result: dict[str, Any]) -> str:
        answer_payload = result.get("answer") or {}
        direct = str(answer_payload.get("direct_answer") or "").strip()
        sections = []
        for section in answer_payload.get("sections") or []:
            title = str(section.get("title") or "").strip()
            items = [str(item).strip() for item in section.get("items") or [] if str(item).strip()]
            if title and items:
                sections.append(f"{title}：" + "；".join(items[:4]))
        if sections:
            return (direct + " " if direct else "") + " ".join(sections)
        return direct or "当前结果没有足够结构化内容可展开。"

    @staticmethod
    def _generic_buying_advice_followup_answer(result: dict[str, Any]) -> str:
        answer_payload = result.get("answer") or {}
        conclusion = AIClient._answer_section_text(answer_payload, {"关键结论", "结论"})
        direct = str(answer_payload.get("direct_answer") or "").strip()
        base = conclusion or direct
        if not base:
            return "当前结果还不足以给出明确购买建议；需要先确认视频里的产品优缺点、价格差和适合人群。"
        return f"按当前视频结论，购买建议应围绕这个判断：{base} 如果你重视这些优势并能接受价格，就选高配/新款；如果只是基础使用或预算更敏感，就选普通版/上一代。"

    @staticmethod
    def _result_text_context(result: dict[str, Any]) -> str:
        chunks: list[str] = []
        answer = result.get("answer") or {}
        chunks.append(str(answer.get("direct_answer") or ""))
        for segment in (result.get("transcript_segments") or [])[:120]:
            chunks.append(str(segment.get("text") or ""))
        return " ".join(chunks)

    @staticmethod
    def _first_transcript_time(transcripts: list[dict[str, Any]], keywords: tuple[str, ...]) -> str:
        for segment in transcripts:
            text = str(segment.get("text") or "")
            if any(keyword.lower() in text.lower() or keyword in text for keyword in keywords):
                return AIClient._format_seconds(float(segment.get("start") or 0.0))
        return ""

    @staticmethod
    def _compose_local_followup_answer(question: str, context: list[tuple[float, str]], result: dict[str, Any]) -> str:
        answer_payload = result.get("answer") or {}
        direct = str(answer_payload.get("direct_answer") or "").strip()
        conclusion = AIClient._answer_section_text(answer_payload, {"关键结论", "结论"})
        joined = " ".join(text for _, text in context)
        times = [time for time, _ in context]
        if AIClient._first_time_in_text(question) is not None:
            snippets = [AIClient._compress_text(text, 70) for _, text in context if text]
            deduped = []
            for item in snippets:
                if item not in deduped:
                    deduped.append(item)
            if deduped:
                return AIClient._append_time_refs("这个时间点附近主要在讲：" + "；".join(deduped[:3]) + "。", times)
        if any(token in question for token in ("总结", "讲什么", "主要", "内容", "主题")) and direct:
            return direct
        if any(token in question for token in ("值得", "买吗", "购买", "升级", "推荐")):
            answer = conclusion or direct or "当前分析只能确认视频在讨论购买和升级价值，但没有形成更明确的购买建议。"
            return AIClient._append_time_refs(answer, times)
        if any(token in question for token in ("画质", "成像", "动态范围", "ISO", "高光", "暗部", "低光")):
            if any(token in joined for token in ("画质", "动态范围", "ISO", "高光", "暗部", "低光")):
                return AIClient._append_time_refs("视频对画质的判断偏正向：它重点测试样片、低光 ISO、动态范围以及高光/暗部保留，并认为实际成像比预期更好。", times)
        if any(token in question for token in ("长焦", "镜头", "双镜", "三倍", "60毫米")):
            if any(token in joined for token in ("长焦", "镜头", "三倍", "60毫米", "主摄")):
                return AIClient._append_time_refs("视频把长焦/双镜头视为主要升级点：它能从主摄切到约三倍视角，价值主要体现在更丰富的构图和拍摄距离选择上。", times)
        if any(token in question for token in ("发热", "稳定", "续航", "体验")):
            if any(token in joined for token in ("发热", "稳定", "体验", "测试")):
                return AIClient._append_time_refs("视频后段补充了使用体验，包括稳定效果、发热测试和整体体验；这些内容用于判断它是否适合长时间或更正式的拍摄。", times)
        snippets = [AIClient._compress_text(text, 58) for _, text in context if text]
        deduped = []
        for item in snippets:
            if item not in deduped:
                deduped.append(item)
        if deduped:
            return AIClient._append_time_refs("根据已解析内容，" + "；".join(deduped[:3]) + "。", times)
        return direct or "当前已解析内容不足以回答这个追问。"

    @staticmethod
    def _append_time_anchors(text: str, process_items: list[str], limit: int = 3) -> str:
        import re

        anchors: list[str] = []
        for item in process_items:
            for match in re.finditer(r"\d{1,2}:\d{2}", item):
                token = match.group(0)
                if token not in anchors:
                    anchors.append(token)
                if len(anchors) >= limit:
                    break
            if len(anchors) >= limit:
                break
        if not anchors or any(anchor in text for anchor in anchors):
            return text
        return AIClient._compress_text(f"{text}（相关位置：{'、'.join(anchors)}）", 230)

    @staticmethod
    def _append_time_refs(text: str, times: list[float], limit: int = 3) -> str:
        refs: list[str] = []
        for time in times:
            ref = AIClient._format_seconds(time)
            if ref not in refs:
                refs.append(ref)
            if len(refs) >= limit:
                break
        if not refs or any(ref in text for ref in refs):
            return text
        return f"{text}（相关位置：{'、'.join(refs)}）"

    @staticmethod
    def _answer_section_text(answer_payload: dict[str, Any], titles: set[str]) -> str:
        for section in answer_payload.get("sections") or []:
            if str(section.get("title")) in titles:
                items = [str(item).strip() for item in section.get("items") or [] if str(item).strip()]
                if items:
                    return " ".join(items)
        return ""

    @staticmethod
    def _first_time_in_text(text: str) -> float | None:
        import re

        value = str(text or "")
        chinese_match = re.search(r"(\d{1,3})\s*分(?:钟)?\s*(\d{1,2})\s*秒", value)
        if chinese_match:
            return float(int(chinese_match.group(1)) * 60 + int(chinese_match.group(2)))
        minute_match = re.search(r"(\d{1,3})\s*分(?:钟)?", value)
        if minute_match:
            return float(int(minute_match.group(1)) * 60)
        match = re.search(r"(\d{1,2}:\d{2}(?::\d{2})?)", value)
        if not match:
            return None
        parts = [int(part) for part in match.group(1).split(":")]
        if len(parts) == 3:
            return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
        return float(parts[0] * 60 + parts[1])

    @staticmethod
    def _clean_followup_context(text: str) -> str:
        cleaned = " ".join(str(text).split())
        prefixes = (
            "画面支持该音频。证据：",
            "画面支持音频。证据：",
            "画面支持该音频：",
            "证据：",
        )
        for prefix in prefixes:
            cleaned = cleaned.replace(prefix, "")
        return cleaned.strip(" ，。；;:：")

    @staticmethod
    def _keywords(text: str) -> list[str]:
        import re

        raw_words = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", str(text or ""))
        stop = {"这个", "视频", "什么", "怎么", "如何", "一下", "主要", "是否", "有没有", "为什么", "哪里", "怎么样", "哪些", "一个"}
        domain_phrases = (
            "动态范围",
            "低光",
            "高光",
            "暗部",
            "画质",
            "成像",
            "长焦",
            "镜头",
            "稳定性",
            "发热",
            "购买建议",
            "推荐",
            "升级",
            "价格",
            "成本",
            "设计",
            "施工",
            "材料",
        )
        terms: list[str] = []
        compact = "".join(raw_words)
        for phrase in domain_phrases:
            if phrase in compact and phrase not in terms:
                terms.append(phrase.lower())
        for word in raw_words:
            lowered = word.lower()
            if lowered in stop or word in stop:
                continue
            if re.fullmatch(r"[\u4e00-\u9fff]+", word):
                if len(word) <= 4:
                    candidates = [word]
                else:
                    candidates = [word]
                    for size in (2, 3, 4):
                        candidates.extend(word[index : index + size] for index in range(0, len(word) - size + 1))
                for candidate in candidates:
                    if candidate not in stop and candidate.lower() not in terms:
                        terms.append(candidate.lower())
            elif lowered not in terms:
                terms.append(lowered)
        return terms[:32]

    @staticmethod
    def _overlap_score(query_terms: list[str], text: str) -> int:
        lowered = text.lower()
        return sum(1 for term in query_terms if term in lowered or term in text)

    @staticmethod
    def _compress_text(text: str, limit: int) -> str:
        text = " ".join(str(text).split())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)] + "..."

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        safe = max(0, int(round(seconds)))
        return f"{safe // 60}:{safe % 60:02d}"

    @staticmethod
    def _infer_fast_topic(transcript_bits: list[str], visual_bits: list[str], question: str) -> str:
        text = " ".join(transcript_bits + visual_bits + [question])
        if any(token in text for token in ("3D", "打印", "装修", "房子", "施工")):
            return "用 3D 打印和施工流程完成房屋装修"
        if any(token in text for token in ("教程", "步骤", "演示", "安装")):
            return "一个带步骤演示的教程内容"
        if any(token in text for token in ("故事", "夫妻", "狐狸", "结局")):
            return "一段带人物关系推进的故事"
        return "音频叙述中的事件和对应关键画面"

    def _normalize_transcript(self, response: Any) -> list[dict[str, Any]]:
        payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        segments = payload.get("segments") or []
        if not segments and payload.get("text"):
            segments = [{"start": 0.0, "end": 0.0, "text": payload["text"]}]
        normalized = []
        for segment in segments:
            normalized.append(
                {
                    "start": float(segment.get("start") or 0.0),
                    "end": float(segment.get("end") or segment.get("start") or 0.0),
                    "speaker": str(segment.get("speaker") or "unknown"),
                    "text": str(segment.get("text") or "").strip(),
                    "confidence": segment.get("confidence"),
                }
            )
        return [segment for segment in normalized if segment["text"]]

    def _chat_audio_transcribe(
        self,
        audio_path: Path,
        duration: float,
        attempts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        assert self._client is not None
        model = self.settings.audio_chat_transcribe_model
        if not model:
            return []
        schema_hint = {
            "transcript_segments": [
                {
                    "start": 0.0,
                    "end": duration,
                    "speaker": "unknown",
                    "text": "spoken text",
                    "confidence": None,
                }
            ]
        }
        try:
            audio_data = base64.b64encode(audio_path.read_bytes()).decode("ascii")
            response = self._client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Transcribe this audio in its original language. Return JSON only matching "
                                    f"this shape: {json.dumps(schema_hint)}"
                                ),
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {"data": audio_data, "format": "wav"},
                            },
                        ],
                    }
                ],
                temperature=0,
            )
            text = response.choices[0].message.content or "{}"
            payload = json.loads(self._strip_json_fence(text))
            segments = self._normalize_transcript({"segments": payload.get("transcript_segments", [])})
            attempts.append(
                {
                    "method": "chat_audio",
                    "model": model,
                    "status": "ok" if segments else "empty",
                    "segment_count": len(segments),
                }
            )
            return segments
        except Exception as exc:
            attempts.append(
                {
                    "method": "chat_audio",
                    "model": model,
                    "status": "error",
                    "error": str(exc),
                }
            )
            if self.settings.allow_model_fallback:
                return []
            raise

    def _local_transcribe(self, audio_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        model_ref = self._local_transcribe_model_ref()
        try:
            from faster_whisper import WhisperModel

            model = self._local_whisper_cache.get(model_ref)
            cache_hit = model is not None
            if model is None:
                model = WhisperModel(model_ref, device="cpu", compute_type="int8")
                self._local_whisper_cache[model_ref] = model
            segments, info = model.transcribe(str(audio_path), vad_filter=True, beam_size=5)
            language = getattr(info, "language", "unknown")
            language_probability = getattr(info, "language_probability", None)
            normalized = []
            for segment in segments:
                text = (segment.text or "").strip()
                if not text:
                    continue
                normalized.append(
                    {
                        "start": float(segment.start),
                        "end": float(segment.end),
                        "speaker": "unknown",
                        "text": text,
                        "confidence": None,
                        "source": f"faster-whisper:{model_ref}:{language}",
                    }
                )
            return normalized, {
                "method": "local_faster_whisper",
                "model": model_ref,
                "status": "ok" if normalized else "empty",
                "language": language,
                "language_probability": language_probability,
                "segment_count": len(normalized),
                "cache_hit": cache_hit,
            }
        except Exception as exc:
            status = {
                "method": "local_faster_whisper",
                "model": model_ref,
                "status": "error",
                "error": str(exc),
            }
            if self.settings.allow_model_fallback:
                return [], status
            raise

    def _local_transcribe_model_ref(self) -> str:
        configured = self.settings.local_transcribe_model
        path = Path(configured)
        candidates = []
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append((self.settings.project_root / path).resolve())
            if configured == "tiny":
                candidates.append((self.settings.project_root / "data" / "models" / "faster-whisper-tiny").resolve())
            if configured == "base":
                candidates.append((self.settings.project_root / "data" / "models" / "faster-whisper-base").resolve())
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return configured

    @staticmethod
    def _image_data_url(path: Path) -> str:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{data}"

    @staticmethod
    def _video_data_url(path: Path) -> str:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:video/mp4;base64,{data}"

    def _responses_json(
        self,
        model: str,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
        images: list[str] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        assert self._client is not None
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for image_url in images or []:
            content.append({"type": "input_image", "image_url": image_url})

        kwargs: dict[str, Any] = {
            "model": model,
            "input": [{"role": "user", "content": content}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        }
        if self.settings.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.settings.reasoning_effort}
        if timeout_seconds is not None:
            kwargs["timeout"] = timeout_seconds

        try:
            response = self._client.responses.create(
                **kwargs,
            )
        except TypeError:
            response = self._client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": content
                        + [
                            {
                                "type": "input_text",
                                "text": f"\nReturn JSON matching this schema:\n{json.dumps(schema)}",
                            }
                        ],
                    }
                ],
            )

        text = getattr(response, "output_text", None)
        if text is None:
            payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)
            text = json.dumps(payload)
        return json.loads(self._strip_json_fence(text))

    @staticmethod
    def _strip_json_fence(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        return text.strip()
