from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

from .config import Settings
from .prediction import classify_prediction_check


class AIClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.mock = settings.use_mock_models
        self._client = None
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
                observations = self._observe_frames_with_joyai(
                    question=question,
                    frames=frames,
                    audio_world_model=audio_world_model,
                )
                self.last_vision_request_count = len(frames)
                return observations
            except Exception:
                if not self.settings.allow_model_fallback:
                    raise
                if self.settings.vision_provider in {"joyai", "joyai_adapter"}:
                    self.last_vision_request_count = len(frames)
                    return self._fallback_frame_observations(frames, "JoyAI local vision endpoint failed.")

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
    ) -> list[dict[str, Any]]:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.settings.joyai_api_key,
            base_url=self.settings.joyai_api_base,
            timeout=self.settings.joyai_timeout_seconds,
            max_retries=0,
        )
        observations: list[dict[str, Any]] = []
        for index, frame in enumerate(frames):
            target = str((frame.get("probe") or {}).get("question") or frame.get("reason") or question)
            audio_context = self._audio_context_for_time(audio_world_model, float(frame["time"]))
            session_id = f"audio-first-{int(time.time() * 1000)}-{index}"
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
                max_tokens=96,
                temperature=0,
                extra_headers={
                    "x-streaming-session": session_id,
                    "x-frame-time-range": f"{float(frame['time']):.2f}s-{float(frame['time']) + 1.0:.2f}s",
                },
            )
            raw_text = response.choices[0].message.content if response.choices else ""
            observations.append(self._joyai_text_to_observation(frame, target, audio_context, raw_text or ""))
        return observations

    def _fallback_frame_observations(self, frames: list[dict[str, Any]], reason: str) -> list[dict[str, Any]]:
        return [
            {
                "filename": frame["filename"],
                "time": frame["time"],
                "scene": "Frame extracted successfully, but the vision endpoint could not inspect pixels.",
                "objects": [],
                "actions": [],
                "visible_text": [],
                "audio_alignment": "uncertain",
                "visual_target": str((frame.get("probe") or {}).get("question") or frame.get("reason", "")),
                "evidence_assessment": reason,
                "notes": "Model fallback observation.",
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
        if not text or raw_text.strip() == "</silence>":
            text = "JoyAI returned silence for this frame."
        lower = text.lower()
        visible_text = []
        for marker in ("囍", "喜", "double happiness", "red cloth", "veil", "盖头", "红布", "红色"):
            if marker.lower() in lower or marker in text:
                visible_text.append(marker)
        match_terms = ("match", "matches", "support", "supports", "一致", "符合", "支持", "出现", "可见")
        conflict_terms = ("conflict", "contradict", "不一致", "冲突", "没有", "未见", "看不到")
        if any(term in lower or term in text for term in conflict_terms):
            alignment = "conflict"
        elif any(term in lower or term in text for term in match_terms):
            alignment = "match"
        else:
            alignment = "uncertain"
        return {
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
            "Answer the user's question about the video in concise Chinese. Ground the answer in audio timeline "
            "evidence, frame observations, and prediction checks. Mention uncertainty explicitly when evidence is "
            "weak. Keep direct_answer under 180 Chinese characters, summary under 450 Chinese characters, and use "
            "at most 8 evidence_refs.\n\n"
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

    def answer_followup(self, *, question: str, result: dict[str, Any]) -> dict[str, Any]:
        if self.mock or self.settings.fast_mode:
            return self._local_followup_answer(question, result)
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "answer": {"type": "string"},
                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                "coverage_note": {"type": "string"},
            },
            "required": ["answer", "evidence_refs", "coverage_note"],
        }
        context = {
            "original_question": result.get("question"),
            "answer": result.get("answer"),
            "timeline": (result.get("timeline") or [])[:24],
            "transcript_segments": (result.get("transcript_segments") or [])[:80],
            "frames": self._compact_frames_for_prompt(result.get("frames") or []),
            "metadata": result.get("metadata") or {},
        }
        prompt = (
            "Answer this follow-up question in Chinese using only the already extracted transcript and key-frame "
            "evidence from a previous video analysis. Do not invent unseen content. If the evidence does not cover "
            "the requested moment, say which time range would need more frames. Keep the answer concise.\n\n"
            f"Follow-up question: {question}\n"
            f"Analysis context: {json.dumps(context, ensure_ascii=False)}"
        )
        try:
            return self._responses_json(self.settings.reasoning_model, prompt, schema, "followup_answer")
        except Exception:
            if self.settings.allow_model_fallback:
                return self._local_followup_answer(question, result)
            raise

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

        if "3D 打印" in title:
            direct = (
                "这是一个 3D 打印装修实验视频：核心是在一套房的装修和家具制作中尝试使用 3D 打印，"
                "内容从设计、打印材料/设备、现场施工延伸到成本和成品效果。"
            )
        else:
            direct = f"视频主要围绕{title}展开，音频给出事件线索，关键画面用于确认主要场景和动作。"
        direct = AIClient._compress_text(direct, 180)

        process_items = AIClient._fast_process_items(events, title)
        if not process_items and transcript_bits:
            process_items = [AIClient._compress_text(bit, 88) for bit in transcript_bits[:4]]
        if not process_items:
            process_items = ["当前没有稳定的音频事件，系统主要依赖稀疏关键画面做概览。"]

        evidence_items = visual_bits[:6] or ["关键画面已经抽取，但视觉模型没有返回可压缩成证据的稳定描述。"]
        coverage_note = AIClient._fast_coverage_notes(events, observations, len(timeline))

        conclusion_items = [
            "这轮结果适合先理解视频主线：它不是逐帧详查，而是用音频定位重点，再用少量画面确认代表性片段。"
        ]
        if "3D 打印" in title:
            conclusion_items = [
                "视频主线是验证 3D 打印在装修和家具制作中的可行性：能做出有层纹和造型的部件，但成本、材料和传统施工衔接仍是重点讨论对象。"
            ]

        sections = [
            {"title": "视频主题", "items": [direct]},
            {"title": "过程脉络", "items": process_items[:5]},
            {"title": "关键画面证据", "items": evidence_items[:6]},
            {"title": "结论", "items": conclusion_items + coverage_note[:1]},
        ]
        summary = " ".join(
            [
                "过程脉络：" + "；".join(process_items[:4]) + "。",
                "关键画面证据：" + "；".join(evidence_items[:4]) + "。",
                "结论：" + conclusion_items[0],
            ]
        )
        return {
            "direct_answer": direct or f"快速模式已处理问题：{question}",
            "summary": summary[:900],
            "sections": sections,
            "evidence_refs": (visual_bits[:6] + [f"音频：{bit}" for bit in transcript_bits[:4]])[:8],
            "uncertainties": coverage_note,
        }

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
    def _fast_process_items(events: list[dict[str, Any]], topic: str) -> list[str]:
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
    def _local_followup_answer(question: str, result: dict[str, Any]) -> dict[str, Any]:
        query_terms = AIClient._keywords(question)
        timeline = result.get("timeline") or []
        transcripts = result.get("transcript_segments") or []
        frames = result.get("frames") or []

        scored_events = []
        for item in timeline:
            text = " ".join([str(item.get("label", "")), str(item.get("evidence", ""))])
            score = AIClient._overlap_score(query_terms, text)
            if score:
                scored_events.append((score, float(item.get("time") or 0.0), text))
        scored_transcripts = []
        for item in transcripts:
            text = str(item.get("text", ""))
            score = AIClient._overlap_score(query_terms, text)
            if score:
                scored_transcripts.append((score, float(item.get("start") or 0.0), text))
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
            score = AIClient._overlap_score(query_terms, text)
            if score:
                scored_frames.append((score, float(frame.get("time") or 0.0), text))

        selected = sorted(scored_events + scored_transcripts + scored_frames, key=lambda item: (-item[0], item[1]))[:6]
        if not selected:
            base_answer = (result.get("answer") or {}).get("direct_answer") or "当前结果里没有命中这个追问的明确证据。"
            return {
                "answer": f"基于已分析内容：{base_answer}",
                "evidence_refs": [],
                "coverage_note": "这个追问没有命中已抽取的音频片段或关键画面，建议对相关时间段加密抽帧。",
            }
        evidence_refs = [f"{AIClient._format_seconds(time)}：{AIClient._compress_text(text, 88)}" for _, time, text in selected]
        answer = "我在已分析证据里找到这些相关线索：" + "；".join(evidence_refs[:4]) + "。"
        return {
            "answer": answer[:520],
            "evidence_refs": evidence_refs[:6],
            "coverage_note": "这是基于当前已抽取音频和关键画面的追问回答；没有重新读取更多视频帧。",
        }

    @staticmethod
    def _keywords(text: str) -> list[str]:
        import re

        words = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", text)
        stop = {"这个", "视频", "什么", "怎么", "如何", "一下", "主要", "是否", "有没有", "为什么", "哪里"}
        return [word.lower() for word in words if word and word not in stop]

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

            model = WhisperModel(model_ref, device="cpu", compute_type="int8")
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

    def _responses_json(
        self,
        model: str,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
        images: list[str] | None = None,
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
