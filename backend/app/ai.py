from __future__ import annotations

import base64
import json
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

        assert self._client is not None
        attempts: list[dict[str, Any]] = []
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
        if self.mock:
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

        observations: list[dict[str, Any]] = []
        for frame in frames:
            image_data = self._image_data_url(Path(frame["path"]))
            schema = {
                "type": "object",
                "additionalProperties": False,
                "properties": {
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
                    "scene",
                    "objects",
                    "actions",
                    "visible_text",
                    "audio_alignment",
                    "visual_target",
                    "evidence_assessment",
                    "notes",
                ],
            }
            probe = frame.get("probe") or {}
            prompt = (
                "Inspect this video frame as targeted evidence for an audio-first video understanding loop. "
                "Do not merely caption the image. First answer the frame-specific visual target, then decide "
                "whether the frame matches, conflicts with, or is uncertain for the audio-derived hypothesis. "
                "If the frame is too early, too late, cropped, or visually ambiguous, mark audio_alignment as uncertain.\n\n"
                f"Frame timestamp: {frame['time']:.2f}s\n"
                f"Reason selected: {frame.get('reason', '')}\n"
                f"Frame-specific visual target: {json.dumps(probe, ensure_ascii=False)}\n"
                f"Question: {question}\n"
                f"Audio world model: {json.dumps(audio_world_model, ensure_ascii=False)}"
            )
            try:
                payload = self._responses_json(
                    self.settings.vision_model,
                    prompt,
                    schema,
                    "frame_observation",
                    images=[image_data],
                )
            except Exception:
                if not self.settings.allow_model_fallback:
                    raise
                payload = {
                    "scene": "Frame extracted successfully, but the configured model endpoint could not inspect images.",
                    "objects": [],
                    "actions": [],
                    "visible_text": [],
                    "audio_alignment": "uncertain",
                    "visual_target": str(probe.get("question") or frame.get("reason", "")),
                    "evidence_assessment": "The image endpoint could not inspect this frame.",
                    "notes": "Model fallback observation.",
                }
            payload["time"] = frame["time"]
            payload["filename"] = frame["filename"]
            observations.append(payload)
        return observations

    def predict_next_events(
        self,
        *,
        audio_world_model: dict[str, Any],
        observations: list[dict[str, Any]],
        duration: float,
    ) -> list[dict[str, Any]]:
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
