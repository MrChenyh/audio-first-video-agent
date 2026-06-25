from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from langgraph.graph import END, StateGraph

try:
    from langgraph.checkpoint.memory import MemorySaver
except Exception:  # pragma: no cover
    MemorySaver = None  # type: ignore[assignment]

from .ai import AIClient
from .candidates import generate_frame_candidates
from .config import Settings
from .keyframes import plan_keyframes
from .models import VideoAgentState
from .storage import JobStore
from .video import VideoProcessor


ProgressMap = dict[str, int]


NODE_PROGRESS: ProgressMap = {
    "ingest_video": 5,
    "extract_audio": 15,
    "transcribe_audio": 25,
    "build_audio_world_model": 38,
    "generate_frame_candidates": 44,
    "plan_keyframes": 48,
    "extract_keyframes": 58,
    "observe_frames": 70,
    "predict_next_events": 80,
    "verify_predictions": 88,
    "synthesize_answer": 96,
}


class VideoUnderstandingWorkflow:
    def __init__(self, settings: Settings, store: JobStore, processor: VideoProcessor, ai: AIClient):
        self.settings = settings
        self.store = store
        self.processor = processor
        self.ai = ai
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(VideoAgentState)
        builder.add_node("ingest_video", self._node("ingest_video", self.ingest_video))
        builder.add_node("extract_audio", self._node("extract_audio", self.extract_audio))
        builder.add_node("transcribe_audio", self._node("transcribe_audio", self.transcribe_audio))
        builder.add_node("build_audio_world_model", self._node("build_audio_world_model", self.build_audio_world_model))
        builder.add_node(
            "generate_frame_candidates",
            self._node("generate_frame_candidates", self.generate_frame_candidates_node),
        )
        builder.add_node("plan_keyframes", self._node("plan_keyframes", self.plan_keyframes_node))
        builder.add_node("extract_keyframes", self._node("extract_keyframes", self.extract_keyframes))
        builder.add_node("observe_frames", self._node("observe_frames", self.observe_frames))
        builder.add_node("predict_next_events", self._node("predict_next_events", self.predict_next_events))
        builder.add_node("verify_predictions", self._node("verify_predictions", self.verify_predictions))
        builder.add_node("synthesize_answer", self._node("synthesize_answer", self.synthesize_answer))

        builder.set_entry_point("ingest_video")
        builder.add_edge("ingest_video", "extract_audio")
        builder.add_edge("extract_audio", "transcribe_audio")
        builder.add_edge("transcribe_audio", "build_audio_world_model")
        builder.add_edge("build_audio_world_model", "generate_frame_candidates")
        builder.add_edge("generate_frame_candidates", "plan_keyframes")
        builder.add_edge("plan_keyframes", "extract_keyframes")
        builder.add_edge("extract_keyframes", "observe_frames")
        builder.add_edge("observe_frames", "predict_next_events")
        builder.add_edge("predict_next_events", "verify_predictions")
        builder.add_conditional_edges(
            "verify_predictions",
            self._next_after_verification,
            {"refine": "plan_keyframes", "finish": "synthesize_answer"},
        )
        builder.add_edge("synthesize_answer", END)

        if MemorySaver is not None:
            return builder.compile(checkpointer=MemorySaver())
        return builder.compile()

    def _node(
        self,
        node_name: str,
        func: Callable[[VideoAgentState], dict[str, Any]],
    ) -> Callable[[VideoAgentState], dict[str, Any]]:
        def wrapped(state: VideoAgentState) -> dict[str, Any]:
            job_id = state["job_id"]
            self.store.update_job(
                job_id,
                status="running",
                current_node=node_name,
                progress=NODE_PROGRESS[node_name],
            )
            updates = func(state)
            merged = {**state, **updates}
            self.store.save_state(job_id, merged)
            return updates

        return wrapped

    def _next_after_verification(self, state: VideoAgentState) -> str:
        if state.get("should_refine") and int(state.get("refinement_rounds", 0)) <= self.settings.max_refinement_rounds:
            return "refine"
        return "finish"

    def run(self, initial_state: VideoAgentState) -> VideoAgentState:
        config = {"configurable": {"thread_id": initial_state["job_id"]}}
        return self.graph.invoke(initial_state, config=config)

    def ingest_video(self, state: VideoAgentState) -> dict[str, Any]:
        video_path = Path(state["video_path"])
        metadata = self.processor.probe_video(video_path)
        self.processor.validate_duration(float(metadata["duration_seconds"]))
        return {
            "duration_seconds": float(metadata["duration_seconds"]),
            "fps": float(metadata["fps"]),
            "width": int(metadata["width"]),
            "height": int(metadata["height"]),
            "has_audio": bool(metadata["has_audio"]),
            "vision_request_count": int(state.get("vision_request_count", 0)),
            "refinement_rounds": int(state.get("refinement_rounds", 0)),
            "refinement_windows": state.get("refinement_windows", []),
            "should_refine": False,
        }

    def extract_audio(self, state: VideoAgentState) -> dict[str, Any]:
        audio_path = self.store.job_dir(state["job_id"]) / "audio.wav"
        extracted = self.processor.extract_audio(Path(state["video_path"]), audio_path, bool(state.get("has_audio", False)))
        return {"audio_path": str(extracted) if extracted else None}

    def transcribe_audio(self, state: VideoAgentState) -> dict[str, Any]:
        audio_path = Path(state["audio_path"]) if state.get("audio_path") else None
        segments = self.ai.transcribe(audio_path, float(state.get("duration_seconds", 0.0)))
        return {"transcript_segments": segments, "transcription_status": self.ai.last_transcription_status}

    def build_audio_world_model(self, state: VideoAgentState) -> dict[str, Any]:
        world_model = self.ai.build_audio_world_model(
            question=state["question"],
            transcript_segments=state.get("transcript_segments", []),
            duration=float(state.get("duration_seconds", 0.0)),
            has_audio=bool(state.get("audio_path")),
        )
        return {"audio_world_model": world_model}

    def generate_frame_candidates_node(self, state: VideoAgentState) -> dict[str, Any]:
        if state.get("refinement_rounds", 0):
            return {"frame_candidates": state.get("frame_candidates", [])}
        candidates = generate_frame_candidates(
            video_path=Path(state["video_path"]),
            duration=float(state.get("duration_seconds", 0.0)),
            audio_world_model=state.get("audio_world_model", {}),
            question=state["question"],
            processor=self.processor,
            settings=self.settings,
        )
        return {"frame_candidates": candidates}

    def plan_keyframes_node(self, state: VideoAgentState) -> dict[str, Any]:
        max_frames = self.settings.max_keyframes
        if self.settings.fast_mode:
            duration = float(state.get("duration_seconds", 0.0))
            adaptive = max(
                self.settings.enhanced_initial_keyframes,
                int((duration / self.settings.fast_seconds_per_frame) + 0.999),
            )
            max_frames = min(self.settings.fast_max_keyframes, adaptive)
        if (
            self.settings.keyframe_strategy == "enhanced"
            and not self.settings.fast_mode
            and not int(state.get("refinement_rounds", 0))
            and state.get("frame_candidates")
        ):
            max_frames = min(max_frames, self.settings.enhanced_initial_keyframes)
        plan = plan_keyframes(
            duration=float(state["duration_seconds"]),
            audio_world_model=state.get("audio_world_model", {}),
            question=state["question"],
            existing_plan=state.get("keyframe_plan", []),
            refinement_windows=state.get("refinement_windows", []),
            frame_candidates=state.get("frame_candidates", []),
            max_frames=max_frames,
            refinement_samples_per_window=self.settings.refinement_samples_per_window,
        )
        return {
            "keyframe_plan": plan,
            "should_refine": False,
            "refinement_windows": [],
        }

    def extract_keyframes(self, state: VideoAgentState) -> dict[str, Any]:
        frames_dir = self.store.job_dir(state["job_id"]) / "frames"
        existing_by_time = {round(float(frame["time"]), 2): frame for frame in state.get("extracted_frames", [])}
        frames: list[dict[str, Any]] = list(existing_by_time.values())
        for item in state.get("keyframe_plan", []):
            time = round(float(item["time"]), 2)
            if time in existing_by_time:
                continue
            extension = "png" if self.settings.use_mock_models and not self.processor.ffmpeg else "jpg"
            filename = f"frame_{int(time * 100):08d}.{extension}"
            output_path = frames_dir / filename
            self.processor.extract_frame(Path(state["video_path"]), time, output_path)
            frames.append(
                {
                    "time": time,
                    "filename": filename,
                    "path": str(output_path),
                    "url": f"/api/jobs/{state['job_id']}/frames/{filename}",
                    "reason": item.get("reason", ""),
                    "probe": item.get("probe"),
                }
            )
        frames = sorted(frames, key=lambda item: item["time"])
        return {"extracted_frames": frames}

    def observe_frames(self, state: VideoAgentState) -> dict[str, Any]:
        already = {obs["filename"] for obs in state.get("frame_observations", [])}
        new_frames = [frame for frame in state.get("extracted_frames", []) if frame["filename"] not in already]
        new_observations = []
        total = max(1, len(new_frames))
        processed = 0
        requests_used = 0
        batch_size = max(1, self.settings.vision_batch_size)
        for batch_start in range(0, len(new_frames), batch_size):
            batch = new_frames[batch_start : batch_start + batch_size]
            frame_observation = self.ai.observe_frames(
                question=state["question"],
                frames=batch,
                audio_world_model=state.get("audio_world_model", {}),
            )
            new_observations.extend(frame_observation)
            processed += len(batch)
            requests_used += self.ai.last_vision_request_count
            partial_observations = sorted(
                list(state.get("frame_observations", [])) + new_observations,
                key=lambda item: item["time"],
            )
            progress = min(79, NODE_PROGRESS["observe_frames"] + int((processed / total) * 9))
            self.store.update_job(
                state["job_id"],
                status="running",
                current_node=f"observe_frames {processed}/{total}",
                progress=progress,
            )
            self.store.save_state(
                state["job_id"],
                {
                    **state,
                    "frame_observations": partial_observations,
                    "vision_request_count": int(state.get("vision_request_count", 0)) + requests_used,
                },
            )
        observations = sorted(
            list(state.get("frame_observations", [])) + new_observations,
            key=lambda item: item["time"],
        )
        return {
            "frame_observations": observations,
            "vision_request_count": int(state.get("vision_request_count", 0)) + requests_used,
        }

    def predict_next_events(self, state: VideoAgentState) -> dict[str, Any]:
        predictions = self.ai.predict_next_events(
            audio_world_model=state.get("audio_world_model", {}),
            observations=state.get("frame_observations", []),
            duration=float(state["duration_seconds"]),
        )
        return {"predictions": predictions}

    def verify_predictions(self, state: VideoAgentState) -> dict[str, Any]:
        checks = self.ai.verify_predictions(
            predictions=state.get("predictions", []),
            observations=state.get("frame_observations", []),
        )
        if self.settings.fast_mode:
            return {
                "prediction_checks": checks,
                "refinement_windows": [],
                "refinement_rounds": int(state.get("refinement_rounds", 0)),
                "should_refine": False,
            }
        conflict_windows = []
        observed_times = [float(obs.get("time", 0.0)) for obs in state.get("frame_observations", [])]
        for check in checks:
            score = float(check.get("conflict_score", 0.0))
            status = check.get("status")
            needs_more_evidence = status == "conflict" and score >= 0.75
            needs_more_evidence = needs_more_evidence or (status == "uncertain" and score >= 0.55)
            if not needs_more_evidence:
                continue

            start = max(0.0, float(check["window_start"]) - 0.75)
            end = min(float(state["duration_seconds"]), float(check["window_end"]) + 0.75)
            inside_count = sum(1 for time in observed_times if start <= time <= end)
            if inside_count >= 3:
                continue
            conflict_windows.append(
                {
                    "start": start,
                    "end": end,
                    "reason": check["evidence"],
                    "hypothesis": check.get("hypothesis", ""),
                    "expected_evidence": check.get("expected_evidence", []),
                    "status": status,
                    "conflict_score": score,
                }
            )
        refinement_rounds = int(state.get("refinement_rounds", 0))
        should_refine = bool(conflict_windows) and refinement_rounds < self.settings.max_refinement_rounds
        return {
            "prediction_checks": checks,
            "refinement_windows": conflict_windows,
            "refinement_rounds": refinement_rounds + (1 if should_refine else 0),
            "should_refine": should_refine,
        }

    def synthesize_answer(self, state: VideoAgentState) -> dict[str, Any]:
        answer = self.ai.synthesize_answer(
            question=state["question"],
            audio_world_model=state.get("audio_world_model", {}),
            observations=state.get("frame_observations", []),
            checks=state.get("prediction_checks", []),
        )
        frame_observation_map = {obs["filename"]: obs for obs in state.get("frame_observations", [])}
        frames = []
        for frame in state.get("extracted_frames", []):
            observation = frame_observation_map.get(frame["filename"])
            frames.append({**frame, "observation": observation})
        result = {
            "job_id": state["job_id"],
            "question": state["question"],
            "answer": answer,
            "timeline": (state.get("audio_world_model") or {}).get("timeline", []),
            "transcript_segments": state.get("transcript_segments", []),
            "audio_world_model": state.get("audio_world_model", {}),
            "frames": frames,
            "prediction_checks": state.get("prediction_checks", []),
            "evidence": answer.get("evidence_refs", []),
            "metadata": {
                "duration_seconds": state.get("duration_seconds"),
                "fps": state.get("fps"),
                "width": state.get("width"),
                "height": state.get("height"),
                "has_audio": state.get("has_audio"),
                "mock_mode": self.settings.use_mock_models,
                "fast_mode": self.settings.fast_mode,
                "vision_provider": self.settings.vision_provider,
                "vision_model": self.settings.joyai_model
                if self.settings.vision_provider in {"joyai", "joyai_adapter"}
                else self.settings.vision_model,
                "vision_request_count": state.get("vision_request_count", 0),
            },
            "transcription_status": state.get("transcription_status", {}),
        }
        return {"answer": answer, "result": result}
