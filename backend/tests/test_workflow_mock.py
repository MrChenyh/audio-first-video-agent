from pathlib import Path

from app.ai import AIClient
from app.config import Settings
from app.storage import JobStore
from app.video import VideoProcessor
from app.workflow import VideoUnderstandingWorkflow


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        openai_api_key=None,
        openai_base_url=None,
        openai_org_id=None,
        mock_mode="true",
        fast_mode=False,
        ffmpeg_path=None,
        ffprobe_path=None,
        transcribe_model="gpt-4o-transcribe-diarize",
        transcribe_fallback_model="gpt-4o-transcribe",
        audio_chat_transcribe_model=None,
        vision_provider="openai",
        vision_model="gpt-5.4-mini",
        joyai_api_base="http://127.0.0.1:8070/v1",
        joyai_api_key="EMPTY",
        joyai_model="JoyAI-VL-Interaction-Preview",
        joyai_timeout_seconds=30,
        reasoning_model="gpt-5.5",
        reasoning_effort=None,
        min_video_seconds=30,
        max_video_seconds=600,
        max_refinement_rounds=1,
        max_keyframes=8,
        enhanced_initial_keyframes=4,
        fast_max_keyframes=12,
        fast_seconds_per_frame=120,
        fast_max_timeline_events=12,
        refinement_samples_per_window=1,
        keyframe_strategy="enhanced",
        candidate_sample_fps=3,
        candidate_max_per_event=2,
        candidate_hash_min_distance=4,
        vision_batch_size=3,
        llm_timeout_seconds=90,
        llm_max_retries=1,
        allow_model_fallback=True,
        local_transcribe_first=False,
        local_transcribe_fallback=True,
        local_transcribe_model="tiny",
        live_window_seconds=4,
        live_max_segments=0,
        live_segment_timeout_seconds=18,
        cors_origins=("http://127.0.0.1:5173",),
    )


def test_mock_workflow_completes(tmp_path):
    settings = make_settings(tmp_path)
    store = JobStore(settings.data_dir)
    store.init()
    video = tmp_path / "source.mp4"
    video.write_bytes(b"mock video")
    store.create_job("job1", "What is happening?", video)

    workflow = VideoUnderstandingWorkflow(settings, store, VideoProcessor(settings), AIClient(settings))
    final_state = workflow.run(
        {
            "job_id": "job1",
            "question": "What is happening?",
            "video_path": str(video),
            "refinement_rounds": 0,
            "refinement_windows": [],
            "should_refine": False,
        }
    )

    assert final_state["result"]["answer"]["direct_answer"]
    assert final_state["result"]["frames"]
    assert final_state["prediction_checks"]


def test_uncertain_prediction_triggers_targeted_refinement(tmp_path):
    settings = make_settings(tmp_path)
    store = JobStore(settings.data_dir)
    store.init()
    workflow = VideoUnderstandingWorkflow(settings, store, VideoProcessor(settings), AIClient(settings))

    update = workflow.verify_predictions(
        {
            "duration_seconds": 10.0,
            "refinement_rounds": 0,
            "predictions": [
                {
                    "window_start": 4.0,
                    "window_end": 7.0,
                    "hypothesis": "The next frames should show a red veil.",
                    "expected_evidence": ["red veil"],
                }
            ],
            "frame_observations": [
                {
                    "time": 4.5,
                    "filename": "frame.jpg",
                    "scene": "A person stands in a doorway.",
                    "objects": ["person"],
                    "actions": [],
                    "audio_alignment": "uncertain",
                }
            ],
        }
    )

    assert update["should_refine"] is True
    assert update["refinement_rounds"] == 1
    assert update["refinement_windows"][0]["status"] == "uncertain"
    assert update["refinement_windows"][0]["expected_evidence"] == ["red veil"]
