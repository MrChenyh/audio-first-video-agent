from pathlib import Path

from app.candidates import classify_audio_event, generate_frame_candidates
from app.storage import JobStore
from app.video import VideoProcessor
from test_workflow_mock import make_settings


def test_classify_audio_event_detects_relationship_and_identity():
    assert classify_audio_event({"label": "wedding", "expected_visuals": ["red veil"]}) == "relationship"
    assert classify_audio_event({"label": "identity", "evidence": "you are the fox"}) == "identity"


def test_mock_frame_candidates_follow_audio_events(tmp_path: Path):
    settings = make_settings(tmp_path)
    store = JobStore(settings.data_dir)
    store.init()
    video = tmp_path / "source.mp4"
    video.write_bytes(b"mock video")

    world_model = {
        "timeline": [
            {
                "time": 5.79,
                "end_time": 7.99,
                "label": "marriage",
                "evidence": "they marry",
                "expected_visuals": ["red veil", "couple"],
                "visual_question": "Is there a wedding action?",
            }
        ]
    }

    candidates = generate_frame_candidates(
        video_path=video,
        duration=10.08,
        audio_world_model=world_model,
        question="What happened?",
        processor=VideoProcessor(settings),
        settings=settings,
    )

    assert candidates
    assert candidates[0]["source"] == "visual_candidate"
    assert candidates[0]["probe"]["question"] == "Is there a wedding action?"
