from pathlib import Path

from app.main import _looks_like_media_response, _state_to_partial


def test_media_response_detection_accepts_direct_video_urls():
    assert _looks_like_media_response("https://example.com/video.mp4", "text/plain") is True
    assert _looks_like_media_response("https://example.com/watch?id=1", "video/mp4") is True
    assert _looks_like_media_response("https://example.com/watch?id=1", "text/html") is False


def test_state_to_partial_merges_observed_frames():
    job = {"job_id": "job1", "question": "What happened?", "status": "running", "progress": 70, "current_node": "observe_frames"}
    state = {
        "duration_seconds": 12.0,
        "extracted_frames": [
            {"time": 1.5, "filename": "frame.jpg", "url": "/frame.jpg", "reason": "audio cue"},
        ],
        "frame_observations": [
            {"time": 1.5, "filename": "frame.jpg", "scene": "A machine is visible."},
        ],
        "audio_world_model": {"timeline": [{"time": 1.0, "label": "cue", "evidence": "machine starts"}]},
        "transcript_segments": [{"start": 1.0, "end": 2.0, "text": "machine starts"}],
    }

    partial = _state_to_partial(job, state)

    assert partial["partial"] is True
    assert partial["frames"][0]["observation"]["scene"] == "A machine is visible."
    assert partial["timeline"][0]["label"] == "cue"
