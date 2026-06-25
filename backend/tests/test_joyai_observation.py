from app.ai import AIClient


def test_joyai_text_to_observation_extracts_wedding_evidence():
    observation = AIClient._joyai_text_to_observation(
        {"time": 6.5, "filename": "frame.jpg"},
        "Does the frame show wedding evidence?",
        "audio context",
        "</response> 男子手持红布，背景可见囍字，画面支持婚礼场景。",
    )

    assert observation["audio_alignment"] == "match"
    assert "囍" in observation["visible_text"]
    assert "红布" in observation["visible_text"]
    assert observation["visual_target"] == "Does the frame show wedding evidence?"


def test_joyai_text_to_observation_handles_silence():
    observation = AIClient._joyai_text_to_observation(
        {"time": 1.0, "filename": "frame.jpg"},
        "target",
        "audio context",
        "</silence>",
    )

    assert observation["audio_alignment"] == "uncertain"
    assert "silence" in observation["evidence_assessment"]
