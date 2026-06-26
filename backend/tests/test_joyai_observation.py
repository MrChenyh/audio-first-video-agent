from app.ai import AIClient
from test_workflow_mock import make_settings


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
    assert "视觉模型没有返回可用画面描述" in observation["evidence_assessment"]
    assert "silence" in observation["vision_error"]


def test_joyai_text_to_observation_parses_live_moderation_json():
    observation = AIClient._joyai_text_to_observation(
        {"time": 2.0, "filename": "frame.jpg"},
        "monitor live violations",
        "audio context",
        '</response> {"risk_level":"medium","caption":"画面有人吸烟","violations":[{"category":"smoking","severity":"medium","confidence":0.88,"evidence":"人物手持香烟并靠近嘴部","visible_text":[]}]}'
    )

    assert observation["audio_alignment"] == "match"
    assert observation["live_moderation"]["risk_level"] == "medium"
    assert observation["live_moderation"]["violations"][0]["category"] == "smoking"


def test_joyai_auto_clip_mode_detects_temporal_questions(tmp_path):
    settings = make_settings(tmp_path)
    settings = settings.__class__(**{**settings.__dict__, "joyai_input_mode": "auto"})
    ai = AIClient(settings)

    assert ai._should_use_joyai_clips("防抖稳定性怎么样", [{"reason": "样片动态范围测试", "probe": {}}]) is True
    assert ai._should_use_joyai_clips("外观是什么样", [{"reason": "外观展示", "probe": {}}]) is False


def test_joyai_clip_observation_preserves_clip_metadata():
    observation = AIClient._joyai_text_to_observation(
        {"time": 477.48, "filename": "frame.jpg"},
        "看动态范围测试",
        "audio context",
        "</response> 片段中从相机画面切到暗部灯光测试，能看到高光和暗部同时保留，画面支持动态范围测试。",
    )
    observation["clip"] = {"start": 475.48, "end": 479.48, "mode": "joyai_video_url"}

    assert observation["audio_alignment"] == "match"
    assert observation["clip"]["mode"] == "joyai_video_url"


def test_joyai_text_to_observation_treats_related_clip_as_match():
    observation = AIClient._joyai_text_to_observation(
        {"time": 603.49, "filename": "frame.jpg"},
        "Does the clip show 3D printed furniture?",
        "audio context",
        "</response> 视频中展示了一个蓝色的3D打印家具，具有明显的分层打印纹理，与音频中提到的成本和家具相关。",
    )

    assert observation["audio_alignment"] == "match"


def test_joyai_clip_mode_is_default_for_static_video(tmp_path):
    settings = make_settings(tmp_path)
    settings = settings.__class__(**{**settings.__dict__, "joyai_input_mode": "clips"})
    ai = AIClient(settings)

    assert ai._should_use_joyai_clips("summarize the video", [{"reason": "audio event", "probe": {}}]) is True


def test_joyai_clip_mode_skips_live_segments(tmp_path):
    settings = make_settings(tmp_path)
    settings = settings.__class__(**{**settings.__dict__, "joyai_input_mode": "clips"})
    ai = AIClient(settings)

    frame = {"reason": "live moderation sample", "probe": {"type": "live_segment"}}
    assert ai._should_use_joyai_clips("monitor live violations", [frame]) is False


def test_joyai_adaptive_clip_seconds_shortens_static_checks(tmp_path):
    settings = make_settings(tmp_path)
    settings = settings.__class__(**{**settings.__dict__, "joyai_clip_seconds": 4, "joyai_adaptive_clip_seconds": True})
    ai = AIClient(settings)

    static_frame = {"reason": "3D printed cabinet", "probe": {"question": "Does this show printed furniture?"}}
    motion_frame = {"reason": "stabilization motion sample", "probe": {"question": "Does this show dynamic range or motion?"}}
    assert ai._clip_seconds_for_frame("summarize", static_frame) == 2.0
    assert ai._clip_seconds_for_frame("防抖稳定性怎么样", motion_frame) == 4.0
