from pathlib import Path

from app.main import (
    _explicit_times_from_question,
    _external_search_terms,
    _followup_needs_visual_probe,
    _followup_probe_times,
    _friendly_ytdlp_error,
    _looks_like_media_response,
    _state_to_partial,
    _video_page_url_candidates,
    _web_search_queries,
)
from app.ai import AIClient


def test_media_response_detection_accepts_direct_video_urls():
    assert _looks_like_media_response("https://example.com/video.mp4", "text/plain") is True
    assert _looks_like_media_response("https://example.com/watch?id=1", "video/mp4") is True
    assert _looks_like_media_response("https://example.com/watch?id=1", "text/html") is False


def test_douyin_jingxuan_modal_url_adds_video_page_candidates():
    candidates = _video_page_url_candidates("https://www.douyin.com/jingxuan?modal_id=7653491239685328174")

    assert candidates[0] == "https://www.douyin.com/jingxuan?modal_id=7653491239685328174"
    assert "https://www.douyin.com/video/7653491239685328174" in candidates
    assert "https://www.douyin.com/note/7653491239685328174" in candidates
    assert "https://www.iesdouyin.com/share/video/7653491239685328174/" in candidates


def test_friendly_ytdlp_error_mentions_cookie_configuration():
    message = _friendly_ytdlp_error("ERROR: [Douyin] 123: Fresh cookies (not necessarily logged in) are needed")

    assert "DOUYIN_COOKIES_FILE" in message
    assert "YTDLP_COOKIES_FROM_BROWSER" in message


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
    assert partial["ai_overview"]["suggested_questions"]


def test_ai_overview_adds_summary_highlights_and_recommended_questions():
    result = {
        "answer": {
            "direct_answer": "这是一期 Pocket 4P 评测视频，核心是长焦、画质和动态范围。",
            "summary": "视频围绕 Pocket 4P 是否值得升级展开，重点看长焦、动态范围、发热和稳定体验。",
            "sections": [
                {"title": "内容脉络", "items": ["0:14：介绍等效 60mm 长焦。", "7:55：测试 ISO1600 动态范围。"]},
                {"title": "关键结论", "items": ["重画质和长焦可以买 Pocket 4P。"]},
            ],
        },
        "timeline": [
            {"time": 14.0, "label": "长焦", "evidence": "等效60毫米长焦"},
            {"time": 475.0, "label": "动态范围", "evidence": "ISO1600 动态范围最好"},
        ],
        "transcript_segments": [{"start": 14.0, "end": 16.0, "text": "等效60毫米长焦"}],
        "frames": [],
        "audio_world_model": {"summary": "产品评测"},
    }

    overview = AIClient.build_ai_overview(result)

    assert "Pocket 4P" in overview["summary"]
    assert any(item["time_label"] == "0:14" for item in overview["highlights"])
    assert "相比上一代有哪些升级？" in overview["suggested_questions"]


def test_followup_probe_times_selects_relevant_video_moments():
    result = {
        "answer": {"direct_answer": "产品评测，重点是画质、长焦和购买建议。"},
        "timeline": [],
        "transcript_segments": [
            {"start": 14.0, "end": 16.0, "text": "等效60毫米长焦"},
            {"start": 304.0, "end": 306.0, "text": "ISO1600 动态范围最好，大概17档"},
            {"start": 699.0, "end": 711.0, "text": "多花的钱主要买画质，别的跟Pocket 4一样"},
        ],
        "frames": [],
        "metadata": {"duration_seconds": 720},
    }

    times = _followup_probe_times("动态范围怎么样", result, duration=720)

    assert times[0] == 304.0
    assert all(0 <= time <= 720 for time in times)


def test_followup_visual_probe_only_for_visual_questions():
    assert _followup_needs_visual_probe("推荐买哪一代") is False
    assert _followup_needs_visual_probe("总结视频内容") is False
    assert _followup_needs_visual_probe("5:04 画面里展示了什么") is True
    assert _followup_needs_visual_probe("外观长什么样") is True


def test_explicit_times_from_followup_question_take_priority():
    assert _explicit_times_from_question("5:04 画面里展示了什么", 720) == [304.0]
    assert _explicit_times_from_question("304秒附近呢", 720) == [304.0]
    assert _explicit_times_from_question("1分02秒的位置", 720) == [62.0]

    result = {"transcript_segments": [{"start": 317.0, "text": "更多的高光信息"}], "frames": []}
    assert _followup_probe_times("5:04 画面里展示了什么", result, duration=720)[0] == 304.0


def test_web_search_queries_split_external_product_from_video_context():
    result = {
        "answer": {"direct_answer": "Pocket 4P 评测，重点是长焦、画质和动态范围。"},
        "transcript_segments": [{"start": 0, "text": "等效60毫米长焦"}],
    }

    queries = _web_search_queries("总结这个视频内容，并对比跟insta360 luna哪个更好，区别是什么？", result)

    assert queries[0] == "insta360 luna specs dynamic range"
    assert "insta360 luna vs DJI Pocket 4P" in queries
    assert _external_search_terms("对比跟insta360 luna哪个更好") == ["insta360 luna"]
