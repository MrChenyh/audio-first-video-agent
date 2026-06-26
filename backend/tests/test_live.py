from app.live import (
    evaluate_live_moderation,
    _extract_stream_urls,
    _is_low_quality_live_audio,
    _prefer_low_latency_stream,
    resolve_live_stream_url,
    summarize_live_segment,
    update_live_world_model,
)


def test_extract_douyin_stream_urls_from_html():
    html = (
        '{"hls_pull_url":"http://pull-hls.example.com/live_sd/playlist.m3u8?wsSecret=abc\\u0026wsTime=123",'
        '"flv_pull_url":{"HD1":"http://pull-flv.example.com/live_hd.flv?wsSecret=def\\u0026wsTime=456"}}'
    )

    hls = _extract_stream_urls(html.replace("\\u0026", "&"), ".m3u8")
    flv = _extract_stream_urls(html.replace("\\u0026", "&"), ".flv")

    assert hls == ["http://pull-hls.example.com/live_sd/playlist.m3u8?wsSecret=abc&wsTime=123"]
    assert flv == ["http://pull-flv.example.com/live_hd.flv?wsSecret=def&wsTime=456"]
    assert _prefer_low_latency_stream(hls + flv).endswith("playlist.m3u8?wsSecret=abc&wsTime=123")


def test_direct_stream_url_is_returned_without_resolution():
    assert resolve_live_stream_url("https://example.com/live/playlist.m3u8") == "https://example.com/live/playlist.m3u8"


def test_live_segment_summary_combines_audio_and_visual():
    summary = summarize_live_segment(
        index=0,
        start_time=0,
        transcript="主播正在介绍商品价格",
        observation={"evidence_assessment": "画面中主播拿起一件商品，屏幕右侧有价格文字。"},
        question="实时总结直播",
    )

    assert "第 1 段" in summary
    assert "音频" in summary
    assert "画面" in summary


def test_live_world_model_accumulates_news_context():
    model = update_live_world_model(
        None,
        {
            "index": 0,
            "start_time": 0.0,
            "end_time": 2.0,
            "transcript": "主持人正常介绍商品，没有违规内容",
            "observation": {"evidence_assessment": "当前帧未发现可见违规。"},
            "moderation": {"has_risk": False, "risk_level": "none", "violations": [], "summary": "未发现违禁风险"},
        },
        "实时监控直播是否有违禁行为",
    )

    assert model["status"] == "ready"
    assert model["program_type"] == "直播合规监控"
    assert model["risk_state"]["scanned_segments"] == 1
    assert model["risk_state"]["risk_segments"] == 0
    assert model["evidence_count"] == 0


def test_live_world_model_hides_joyai_failure_from_visual_evidence():
    model = update_live_world_model(
        None,
        {
            "index": 0,
            "start_time": 0.0,
            "end_time": 2.0,
            "transcript": "主持人正在播报一条新闻",
            "observation": {
                "evidence_assessment": "JoyAI local vision endpoint failed.",
                "vision_error": "JoyAI local vision endpoint failed.",
            },
        },
        "实时总结直播",
    )

    assert model["visual_evidence_count"] == 0
    assert "JoyAI" not in model["stable_summary"]


def test_live_world_model_uses_visual_focus_when_short_asr_is_noisy():
    moderation = evaluate_live_moderation(
        index=0,
        start_time=0.0,
        end_time=2.0,
        transcript="",
        raw_transcript="石蜜丝 鬼",
        observation={"evidence_assessment": "当前帧未发现可见违规。"},
    )
    model = update_live_world_model(
        None,
        {
            "index": 0,
            "start_time": 0.0,
            "end_time": 2.0,
            "transcript": "石蜜丝 鬼",
            "observation": {"evidence_assessment": "这是新闻画面，屏幕显示 Smithfield Since 1936，主持人正在介绍品牌历史。"},
            "moderation": moderation,
        },
        "实时监控直播是否有违禁行为",
    )

    assert model["risk_state"]["risk_segments"] == 0
    assert model["audio_evidence_count"] == 0


def test_news_live_visual_title_beats_weak_short_audio():
    moderation = evaluate_live_moderation(
        index=0,
        start_time=0.0,
        end_time=2.0,
        transcript="",
        raw_transcript="184种 利用",
        observation={"evidence_assessment": "当前帧未发现可见违规。"},
    )
    model = update_live_world_model(
        None,
        {
            "index": 0,
            "start_time": 0.0,
            "end_time": 2.0,
            "transcript": "184种 利用",
            "observation": {"evidence_assessment": "这是一档新闻资讯直播，画面正在播报关于黄岩岛蓝洞形成时间的科学报道，字幕显示黄岩岛蓝洞至少形成于3200年前。"},
            "moderation": moderation,
        },
        "实时监控直播是否有违禁行为",
    )

    assert model["risk_state"]["risk_segments"] == 0
    assert "184种" not in model["stable_summary"]


def test_live_audio_quality_filter_keeps_meaningful_news_terms():
    assert _is_low_quality_live_audio("石蜜丝 鬼") is True
    assert _is_low_quality_live_audio("卧槽你别这样") is False


def test_live_moderation_detects_audio_profanity_and_updates_alert_state():
    moderation = evaluate_live_moderation(
        index=1,
        start_time=2.0,
        end_time=4.0,
        transcript="卧槽你别这样",
        raw_transcript="卧槽你别这样",
        observation={"evidence_assessment": "当前帧未发现可见违规。"},
    )
    model = update_live_world_model(
        None,
        {
            "index": 1,
            "start_time": 2.0,
            "end_time": 4.0,
            "transcript": "卧槽你别这样",
            "observation": {"evidence_assessment": "当前帧未发现可见违规。"},
            "moderation": moderation,
        },
        "实时监控直播是否有违禁行为",
    )

    assert moderation["has_risk"] is True
    assert moderation["violations"][0]["category"] == "profanity"
    assert model["risk_state"]["risk_segments"] == 1
    assert model["audio_evidence_count"] == 1
    assert "违禁词" in model["stable_summary"]


def test_live_moderation_uses_visual_json_for_smoking():
    moderation = evaluate_live_moderation(
        index=0,
        start_time=0.0,
        end_time=2.0,
        transcript="",
        raw_transcript="",
        observation={
            "live_moderation": {
                "risk_level": "medium",
                "caption": "画面有人吸烟",
                "violations": [
                    {
                        "category": "smoking",
                        "severity": "medium",
                        "confidence": 0.86,
                        "evidence": "人物手持香烟并有吸烟动作",
                        "visible_text": [],
                    }
                ],
            }
        },
    )

    assert moderation["has_risk"] is True
    assert moderation["violations"][0]["category"] == "smoking"
    assert moderation["risk_level"] == "medium"
