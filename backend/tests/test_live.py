from app.live import _extract_stream_urls, _prefer_low_latency_stream, resolve_live_stream_url, summarize_live_segment


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
