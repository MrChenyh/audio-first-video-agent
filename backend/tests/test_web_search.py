from pathlib import Path

import httpx

from app.config import Settings
from app.web_search import _dedupe_results, _normalize_duckduckgo_url, search_web


def make_search_settings(tmp_path: Path) -> Settings:
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
        joyai_input_mode="frames",
        joyai_clip_seconds=4,
        joyai_adaptive_clip_seconds=True,
        joyai_clip_width=640,
        joyai_max_clips_per_job=4,
        reasoning_model="gpt-5.5",
        reasoning_effort=None,
        followup_model=None,
        followup_timeout_seconds=20,
        followup_max_chunks=12,
        web_search_enabled=True,
        web_search_provider="custom",
        web_search_base_url="https://search.example.test/api",
        web_search_api_key=None,
        web_search_timeout_seconds=8,
        web_search_max_results=3,
        min_video_seconds=0,
        max_video_seconds=0,
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
        live_fast_capture=True,
        live_frame_width=640,
        cors_origins=("http://127.0.0.1:5173",),
    )


def test_search_web_custom_endpoint_normalizes_results(monkeypatch, tmp_path):
    settings = make_search_settings(tmp_path)

    def fake_get(self, url, params=None):
        assert url == settings.web_search_base_url
        assert params["q"] == "Pocket 4P"
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json={"results": [{"title": "DJI Pocket 4P", "snippet": "dual cameras", "url": "https://example.test/pocket"}]},
        )

    monkeypatch.setattr(httpx.Client, "get", fake_get)

    payload = search_web("Pocket 4P", settings)

    assert payload["enabled"] is True
    assert payload["results"][0]["title"] == "DJI Pocket 4P"
    assert payload["results"][0]["snippet"] == "dual cameras"


def test_duckduckgo_redirect_url_is_unwrapped():
    url = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%3Fb%3D1"

    assert _normalize_duckduckgo_url(url) == "https://example.com/a?b=1"


def test_dedupe_results_keeps_first_unique_item():
    results = _dedupe_results(
        [
            {"title": "A", "snippet": "one", "url": "https://example.test/a"},
            {"title": "A", "snippet": "two", "url": "https://example.test/a"},
            {"title": "B", "snippet": "three", "url": "https://example.test/b"},
        ]
    )

    assert [item["title"] for item in results] == ["A", "B"]
