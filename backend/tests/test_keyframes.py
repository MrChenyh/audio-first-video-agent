from app.keyframes import plan_keyframes


def test_plan_keyframes_uses_audio_cues_and_dedupes_times():
    world_model = {
        "timeline": [
            {
                "time": 10.0,
                "label": "glass breaks",
                "expected_visuals": ["broken glass", "person reacting"],
                "visual_question": "Is there visible broken glass or a startled reaction?",
            },
            {
                "time": 10.1,
                "label": "person reacts",
                "expected_visuals": ["person reacting"],
                "visual_question": "Is a person reacting to the sound?",
            },
        ]
    }

    plan = plan_keyframes(duration=60.0, audio_world_model=world_model, question="What happened?")
    times = [item["time"] for item in plan]

    assert 10.0 in times
    assert times == sorted(times)
    assert len(times) == len(set(times))
    assert any(item.get("probe", {}).get("question") == "Is there visible broken glass or a startled reaction?" for item in plan)
    assert any("Expected: broken glass" in item["reason"] for item in plan)


def test_plan_keyframes_adds_refinement_windows():
    plan = plan_keyframes(
        duration=60.0,
        audio_world_model={"timeline": []},
        question="What changed?",
        refinement_windows=[{"start": 20.0, "end": 24.0}],
    )

    times = [item["time"] for item in plan]
    assert 20.0 in times
    assert 22.0 in times
    assert 24.0 in times


def test_plan_keyframes_budget_keeps_late_video_coverage():
    world_model = {
        "timeline": [
            {"time": 0.0, "end_time": 1.39, "label": "opening"},
            {"time": 1.39, "end_time": 3.99, "label": "fox rescue"},
            {"time": 3.99, "end_time": 5.29, "label": "identity question"},
            {"time": 5.29, "end_time": 5.79, "label": "identity confirmation"},
            {"time": 5.79, "end_time": 7.99, "label": "marriage"},
            {"time": 7.99, "end_time": 9.99, "label": "happy ending"},
        ]
    }

    plan = plan_keyframes(
        duration=10.08,
        audio_world_model=world_model,
        question="Summarize the video",
        max_frames=8,
    )
    times = [item["time"] for item in plan]

    assert len(times) == 8
    assert times == sorted(times)
    assert 9.5 <= max(times) < 10.08
    assert any(5.5 <= time <= 8.2 for time in times)
    assert len({item.get("group") for item in plan if item.get("group", "").startswith("audio_event_")}) >= 5


def test_plan_keyframes_budget_prefers_audio_probes_over_global_cover():
    world_model = {
        "timeline": [
            {
                "time": 1.0,
                "end_time": 2.0,
                "label": "door opens",
                "expected_visuals": ["door"],
                "visual_question": "Is the door visible?",
            },
            {
                "time": 8.0,
                "end_time": 9.0,
                "label": "wedding cue",
                "expected_visuals": ["red veil", "couple"],
                "visual_question": "Is there a red veil or couple?",
            },
        ]
    }

    plan = plan_keyframes(
        duration=10.0,
        audio_world_model=world_model,
        question="What happens?",
        max_frames=3,
    )

    assert len(plan) == 3
    assert all(item["source"] != "global" for item in plan)
    assert any("red veil" in " ".join(item.get("probe", {}).get("expected_visuals", [])) for item in plan)


def test_plan_keyframes_avoids_exact_video_end_for_extraction():
    plan = plan_keyframes(
        duration=10.08,
        audio_world_model={"timeline": []},
        question="What happens?",
        refinement_windows=[{"start": 8.8, "end": 10.08}],
        max_frames=3,
    )

    assert max(item["time"] for item in plan) < 10.08
