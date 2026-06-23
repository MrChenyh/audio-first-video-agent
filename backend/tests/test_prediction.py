from app.prediction import classify_prediction_check


def test_prediction_check_match():
    check = classify_prediction_check(
        {"window_start": 1, "window_end": 5, "hypothesis": "person appears", "expected_evidence": ["person"]},
        [{"time": 3, "scene": "A person appears", "objects": ["person"], "actions": [], "audio_alignment": "match"}],
    )

    assert check["status"] == "match"
    assert check["conflict_score"] < 0.2


def test_prediction_check_conflict():
    check = classify_prediction_check(
        {"window_start": 1, "window_end": 5, "hypothesis": "person appears", "expected_evidence": ["person"]},
        [{"time": 3, "scene": "Empty room", "objects": [], "actions": [], "audio_alignment": "conflict"}],
    )

    assert check["status"] == "conflict"
    assert check["conflict_score"] > 0.75


def test_prediction_check_uncertain_without_observations():
    check = classify_prediction_check(
        {"window_start": 1, "window_end": 5, "hypothesis": "person appears", "expected_evidence": ["person"]},
        [],
    )

    assert check["status"] == "uncertain"
