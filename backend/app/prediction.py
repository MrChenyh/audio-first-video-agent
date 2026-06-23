from __future__ import annotations

from typing import Any, Literal


CheckStatus = Literal["match", "conflict", "uncertain"]


def classify_prediction_check(prediction: dict[str, Any], observations: list[dict[str, Any]]) -> dict[str, Any]:
    start = float(prediction.get("window_start", 0.0))
    end = float(prediction.get("window_end", start + 3.0))
    expected = " ".join(prediction.get("expected_evidence") or []).lower()
    hypothesis = str(prediction.get("hypothesis", ""))

    relevant = [
        obs
        for obs in observations
        if start <= float(obs.get("time", 0.0)) <= end
    ]
    if not relevant:
        return {
            "window_start": start,
            "window_end": end,
            "hypothesis": hypothesis,
            "status": "uncertain",
            "conflict_score": 0.5,
            "evidence": "No visual observation was available inside this prediction window.",
        }

    joined = " ".join(
        " ".join(
            [
                str(obs.get("scene", "")),
                " ".join(obs.get("objects") or []),
                " ".join(obs.get("actions") or []),
                " ".join(obs.get("visible_text") or []),
                str(obs.get("visual_target", "")),
                str(obs.get("evidence_assessment", "")),
                str(obs.get("notes", "")),
            ]
        )
        for obs in relevant
    ).lower()

    if expected and any(token for token in expected.split() if len(token) >= 4 and token in joined):
        status: CheckStatus = "match"
        score = 0.1
        evidence = "Expected evidence appears in later frame observations."
    elif any(obs.get("audio_alignment") == "conflict" for obs in relevant):
        status = "conflict"
        score = 0.85
        evidence = "Later frame observations conflict with the audio-derived expectation."
    elif any(obs.get("audio_alignment") == "match" for obs in relevant):
        status = "match"
        score = 0.2
        evidence = "Later frame observations were marked as matching the audio-derived expectation."
    else:
        status = "uncertain"
        score = 0.62
        evidence = "Later frames do not clearly confirm or reject the hypothesis, so targeted local sampling may help."

    return {
        "window_start": start,
        "window_end": end,
        "hypothesis": hypothesis,
        "status": status,
        "conflict_score": score,
        "evidence": evidence,
    }
