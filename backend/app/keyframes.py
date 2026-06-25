from __future__ import annotations

from typing import Any


def _clamp_time(value: float, duration: float) -> float:
    if duration <= 0:
        return 0.0
    extractable_end = max(0.0, duration - 0.05)
    return max(0.0, min(extractable_end, round(float(value), 2)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def dedupe_times(times: list[tuple[float, str, str]], duration: float, min_gap: float = 0.35) -> list[dict[str, Any]]:
    sorted_items = sorted(times, key=lambda item: item[0])
    planned: list[dict[str, Any]] = []
    for raw_time, reason, source in sorted_items:
        time = _clamp_time(raw_time, duration)
        if planned and abs(time - planned[-1]["time"]) < min_gap:
            planned[-1]["reason"] = f"{planned[-1]['reason']}; {reason}"
            continue
        planned.append({"time": time, "reason": reason, "source": source})
    return planned


def _add_candidate(
    candidates: list[dict[str, Any]],
    time: float,
    reason: str,
    source: str,
    priority: int,
    *,
    group: str | None = None,
    probe: dict[str, Any] | None = None,
) -> None:
    candidates.append(
        {
            "time": time,
            "reason": reason,
            "source": source,
            "priority": priority,
            "group": group,
            "probe": probe,
        }
    )


def _dedupe_candidates(
    candidates: list[dict[str, Any]],
    duration: float,
    min_gap: float = 0.35,
) -> list[dict[str, Any]]:
    sorted_items = sorted(candidates, key=lambda item: (-int(item["priority"]), float(item["time"])))
    selected: list[dict[str, Any]] = []
    for candidate in sorted_items:
        raw_time = float(candidate["time"])
        reason = str(candidate["reason"])
        source = str(candidate["source"])
        priority = int(candidate["priority"])
        time = _clamp_time(raw_time, duration)
        match = next((item for item in selected if abs(float(item["time"]) - time) < min_gap), None)
        if match:
            match["reason"] = f"{match['reason']}; {reason}"
            match["priority"] = max(int(match.get("priority", 0)), priority)
            if source not in str(match.get("source", "")).split("+"):
                match["source"] = f"{match.get('source', '')}+{source}".strip("+")
            if not match.get("probe") and candidate.get("probe"):
                match["probe"] = candidate["probe"]
            if not match.get("group") and candidate.get("group"):
                match["group"] = candidate["group"]
            continue
        selected.append(
            {
                "time": time,
                "reason": reason,
                "source": source,
                "priority": priority,
                "group": candidate.get("group"),
                "probe": candidate.get("probe"),
            }
        )
    return selected


def _binary_cover_times(duration: float, depth: int = 3) -> list[float]:
    if duration <= 0:
        return [0.0]
    points = {0.0, duration}
    intervals = [(0.0, duration)]
    for _ in range(depth):
        next_intervals: list[tuple[float, float]] = []
        for start, end in intervals:
            midpoint = (start + end) / 2
            points.add(midpoint)
            next_intervals.extend([(start, midpoint), (midpoint, end)])
        intervals = next_intervals
    return sorted(points)


def _spread_select(items: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    if budget <= 0:
        return []
    if len(items) <= budget:
        return sorted(items, key=lambda item: item["time"])

    by_time = sorted(items, key=lambda item: item["time"])
    if budget == 1:
        return [max(by_time, key=lambda item: int(item.get("priority", 0)))]

    selected = [by_time[0], by_time[-1]]
    while len(selected) < budget:
        selected_times = [float(item["time"]) for item in selected]
        best_item = None
        best_score = -1.0
        for item in by_time:
            if item in selected:
                continue
            nearest_gap = min(abs(float(item["time"]) - time) for time in selected_times)
            score = nearest_gap + (int(item.get("priority", 0)) / 100.0)
            if score > best_score:
                best_score = score
                best_item = item
        if best_item is None:
            break
        selected.append(best_item)
    return sorted(selected, key=lambda item: item["time"])


def _choose_budgeted_frames(candidates: list[dict[str, Any]], max_frames: int) -> list[dict[str, Any]]:
    if max_frames <= 0:
        return []
    if len(candidates) <= max_frames:
        return sorted(candidates, key=lambda item: item["time"])

    selected: list[dict[str, Any]] = []

    def add(item: dict[str, Any]) -> None:
        if item not in selected and len(selected) < max_frames:
            selected.append(item)

    refinement_items = [item for item in candidates if str(item.get("source", "")).startswith("refinement")]
    for item in _spread_select(refinement_items, min(len(refinement_items), max_frames)):
        add(item)

    by_group: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        group = str(item.get("group") or "")
        if group.startswith("audio_event_"):
            by_group.setdefault(group, []).append(item)

    event_probes = [
        max(items, key=lambda item: (int(item.get("priority", 0)), float(item["time"])))
        for _, items in sorted(by_group.items(), key=lambda pair: min(float(item["time"]) for item in pair[1]))
    ]
    for item in _spread_select(event_probes, max_frames - len(selected)):
        add(item)

    for item in sorted(candidates, key=lambda item: (-int(item.get("priority", 0)), item["time"])):
        if len(selected) >= max_frames:
            break
        if item in selected:
            continue
        if not selected:
            add(item)
            continue
        selected_times = sorted(float(chosen["time"]) for chosen in selected)
        nearest_gap = min(abs(float(item["time"]) - time) for time in selected_times)
        if nearest_gap >= 0.35:
            add(item)

    while len(selected) < max_frames:
        selected_times = sorted(float(item["time"]) for item in selected)
        best_item = None
        best_gap = -1.0
        for item in candidates:
            if item in selected:
                continue
            if not selected_times:
                best_item = item
                best_gap = float("inf")
                break
            nearest_gap = min(abs(float(item["time"]) - time) for time in selected_times)
            if nearest_gap > best_gap:
                best_gap = nearest_gap
                best_item = item
        if best_item is None:
            break
        add(best_item)

    return sorted(selected, key=lambda item: item["time"])


def _probe_for_event(event: dict[str, Any], index: int, duration: float, question: str) -> dict[str, Any]:
    start = _clamp_time(_safe_float(event.get("time"), 0.0), duration)
    raw_end = event.get("end_time")
    end = _clamp_time(_safe_float(raw_end, min(duration, start + 2.0)), duration) if raw_end is not None else start
    if end < start:
        start, end = end, start

    label = str(event.get("label") or "audio event")
    evidence = str(event.get("evidence") or "")
    expected_visuals = [str(item) for item in (event.get("expected_visuals") or []) if str(item).strip()]
    visual_question = str(event.get("visual_question") or "").strip()
    if not visual_question:
        expected = ", ".join(expected_visuals[:3]) or "visible evidence that would support this audio cue"
        visual_question = f"Does the frame show {expected}?"

    return {
        "type": "audio_event",
        "event_index": index,
        "label": label,
        "window_start": start,
        "window_end": end,
        "question": visual_question,
        "expected_visuals": expected_visuals,
        "audio_evidence": evidence,
        "user_question": question,
    }


def _probe_reason(probe: dict[str, Any], checkpoint: str) -> str:
    expected = ", ".join(probe.get("expected_visuals") or []) or "concrete visible evidence"
    return f"{checkpoint}: {probe['question']} Expected: {expected}"


def plan_keyframes(
    *,
    duration: float,
    audio_world_model: dict[str, Any] | None,
    question: str,
    existing_plan: list[dict[str, Any]] | None = None,
    refinement_windows: list[dict[str, Any]] | None = None,
    frame_candidates: list[dict[str, Any]] | None = None,
    max_frames: int = 48,
    refinement_samples_per_window: int = 3,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    world_model = audio_world_model or {}
    timeline = world_model.get("timeline") or []

    for index, event in enumerate(timeline):
        probe = _probe_for_event(event, index, duration, question)
        start = float(probe["window_start"])
        end = float(probe["window_end"])
        group = f"audio_event_{index}"
        has_span = end > start
        midpoint = (start + end) / 2 if has_span else start

        _add_candidate(
            candidates,
            start,
            _probe_reason(probe, "audio-guided start check"),
            "audio",
            82,
            group=group,
            probe=probe,
        )
        if has_span:
            _add_candidate(
                candidates,
                midpoint,
                _probe_reason(probe, "audio-guided midpoint check"),
                "audio",
                90,
                group=group,
                probe=probe,
            )
            _add_candidate(
                candidates,
                end,
                _probe_reason(probe, "audio-guided proof checkpoint"),
                "audio",
                94,
                group=group,
                probe=probe,
            )
        for offset in (-1.0, 1.0):
            _add_candidate(
                candidates,
                start + offset,
                _probe_reason(probe, "audio local context check"),
                "audio_context",
                45,
                group=group,
                probe=probe,
            )

    if duration > 0:
        if timeline:
            _add_candidate(
                candidates,
                0.0,
                f"safety cover only if audio probes leave budget: {question[:80]}",
                "global",
                25,
                group="global_safety",
            )
            _add_candidate(
                candidates,
                duration,
                f"safety cover only if audio probes leave budget: {question[:80]}",
                "global",
                25,
                group="global_safety",
            )
        else:
            for time in _binary_cover_times(duration, depth=3):
                _add_candidate(candidates, time, f"binary global cover for question: {question[:80]}", "global", 60)

    for candidate in frame_candidates or []:
        event_index = candidate.get("event_index")
        group = f"audio_event_{event_index}" if event_index is not None else "visual_candidate"
        score = max(0.0, min(1.0, _safe_float(candidate.get("score"), 0.5)))
        _add_candidate(
            candidates,
            _safe_float(candidate.get("time"), 0.0),
            str(candidate.get("reason") or "enhanced visual candidate"),
            str(candidate.get("source") or "visual_candidate"),
            70 + int(score * 25),
            group=group,
            probe=candidate.get("probe"),
        )

    for item in existing_plan or []:
        _add_candidate(
            candidates,
            float(item["time"]),
            str(item.get("reason", "existing plan")),
            str(item.get("source", "existing")),
            int(item.get("priority", 50)),
            group=str(item.get("group")) if item.get("group") else None,
            probe=item.get("probe"),
        )

    for index, window in enumerate(refinement_windows or []):
        start = float(window.get("start", 0.0))
        end = float(window.get("end", start + 2.0))
        midpoint = (start + end) / 2
        probe = {
            "type": "prediction_refinement",
            "window_start": _clamp_time(start, duration),
            "window_end": _clamp_time(end, duration),
            "question": str(window.get("reason") or "Verify the uncertain or conflicting prediction."),
            "expected_visuals": [str(item) for item in window.get("expected_evidence", [])],
            "audio_evidence": str(window.get("hypothesis", "")),
            "user_question": question,
        }
        group = f"refinement_{index}"
        if refinement_samples_per_window >= 3:
            _add_candidate(
                candidates,
                start,
                "targeted refinement start: verify missing or conflicting prediction evidence",
                "refinement",
                108,
                group=group,
                probe=probe,
            )
        elif refinement_samples_per_window >= 2:
            quarter = start + ((end - start) * 0.25)
            _add_candidate(
                candidates,
                quarter,
                "targeted refinement lower binary probe: verify missing or conflicting prediction evidence",
                "refinement",
                112,
                group=group,
                probe=probe,
            )
        _add_candidate(
            candidates,
            midpoint,
            "targeted refinement binary midpoint: verify missing or conflicting prediction evidence",
            "refinement",
            120,
            group=group,
            probe=probe,
        )
        if refinement_samples_per_window >= 3:
            _add_candidate(
                candidates,
                end,
                "targeted refinement end: verify missing or conflicting prediction evidence",
                "refinement",
                108,
                group=group,
                probe=probe,
            )
        elif refinement_samples_per_window >= 2:
            upper = start + ((end - start) * 0.75)
            _add_candidate(
                candidates,
                upper,
                "targeted refinement upper binary probe: verify missing or conflicting prediction evidence",
                "refinement",
                112,
                group=group,
                probe=probe,
            )

    planned = _dedupe_candidates(candidates, duration)
    budgeted = _choose_budgeted_frames(planned, max_frames)
    for item in budgeted:
        item.pop("priority", None)
        if item.get("group") is None:
            item.pop("group", None)
        if item.get("probe") is None:
            item.pop("probe", None)
    return budgeted
