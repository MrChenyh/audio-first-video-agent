from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .config import Settings
from .video import VideoProcessor


def generate_frame_candidates(
    *,
    video_path: Path,
    duration: float,
    audio_world_model: dict[str, Any],
    question: str,
    processor: VideoProcessor,
    settings: Settings,
) -> list[dict[str, Any]]:
    """Generate cheap local candidate frames before spending VLM calls.

    The candidate pass samples only short windows implied by the audio world
    model, extracts tiny PPM frames, and scores them with local image features.
    This keeps the expensive multimodal calls focused on likely evidence.
    """
    if settings.keyframe_strategy != "enhanced":
        return []
    if settings.use_mock_models:
        return _mock_candidates(duration, audio_world_model, question)

    timeline = audio_world_model.get("timeline") or []
    if not timeline:
        return []

    candidates: list[dict[str, Any]] = []
    with TemporaryDirectory(prefix="audio_first_candidates_") as tmp:
        tmp_dir = Path(tmp)
        for event_index, event in enumerate(timeline):
            candidates.extend(
                _event_candidates(
                    video_path=video_path,
                    duration=duration,
                    event=event,
                    event_index=event_index,
                    question=question,
                    processor=processor,
                    settings=settings,
                    tmp_dir=tmp_dir,
                )
            )
    return sorted(candidates, key=lambda item: (-float(item.get("score", 0.0)), float(item["time"])))


def _event_candidates(
    *,
    video_path: Path,
    duration: float,
    event: dict[str, Any],
    event_index: int,
    question: str,
    processor: VideoProcessor,
    settings: Settings,
    tmp_dir: Path,
) -> list[dict[str, Any]]:
    start = _clamp(float(event.get("time") or 0.0) - 0.35, duration)
    end = _clamp(float(event.get("end_time") or start + 1.5) + 0.35, duration)
    if end <= start:
        end = min(duration, start + 1.0)

    times = _sample_times(start, end, settings.candidate_sample_fps, duration)
    expected_text = " ".join(
        [
            str(event.get("label", "")),
            str(event.get("evidence", "")),
            str(event.get("visual_question", "")),
            " ".join(str(item) for item in event.get("expected_visuals", [])),
            question,
        ]
    )
    event_type = classify_audio_event(event)
    scored = []
    prev_hash: int | None = None
    prev_brightness: float | None = None

    for index, time in enumerate(times):
        image_path = tmp_dir / f"event_{event_index}_{index}.ppm"
        try:
            processor.extract_analysis_frame(video_path, time, image_path, size=96)
            features = _image_features(image_path)
        except Exception:
            continue

        hash_distance = _hamming_distance(prev_hash, features["hash"]) if prev_hash is not None else 0
        brightness_delta = abs(features["brightness"] - prev_brightness) if prev_brightness is not None else 0.0
        prev_hash = features["hash"]
        prev_brightness = features["brightness"]

        score = _score_candidate(
            time=time,
            start=start,
            end=end,
            event_type=event_type,
            expected_text=expected_text,
            hash_distance=hash_distance,
            brightness_delta=brightness_delta,
            edge_density=features["edge_density"],
        )
        scored.append(
            {
                "time": time,
                "score": score,
                "source": "visual_candidate",
                "reason": (
                    f"enhanced candidate for {event.get('label', 'audio event')} "
                    f"type={event_type} hash_delta={hash_distance} brightness_delta={brightness_delta:.3f}"
                ),
                "event_index": event_index,
                "event_type": event_type,
                "features": {
                    "hash_distance": hash_distance,
                    "brightness_delta": round(brightness_delta, 4),
                    "edge_density": round(features["edge_density"], 4),
                    "brightness": round(features["brightness"], 4),
                },
                "probe": _probe_for_candidate(event, event_index, question),
            }
        )

    return _select_diverse(scored, settings.candidate_max_per_event, settings.candidate_hash_min_distance)


def classify_audio_event(event: dict[str, Any]) -> str:
    text = " ".join(
        [
            str(event.get("label", "")),
            str(event.get("evidence", "")),
            str(event.get("visual_question", "")),
            " ".join(str(item) for item in event.get("expected_visuals", [])),
        ]
    ).lower()

    # Ordering matters: story-memory cues often mention foxes, so classify them
    # before generic identity/reveal cues.
    if _contains_any(text, ["\u6551", "\u96ea\u5c71", "\u56de\u5fc6", "rescue", "snow"]):
        return "memory"
    if _contains_any(text, ["\u5c71\u6797", "\u751f\u6d3b", "\u7ed3\u5c40", "happy", "ending", "forest"]):
        return "ending"
    if _contains_any(
        text,
        ["\u5a5a", "\u592b\u59bb", "\u592b\u5987", "\u7ea2\u76d6\u5934", "wedding", "marriage", "couple"],
    ):
        return "relationship"
    if _contains_any(
        text,
        ["\u72d0\u72f8", "\u767d\u72d0", "\u8eab\u4efd", "\u5c31\u662f", "\u786e\u8ba4", "fox", "identity"],
    ):
        return "identity"
    if _contains_any(text, ["?", "\uff1f", "\u96be\u9053", "\u662f\u4e0d\u662f", "question"]):
        return "dialogue"
    return "generic"


def _probe_for_candidate(event: dict[str, Any], event_index: int, question: str) -> dict[str, Any]:
    return {
        "type": "visual_candidate",
        "event_index": event_index,
        "label": str(event.get("label") or "audio event"),
        "window_start": float(event.get("time") or 0.0),
        "window_end": float(event.get("end_time") or event.get("time") or 0.0),
        "question": str(event.get("visual_question") or "Does this frame provide strong visual evidence for the audio cue?"),
        "expected_visuals": [str(item) for item in event.get("expected_visuals", [])],
        "audio_evidence": str(event.get("evidence", "")),
        "user_question": question,
    }


def _score_candidate(
    *,
    time: float,
    start: float,
    end: float,
    event_type: str,
    expected_text: str,
    hash_distance: int,
    brightness_delta: float,
    edge_density: float,
) -> float:
    target_ratio = {
        "relationship": 0.72,
        "identity": 0.58,
        "ending": 0.72,
        "memory": 0.50,
        "dialogue": 0.50,
        "generic": 0.50,
    }.get(event_type, 0.50)
    center = start + ((end - start) * target_ratio)
    span = max(0.25, end - start)
    center_score = 1.0 - min(1.0, abs(time - center) / span)
    motion_score = min(1.0, (hash_distance / 24.0) + brightness_delta)
    detail_score = min(1.0, edge_density * 5.0)
    keyword_boost = _keyword_boost(expected_text, event_type)

    weights = {
        "relationship": (0.45, 0.25, 0.20),
        "identity": (0.35, 0.30, 0.25),
        "ending": (0.30, 0.35, 0.20),
        "memory": (0.35, 0.35, 0.15),
        "dialogue": (0.55, 0.20, 0.15),
        "generic": (0.40, 0.30, 0.20),
    }.get(event_type, (0.40, 0.30, 0.20))
    return round((weights[0] * center_score) + (weights[1] * motion_score) + (weights[2] * detail_score) + keyword_boost, 4)


def _keyword_boost(text: str, event_type: str) -> float:
    normalized = text.lower()
    keywords = {
        "relationship": ["\u7ea2", "\u5a5a", "\u592b\u59bb", "couple", "wedding"],
        "identity": ["\u72d0", "\u8eab\u4efd", "\u786e\u8ba4", "fox"],
        "ending": ["\u5c71", "\u6797", "\u5e78\u798f", "\u751f\u6d3b", "forest"],
        "memory": ["\u96ea", "\u6551", "\u56de\u5fc6", "snow", "rescue"],
        "dialogue": ["?", "\uff1f", "\u95ee"],
    }.get(event_type, [])
    hits = sum(1 for keyword in keywords if keyword in normalized)
    return min(0.18, hits * 0.04)


def _select_diverse(items: list[dict[str, Any]], limit: int, min_hash_distance: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    selected: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda candidate: (-float(candidate["score"]), float(candidate["time"]))):
        if len(selected) >= limit:
            break
        if all(abs(float(item["time"]) - float(existing["time"])) >= 0.25 for existing in selected):
            if int(item.get("features", {}).get("hash_distance", min_hash_distance)) >= min_hash_distance or not selected:
                selected.append(item)
    return sorted(selected, key=lambda candidate: candidate["time"])


def _mock_candidates(duration: float, audio_world_model: dict[str, Any], question: str) -> list[dict[str, Any]]:
    candidates = []
    for index, event in enumerate(audio_world_model.get("timeline") or []):
        start = float(event.get("time") or 0.0)
        end = float(event.get("end_time") or start + 1.0)
        time = _clamp((start + end) / 2, duration)
        candidates.append(
            {
                "time": time,
                "score": 0.75,
                "source": "visual_candidate",
                "reason": f"mock enhanced candidate for {event.get('label', 'audio event')}",
                "event_index": index,
                "event_type": classify_audio_event(event),
                "features": {},
                "probe": _probe_for_candidate(event, index, question),
            }
        )
    return candidates


def _sample_times(start: float, end: float, fps: float, duration: float) -> list[float]:
    fps = max(0.5, fps)
    step = 1.0 / fps
    times = []
    current = start
    while current <= end + 0.001:
        times.append(_clamp(current, duration))
        current += step
    midpoint = _clamp((start + end) / 2, duration)
    times.append(midpoint)
    return sorted(set(times))


def _image_features(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    width, height, max_value, pixels = _parse_ppm(data)
    gray = []
    for offset in range(0, len(pixels), 3):
        r, g, b = pixels[offset], pixels[offset + 1], pixels[offset + 2]
        gray.append((0.299 * r + 0.587 * g + 0.114 * b) / max_value)

    brightness = sum(gray) / max(1, len(gray))
    hash_value = _average_hash(gray, width, height)
    edge_density = _edge_density(gray, width, height)
    return {"hash": hash_value, "brightness": brightness, "edge_density": edge_density}


def _parse_ppm(data: bytes) -> tuple[int, int, int, bytes]:
    tokens: list[bytes] = []
    index = 0
    while len(tokens) < 4:
        while index < len(data) and data[index] in b" \t\r\n":
            index += 1
        if index < len(data) and data[index] == ord("#"):
            while index < len(data) and data[index] not in b"\r\n":
                index += 1
            continue
        start = index
        while index < len(data) and data[index] not in b" \t\r\n":
            index += 1
        tokens.append(data[start:index])
    if tokens[0] != b"P6":
        raise ValueError("Expected binary PPM image.")
    while index < len(data) and data[index] in b" \t\r\n":
        index += 1
    return int(tokens[1]), int(tokens[2]), int(tokens[3]), data[index:]


def _average_hash(gray: list[float], width: int, height: int) -> int:
    cells = []
    for y in range(8):
        for x in range(8):
            values = []
            x0, x1 = int(x * width / 8), int((x + 1) * width / 8)
            y0, y1 = int(y * height / 8), int((y + 1) * height / 8)
            for yy in range(y0, max(y0 + 1, y1)):
                row = yy * width
                for xx in range(x0, max(x0 + 1, x1)):
                    values.append(gray[row + xx])
            cells.append(sum(values) / max(1, len(values)))
    avg = sum(cells) / len(cells)
    value = 0
    for cell in cells:
        value = (value << 1) | int(cell >= avg)
    return value


def _edge_density(gray: list[float], width: int, height: int) -> float:
    if width < 2 or height < 2:
        return 0.0
    total = 0.0
    count = 0
    for y in range(1, height):
        row = y * width
        prev_row = (y - 1) * width
        for x in range(1, width):
            total += abs(gray[row + x] - gray[row + x - 1])
            total += abs(gray[row + x] - gray[prev_row + x])
            count += 2
    return total / max(1, count)


def _hamming_distance(left: int | None, right: int | None) -> int:
    if left is None or right is None:
        return 64
    return int((left ^ right).bit_count())


def _contains_any(text: str, tokens: list[str]) -> bool:
    return any(token in text for token in tokens)


def _clamp(value: float, duration: float) -> float:
    if duration <= 0:
        return 0.0
    return round(max(0.0, min(max(0.0, duration - 0.05), value)), 2)
