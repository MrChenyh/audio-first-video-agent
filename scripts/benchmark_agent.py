from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.ai import AIClient
from app.config import load_settings
from app.storage import JobStore
from app.video import VideoProcessor
from app.workflow import VideoUnderstandingWorkflow


DEFAULT_QUESTION = (
    "\u8fd9\u4e2a\u89c6\u9891\u4e3b\u8981\u53d1\u751f\u4e86\u4ec0\u4e48\uff1f"
    "\u8bf7\u6309\u65f6\u95f4\u7ebf\u603b\u7ed3\uff0c"
    "\u5e76\u7ed9\u51fa\u5173\u952e\u97f3\u9891\u548c\u89c6\u89c9\u8bc1\u636e\u3002"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark legacy vs enhanced audio-first video agent strategies.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--strategies", default="legacy,enhanced")
    parser.add_argument("--output", type=Path, default=Path("data") / "benchmarks" / "latest.json")
    args = parser.parse_args()

    settings = load_settings()
    results = []
    for strategy in [item.strip() for item in args.strategies.split(",") if item.strip()]:
        run_settings = replace(settings, keyframe_strategy=strategy)
        run_result = run_once(run_settings, args.video, args.question, strategy)
        results.append(run_result)

    report = {"video": str(args.video), "question": args.question, "runs": results, "comparison": compare(results)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def run_once(settings, video_path: Path, question: str, strategy: str) -> dict[str, Any]:
    data_dir = settings.data_dir / "benchmarks" / strategy / uuid4().hex
    store = JobStore(data_dir)
    store.init()
    job_id = f"bench_{strategy}_{uuid4().hex[:8]}"
    upload_path = store.upload_dir(job_id) / "source.mp4"
    shutil.copyfile(video_path, upload_path)
    store.create_job(job_id, question, upload_path)

    workflow = VideoUnderstandingWorkflow(settings, store, VideoProcessor(settings), AIClient(settings))
    initial_state = {
        "job_id": job_id,
        "question": question,
        "video_path": str(upload_path),
        "refinement_rounds": 0,
        "refinement_windows": [],
        "should_refine": False,
    }
    started = time.perf_counter()
    final_state = workflow.run(initial_state)
    elapsed = time.perf_counter() - started
    result = final_state.get("result") or {}
    frames = result.get("frames") or final_state.get("extracted_frames", [])
    observations = final_state.get("frame_observations", [])
    candidates = final_state.get("frame_candidates", [])
    checks = final_state.get("prediction_checks", [])
    answer = result.get("answer") or final_state.get("answer", {})

    score = score_result(result=result, final_state=final_state)
    return {
        "strategy": strategy,
        "job_id": job_id,
        "elapsed_seconds": round(elapsed, 2),
        "candidate_count": len(candidates),
        "planned_frame_count": len(final_state.get("keyframe_plan", [])),
        "extracted_frame_count": len(frames),
        "observed_frame_count": len(observations),
        "vision_request_count": int(final_state.get("vision_request_count", 0)),
        "refinement_rounds": final_state.get("refinement_rounds", 0),
        "frame_times": [round(float(frame.get("time", 0.0)), 2) for frame in frames],
        "prediction_checks": [
            {
                "status": check.get("status"),
                "conflict_score": check.get("conflict_score"),
                "window_start": check.get("window_start"),
                "window_end": check.get("window_end"),
            }
            for check in checks
        ],
        "quality_score": score,
        "answer_summary": answer.get("summary", ""),
        "answer_direct": answer.get("direct_answer", ""),
    }


def score_result(*, result: dict[str, Any], final_state: dict[str, Any]) -> dict[str, Any]:
    frames = result.get("frames") or final_state.get("extracted_frames", [])
    observations = final_state.get("frame_observations", [])
    checks = final_state.get("prediction_checks", [])
    answer = result.get("answer") or final_state.get("answer", {})
    text = json.dumps({"observations": observations, "checks": checks, "answer": answer}, ensure_ascii=False).lower()
    frame_times = [float(frame.get("time", 0.0)) for frame in frames]

    metrics = {
        "audio_transcript_present": bool(final_state.get("transcript_segments")),
        "wedding_visual_hit": any(5.6 <= time <= 8.2 for time in frame_times)
        and _contains_any(text, ["\u7ea2", "\u76d6\u5934", "\u5a5a", "\u592b\u59bb", "wedding", "veil", "couple"]),
        "identity_uncertainty_preserved": _contains_any(
            text,
            ["\u4e0d\u786e\u5b9a", "uncertain", "\u7f3a\u5c11", "\u4e0d\u8db3", "ambiguous"],
        )
        and _contains_any(text, ["\u72d0", "fox", "\u8eab\u4efd", "identity"]),
        "ending_uncertainty_or_conflict": _contains_any(text, ["\u5c71\u6797", "forest", "\u751f\u6d3b", "ending"])
        and _contains_any(
            text,
            ["conflict", "\u51b2\u7a81", "\u4e0d\u786e\u5b9a", "\u4e0d\u8db3", "uncertain"],
        ),
        "has_prediction_checks": bool(checks),
    }
    raw_score = sum(1 for value in metrics.values() if value)
    return {"score": raw_score, "max_score": len(metrics), "metrics": metrics}


def compare(results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(results) < 2:
        return {}
    base = results[0]
    comparison = {}
    for current in results[1:]:
        comparison[f"{base['strategy']}_vs_{current['strategy']}"] = {
            "elapsed_delta_seconds": round(current["elapsed_seconds"] - base["elapsed_seconds"], 2),
            "elapsed_delta_percent": percent_delta(current["elapsed_seconds"], base["elapsed_seconds"]),
            "observed_frame_delta": current["observed_frame_count"] - base["observed_frame_count"],
            "vision_request_delta": current.get("vision_request_count", 0) - base.get("vision_request_count", 0),
            "vision_request_delta_percent": percent_delta(
                current.get("vision_request_count", 0),
                base.get("vision_request_count", 0),
            ),
            "quality_score_delta": current["quality_score"]["score"] - base["quality_score"]["score"],
            "candidate_count_delta": current["candidate_count"] - base["candidate_count"],
        }
    return comparison


def percent_delta(current: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return round(((current - baseline) / baseline) * 100, 2)


def _contains_any(text: str, tokens: list[str]) -> bool:
    return any(token in text for token in tokens)


if __name__ == "__main__":
    main()
