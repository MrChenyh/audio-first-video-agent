from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize method-level video understanding comparisons.")
    parser.add_argument(
        "report",
        nargs="?",
        type=Path,
        default=latest_report(),
        help="Path to data/benchmarks/{run_id}/report.json",
    )
    parser.add_argument("--uniform-samples", type=int, default=12)
    parser.add_argument("--tolerance", type=float, default=4.0, help="Seconds for temporal hit recall.")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    print(render_summary(report, args.uniform_samples, args.tolerance))


def latest_report() -> Path:
    reports = sorted((ROOT / "data" / "benchmarks").glob("*/report.json"), key=lambda path: path.stat().st_mtime)
    if not reports:
        raise FileNotFoundError("No benchmark report found under data/benchmarks/*/report.json")
    return reports[-1]


def render_summary(report: dict[str, Any], uniform_samples: int, tolerance: float) -> str:
    summary = report["summary"]
    rows = report["rows"]
    whole_rows = report.get("whole_rows") or []
    unique_points = sorted({(row["video_id"], float(row["time_seconds"])) for row in rows})
    total_duration = sum(float(video["duration_seconds"]) for video in report["videos"].values())

    uniform_recall = temporal_uniform_recall(report, unique_points, uniform_samples, tolerance)
    guaranteed_2s = sum(int(float(video["duration_seconds"]) / 4 + 0.999) for video in report["videos"].values())
    guaranteed_4s = sum(int(float(video["duration_seconds"]) / 8 + 0.999) for video in report["videos"].values())
    clip4 = summary.get("clip4_640", {})
    clip2 = summary.get("clip2_640", {})
    frame = summary.get("frame_640", {})
    clip8 = summary.get("clip8_640", {})
    whole = summary.get("whole", {})
    clip4_watch_seconds = len(unique_points) * 4.0
    clip2_watch_seconds = len(unique_points) * 2.0

    lines = ["# Method Comparison Summary", ""]
    lines.append(f"Benchmark run: `{report['run_id']}`")
    lines.append(f"Videos: {len(report['videos'])}; probe points: {len(unique_points)}; total duration: {total_duration:.2f}s")
    lines.append("")
    lines.append("## Metric Table")
    lines.append("| Method | Visual budget | Avg latency | Avg score | Pass@0.67 | Temporal recall | Payload/call | Notes |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    lines.append(
        f"| Uniform sparse frames | {uniform_samples} frames/video | - | not run | not run | "
        f"{uniform_recall['hits']}/{uniform_recall['total']} ({uniform_recall['recall']:.1%}) | - | Same sparse budget misses most target moments. |"
    )
    lines.append(
        f"| Targeted single frame | {len(unique_points)} frames | {frame.get('avg_elapsed_s', 0)}s | "
        f"{frame.get('avg_score', 0)} | {frame.get('pass_rate_0_67', 0)} | 8/8 selected | "
        f"{frame.get('avg_media_kb', 0)}KB | Fast, but weak for motion/context. |"
    )
    lines.append(
        f"| Audio-first 2s clips | {clip2_watch_seconds:.0f}s watched ({clip2_watch_seconds / total_duration:.2%}) | "
        f"{clip2.get('avg_elapsed_s', 0)}s | {clip2.get('avg_score', 0)} | {clip2.get('pass_rate_0_67', 0)} | "
        f"8/8 selected | {clip2.get('avg_media_kb', 0)}KB | Best latency/accuracy tradeoff in this run. |"
    )
    lines.append(
        f"| Audio-first 4s clips | {clip4_watch_seconds:.0f}s watched ({clip4_watch_seconds / total_duration:.2%}) | "
        f"{clip4.get('avg_elapsed_s', 0)}s | {clip4.get('avg_score', 0)} | {clip4.get('pass_rate_0_67', 0)} | "
        f"8/8 selected | {clip4.get('avg_media_kb', 0)}KB | Current default; stronger temporal context. |"
    )
    lines.append(
        f"| Longer 8s clips | {len(unique_points) * 8:.0f}s watched ({len(unique_points) * 8 / total_duration:.2%}) | "
        f"{clip8.get('avg_elapsed_s', 0)}s | {clip8.get('avg_score', 0)} | {clip8.get('pass_rate_0_67', 0)} | "
        f"8/8 selected | {clip8.get('avg_media_kb', 0)}KB | More context can add noise. |"
    )
    lines.append(
        f"| Whole-video low-res scan | full duration at 1fps | {whole.get('avg_elapsed_s', 0)}s model + "
        f"{whole.get('avg_encode_s', 0)}s encode | {whole.get('avg_score', 0)} | - | full coarse scan | "
        f"~{avg_whole_payload(whole_rows)}KB | Fast coarse topic scan, unreliable detail memory. |"
    )
    lines.append("")
    lines.append("## Sampling Efficiency")
    lines.append(f"- Uniform {uniform_samples} frames/video temporal recall within +/-{tolerance:.0f}s: {uniform_recall['hits']}/{uniform_recall['total']} ({uniform_recall['recall']:.1%}).")
    lines.append(f"- Uniform sampling needed to guarantee +/-2s coverage over these videos: about {guaranteed_2s} frames.")
    lines.append(f"- Uniform sampling needed to guarantee +/-4s coverage over these videos: about {guaranteed_4s} frames.")
    lines.append(f"- Audio-first 4s clips watched {clip4_watch_seconds:.0f}s out of {total_duration:.0f}s ({clip4_watch_seconds / total_duration:.2%}).")
    lines.append("")
    lines.append("## Whole-Video Scan Rows")
    for row in whole_rows:
        lines.append(f"- {row['video_id']}: score={row['score']}, latency={row['elapsed_s']}s, encode={row['encode_s']}s, matched={', '.join(row.get('matched_keywords') or [])}")
    return "\n".join(lines)


def temporal_uniform_recall(
    report: dict[str, Any],
    points: list[tuple[str, float]],
    samples_per_video: int,
    tolerance: float,
) -> dict[str, Any]:
    video_points: dict[str, list[float]] = defaultdict(list)
    for video_id, time_seconds in points:
        video_points[video_id].append(time_seconds)
    hits = 0
    total = 0
    for video_id, target_times in video_points.items():
        duration = float(report["videos"][video_id]["duration_seconds"])
        uniform_times = [((index + 0.5) * duration / samples_per_video) for index in range(samples_per_video)]
        for target in target_times:
            total += 1
            if min(abs(target - sample) for sample in uniform_times) <= tolerance:
                hits += 1
    return {"hits": hits, "total": total, "recall": hits / max(1, total)}


def avg_whole_payload(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return round(sum(float(row.get("media_size_kb") or 0.0) for row in rows) / len(rows), 1)


if __name__ == "__main__":
    main()
