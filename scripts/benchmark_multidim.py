from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.config import load_settings  # noqa: E402
from app.video import VideoProcessor  # noqa: E402


@dataclass(frozen=True)
class ProbePoint:
    video_id: str
    time_seconds: float
    question: str
    audio_context: str
    expected_groups: list[list[str]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multidimensional JoyAI video understanding benchmarks.")
    parser.add_argument("--limit-points", type=int, default=0, help="Only run the first N probe points.")
    parser.add_argument("--skip-whole", action="store_true", help="Skip whole-video compressed scan tests.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "benchmarks")
    args = parser.parse_args()

    settings = load_settings()
    processor = VideoProcessor(settings)
    ffmpeg = processor.ffmpeg
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for benchmark clip extraction.")

    videos = resolve_videos(settings)
    points = build_probe_points()
    if args.limit_points:
        points = points[: args.limit_points]

    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    from openai import OpenAI

    client = OpenAI(
        api_key=settings.joyai_api_key,
        base_url=settings.joyai_api_base,
        timeout=max(60.0, settings.joyai_timeout_seconds),
        max_retries=0,
    )

    video_meta: dict[str, Any] = {}
    for video_id, path in videos.items():
        video_meta[video_id] = {"path": str(path), **processor.probe_video(path)}

    variants = [
        {"name": "frame_640", "kind": "frame", "width": 640},
        {"name": "clip2_640", "kind": "clip", "seconds": 2.0, "width": 640, "audio": False},
        {"name": "clip4_640", "kind": "clip", "seconds": 4.0, "width": 640, "audio": False},
        {"name": "clip8_640", "kind": "clip", "seconds": 8.0, "width": 640, "audio": False},
    ]
    subset_variants = [
        {"name": "clip4_320", "kind": "clip", "seconds": 4.0, "width": 320, "audio": False},
        {"name": "clip4_640_audio", "kind": "clip", "seconds": 4.0, "width": 640, "audio": True},
    ]

    rows: list[dict[str, Any]] = []
    for index, point in enumerate(points, start=1):
        video_path = videos[point.video_id]
        active_variants = variants + (subset_variants if index in {2, 4, 6, 8} else [])
        print(f"[{index}/{len(points)}] {point.video_id} @ {point.time_seconds:.2f}s")
        for variant in active_variants:
            row = run_variant(
                client=client,
                model=settings.joyai_model,
                ffmpeg=ffmpeg,
                output_dir=output_dir,
                video_path=video_path,
                point=point,
                variant=variant,
            )
            rows.append(row)
            print(
                f"  {variant['name']}: {row['elapsed_s']}s score={row['score']:.2f} "
                f"groups={row['hit_groups']}/{row['total_groups']} err={bool(row.get('error'))}"
            )

    whole_rows: list[dict[str, Any]] = []
    if not args.skip_whole:
        for video_id, video_path in videos.items():
            print(f"[whole] {video_id}")
            whole_rows.append(
                run_whole_scan(
                    client=client,
                    model=settings.joyai_model,
                    ffmpeg=ffmpeg,
                    output_dir=output_dir,
                    video_id=video_id,
                    video_path=video_path,
                    audio=True,
                )
            )

    report = {
        "run_id": run_id,
        "settings": {
            "joyai_api_base": settings.joyai_api_base,
            "joyai_model": settings.joyai_model,
            "joyai_input_mode": settings.joyai_input_mode,
            "joyai_clip_seconds": settings.joyai_clip_seconds,
            "joyai_clip_width": settings.joyai_clip_width,
            "vision_provider": settings.vision_provider,
        },
        "videos": video_meta,
        "rows": rows,
        "whole_rows": whole_rows,
        "summary": summarize(rows, whole_rows),
    }
    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"\nJSON: {json_path}")
    print(f"Markdown: {md_path}")


def resolve_videos(settings: Any) -> dict[str, Path]:
    candidates = {
        "3d_house": [
            Path.home() / "Desktop" / "240天我们用3D打印装修了一整套房子..._哔哩哔哩_bilibili.mp4",
            settings.data_dir / "uploads" / "a6be2a0bbc5544fd88e313d004895a95" / "source.mp4",
        ],
        "pocket4p": [
            settings.data_dir / "uploads" / "0f4a637b103b49fd9a342f2e4c8c9941" / "source.mp4",
            settings.data_dir / "uploads" / "c1d6c94597824e16a974bd1ba3a4fbd0" / "source.mp4",
        ],
    }
    resolved: dict[str, Path] = {}
    for video_id, paths in candidates.items():
        for path in paths:
            if path.exists() and path.stat().st_size > 1024 * 1024:
                resolved[video_id] = path
                break
        if video_id not in resolved:
            raise FileNotFoundError(f"Could not find benchmark video: {video_id}")
    return resolved


def build_probe_points() -> list[ProbePoint]:
    return [
        ProbePoint(
            "3d_house",
            264.59,
            "这个时间点是否展示3D打印装修施工现场，以及天花/墙面施工细节？",
            "音频提到除了瓷砖，墙面等装修部分和施工现场有关。",
            [["施工", "装修", "现场"], ["天花", "吊顶", "墙面", "灯"], ["3D", "打印", "结构"]],
        ),
        ProbePoint(
            "3d_house",
            603.49,
            "这个时间点是否展示3D打印家具、分层打印纹理，能否支持成本讨论？",
            "音频提到3D打印装修成本，画面应检查打印家具、层纹、材料形态。",
            [["3D", "打印"], ["家具", "柜", "柜体", "架"], ["分层", "层纹", "堆叠"], ["成本", "材料"]],
        ),
        ProbePoint(
            "3d_house",
            791.53,
            "这个时间点是否真的出现当地打印、打印设备或材料堆叠？",
            "音频提到当地打印；需要验证画面里是否有打印设备、打印材料或只是人物对话。",
            [["打印", "3D"], ["设备", "机器", "材料", "堆叠"], ["对话", "人物", "室内"]],
        ),
        ProbePoint(
            "3d_house",
            869.43,
            "这个时间点是否展示家具设计、模型、图纸或安装部件？",
            "音频提到涉及家具设计；需要看设计图、模型、部件或现场安装。",
            [["家具", "设计"], ["模型", "图纸", "建模", "部件"], ["安装", "手", "现场", "实物"]],
        ),
        ProbePoint(
            "pocket4p",
            164.59,
            "这个时间点是否在展示Pocket 4P稳定/防抖效果或运动样片？",
            "音频说稳定效果你看一下；需要检查是否有云台、户外运动、样片或防抖对比。",
            [["稳定", "防抖", "云台"], ["运动", "走动", "骑", "户外"], ["样片", "效果", "对比"]],
        ),
        ProbePoint(
            "pocket4p",
            189.79,
            "这个时间点是否讲到长焦镜头、三倍视角或60mm焦段？",
            "音频说重点可能在长焦；画面应检查镜头/焦段/样片切换。",
            [["长焦", "镜头"], ["三倍", "3倍", "60", "60mm"], ["视角", "切换", "焦段", "构图"]],
        ),
        ProbePoint(
            "pocket4p",
            477.48,
            "这个时间点是否展示ISO1600、动态范围、高光暗部或低光测试？",
            "音频提到ISO1600测试；需要检查动态范围图表、暗部灯光或样片。",
            [["ISO", "1600"], ["动态范围", "高光", "暗部", "低光"], ["测试", "图表", "样片"]],
        ),
        ProbePoint(
            "pocket4p",
            608.68,
            "这个时间点是否展示发热测试、温度或长时间使用体验？",
            "音频说发热也测试了；需要检查温度、机身、测试过程或体验总结。",
            [["发热", "温度", "热"], ["测试", "体验", "过程"], ["机身", "Pocket", "相机", "设备"]],
        ),
    ]


def run_variant(
    *,
    client: Any,
    model: str,
    ffmpeg: str,
    output_dir: Path,
    video_path: Path,
    point: ProbePoint,
    variant: dict[str, Any],
) -> dict[str, Any]:
    media_path = output_dir / safe_name(f"{point.video_id}_{point.time_seconds:.2f}_{variant['name']}")
    content_type = "image_url"
    payload_key = "image_url"
    prompt = build_prompt(point, variant)
    error = ""
    response_text = ""
    elapsed = 0.0
    media_size = 0
    try:
        if variant["kind"] == "frame":
            media_path = media_path.with_suffix(".jpg")
            extract_frame(ffmpeg, video_path, point.time_seconds, media_path, width=int(variant["width"]))
            data_url = image_data_url(media_path)
        else:
            media_path = media_path.with_suffix(".mp4")
            seconds = float(variant["seconds"])
            start = max(0.0, point.time_seconds - seconds / 2)
            extract_clip(
                ffmpeg,
                video_path,
                start,
                seconds,
                media_path,
                width=int(variant["width"]),
                audio=bool(variant.get("audio")),
            )
            data_url = video_data_url(media_path)
            content_type = "video_url"
            payload_key = "video_url"
        media_size = media_path.stat().st_size
        start_time = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": content_type, payload_key: {"url": data_url}},
                    ],
                }
            ],
            max_tokens=260,
            temperature=0,
            extra_headers={
                "x-streaming-session": safe_name(f"bench-{point.video_id}-{variant['name']}-{time.time()}"),
                "x-frame-time-range": f"{point.time_seconds:.2f}s",
            },
        )
        elapsed = time.perf_counter() - start_time
        response_text = response.choices[0].message.content or ""
    except Exception as exc:
        error = str(exc)
    hit_groups, total_groups, matched = score_text(response_text, point.expected_groups)
    return {
        "video_id": point.video_id,
        "time_seconds": point.time_seconds,
        "question": point.question,
        "variant": variant["name"],
        "kind": variant["kind"],
        "seconds": variant.get("seconds"),
        "width": variant.get("width"),
        "audio_in_clip": bool(variant.get("audio")),
        "elapsed_s": round(elapsed, 3),
        "media_size_kb": round(media_size / 1024, 1) if media_size else 0,
        "hit_groups": hit_groups,
        "total_groups": total_groups,
        "score": round(hit_groups / max(1, total_groups), 3),
        "matched_keywords": matched,
        "response_chars": len(response_text),
        "response": response_text.strip(),
        "error": error,
    }


def run_whole_scan(
    *,
    client: Any,
    model: str,
    ffmpeg: str,
    output_dir: Path,
    video_id: str,
    video_path: Path,
    audio: bool,
) -> dict[str, Any]:
    out = output_dir / f"whole_{video_id}_{'audio' if audio else 'silent'}_320p_1fps.mp4"
    start_encode = time.perf_counter()
    extract_whole_lowres(ffmpeg, video_path, out, audio=audio)
    encode_s = time.perf_counter() - start_encode
    prompt = (
        "请完整粗扫这条视频。它保留完整时长，但被压成320p、1fps。请用中文输出："
        "主题一句话；按顺序列出5-8个关键阶段；最后说明是否足以支撑高质量问答。不要泛泛宣传。"
    )
    elapsed = 0.0
    response_text = ""
    error = ""
    try:
        data_url = video_data_url(out)
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "video_url", "video_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=700,
            temperature=0,
            extra_headers={"x-streaming-session": safe_name(f"bench-whole-{video_id}-{time.time()}")},
        )
        elapsed = time.perf_counter() - start
        response_text = response.choices[0].message.content or ""
    except Exception as exc:
        error = str(exc)
    topic_groups = {
        "3d_house": [["3D", "打印"], ["房", "装修"], ["家具", "施工", "成本", "设计"]],
        "pocket4p": [["Pocket", "Pockets", "大疆"], ["画质", "长焦", "镜头"], ["动态范围", "稳定", "发热"]],
    }[video_id]
    hit_groups, total_groups, matched = score_text(response_text, topic_groups)
    return {
        "video_id": video_id,
        "variant": "whole_320p_1fps_audio" if audio else "whole_320p_1fps_silent",
        "encode_s": round(encode_s, 3),
        "elapsed_s": round(elapsed, 3),
        "media_size_kb": round(out.stat().st_size / 1024, 1),
        "hit_groups": hit_groups,
        "total_groups": total_groups,
        "score": round(hit_groups / max(1, total_groups), 3),
        "matched_keywords": matched,
        "response_chars": len(response_text),
        "response": response_text.strip(),
        "error": error,
    }


def build_prompt(point: ProbePoint, variant: dict[str, Any]) -> str:
    medium = "连续短视频片段" if variant["kind"] == "clip" else "单张关键帧"
    return (
        f"你是音频优先视频理解agent的视觉验证器。现在给你{medium}。"
        "不要做泛泛描述，要围绕用户问题和音频线索判断画面是否支持。"
        "请用中文回答，结构为：观察；是否支持；缺失或不确定点。"
        "最后必须写：结论：支持 / 结论：冲突 / 结论：不确定。\n"
        f"用户问题：{point.question}\n"
        f"时间点：{point.time_seconds:.2f}s\n"
        f"音频线索：{point.audio_context}\n"
        f"期望检查关键词组：{json.dumps(point.expected_groups, ensure_ascii=False)}"
    )


def extract_frame(ffmpeg: str, video: Path, time_seconds: float, out: Path, width: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-ss",
        f"{time_seconds:.2f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:-2",
        "-q:v",
        "2",
        str(out),
    ]
    run_ffmpeg(command)


def extract_clip(ffmpeg: str, video: Path, start: float, seconds: float, out: Path, width: int, audio: bool) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start:.2f}",
        "-i",
        str(video),
        "-t",
        f"{seconds:.2f}",
        "-vf",
        f"scale={width}:-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
    ]
    if audio:
        command += ["-c:a", "aac", "-ac", "1", "-ar", "16000", "-b:a", "32k"]
    else:
        command += ["-an"]
    command += ["-movflags", "+faststart", str(out)]
    run_ffmpeg(command)


def extract_whole_lowres(ffmpeg: str, video: Path, out: Path, audio: bool) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video),
        "-vf",
        "fps=1,scale=320:-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "32",
    ]
    if audio:
        command += ["-c:a", "aac", "-ac", "1", "-ar", "16000", "-b:a", "32k"]
    else:
        command += ["-an"]
    command += ["-movflags", "+faststart", str(out)]
    run_ffmpeg(command)


def run_ffmpeg(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True)
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(stderr[-2000:])


def image_data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def video_data_url(path: Path) -> str:
    return "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def score_text(text: str, groups: list[list[str]]) -> tuple[int, int, list[str]]:
    haystack = str(text or "").lower()
    hit = 0
    matched: list[str] = []
    for group in groups:
        group_hit = ""
        for keyword in group:
            if keyword.lower() in haystack:
                group_hit = keyword
                break
        if group_hit:
            hit += 1
            matched.append(group_hit)
    return hit, len(groups), matched


def summarize(rows: list[dict[str, Any]], whole_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    summary: dict[str, Any] = {}
    for variant, items in by_variant.items():
        ok_items = [item for item in items if not item.get("error")]
        summary[variant] = {
            "count": len(items),
            "errors": len(items) - len(ok_items),
            "avg_elapsed_s": round(sum(item["elapsed_s"] for item in ok_items) / max(1, len(ok_items)), 3),
            "avg_score": round(sum(item["score"] for item in ok_items) / max(1, len(ok_items)), 3),
            "pass_rate_0_67": round(sum(1 for item in ok_items if item["score"] >= 0.67) / max(1, len(ok_items)), 3),
            "avg_media_kb": round(sum(item["media_size_kb"] for item in ok_items) / max(1, len(ok_items)), 1),
        }
    if whole_rows:
        summary["whole"] = {
            "count": len(whole_rows),
            "avg_elapsed_s": round(sum(item["elapsed_s"] for item in whole_rows) / len(whole_rows), 3),
            "avg_encode_s": round(sum(item["encode_s"] for item in whole_rows) / len(whole_rows), 3),
            "avg_score": round(sum(item["score"] for item in whole_rows) / len(whole_rows), 3),
        }
    return summary


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# Multidimensional Video Benchmark {report['run_id']}", ""]
    lines.append("## Settings")
    for key, value in report["settings"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Variant Summary")
    lines.append("| Variant | Count | Errors | Avg Latency | Avg Score | Pass@0.67 | Avg Media |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for variant, item in report["summary"].items():
        if variant == "whole":
            continue
        lines.append(
            f"| {variant} | {item['count']} | {item['errors']} | {item['avg_elapsed_s']}s | "
            f"{item['avg_score']} | {item['pass_rate_0_67']} | {item['avg_media_kb']} KB |"
        )
    if "whole" in report["summary"]:
        item = report["summary"]["whole"]
        lines.extend(["", "## Whole-Video Scans"])
        lines.append(
            f"- count={item['count']}, avg encode={item['avg_encode_s']}s, "
            f"avg model latency={item['avg_elapsed_s']}s, avg score={item['avg_score']}"
        )
    lines.extend(["", "## Detailed Rows"])
    lines.append("| Video | Time | Variant | Latency | Score | Matched | Response |")
    lines.append("|---|---:|---|---:|---:|---|---|")
    for row in report["rows"]:
        response = " ".join(str(row.get("response") or row.get("error") or "").split())[:160]
        matched = ", ".join(row.get("matched_keywords") or [])
        lines.append(
            f"| {row['video_id']} | {row['time_seconds']} | {row['variant']} | {row['elapsed_s']}s | "
            f"{row['score']} | {matched} | {response} |"
        )
    if report.get("whole_rows"):
        lines.extend(["", "## Whole Responses"])
        for row in report["whole_rows"]:
            response = " ".join(str(row.get("response") or row.get("error") or "").split())[:500]
            lines.append(f"### {row['video_id']} {row['variant']}")
            lines.append(f"- encode: {row['encode_s']}s; latency: {row['elapsed_s']}s; score: {row['score']}")
            lines.append(f"- response: {response}")
    lines.append("")
    return "\n".join(lines)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)[:120]


if __name__ == "__main__":
    main()
