from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import Settings


# Valid 1x1 PNG bytes. Mock mode writes this through the frame endpoint so the UI
# can render a real image even when FFmpeg is unavailable.
PLACEHOLDER_IMAGE = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/ax+2kAAAAAASUVORK5CYII="
)


class VideoProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _find_binary(self, configured: str | None, name: str) -> str | None:
        if configured:
            return configured
        found = shutil.which(name)
        if found:
            return found
        if name == "ffmpeg":
            try:
                import imageio_ffmpeg

                return imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                return None
        return None

    @property
    def ffmpeg(self) -> str | None:
        return self._find_binary(self.settings.ffmpeg_path, "ffmpeg")

    @property
    def ffprobe(self) -> str | None:
        return self._find_binary(self.settings.ffprobe_path, "ffprobe")

    def probe_video(self, video_path: Path) -> dict[str, Any]:
        if self.settings.use_mock_models and not self.settings.ffmpeg_path and not self.settings.ffprobe_path:
            return {
                "duration_seconds": 120.0,
                "fps": 30.0,
                "width": 1280,
                "height": 720,
                "has_audio": True,
                "mock_metadata": True,
            }
        ffprobe = self.ffprobe
        if not ffprobe and self.ffmpeg:
            return self._probe_with_ffmpeg(video_path)
        if not ffprobe:
            if self.settings.use_mock_models:
                return {
                    "duration_seconds": 120.0,
                    "fps": 30.0,
                    "width": 1280,
                    "height": 720,
                    "has_audio": True,
                    "mock_metadata": True,
                }
            raise RuntimeError("ffprobe was not found. Install FFmpeg or set FFPROBE_PATH.")

        command = [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(video_path),
        ]
        completed = self._run_media_command(command)
        if completed.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {completed.stderr.strip()}")

        payload = json.loads(completed.stdout or "{}")
        streams = payload.get("streams") or []
        video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
        if not video_stream:
            raise RuntimeError("No video stream was found in the uploaded file.")

        duration = float((payload.get("format") or {}).get("duration") or video_stream.get("duration") or 0)
        fps_text = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/1"
        fps = self._parse_rate(fps_text)
        has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
        return {
            "duration_seconds": duration,
            "fps": fps,
            "width": int(video_stream.get("width") or 0),
            "height": int(video_stream.get("height") or 0),
            "has_audio": has_audio,
        }

    def _probe_with_ffmpeg(self, video_path: Path) -> dict[str, Any]:
        command = [self.ffmpeg or "ffmpeg", "-hide_banner", "-i", str(video_path)]
        completed = self._run_media_command(command)
        text = f"{completed.stdout}\n{completed.stderr}"
        if "No such file" in text or "Invalid data" in text:
            raise RuntimeError(f"ffmpeg probe failed: {text.strip()}")

        import re

        duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
        duration = 0.0
        if duration_match:
            hours, minutes, seconds = duration_match.groups()
            duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)

        video_match = re.search(r"Video:.*?,\s*(\d+)x(\d+)[,\s]", text)
        width = int(video_match.group(1)) if video_match else 0
        height = int(video_match.group(2)) if video_match else 0

        fps_match = re.search(r"(\d+(?:\.\d+)?)\s*fps", text)
        fps = float(fps_match.group(1)) if fps_match else 0.0
        has_audio = "Audio:" in text

        if duration <= 0 or width <= 0 or height <= 0:
            raise RuntimeError("Could not read video metadata from ffmpeg output.")
        return {
            "duration_seconds": duration,
            "fps": fps,
            "width": width,
            "height": height,
            "has_audio": has_audio,
        }

    @staticmethod
    def _parse_rate(value: str) -> float:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            denominator_float = float(denominator or 1)
            return float(numerator or 0) / denominator_float if denominator_float else 0.0
        return float(value or 0)

    def validate_duration(self, duration: float) -> None:
        if duration <= 0:
            raise ValueError("Video duration could not be read.")

    def extract_audio(self, video_path: Path, output_path: Path, has_audio: bool) -> Path | None:
        if not has_audio:
            return None
        if self.settings.use_mock_models and not self.settings.ffmpeg_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"mock audio")
            return output_path
        ffmpeg = self.ffmpeg
        if not ffmpeg:
            if self.settings.use_mock_models:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"mock audio")
                return output_path
            raise RuntimeError("ffmpeg was not found. Install FFmpeg or set FFMPEG_PATH.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_path),
        ]
        completed = self._run_media_command(command)
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            if "audio" in stderr.lower() and ("stream" in stderr.lower() or "matches no streams" in stderr.lower()):
                return None
            raise RuntimeError(f"ffmpeg audio extraction failed: {stderr}")
        return output_path

    def extract_frame(self, video_path: Path, time_seconds: float, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.settings.use_mock_models and not self.settings.ffmpeg_path:
            output_path.write_bytes(PLACEHOLDER_IMAGE)
            return
        ffmpeg = self.ffmpeg
        if not ffmpeg:
            if self.settings.use_mock_models:
                output_path.write_bytes(PLACEHOLDER_IMAGE)
                return
            raise RuntimeError("ffmpeg was not found. Install FFmpeg or set FFMPEG_PATH.")

        command = [
            ffmpeg,
            "-y",
            "-ss",
            f"{time_seconds:.2f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ]
        completed = self._run_media_command(command)
        if completed.returncode != 0:
            fallback_path = output_path.with_suffix(".png")
            fallback_command = [
                ffmpeg,
                "-y",
                "-ss",
                f"{max(0.0, time_seconds - 0.05):.2f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                str(fallback_path),
            ]
            fallback = self._run_media_command(fallback_command)
            if fallback.returncode == 0 and fallback_path.exists() and fallback_path.stat().st_size > 0:
                os.replace(fallback_path, output_path)
                return
            if fallback_path.exists():
                fallback_path.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg frame extraction failed: {completed.stderr.strip()}")

    def extract_clip(self, video_path: Path, start_seconds: float, duration_seconds: float, output_path: Path, width: int = 640) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = self.ffmpeg
        if not ffmpeg:
            raise RuntimeError("ffmpeg was not found. Install FFmpeg or set FFMPEG_PATH.")
        command = [
            ffmpeg,
            "-y",
            "-ss",
            f"{max(0.0, start_seconds):.2f}",
            "-i",
            str(video_path),
            "-t",
            f"{max(0.1, duration_seconds):.2f}",
            "-an",
            "-vf",
            f"scale={width}:-2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        completed = self._run_media_command(command)
        if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg clip extraction failed: {completed.stderr.strip()}")

    def extract_analysis_frame(self, video_path: Path, time_seconds: float, output_path: Path, size: int = 96) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.settings.use_mock_models and not self.settings.ffmpeg_path:
            output_path.write_bytes(_mock_ppm(size, size))
            return
        ffmpeg = self.ffmpeg
        if not ffmpeg:
            if self.settings.use_mock_models:
                output_path.write_bytes(_mock_ppm(size, size))
                return
            raise RuntimeError("ffmpeg was not found. Install FFmpeg or set FFMPEG_PATH.")

        command = [
            ffmpeg,
            "-y",
            "-ss",
            f"{max(0.0, time_seconds):.2f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            f"scale={size}:{size}:force_original_aspect_ratio=decrease,pad={size}:{size}:(ow-iw)/2:(oh-ih)/2,format=rgb24",
            "-f",
            "image2",
            "-vcodec",
            "ppm",
            str(output_path),
        ]
        completed = self._run_media_command(command)
        if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
            retry_command = command.copy()
            retry_command[retry_command.index(f"{max(0.0, time_seconds):.2f}")] = f"{max(0.0, time_seconds - 0.05):.2f}"
            retry = self._run_media_command(retry_command)
            if retry.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError(f"ffmpeg analysis frame extraction failed: {completed.stderr.strip()}")

    @staticmethod
    def _run_media_command(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )


def _mock_ppm(width: int, height: int) -> bytes:
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            pixels.extend((x % 256, y % 256, (x + y) % 256))
    return header + bytes(pixels)
