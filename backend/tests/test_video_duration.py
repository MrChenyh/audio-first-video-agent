from pathlib import Path

import pytest

from app.video import VideoProcessor
from test_workflow_mock import make_settings


def test_validate_duration_allows_short_and_long_static_videos(tmp_path: Path):
    processor = VideoProcessor(make_settings(tmp_path))

    processor.validate_duration(1.0)
    processor.validate_duration(60 * 60 * 3)


def test_validate_duration_rejects_unreadable_duration(tmp_path: Path):
    processor = VideoProcessor(make_settings(tmp_path))

    with pytest.raises(ValueError):
        processor.validate_duration(0.0)
