"""Job definition for video segment and summary proxy compression."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, NonNegativeFloat, NonNegativeInt, PositiveInt

from ai_sub.jobs.job_base import BaseJob
from ai_sub.video import reencode_video


@dataclass(frozen=True)
class ReencodeResult:
    """Result returned upon completing a media re-encoding job."""

    path: Path
    fps: float


class ReencodeJob(BaseJob):
    """Represents a video segment, proxy, or audio-only compression task using FFmpeg."""

    input_file: Path = Field(description="Source video segment or raw video path.")
    output_file: Path = Field(description="Destination path for the compressed output.")
    fps: NonNegativeFloat | Literal["keyframes"] = Field(
        description="Target framerate in FPS (0 for audio-only), or 'keyframes' for native keyframe extraction.",
    )
    height: PositiveInt = Field(description="Target resolution height in pixels.")
    bitrate: PositiveInt = Field(description="Target video bitrate in bytes per second.")
    duration_tolerance_ms: NonNegativeInt = Field(
        description="Allowed duration difference in milliseconds to consider an existing re-encoded file valid.",
    )
    encoder: str | None = Field(
        default=None,
        description="Hardware or software encoder to use.",
    )
    keyframe_only: bool = Field(
        default=False,
        description="Whether to decode only keyframes from the input video.",
    )
    segment_index: int | None = Field(
        default=None,
        description="Segment chunk index if this job compresses a segment, or None for summary proxy.",
    )
    is_summary_proxy: bool = Field(
        default=False,
        description="Whether this job compresses the whole-video summary proxy.",
    )
    fallback_tier: NonNegativeInt = Field(
        default=0,
        description="Fallback re-encoding tier index (0 = initial, 1+ = fallback FPS tier).",
    )

    async def execute(self) -> ReencodeResult:
        """Executes FFmpeg re-encoding and returns the output path and effective framerate.

        Returns:
            ReencodeResult: Output file path and effective numeric framerate (0.0 for audio-only).
        """
        effective_fps = await reencode_video(
            input_path=self.input_file,
            output_path=self.output_file,
            fps=self.fps,
            height=self.height,
            bitrate=self.bitrate,
            encoder=self.encoder or "libx264",
            duration_tolerance_ms=self.duration_tolerance_ms,
            keyframe_only=self.keyframe_only,
        )
        return ReencodeResult(path=self.output_file, fps=effective_fps)
