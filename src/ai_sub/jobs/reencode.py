"""Job definition for video segment compression and re-encoding."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt

from ai_sub.jobs.base import BaseJob
from ai_sub.video import reencode_video


class ReencodeJob(BaseJob):
    """Represents a video segment compression task using FFmpeg."""

    input_file: Path = Field(description="Source video segment path.")
    output_file: Path = Field(description="Destination path for the compressed segment.")
    fps: PositiveFloat = Field(description="Target framerate for the re-encoded video.")
    height: PositiveInt = Field(description="Target resolution height.")
    bitrate_kb: PositiveInt = Field(description="Target video bitrate in Kilobytes per second.")
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

    async def execute(self) -> Path:
        """Executes FFmpeg re-encoding for this segment.

        Returns:
            Path: Path to the validated compressed video file.
        """
        await reencode_video(
            input_path=self.input_file,
            output_path=self.output_file,
            fps=self.fps,
            height=self.height,
            bitrate_kb=self.bitrate_kb,
            encoder=self.encoder or "libx264",
            duration_tolerance_ms=self.duration_tolerance_ms,
            keyframe_only=self.keyframe_only,
        )
        return self.output_file
