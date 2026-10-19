"""FFmpeg-based video manipulation utilities for segmentation and compression."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import logfire
import static_ffmpeg


async def _run_ffmpeg(cmd: list[str]) -> str:
    """Executes an FFmpeg command asynchronously with cancellation handling.

    Args:
        cmd: Command arguments as a list of strings.

    Returns:
        str: Standard output from the command.

    Raises:
        subprocess.CalledProcessError: If the process returns a non-zero exit code.
    """
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await process.communicate()
    except asyncio.CancelledError:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        await process.wait()
        raise

    stdout_str = stdout.decode("utf-8", errors="replace")
    stderr_str = stderr.decode("utf-8", errors="replace")

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode or 1, cmd, output=stdout_str, stderr=stderr_str)
    return stdout_str


async def get_video_duration_ms(video_path: Path) -> int:
    """Calculates the duration of a video file in milliseconds.

    Args:
        video_path: Path to the target media file.

    Returns:
        int: Duration in milliseconds.

    Raises:
        RuntimeError: If the duration cannot be probed.
    """
    static_ffmpeg.add_paths(weak=True)
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        stdout = await _run_ffmpeg(cmd)
        return int(float(stdout.strip()) * 1000)
    except (subprocess.CalledProcessError, ValueError) as e:
        logfire.exception(f"Could not determine duration for {video_path.name}")
        raise RuntimeError(f"Could not determine duration for video file: {video_path.name}") from e


async def get_video_fps(video_path: Path) -> float:
    """Calculates the average framerate (FPS) of a video file.

    Args:
        video_path: Path to the target media file.

    Returns:
        float: Frames per second.

    Raises:
        RuntimeError: If the framerate cannot be determined.
    """
    static_ffmpeg.add_paths(weak=True)
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        stdout = await _run_ffmpeg(cmd)
        num, den = map(int, stdout.strip().split("/"))
        return num / den
    except (subprocess.CalledProcessError, ValueError, ZeroDivisionError) as e:
        logfire.exception(f"Could not determine FPS for {video_path.name}")
        raise RuntimeError(f"Could not determine FPS for video file: {video_path.name}") from e


async def get_working_encoder() -> str:
    """Discovers available hardware-accelerated H.264 encoders.

    Returns:
        str: Preferred encoder name, falling back to 'libx264'.
    """
    static_ffmpeg.add_paths(weak=True)
    candidates = ["h264_nvenc", "h264_qsv", "h264_amf", "h264_videotoolbox", "h264_mf"]

    for encoder in candidates:
        try:
            await _run_ffmpeg(
                cmd=[
                    "ffmpeg",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=64x64:d=0.01",
                    "-c:v",
                    encoder,
                    "-b:v",
                    "1000k",
                    "-f",
                    "null",
                    "-",
                ]
            )
            return encoder
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    return "libx264"


@logfire.instrument("Splitting video into segments")
async def split_video(
    input_video: Path,
    output_dir: Path,
    split_duration_s: int,
    output_pattern: str = "part_%03d",
    duration_tolerance_ms: int = 100,
) -> list[Path]:
    """Splits an input video into uniform segments using FFmpeg stream copying.

    Skips splitting if valid segments matching the pattern and duration already exist.

    Args:
        input_video: Path to the input video.
        output_dir: Directory where segments will be saved.
        split_duration_s: Target segment length in seconds.
        output_pattern: Pattern formatting segment filenames. Defaults to 'part_%03d'.
        duration_tolerance_ms: Allowed total duration variance in milliseconds.

    Returns:
        list[Path]: Sorted list of generated segment paths.

    Raises:
        subprocess.CalledProcessError: If FFmpeg segmentation fails.
    """
    ext = input_video.suffix
    glob_pattern = output_pattern.replace("%03d", "*") + ext
    existing_segments = sorted(output_dir.glob(glob_pattern))

    if existing_segments:
        try:
            input_duration = await get_video_duration_ms(input_video)
            durations = await asyncio.gather(*(get_video_duration_ms(s) for s in existing_segments))
            total_segment_duration = sum(durations)
            threshold = duration_tolerance_ms * len(existing_segments)

            if abs(input_duration - total_segment_duration) < threshold:
                logfire.info(f"Reusing {len(existing_segments)} existing valid segments for {input_video.name}.")
                return existing_segments

            logfire.warning(
                f"Re-splitting {input_video.name}: existing duration ({total_segment_duration}ms) "
                f"mismatches input ({input_duration}ms)."
            )
            for segment in existing_segments:
                try:
                    segment.unlink()
                except OSError:
                    pass
        except Exception:
            logfire.warning(f"Re-splitting {input_video.name}: existing segments verification failed.")

    static_ffmpeg.add_paths(weak=True)
    escaped_dir = str(output_dir).replace("%", "%%")
    full_output_pattern = str(Path(escaped_dir) / f"{output_pattern}{ext}")

    cmd = [
        "ffmpeg",
        "-i",
        str(input_video),
        "-c",
        "copy",
        "-map",
        "0",
        "-f",
        "segment",
        "-segment_time",
        str(split_duration_s),
        "-reset_timestamps",
        "1",
        full_output_pattern,
    ]

    try:
        await _run_ffmpeg(cmd)
    except subprocess.CalledProcessError as e:
        logfire.exception(f"FFmpeg split failed. Stdout: {e.stdout}, Stderr: {e.stderr}")
        raise

    return sorted(output_dir.glob(glob_pattern))


async def reencode_video(
    input_path: Path,
    output_path: Path,
    fps: float,
    height: int,
    bitrate_kb: int,
    encoder: str,
    duration_tolerance_ms: int = 100,
) -> None:
    """Compresses a video segment using low framerates and downscaled resolution.

    Args:
        input_path: Source video segment path.
        output_path: Destination path for compressed output.
        fps: Target framerate.
        height: Target height resolution.
        bitrate_kb: Target bitrate in Kilobytes per second.
        encoder: Encoder codec to employ.
        duration_tolerance_ms: Allowed duration variance in milliseconds.

    Raises:
        subprocess.CalledProcessError: If FFmpeg re-encoding fails.
    """
    if output_path.exists():
        try:
            input_duration, output_duration, output_fps = await asyncio.gather(
                get_video_duration_ms(input_path),
                get_video_duration_ms(output_path),
                get_video_fps(output_path),
            )
            effective_tolerance = duration_tolerance_ms + (int(1000 / fps) if fps < 1.0 else 0)
            if abs(input_duration - output_duration) < effective_tolerance and abs(fps - output_fps) < 0.01:
                logfire.info(f"Reusing existing valid compressed segment: {output_path.name}")
                return
        except Exception:
            pass

    static_ffmpeg.add_paths(weak=True)
    video_bytes_per_sec = bitrate_kb * 1024

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        f"fps={fps},scale=-2:{height}",
        "-c:v",
        encoder,
        "-g",
        str(int(fps * 10)),
        "-b:v",
        str(video_bytes_per_sec * 8),
        "-maxrate",
        str(video_bytes_per_sec * 8),
        "-bufsize",
        str(video_bytes_per_sec * 8 * 2),
        "-c:a",
        "pcm_u8",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]

    try:
        await _run_ffmpeg(cmd)
    except subprocess.CalledProcessError as e:
        logfire.exception(f"FFmpeg re-encoding failed for {input_path.name}. Stderr: {e.stderr}")
        raise
