"""FFmpeg-based video manipulation utilities for segmentation and compression."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path

import logfire
import static_ffmpeg

_FFMPEG_INITIALIZED: bool = False
_FFMPEG_PATH: str = "ffmpeg"
_FFPROBE_PATH: str = "ffprobe"
_FFMPEG_SEMAPHORE: asyncio.Semaphore | None = None


def set_ffmpeg_concurrency(max_concurrent: int) -> None:
    """Sets the maximum number of concurrent FFmpeg operations.

    Args:
        max_concurrent: Maximum concurrent FFmpeg subprocesses allowed.
    """
    global _FFMPEG_SEMAPHORE
    _FFMPEG_SEMAPHORE = asyncio.Semaphore(max(1, max_concurrent))


def _get_ffmpeg_semaphore() -> asyncio.Semaphore:
    """Retrieves or initializes the global FFmpeg concurrency semaphore.

    Returns:
        asyncio.Semaphore: Global semaphore instance.
    """
    global _FFMPEG_SEMAPHORE
    if _FFMPEG_SEMAPHORE is None:
        _FFMPEG_SEMAPHORE = asyncio.Semaphore(1)
    return _FFMPEG_SEMAPHORE


def _find_system_binary(binary_name: str) -> str | None:
    """Finds an executable in the system PATH, ignoring any static_ffmpeg directory.

    Args:
        binary_name: Name of the binary to find (e.g., 'ffmpeg', 'ffprobe').

    Returns:
        str | None: Absolute path to the system binary if found, or None.
    """
    path_env = os.environ.get("PATH", "")
    clean_entries: list[str] = []
    for entry in path_env.split(os.pathsep):
        cleaned = entry.strip().strip('"').strip("'")
        if cleaned and "static_ffmpeg" not in cleaned.lower() and Path(cleaned).is_dir():
            clean_entries.append(cleaned)

    clean_path = os.pathsep.join(clean_entries)

    for name in (binary_name, f"{binary_name}.exe"):
        found = shutil.which(name, path=clean_path)
        if found and Path(found).is_file():
            return str(Path(found).resolve())

    return None


def _probe_binary_info(binary_path: str) -> tuple[str, str]:
    """Probes version and configuration flags from an FFmpeg or FFprobe binary.

    Args:
        binary_path: Path to the executable.

    Returns:
        tuple[str, str]: (version_line, configuration_line).
    """
    try:
        proc = subprocess.run(
            [binary_path, "-version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        version_line = "unknown"
        config_line = "unknown"
        for line in proc.stdout.splitlines():
            stripped = line.strip()
            if not version_line or version_line == "unknown":
                if stripped.startswith(("ffmpeg version", "ffprobe version")):
                    version_line = stripped
            if stripped.startswith("configuration:"):
                config_line = stripped
                break
        return version_line, config_line
    except Exception as e:
        return f"error probing version: {e}", "unknown"


def ensure_ffmpeg() -> tuple[str, str]:
    """Ensures ffmpeg and ffprobe are available, preferring system binaries over static-ffmpeg.

    Returns:
        tuple[str, str]: Absolute or command paths for (ffmpeg, ffprobe).
    """
    global _FFMPEG_INITIALIZED, _FFMPEG_PATH, _FFPROBE_PATH
    if _FFMPEG_INITIALIZED:
        return _FFMPEG_PATH, _FFPROBE_PATH

    sys_ffmpeg = _find_system_binary("ffmpeg")
    sys_ffprobe = _find_system_binary("ffprobe")

    if sys_ffmpeg and sys_ffprobe:
        _FFMPEG_PATH = sys_ffmpeg
        _FFPROBE_PATH = sys_ffprobe
    else:
        try:
            static_ffmpeg.add_paths(weak=True)
            resolved_ffmpeg = shutil.which("ffmpeg")
            resolved_ffprobe = shutil.which("ffprobe")
            _FFMPEG_PATH = sys_ffmpeg or resolved_ffmpeg or "ffmpeg"
            _FFPROBE_PATH = sys_ffprobe or resolved_ffprobe or "ffprobe"
        except Exception as e:
            logfire.warning(f"Could not load static-ffmpeg fallback: {e}")
            _FFMPEG_PATH = sys_ffmpeg or "ffmpeg"
            _FFPROBE_PATH = sys_ffprobe or "ffprobe"

    _FFMPEG_INITIALIZED = True
    return _FFMPEG_PATH, _FFPROBE_PATH


async def init_ffmpeg_engine(requested_encoder: str | None = None) -> tuple[str, str, str]:
    """Initializes FFmpeg and FFprobe paths and discovers working encoder within a single Logfire span.

    Args:
        requested_encoder: User-configured encoder if specified, or None for auto-detection.

    Returns:
        tuple[str, str, str]: (ffmpeg_path, ffprobe_path, working_encoder).
    """
    ffmpeg_bin, ffprobe_bin = ensure_ffmpeg()
    ffmpeg_version, ffmpeg_config = _probe_binary_info(ffmpeg_bin)
    ffprobe_version, _ = _probe_binary_info(ffprobe_bin)

    with logfire.span(
        "FFmpeg Setup",
        ffmpeg_path=ffmpeg_bin,
        ffmpeg_version=ffmpeg_version,
        ffmpeg_configuration=ffmpeg_config,
        ffprobe_path=ffprobe_bin,
        ffprobe_version=ffprobe_version,
    ):
        logfire.info("FFmpeg: {path}", path=ffmpeg_bin, version=ffmpeg_version)
        logfire.info("FFprobe: {path}", path=ffprobe_bin, version=ffprobe_version)

        if requested_encoder:
            encoder = requested_encoder
        else:
            encoder = await get_working_encoder()

        logfire.info("Encoder: {encoder}", encoder=encoder)

    return ffmpeg_bin, ffprobe_bin, encoder


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
        stdin=asyncio.subprocess.DEVNULL,
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
    _, ffprobe_bin = ensure_ffmpeg()
    try:
        cmd = [
            ffprobe_bin,
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
    _, ffprobe_bin = ensure_ffmpeg()
    try:
        cmd = [
            ffprobe_bin,
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
    ffmpeg_bin, _ = ensure_ffmpeg()
    candidates = ["h264_nvenc", "h264_qsv", "h264_amf", "h264_videotoolbox", "h264_mf"]

    for encoder in candidates:
        try:
            await _run_ffmpeg(
                cmd=[
                    ffmpeg_bin,
                    "-nostdin",
                    "-y",
                    "-loglevel",
                    "error",
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

    ffmpeg_bin, _ = ensure_ffmpeg()
    escaped_dir = str(output_dir).replace("%", "%%")
    full_output_pattern = str(Path(escaped_dir) / f"{output_pattern}{ext}")

    cmd = [
        ffmpeg_bin,
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
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

    sem = _get_ffmpeg_semaphore()
    async with sem:
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
    keyframe_only: bool = False,
) -> None:
    """Compresses a video segment or full video using low framerates and downscaled resolution.

    Args:
        input_path: Source video media path.
        output_path: Destination path for compressed output.
        fps: Target framerate.
        height: Target height resolution.
        bitrate_kb: Target bitrate in Kilobytes per second.
        encoder: Encoder codec to employ.
        duration_tolerance_ms: Allowed duration variance in milliseconds.
        keyframe_only: Whether to decode only keyframes from the input video.

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
                logfire.info(f"Reusing existing valid compressed media: {output_path.name}")
                return
        except Exception:
            pass

    ffmpeg_bin, _ = ensure_ffmpeg()
    video_bytes_per_sec = bitrate_kb * 1024

    # Container-aware audio encoding: .mov supports pcm_u8; .mp4 requires aac
    if output_path.suffix.lower() == ".mp4":
        audio_args = ["-c:a", "aac", "-b:a", "64k", "-ac", "1", "-ar", "16000"]
    else:
        audio_args = ["-c:a", "pcm_u8", "-ac", "1", "-ar", "16000"]

    input_args = ["-skip_frame", "nokey"] if keyframe_only else []
    vf_filter = f"scale=-2:{height},fps={fps}" if keyframe_only else f"fps={fps},scale=-2:{height}"

    cmd = [
        ffmpeg_bin,
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-nostats",
        *input_args,
        "-i",
        str(input_path),
        "-vf",
        vf_filter,
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
        *audio_args,
        str(output_path),
    ]

    sem = _get_ffmpeg_semaphore()
    async with sem:
        try:
            await _run_ffmpeg(cmd)
        except subprocess.CalledProcessError as e:
            logfire.exception(f"FFmpeg re-encoding failed for {input_path.name}. Stderr: {e.stderr}")
            raise
