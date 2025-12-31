import subprocess
from pathlib import Path

import logfire
import static_ffmpeg


def get_video_duration_ms(video_path: Path) -> int:
    """Retrieves the duration of a video file in milliseconds.

    Args:
        video_path (Path): The path to the video file.

    Returns:
        int: The duration of the video in milliseconds. Returns 0 if duration cannot be determined.
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
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, encoding="utf-8"
        )
        return int(float(result.stdout) * 1000)
    except (subprocess.CalledProcessError, ValueError):
        return 0


def get_working_encoder() -> str:
    """
    Checks for available hardware acceleration for H.264 encoding.
    Returns the name of the encoder to use (e.g., 'h264_nvenc', 'libx264').
    """
    # List of hardware encoders to check in order of preference
    candidates = ["h264_nvenc", "h264_qsv", "h264_amf", "h264_videotoolbox", "h264_mf"]

    for encoder in candidates:
        try:
            # Attempt to encode a 1-frame dummy video to null output
            subprocess.run(
                [
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
                ],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            return encoder
        except subprocess.CalledProcessError as e:
            logfire.debug(
                f"Encoder {encoder} check failed.\nstdout: {e.stdout}\nstderr: {e.stderr}"
            )
            continue
        except FileNotFoundError:
            logfire.debug(f"Encoder {encoder} check failed: FFmpeg not found.")
            continue

    return "libx264"


@logfire.instrument("Splitting video into segments")
def split_video(
    input_video: Path,
    output_dir: Path,
    split_duration_s: int,
    output_pattern: str = "part_%03d",
) -> list[Path]:
    """Splits a video file into segments of a specified duration using FFmpeg.

    If the first expected segment already exists in the output directory, the function
    assumes the video has been previously split and skips the FFmpeg operation.
    Otherwise, it creates the output directory (if it doesn't exist) and executes
    an FFmpeg command to split the video.

    Args:
        input_video (Path): The path to the input video file.
        output_dir (Path): The directory where the video segments will be saved.
        split_duration_s (int): The target duration of each video segment in seconds.
        output_pattern (str): The filename pattern for the output segments. Defaults to "part_%03d".

    Returns:
        list[Path]: A sorted list of Path objects, each pointing to a generated video segment.

    Raises:
        subprocess.CalledProcessError: If the FFmpeg command fails.
    """
    ext = input_video.suffix  # Includes the dot, e.g., ".mp4"

    # Check if we have files matching the pattern
    glob_pattern = output_pattern.replace("%03d", "*") + ext

    # Check for the first file to see if we can skip
    first_file_name = output_pattern % 0 + ext
    expected_first_segment_path = output_dir / first_file_name

    if expected_first_segment_path.exists():
        logfire.info(
            f"Assuming video has already been split because {first_file_name} already exists"
        )
        return list(sorted(output_dir.glob(glob_pattern)))
    else:
        static_ffmpeg.add_paths(weak=True)

        full_output_pattern = str(output_dir / f"{output_pattern}{ext}")
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
            subprocess.run(
                cmd, check=True, capture_output=True, text=True, encoding="utf-8"
            )
        except subprocess.CalledProcessError as e:
            logfire.error(
                f"FFmpeg command failed. Stdout: {e.stdout}, Stderr: {e.stderr}"
            )
            raise

    result = list(sorted(output_dir.glob(glob_pattern)))
    logfire.info(f"Split into {len(result)} segments")
    return result


def reencode_video(
    input_path: Path,
    output_path: Path,
    fps: int,
    height: int,
    bitrate_kb: int,
) -> None:
    """Re-encodes a video file to a specific format.

    If the output file already exists, re-encoding is skipped.

    Args:
        input_path (Path): The path to the input video file.
        output_path (Path): The path where the re-encoded video will be saved.
        fps (int): The target framerate.
        height (int): The target height (resolution).
        bitrate_kb (int): The target bitrate in KB/s.
    """

    # If output file already exists, we can just skip the re-encode
    if output_path.exists():
        logfire.info(
            f"Skipping re-encode for {input_path.name} as {output_path.name} already exists."
        )
        return

    static_ffmpeg.add_paths(weak=True)
    encoder = get_working_encoder()
    video_bytes_per_sec = bitrate_kb * 1024

    cmd_encode = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        f"fps={fps},scale=-2:{height}",
        "-c:v",
        encoder,
        "-g",
        str(fps * 10),
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
        subprocess.run(
            cmd_encode,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except subprocess.CalledProcessError as e:
        logfire.error(
            f"FFmpeg re-encode failed for {input_path.name}. Stdout: {e.stdout}, Stderr: {e.stderr}"
        )
        raise
