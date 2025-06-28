import logging
import subprocess
from pathlib import Path

import static_ffmpeg
from pymediainfo import MediaInfo


def get_video_duration_ms(video_path: Path) -> int:
    """Retrieves the duration of a video file in milliseconds.

    Args:
        video_path (Path): The path to the video file.

    Returns:
        int: The duration of the video in milliseconds. Returns 0 if duration cannot be determined.
    """
    media_info = MediaInfo.parse(video_path)
    # Assuming the first track is the video track and contains duration
    for track in media_info.tracks:
        if track.track_type == "Video":
            return int(float(track.duration))
    return 0


def split_video(
    input_video: Path, output_dir: Path, split_duration_s: int
) -> list[Path]:
    """Splits a video file into segments of a specified duration using FFmpeg.

    If the first expected segment already exists in the output directory, the function
    assumes the video has been previously split and skips the FFmpeg operation.
    Otherwise, it creates the output directory (if it doesn't exist) and executes
    an FFmpeg command to split the video.

    Args:
        input_video (Path): The path to the input video file.
        output_dir (Path): The directory where the video segments will be saved.
        split_duration_s (int): The duration of each video segment in seconds.

    Returns:
        list[Path]: A sorted list of Path objects, each pointing to a generated video segment.

    Raises:
        subprocess.CalledProcessError: If the FFmpeg command fails.
    """
    ext = input_video.suffix  # Includes the dot, e.g., ".mp4"

    expected_first_segment_path = output_dir / f"part_000{ext}"
    if expected_first_segment_path.exists():
        logging.info(
            f"First expected segment '{expected_first_segment_path}' already exists. "
            f"Assuming video '{input_video}' has already been split to '{output_dir}'. Skipping."
        )
    else:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Construct the output file pattern for ffmpeg
        output_pattern_filename = f"part_%03d{ext}"
        output_pattern = str(
            output_dir / output_pattern_filename
        )  # ffmpeg needs string path

        static_ffmpeg.add_paths()

        # ffmpeg command for splitting
        cmd = [
            "ffmpeg",  # This is a string from static_ffmpeg
            "-i",
            str(input_video),  # Convert Path to string for subprocess
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
            output_pattern,  # Already a string
        ]

        try:
            subprocess.run(
                cmd, check=True, capture_output=True, text=True, encoding="utf-8"
            )
        except subprocess.CalledProcessError as e:
            logging.error(
                f"FFmpeg command failed. Stdout: {e.stdout}, Stderr: {e.stderr}"
            )
            raise  # Re-raise the exception after logging/printing

    return list(sorted(output_dir.glob(f"*{ext}")))
