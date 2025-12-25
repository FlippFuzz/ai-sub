import subprocess
from pathlib import Path

import logfire
import static_ffmpeg
from pymediainfo import MediaInfo


def get_video_duration_ms(video_path: Path) -> int:
    """Retrieves the duration of a video file in milliseconds.

    Args:
        video_path (Path): The path to the video file.

    Returns:
        int: The duration of the video in milliseconds. Returns 0 if duration cannot be determined.
    """
    # Parse media information from the video file
    media_info = MediaInfo.parse(video_path)
    # Assuming the first track is the video track and contains duration
    for track in media_info.tracks:
        if track.track_type == "Video":
            return int(float(track.duration))
    return 0


@logfire.instrument("Splitting video into segments, reencode={reencode}")
def split_video(
    input_video: Path,
    output_dir: Path,
    split_duration_s: int,
    max_bytes: int,
    reencode: bool = False,
    reencode_fps: int = 1,
    reencode_height: int = 360,
    reencode_bitrate_kb: int = 30,
) -> list[Path]:
    """Splits a video file into segments of a specified duration using FFmpeg.

    If the first expected segment already exists in the output directory, the function
    assumes the video has been previously split and skips the FFmpeg operation.
    Otherwise, it creates the output directory (if it doesn't exist) and executes
    an FFmpeg command to split the video.

    The function estimates the bitrate (either from the source file or the target
    re-encode settings) and adjusts the split duration downwards if necessary to
    ensure each segment stays within `max_bytes`.

    Args:
        input_video (Path): The path to the input video file.
        output_dir (Path): The directory where the video segments will be saved.
        split_duration_s (int): The target duration of each video segment in seconds.
        max_bytes (int): The maximum allowed size in bytes for each segment.
        reencode (bool): If True, re-encodes the video to 1fps 360p. Defaults to False.
        reencode_fps (int): The framerate to re-encode the video to. Defaults to 1.
        reencode_height (int): The height (resolution) to re-encode the video to. Defaults to 360.
        reencode_bitrate_kb (int): The bitrate in KB/s to re-encode the video to. Defaults to 30.

    Returns:
        list[Path]: A sorted list of Path objects, each pointing to a generated video segment.

    Raises:
        subprocess.CalledProcessError: If the FFmpeg command fails.
    """
    ext = input_video.suffix  # Includes the dot, e.g., ".mp4"

    # TODO: Work on better logic to determine whether or not video has already been split
    expected_first_segment_path = output_dir / f"part_000{ext}"
    if expected_first_segment_path.exists():
        logfire.info(
            f"Assuming video has already been split because part_000{ext} already exists"
        )
    else:
        # Construct the output file pattern for ffmpeg
        output_pattern_filename = f"part_%03d{ext}"
        output_pattern = str(
            output_dir / output_pattern_filename
        )  # ffmpeg needs string path

        static_ffmpeg.add_paths(weak=True)

        # ffmpeg command for splitting
        if not reencode:
            # ffmpeg does not provide a method to split by size.
            # We will use the average bytes_per_sec to decide the durations to split at/
            # Adjust split_duration_s if the estimated segment size exceeds max_bytes
            duration_ms = get_video_duration_ms(input_video)
            file_size = input_video.stat().st_size
            if duration_ms > 0 and file_size > 0:
                bytes_per_sec = file_size / (duration_ms / 1000)
                file_size_split_duration_s = int(max_bytes / bytes_per_sec)
            else:
                file_size_split_duration_s = split_duration_s

            if split_duration_s > file_size_split_duration_s:
                split_duration_s = file_size_split_duration_s
                logfire.info(
                    f"Overwriting split_duration_s to fit max_bytes. split_duration_s={split_duration_s}"
                )

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
        else:
            # Re-encoding to specified fps, height, and bitrate.
            # We use the known bitrate to split the video into max_bytes chunks.
            bytes_per_sec = reencode_bitrate_kb * 1024
            file_size_split_duration_s = int(max_bytes / bytes_per_sec)

            if split_duration_s > file_size_split_duration_s:
                split_duration_s = file_size_split_duration_s
                logfire.info(
                    f"Overwriting split_duration_s to fit max_bytes (re-encode estimate). split_duration_s={split_duration_s}"
                )

            cmd = [
                "ffmpeg",
                "-i",
                str(input_video),
                "-vf",
                f"fps={reencode_fps},scale=-2:{reencode_height}",  # Set to specified FPS and height
                "-b:v",
                str(bytes_per_sec * 8),  # ffmpeg expects bits per second
                "-map",
                "0",
                "-f",
                "segment",
                "-segment_time",
                str(split_duration_s),
                "-reset_timestamps",
                "1",
                output_pattern,
            ]

        try:
            subprocess.run(
                cmd, check=True, capture_output=True, text=True, encoding="utf-8"
            )
        except subprocess.CalledProcessError as e:
            logfire.error(
                f"FFmpeg command failed. Stdout: {e.stdout}, Stderr: {e.stderr}"
            )
            raise  # Re-raise the exception after logging/printing

    result = list(sorted(output_dir.glob(f"*{ext}")))
    logfire.info(f"Split into {len(result)} segments")
    return result
