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

    Args:
        input_video (Path): The path to the input video file.
        output_dir (Path): The directory where the video segments will be saved.
        split_duration_s (int): The target duration of each video segment in seconds.
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
    final_ext = ".mov" if reencode else ext

    # TODO: Work on better logic to determine whether or not video has already been split
    expected_first_segment_path = output_dir / f"part_000{final_ext}"
    if expected_first_segment_path.exists():
        logfire.info(
            f"Assuming video has already been split because part_000{final_ext} already exists"
        )
        return list(sorted(output_dir.glob(f"part_*{final_ext}")))
    else:
        static_ffmpeg.add_paths(weak=True)

        if reencode:
            # TODO: In the future, have re-encoding jobs just like gemini upload jobs
            # This will allow us to re-encode while starting the subtitle generation

            # Strategy: Split the original video first (copy mode) to preserve exact timing,
            # then re-encode the chunks. This prevents drift caused by re-encoding the whole file first.
            encoder = get_working_encoder()
            video_bytes_per_sec = reencode_bitrate_kb * 1024

            # 1. Split original video into temporary chunks
            temp_pattern = str(output_dir / f"temp_%03d{ext}")
            with logfire.span("Splitting original video (copy mode)"):
                cmd_split = [
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
                    temp_pattern,
                ]
                try:
                    subprocess.run(
                        cmd_split,
                        check=True,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                    )
                except subprocess.CalledProcessError as e:
                    logfire.error(
                        f"FFmpeg split failed. Stdout: {e.stdout}, Stderr: {e.stderr}"
                    )
                    raise

            # 2. Re-encode each chunk
            temp_files = sorted(output_dir.glob(f"temp_*{ext}"))
            with logfire.span(
                f"Re-encoding {len(temp_files)} segments to {reencode_fps}fps, {reencode_height}p"
            ):
                for temp_file in temp_files:
                    # Map temp_001.mp4 -> part_001.mov
                    part_num = temp_file.stem.split("_")[-1]
                    output_file = output_dir / f"part_{part_num}{final_ext}"

                    cmd_encode = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(temp_file),
                        "-vf",
                        f"fps={reencode_fps},scale=-2:{reencode_height}",
                        "-c:v",
                        encoder,
                        "-g",
                        str(reencode_fps * 10),
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
                        str(output_file),
                    ]

                    try:
                        subprocess.run(
                            cmd_encode,
                            check=True,
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                        )
                        temp_file.unlink()  # Remove temp file
                    except subprocess.CalledProcessError as e:
                        logfire.error(
                            f"FFmpeg re-encode failed for {temp_file.name}. Stdout: {e.stdout}, Stderr: {e.stderr}"
                        )
                        raise

        else:
            # Split the video directly without re-encoding
            output_pattern = str(output_dir / f"part_%03d{ext}")
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
                raise

    result = list(sorted(output_dir.glob(f"part_*{final_ext}")))
    logfire.info(f"Split into {len(result)} segments")
    return result
