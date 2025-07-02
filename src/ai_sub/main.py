import concurrent.futures
import json
import logging
import threading
from argparse import Namespace
from datetime import datetime
from importlib.metadata import version as get_version
from pathlib import Path

from google.genai.types import File
from pysubs2 import SSAEvent, SSAFile

from ai_sub.config import configure_logging, generate_output_paths, parse_arguments
from ai_sub.gemini import PROMPT_GENERATE_SUBTITLE_TEMPLATE, Gemini
from ai_sub.models import State
from ai_sub.video import get_video_duration_ms, split_video

logger = logging.getLogger(__name__)


def single_run(video_file: File, args: Namespace, gemini: Gemini) -> None:
    """Processes a single video file to generate and save subtitles.

    This function handles the generation of subtitles for a given video segment
    using the Gemini model. It manages the state of subtitle generation,
    re-uses previously generated subtitles if available, and saves the
    resulting SSAFile to a specified output path.

    Args:
        video_file (File): The video file object to process.
        args (Namespace): An argparse Namespace object containing command-line arguments
                          such as output directory, log level, etc.
        gemini (Gemini): An instance of the Gemini class for interacting with the
                         Gemini API.
    """
    video_file_display_name = str(video_file.display_name)
    threading.current_thread().name = video_file_display_name

    logger.info("=" * 70)
    logger.info(f"Working on partial video file: {video_file_display_name}")

    # Sort out filenames
    output_subtitle_path, output_state_path = generate_output_paths(video_file, args)

    # Load state from file
    state = State.load_or_return_new(output_state_path)

    # Start processing with Gemini
    try:
        if state.generateSubtitleResponse is None:
            logger.info("  Using Gemini to generate subtitles.")
            generate_subtitle_response = gemini.generate_subtitles(video_file)
            state.generateSubtitleResponse = generate_subtitle_response
            state.save(output_state_path)
        else:
            logger.info(
                f"  Re-using previously generated subtitles from {output_state_path.stem}"
            )

        if state.generateSubtitleResponse is not None:
            current_subtitles = state.generateSubtitleResponse.get_ssafile()
            current_subtitles.save(path=str(output_subtitle_path))

        logger.info(f"Successfully processed video: {video_file_display_name}")
    except Exception as e:
        logger.exception(
            f"Unrecoverable Error processing {video_file_display_name}: {e}"
        )
        logger.error(
            "  This video segment will be skipped. Re-run this script to retry."
        )


def main():
    """Main function to orchestrate the video subtitle generation process.

    This function parses command-line arguments, configures logging, splits
    the input video into segments, uploads segments to Gemini, generates
    subtitles in parallel using multiple threads, and finally combines all
    partial subtitles into a single output file.
    """
    args = parse_arguments()
    configure_logging(args.log_level)
    logger.info(f"ai-sub version: {get_version('ai-sub')}")

    gemini = Gemini(
        args.api_key,
        model=args.model,
        thinking_budget=args.thinking_budget,
        rpm=args.rpm,
        tpm=args.tpm,
        max_subtitle_chars=args.max_subtitle_chars,
        num_upload_threads=args.num_upload_threads,
    )

    # Split video
    logger.info(f"Splitting input into {args.split_seconds}s segments")
    all_video_paths = split_video(args.input_file, args.temp_dir, args.split_seconds)

    # We only need to work on videos parts that haven't been worked on
    video_paths = []
    for video_path in all_video_paths:
        _, output_state_path = generate_output_paths(video_path, args)
        state = State.load_or_return_new(output_state_path)
        if state.generateSubtitleResponse is None:
            video_paths.append(video_path)
        else:
            logger.info(f"  Skipping previously processed part: {video_path.name}")

    # Upload videos to Gemini
    logger.info("Uploading files")
    video_files = gemini.upload_files(video_paths)

    start_time = datetime.now()
    logger.info(
        f"Generating subtitles using {args.num_processing_threads} processing threads."
    )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.num_processing_threads
    ) as executor:
        executor.map(
            lambda video_file: single_run(video_file, args, gemini), video_files
        )
    end_time = datetime.now()
    logger.info("=" * 70)
    logger.info(f"Time taken (excluding splitting/uploading): {end_time - start_time}")

    ## Combine all the partial SRTs into a final one
    all_subtitles = SSAFile()
    all_duration_ms = 0
    unprocessed_segments = []

    for video_path in all_video_paths:
        _, output_state_path = generate_output_paths(video_path, args)
        state = State.load_or_return_new(output_state_path)

        if state.generateSubtitleResponse is not None:
            current_subtitles = state.generateSubtitleResponse.get_ssafile()
        else:
            unprocessed_segments.append(video_path.name)
            current_subtitles = SSAFile()
            current_subtitles.append(
                SSAEvent(
                    start=0,
                    end=get_video_duration_ms(video_path),
                    text="Error processing subtitles for this segment.",
                )
            )

        current_subtitles.shift(ms=all_duration_ms)
        all_subtitles += current_subtitles
        all_duration_ms += get_video_duration_ms(video_path)

    # Add version and config to the subtitle file as comments in the first millisecond
    version = get_version("ai-sub")
    config_dict = vars(args).copy()  # Create a copy to modify
    # Remove sensitive and unnecessary information
    keys_to_remove = ["api_key", "input_file", "output_dir", "temp_dir", "log_level"]
    for key in keys_to_remove:
        if key in config_dict:
            del config_dict[key]

    # Convert Path and File objects to strings for JSON serialization
    for key, value in config_dict.items():
        if isinstance(value, Path):
            config_dict[key] = str(value)
        elif isinstance(value, File):
            config_dict[key] = str(value.display_name)

    # Insert version, config, and prompt template as a single SSAEvent at the beginning (0-1ms)
    # JSON curly braces {} are treated as formatting codes in SRT, so replace them.
    config_json = json.dumps(config_dict, indent=2).replace("{", "(").replace("}", ")")
    prompt_template_str = PROMPT_GENERATE_SUBTITLE_TEMPLATE.replace("{", "(").replace(
        "}", ")"
    )
    info = f"Generated by ai-sub version: {version}\nConfig: {config_json}\nPrompt Template: {prompt_template_str}"
    all_subtitles.insert(0, SSAEvent(start=0, end=1, text=info))

    output_file_path = args.output_dir / f"{args.input_file.stem}.srt"
    all_subtitles.save(str(output_file_path))
    logging.info(f"Subtitles saved to {output_file_path}")

    if len(unprocessed_segments) > 0:
        logger.error("=" * 70)
        logger.error("The following video segments could not be processed:")
        for segment in unprocessed_segments:
            logger.error(f"  - {segment}")
        logger.error("Re-run the script to retry processing these segments.")


if __name__ == "__main__":
    main()
