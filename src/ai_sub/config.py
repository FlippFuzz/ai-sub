import logging
import os
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from pathlib import Path
from typing import Tuple

from google.genai.types import File


def check_file_exists(filepath_str: str) -> Path:
    """Checks if a given file path string corresponds to an existing file.

    Args:
        filepath_str (str): The string representation of the file path.

    Returns:
        Path: A resolved Path object if the file exists.

    Raises:
        ArgumentTypeError: If the file does not exist or is not a file.
    """
    # Resolve the path to get an absolute, normalized path, resolving symlinks
    file_path = Path(filepath_str).resolve()

    # Check if the path points to an actual file
    if not file_path.is_file():
        raise ArgumentTypeError(
            f"Input file '{filepath_str}' does not exist or is not a file."
        )

    return file_path


def parse_arguments() -> Namespace:
    """Parses command-line arguments for the Gemini TL application.

    This function sets up an ArgumentParser with various options for API
    configuration, file and directory handling, processing parameters, and
    logging. It also performs validation for the API key and sets default
    values for temporary and output directories if not provided.

    Returns:
        Namespace: An argparse Namespace object containing the parsed arguments.

    Raises:
        ArgumentTypeError: If the input file does not exist.
        SystemExit: If no Gemini API key is provided.
    """
    parser = ArgumentParser(
        description="AI-Powered Subtitle Generation with Translation.",
        prog="ai-sub",
    )
    parser.add_argument(
        "input_file", type=check_file_exists, help="Path to the input video file."
    )

    api_group = parser.add_argument_group("API Options")
    api_group.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("GOOGLE_API_KEY"),
        help="Your Gemini API key (or set GOOGLE_API_KEY environment variable).",
    )
    api_group.add_argument(
        "--rpm",
        type=int,
        default=5,
        help="Requests per minute for Gemini API (default: 5).",
    )
    api_group.add_argument(
        "--tpm",
        type=int,
        default=250000,
        help="Tokens per minute for Gemini API (default: 250000).",
    )
    api_group.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-pro",
        help="Gemini model to use (default: gemini-2.5-pro).",
    )
    api_group.add_argument(
        "--thinking_budget",
        type=int,
        default=32768,
        help="Thinking budget for Gemini API (default: 32768).",
    )

    file_group = parser.add_argument_group("File and Directory Options")
    file_group.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save output files (default: input_file's parent directory).",
    )
    file_group.add_argument(
        "--temp_dir",
        type=Path,
        help="Directory to store temporary files (default: tmp_<input_file_name>}).",
    )

    processing_group = parser.add_argument_group("Processing Options")
    processing_group.add_argument(
        "--max_subtitle_chars",
        type=int,
        default=60,
        help="Maximum character length for each subtitle entry (default: 60).",
    )
    processing_group.add_argument(
        "--num_processing_threads",
        type=int,
        default=4,
        help="Number of threads to use for parallel subtitle processing (default: 4).",
    )
    processing_group.add_argument(
        "--num_upload_threads",
        type=int,
        default=4,
        help="Number of threads to use for parallel file uploads (default: 4).",
    )
    processing_group.add_argument(
        "--split_seconds",
        type=int,
        default=180,
        help="Duration in seconds to split the video into segments (default: 180s).",
    )

    logging_group = parser.add_argument_group("Logging Options")
    logging_group.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO).",
    )

    args = parser.parse_args()

    if args.api_key is None:
        parser.error(
            "No Gemini API key provided. Use --api_key or set the GOOGLE_API_KEY "
            "environment variable."
        )

    # Set default temp_dir if not provided
    if args.temp_dir is None:
        args.temp_dir = args.input_file.parent / f"tmp_{args.input_file.stem}"
    args.temp_dir.mkdir(parents=True, exist_ok=True)

    # Set default output_dir if not provided
    if args.output_dir is None:
        args.output_dir = args.input_file.parent

    return args


def configure_logging(log_level: str):
    """Configures the logging for the application.

    This function sets up a stream handler for logging, defines the log format,
    and sets the overall logging level. It also suppresses noisy INFO level
    logs from specific external libraries like 'httpx' and 'google_genai.models'.

    Args:
        log_level (str): The desired logging level (e.g., "INFO", "DEBUG").
    """
    # Remove all existing handlers from the root logger to ensure a clean slate
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()

    # Create a formatter with the desired format (no date)
    formatter = logging.Formatter("%(threadName)s %(levelname)s %(message)s")

    # Create a stream handler and set the formatter
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # Get the root logger and add the new handler
    root_logger = logging.getLogger()
    root_logger.addHandler(stream_handler)
    root_logger.setLevel(log_level)

    # Suppress INFO level logs from 'httpx' to reduce noise from HTTP request/response logging.
    # Example noisy log: "INFO HTTP Request: GET https://generativelanguage.googleapis.com/v1beta/files?pageSize=100 'HTTP/1.1 200 OK'"
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Suppress INFO level logs from 'google.genai.models' to reduce noise from internal model operations.
    # Example noisy log: "INFO AFC is enabled with max remote calls: 10."
    # https://github.com/googleapis/python-genai/issues/278
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)


def generate_output_paths(
    video_file: Path | File, args: Namespace
) -> Tuple[Path, Path]:
    """Generates the output paths for subtitle and state files.

    Based on the input video file (either a local Path or a Gemini File object)
    and the provided arguments, this function constructs the full paths for
    where the generated subtitle file (.srt) and the processing state file (.json)
    should be saved.

    Args:
        video_file (Path | File): The input video file, which can be a pathlib.Path
                                  object for local files or a google.genai.types.File
                                  object for uploaded files.
        args (Namespace): An argparse Namespace object containing command-line arguments,
                          specifically `temp_dir` for the temporary directory.

    Returns:
        Tuple[Path, Path]: A tuple containing two Path objects:
                           - The full path for the output subtitle file (.srt).
                           - The full path for the output state file (.json).
    """
    stem = ""
    if isinstance(video_file, Path):
        stem = video_file.stem
    elif isinstance(video_file, File):
        stem = Path(str(video_file.display_name)).stem

    output_subtitle_path = args.temp_dir / f"{stem}.srt"
    output_state_path = args.temp_dir / f"{stem}.json"

    return output_subtitle_path, output_state_path
