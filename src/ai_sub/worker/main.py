"""CLI entrypoint for ai-sub-worker."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import logfire

from ai_sub.config import LoggingSettings, find_config_file
from ai_sub.telemetry import setup_logging
from ai_sub.worker.config import WorkerSettings
from ai_sub.worker.runner import WorkerRunner


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for ai-sub-worker.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="AI Sub remote worker execution node.")
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=None,
        help="Path to worker configuration YAML file (defaults to <hostname>.worker.yaml or worker.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    """Initializes and runs the ephemeral ai-sub-worker loop."""
    args = parse_args()

    try:
        config_path = find_config_file("worker", args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    settings = WorkerSettings.from_yaml(config_path)

    # Register worker_id as the OpenTelemetry service_name
    setup_logging(LoggingSettings(level="info"), service_name=settings.worker_id)
    logfire.info(f"Loaded worker configuration from {config_path}")

    runner = WorkerRunner(settings)
    exit_code = asyncio.run(runner.run())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
