"""CLI entrypoint for ai-sub-worker."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import logfire
from pydantic import HttpUrl

from ai_sub.config import LoggingSettings
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
        "--config",
        type=Path,
        default=None,
        help="Path to optional worker configuration YAML file.",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="Base URL of the central ai-sub-server (e.g. 'https://sub.example.com').",
    )
    parser.add_argument(
        "--server-auth",
        type=str,
        default=None,
        help="Basic Auth credentials formatted as 'username:password'.",
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Unique identifier for this worker node.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum tasks to execute before exiting (0 runs until queue is empty or quotas exhausted).",
    )
    return parser.parse_args()


def main() -> None:
    """Initializes and runs the ephemeral ai-sub-worker loop."""
    args = parse_args()

    if args.config and args.config.is_file():
        logfire.info(f"Loading worker settings from {args.config}")
        settings = WorkerSettings.from_yaml(args.config)
    else:
        settings = WorkerSettings()

    if args.server_url:
        settings.server_url = HttpUrl(args.server_url)
    if args.server_auth:
        settings.server_auth = args.server_auth
    if args.worker_id:
        settings.worker_id = args.worker_id
    if args.max_tasks is not None:
        settings.max_tasks = args.max_tasks

    # Register worker_id as the OpenTelemetry service_name
    setup_logging(LoggingSettings(level="info"), service_name=settings.worker_id)

    runner = WorkerRunner(settings)
    exit_code = asyncio.run(runner.run())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
