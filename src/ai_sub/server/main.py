"""CLI entrypoint for ai-sub-server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import logfire
import uvicorn

from ai_sub.config import LoggingSettings, find_config_file
from ai_sub.server.app import create_app
from ai_sub.server.config import ServerSettings
from ai_sub.telemetry import setup_logging


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for ai-sub-server.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="AI Sub central server coordinator.")
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=None,
        help="Path to server configuration YAML file (defaults to <hostname>.server.yaml or server.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    """Starts the ai-sub-server coordinator with uvicorn using pure YAML configuration."""
    args = parse_args()

    try:
        config_path = find_config_file("server", args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    setup_logging(LoggingSettings(level="info"))
    logfire.info(f"Loading server configuration from {config_path}")
    settings = ServerSettings.from_yaml(config_path)

    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        log_config=None,
    )


if __name__ == "__main__":
    main()
