"""CLI entrypoint for ai-sub-server."""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import sys
from pathlib import Path

import logfire

from ai_sub.config import LoggingSettings, find_config_file
from ai_sub.telemetry import setup_logging


def _ensure_server_dependencies() -> None:
    """Verifies that all dependencies declared under the 'server' extra are installed."""
    missing: list[str] = []

    # Inspect distribution requirements defined for the 'server' extra in pyproject.toml
    for req in importlib.metadata.requires("ai-sub") or []:
        if re.search(r"""extra\s*==\s*['"]server['"]""", req):
            if match := re.match(r"^[\w.-]+", req):
                dist_name = match.group(0)
                try:
                    importlib.metadata.distribution(dist_name)
                except importlib.metadata.PackageNotFoundError:
                    missing.append(dist_name)

    if missing:
        packages_str = ", ".join(missing)
        print(
            f"Error: 'ai-sub-server' requires missing server dependencies ({packages_str}).\n"
            "Please install them via: pip install 'ai-sub[server]'",
            file=sys.stderr,
        )
        sys.exit(1)


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
    # Ensure server dependencies are installed before importing web server modules
    _ensure_server_dependencies()

    # Local imports prevent ModuleNotFoundError before the preflight check runs
    import uvicorn

    from ai_sub.server.app import create_app
    from ai_sub.server.config import ServerSettings

    args = parse_args()

    # Discover and validate the server YAML configuration path
    try:
        config_path = find_config_file("server", args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize Logfire telemetry and application logging
    setup_logging(LoggingSettings(level="info"))
    logfire.info(f"Loading server configuration from {config_path}")
    settings = ServerSettings.from_yaml(config_path)

    # Instantiate the FastAPI coordinator application and launch via Uvicorn
    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        log_config=None,
    )


if __name__ == "__main__":
    main()
