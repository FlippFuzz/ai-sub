"""CLI entrypoint for ai-sub-server."""

from __future__ import annotations

import sys
from pathlib import Path

import logfire
import uvicorn

from ai_sub.config import LoggingSettings
from ai_sub.server.app import create_app
from ai_sub.server.config import ServerSettings
from ai_sub.telemetry import setup_logging


def main() -> None:
    """Starts the ai-sub-server coordinator with uvicorn using pure YAML configuration."""
    setup_logging(LoggingSettings(level="info"))

    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("server.yaml")
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
