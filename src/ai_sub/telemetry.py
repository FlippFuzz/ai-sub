"""Centralized logging and OpenTelemetry/Logfire telemetry configuration."""

from __future__ import annotations

import io
import os
import socket
import warnings
from importlib.metadata import version
from typing import Any

import logfire
from rich.console import Console

from ai_sub.config import LoggingSettings
from ai_sub.standalone_config import StandaloneLoggingSettings


class RichConsoleLogfireWriter(io.TextIOBase):
    """Routes Logfire console messages through Rich Console to prevent progress bar corruption."""

    def __init__(self, console: Console) -> None:
        """Initializes the console writer.

        Args:
            console: Target Rich console.
        """
        self.console = console

    def write(self, message: str) -> int:
        """Writes message through Rich console, pausing and redrawing progress cleanly.

        Args:
            message: Formatted log message string.

        Returns:
            int: Length of message written.
        """
        if cleaned := message.rstrip("\r\n"):
            self.console.print(cleaned, markup=False, highlight=False)
        return len(message)

    def flush(self) -> None:
        """Flushes the writer. This is a no-op."""
        pass


def get_app_version() -> str:
    """Retrieves installed package version string.

    Returns:
        str: Package version or fallback identifier.
    """
    try:
        from ai_sub._version import __version__  # pyright: ignore[reportMissingImports]

        return __version__
    except ImportError:
        pass

    try:
        return version("ai-sub")
    except Exception:
        return "0.0.0.dev0"


def setup_logging(
    settings: LoggingSettings,
    console: Console | None = None,
    service_name: str | None = None,
) -> None:
    """Configures Logfire telemetry, OpenTelemetry instrumentation, and console output.

    Args:
        settings: Logging settings configuration.
        console: Optional Rich console for synchronized terminal rendering.
        service_name: Optional custom service name (defaults to machine hostname).
    """
    warnings.filterwarnings("ignore", message=".*non-text parts in the response.*")

    resolved_service = service_name or socket.gethostname()
    app_version = get_app_version()

    # Instruct OpenTelemetry to use hostname and capture full conversation messages in spans
    os.environ.setdefault("OTEL_SERVICE_NAME", resolved_service)
    os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY")
    os.environ.setdefault("OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental")

    output_stream: Any = None
    if console is not None and isinstance(settings, StandaloneLoggingSettings) and settings.progress:
        output_stream = RichConsoleLogfireWriter(console)

    token = settings.token.get_secret_value() if settings.token is not None else None

    # 1. Main Logfire client (Handles high-level spans and user logs for the terminal console)
    logfire.configure(
        token=token,
        console=logfire.ConsoleOptions(
            output=output_stream,
            min_log_level=settings.level,
            include_timestamps=settings.timestamps,
        ),
        service_name=resolved_service,
        service_version=app_version,
        send_to_logfire="if-token-present",
        scrubbing=None if settings.scrub else False,
        distributed_tracing=True,
    )

    # 2. Silent Logfire client for low-level SDK and HTTP tracing (Google GenAI, HTTPX)
    silent_telemetry_client = logfire.configure(
        local=True,
        token=token,
        console=False,
        service_name=resolved_service,
        service_version=app_version,
        send_to_logfire="if-token-present",
        scrubbing=None if settings.scrub else False,
        distributed_tracing=True,
    )

    # 3. Completely suppress noisy FastAPI instrumentation that is generated even without being explicitly enabled.
    logfire.suppress_scopes("fastapi")

    silent_telemetry_client.instrument_httpx(capture_all=True)
    silent_telemetry_client.instrument_google_genai()
