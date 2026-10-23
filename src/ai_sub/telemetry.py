"""Centralized logging and OpenTelemetry/Logfire telemetry configuration."""

from __future__ import annotations

import io
import os
import socket
import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import logfire
from rich.console import Console

from ai_sub.config import LoggingSettings, Settings


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
        return version("ai-sub")
    except PackageNotFoundError:
        return "4.0.0"


def setup_logging(
    settings: Settings | LoggingSettings,
    console: Console | None = None,
    service_name: str | None = None,
) -> None:
    """Configures Logfire telemetry, OpenTelemetry instrumentation, and console output.

    Args:
        settings: Application or logging settings.
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

    log_cfg = settings.log if isinstance(settings, Settings) else settings

    output_stream: Any = None
    if console is not None and log_cfg.progress:
        output_stream = RichConsoleLogfireWriter(console)

    # 1. Main Logfire client (Handles high-level spans and user logs for the terminal console)
    logfire.configure(
        console=logfire.ConsoleOptions(
            output=output_stream,
            min_log_level=log_cfg.level,
            include_timestamps=log_cfg.timestamps,
        ),
        service_name=resolved_service,
        service_version=app_version,
        send_to_logfire="if-token-present",
        scrubbing=None if log_cfg.scrub else False,
    )

    # 2. Silent Logfire client for low-level SDK and HTTP tracing (Google GenAI, HTTPX)
    # Telemetry is sent to Logfire cloud if a token is present, but kept off the terminal console
    silent_telemetry_client = logfire.configure(
        local=True,
        console=False,
        service_name=resolved_service,
        service_version=app_version,
        send_to_logfire="if-token-present",
        scrubbing=None if log_cfg.scrub else False,
    )
    silent_telemetry_client.instrument_httpx(capture_all=True)

    if hasattr(silent_telemetry_client, "instrument_google_genai"):
        try:
            silent_telemetry_client.instrument_google_genai()
        except Exception:
            pass
    elif hasattr(logfire, "instrument_google_genai"):
        try:
            tracer_prov = getattr(silent_telemetry_client, "tracer_provider", None) or getattr(
                silent_telemetry_client, "_tracer_provider", None
            )
            if tracer_prov is not None:
                logfire.instrument_google_genai(tracer_provider=tracer_prov)
            else:
                logfire.instrument_google_genai()
        except Exception:
            pass
