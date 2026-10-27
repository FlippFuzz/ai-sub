"""Worker-side daily API quota persistence and Pacific Time reset management."""

from __future__ import annotations

import json
import zoneinfo
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Self

import logfire
from pydantic import BaseModel, Field


def get_next_pacific_midnight() -> datetime:
    """Calculates the upcoming midnight in Pacific Time (00:00 PT) represented in UTC.

    Returns:
        datetime: Next 00:00 Pacific Time as an aware UTC datetime object.
    """
    pt_tz = zoneinfo.ZoneInfo("America/Los_Angeles")
    now_pt = datetime.now(pt_tz)
    tomorrow_pt_date = now_pt.date() + timedelta(days=1)
    midnight_pt = datetime.combine(tomorrow_pt_date, time.min, tzinfo=pt_tz)
    return midnight_pt.astimezone(timezone.utc)


class QuotaModelEntry(BaseModel):
    """Details for a single quota-exhausted model."""

    status: str = Field(default="quota_exhausted", description="Current status of the model.")
    reason: str = Field(
        default="Google Free Tier daily request limit reached (HTTP 429)",
        description="Reason for exhaustion.",
    )
    exhausted_at_utc: str = Field(description="ISO-8601 UTC timestamp when exhaustion occurred.")
    resets_at_utc: str = Field(description="ISO-8601 UTC timestamp when the quota resets.")


class QuotaState(BaseModel):
    """Root model for the human-readable .ai-sub-quota.json file."""

    comment: str = Field(
        alias="_comment",
        default="Tracks daily API quota exhaustion for Gemini models. Daily quotas reset at 00:00 Pacific Time.",
        description="Human-readable file explanation.",
    )
    models: dict[str, QuotaModelEntry] = Field(
        default_factory=dict,
        description="Mapping of exhausted model identifiers to their quota status.",
    )


class WorkerQuotaTracker:
    """Manages reading, pruning, and recording daily quota exhaustion on the worker."""

    quota_file: Path
    state: QuotaState

    def __init__(self, quota_file: Path | str = ".ai-sub-quota.json") -> None:
        """Initializes the worker quota tracker.

        Args:
            quota_file: Path to the JSON state file. Defaults to '.ai-sub-quota.json'.
        """
        self.quota_file = Path(quota_file)
        self.state = QuotaState()
        self.load()

    def load(self) -> Self:
        """Loads and prunes expired model entries from disk.

        Returns:
            Self: Loaded tracker instance.
        """
        if not self.quota_file.is_file():
            self.state = QuotaState()
            return self

        try:
            with open(self.quota_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.state = QuotaState.model_validate(data)
        except Exception as e:
            logfire.warning(f"Could not parse quota file {self.quota_file.name}: {e}. Initializing fresh.")
            self.state = QuotaState()
            return self

        now_utc = datetime.now(timezone.utc)
        active_models: dict[str, QuotaModelEntry] = {}

        for model_name, entry in self.state.models.items():
            try:
                resets_at = datetime.fromisoformat(entry.resets_at_utc)
                if resets_at > now_utc:
                    active_models[model_name] = entry
                else:
                    logfire.info(f"Model quota reset for {model_name} (passed 00:00 PT). Re-enabling.")
            except Exception:
                continue

        if len(active_models) != len(self.state.models):
            self.state.models = active_models
            self.save()

        return self

    def save(self) -> None:
        """Persists the current quota state to disk."""
        try:
            self.quota_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.quota_file, "w", encoding="utf-8") as f:
                json.dump(self.state.model_dump(by_alias=True, mode="json"), f, indent=2)
        except Exception as e:
            logfire.warning(f"Failed to write quota file {self.quota_file.name}: {e}")

    def is_model_exhausted(self, model: str) -> bool:
        """Checks if a model is currently marked as quota-exhausted.

        Args:
            model: Gemini model identifier.

        Returns:
            bool: True if model is exhausted; False otherwise.
        """
        self.load()
        return model in self.state.models

    def mark_exhausted(
        self,
        model: str,
        reason: str = "Google Free Tier daily request limit reached (HTTP 429)",
    ) -> None:
        """Marks a model as exhausted until the next midnight Pacific Time.

        Args:
            model: Gemini model identifier.
            reason: Explanation of the quota failure.
        """
        now_utc = datetime.now(timezone.utc)
        resets_at_utc = get_next_pacific_midnight()

        self.state.models[model] = QuotaModelEntry(
            status="quota_exhausted",
            reason=reason,
            exhausted_at_utc=now_utc.isoformat(),
            resets_at_utc=resets_at_utc.isoformat(),
        )
        self.save()
        logfire.warning(f"Marked model '{model}' as quota-exhausted until {resets_at_utc.isoformat()} (00:00 PT).")

    def filter_available_models(self, candidate_models: list[str]) -> list[str]:
        """Filters a candidate model list down to only unexhausted models.

        Args:
            candidate_models: List of model identifiers.

        Returns:
            list[str]: Filtered list of models with remaining quota.
        """
        self.load()
        return [m for m in candidate_models if m not in self.state.models]
