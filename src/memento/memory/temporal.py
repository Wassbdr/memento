"""Temporal helpers used by memory ranking and reorientation."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import re

_TIME_PATTERN = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")


def minutes_until_next_occurrence(schedule: str, *, reference_datetime: datetime) -> int | None:
    """Return minutes until next HH:MM occurrence parsed from one schedule string."""

    hour_minute = _extract_hour_minute(schedule)
    if hour_minute is None:
        return None

    hour, minute = hour_minute
    target = reference_datetime.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target < reference_datetime:
        target = target + timedelta(days=1)
    return int((target - reference_datetime).total_seconds() // 60)


def routine_temporal_label(minutes_until: int | None) -> str:
    if minutes_until is None:
        return "inconnu"
    if minutes_until <= 15:
        return "maintenant"
    if minutes_until <= 90:
        return "bientot"
    if minutes_until <= 360:
        return "aujourd_hui"
    return "plus_tard"


def routine_time_bonus(minutes_until: int | None) -> float:
    if minutes_until is None:
        return 0.0
    if minutes_until <= 15:
        return 0.25
    if minutes_until <= 90:
        return 0.18
    if minutes_until <= 360:
        return 0.1
    return 0.03


def days_since_date(value: str, *, reference_date: date) -> int | None:
    """Return non-negative day distance from one ISO date string."""

    text = value.strip()
    if not text:
        return None

    try:
        happened_on = datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None

    return max((reference_date - happened_on).days, 0)


def recency_bonus(days_since: int | None) -> float:
    if days_since is None:
        return 0.0
    if days_since <= 30:
        return 0.2
    if days_since <= 180:
        return 0.12
    if days_since <= 730:
        return 0.06
    return 0.01


def _extract_hour_minute(schedule: str) -> tuple[int, int] | None:
    match = _TIME_PATTERN.search(schedule)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))
