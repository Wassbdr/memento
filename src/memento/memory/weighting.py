"""Dynamic clinical weighting profiles used by memory recall ranking."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from .emotion import EmotionalState


@dataclass(frozen=True)
class ClinicalWeightProfile:
    """Weight profile that controls ranking signal importance."""

    name: str = "baseline"
    semantic_weight: float = 1.0
    context_weight_factor: float = 0.03
    affective_weight_factor: float = 0.2
    trusted_person_match_bonus: float = 0.05
    max_trusted_bonus: float = 0.15
    anchor_match_bonus: float = 0.06
    max_anchor_bonus: float = 0.18
    routine_time_multiplier: float = 1.0
    recency_multiplier: float = 1.0
    staleness_penalty_multiplier: float = 1.0


@dataclass(frozen=True)
class ResolvedWeightProfile:
    """Resolved profile plus explainability signals for one recall call."""

    profile: ClinicalWeightProfile
    signals: tuple[str, ...]


def resolve_weight_profile(
    *,
    reference_datetime: datetime,
    emotional_state: EmotionalState | None,
) -> ResolvedWeightProfile:
    """Build one adaptive weight profile for the current context."""

    profile = ClinicalWeightProfile()
    signals: list[str] = ["profile_baseline"]

    hour = reference_datetime.hour
    if hour >= 21 or hour < 6:
        profile = replace(
            profile,
            name="night_support",
            routine_time_multiplier=1.25,
            recency_multiplier=0.85,
            anchor_match_bonus=0.075,
            max_anchor_bonus=0.22,
        )
        signals.append("night_hours")

    if emotional_state is None:
        return ResolvedWeightProfile(profile=profile, signals=tuple(signals))

    normalized_label = emotional_state.label.strip().lower()
    intensity = max(0.0, min(1.0, emotional_state.intensity))
    confidence = max(0.0, min(1.0, emotional_state.confidence))
    confidence_factor = 0.5 + confidence * 0.5

    if normalized_label in {"agite", "agitee", "anxieux", "anxieuse", "confus", "confuse"}:
        multiplier = 1.0 + intensity * 0.4 * confidence_factor
        profile = replace(
            profile,
            name="agitation_support",
            routine_time_multiplier=profile.routine_time_multiplier * multiplier,
            anchor_match_bonus=profile.anchor_match_bonus * multiplier,
            max_anchor_bonus=profile.max_anchor_bonus * multiplier,
            affective_weight_factor=profile.affective_weight_factor * 0.7,
            recency_multiplier=profile.recency_multiplier * 0.75,
        )
        signals.append("emotion_agite")
    elif normalized_label in {"triste", "deprime", "deprimee"}:
        multiplier = 1.0 + intensity * 0.25 * confidence_factor
        profile = replace(
            profile,
            name="sadness_support",
            trusted_person_match_bonus=profile.trusted_person_match_bonus * multiplier,
            max_trusted_bonus=profile.max_trusted_bonus * multiplier,
            affective_weight_factor=profile.affective_weight_factor * (1.0 + intensity * 0.3),
            anchor_match_bonus=profile.anchor_match_bonus * 1.1,
        )
        signals.append("emotion_triste")
    elif normalized_label in {"calme", "apaise", "apaisee"}:
        profile = replace(
            profile,
            name="calm_context",
            semantic_weight=1.05,
            recency_multiplier=1.1,
        )
        signals.append("emotion_calme")
    else:
        signals.append(f"emotion_{normalized_label}")

    if emotional_state.source.strip():
        signals.append(f"emotion_source_{emotional_state.source.strip().lower()}")

    return ResolvedWeightProfile(profile=profile, signals=tuple(signals))
