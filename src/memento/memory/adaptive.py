"""Adaptive clinical weighting and lightweight emotional context detection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Protocol


@dataclass(frozen=True)
class RealTimeEmotionalState:
    """One emotional signal inferred from the current interaction."""

    label: str
    intensity: float
    confidence: float
    source: str = ""

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("label must not be empty")
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError("intensity must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass(frozen=True)
class ClinicalWeightProfile:
    """Resolved weight profile used for one recall scoring pass."""

    name: str = "baseline"
    context_weight_factor: float = 0.03
    affective_weight_factor: float = 0.2
    trusted_person_match_bonus: float = 0.05
    max_trusted_bonus: float = 0.15
    anchor_match_bonus: float = 0.06
    max_anchor_bonus: float = 0.18
    routine_bonus_multiplier: float = 1.0
    recency_bonus_multiplier: float = 1.0
    staleness_penalty_multiplier: float = 1.0
    signals: tuple[str, ...] = ()


class EmotionalStateDetector(Protocol):
    """Detect one emotional state from user interaction text."""

    def detect(self, utterance: str) -> RealTimeEmotionalState:
        """Return one detected emotional state."""


class LexicalEmotionDetector:
    """Small rule-based detector used as a fallback before audio ML integration."""

    _AGITATED_TERMS = {
        "perdu",
        "perdue",
        "angoisse",
        "angoissee",
        "anxieux",
        "anxieuse",
        "stress",
        "panique",
        "agite",
        "agitee",
        "vite",
        "urgence",
    }
    _SAD_TERMS = {
        "triste",
        "seul",
        "seule",
        "pleure",
        "deprime",
        "deprimee",
        "melancolie",
        "manque",
    }
    _CALM_TERMS = {
        "calme",
        "serein",
        "sereine",
        "tranquille",
        "doux",
        "douce",
    }

    def detect(self, utterance: str) -> RealTimeEmotionalState:
        tokens = re.findall(r"\w+", utterance.lower(), flags=re.UNICODE)
        if not tokens:
            return RealTimeEmotionalState(
                label="neutral",
                intensity=0.0,
                confidence=0.5,
                source="lexical",
            )

        agitated = sum(1 for token in tokens if token in self._AGITATED_TERMS)
        sad = sum(1 for token in tokens if token in self._SAD_TERMS)
        calm = sum(1 for token in tokens if token in self._CALM_TERMS)

        if agitated == sad == calm == 0:
            return RealTimeEmotionalState(
                label="neutral",
                intensity=0.1,
                confidence=0.55,
                source="lexical",
            )

        best_label = "agitated"
        best_score = agitated
        if sad > best_score:
            best_label = "sad"
            best_score = sad
        if calm > best_score:
            best_label = "calm"
            best_score = calm

        normalized_intensity = min(best_score / max(len(tokens), 1) * 4.0, 1.0)
        confidence = min(0.6 + normalized_intensity * 0.35, 0.98)
        return RealTimeEmotionalState(
            label=best_label,
            intensity=round(normalized_intensity, 4),
            confidence=round(confidence, 4),
            source="lexical",
        )


def resolve_dynamic_clinical_weights(
    *,
    reference_datetime: datetime,
    emotional_state: RealTimeEmotionalState | None = None,
) -> ClinicalWeightProfile:
    """Resolve one adaptive profile from time-of-day and emotion cues."""

    profile = ClinicalWeightProfile()
    signals: list[str] = []

    hour = reference_datetime.hour
    if hour >= 21 or hour < 6:
        profile = _with(
            profile,
            name="night_reassurance",
            routine_bonus_multiplier=1.35,
            recency_bonus_multiplier=0.85,
            anchor_match_bonus=0.08,
            max_anchor_bonus=0.22,
        )
        signals.append("time_night")
    elif 6 <= hour < 11:
        profile = _with(
            profile,
            name="morning_reassurance",
            routine_bonus_multiplier=1.2,
        )
        signals.append("time_morning")
    elif 18 <= hour < 21:
        profile = _with(
            profile, name="evening_support", trusted_person_match_bonus=0.06, max_trusted_bonus=0.18
        )
        signals.append("time_evening")

    if emotional_state is not None:
        label = emotional_state.label.strip().lower()
        if label in {"agitated", "anxious", "confused"}:
            profile = _with(
                profile,
                name=f"{profile.name}+agitated",
                routine_bonus_multiplier=round(profile.routine_bonus_multiplier * 1.4, 4),
                trusted_person_match_bonus=round(profile.trusted_person_match_bonus * 1.25, 4),
                max_trusted_bonus=round(profile.max_trusted_bonus * 1.2, 4),
                anchor_match_bonus=round(profile.anchor_match_bonus * 1.3, 4),
                max_anchor_bonus=round(profile.max_anchor_bonus * 1.2, 4),
                affective_weight_factor=round(profile.affective_weight_factor * 0.75, 4),
                staleness_penalty_multiplier=round(profile.staleness_penalty_multiplier * 1.2, 4),
            )
            signals.append("emotion_agitated")
        elif label in {"sad", "distressed"}:
            profile = _with(
                profile,
                name=f"{profile.name}+sad",
                affective_weight_factor=round(profile.affective_weight_factor * 1.2, 4),
                trusted_person_match_bonus=round(profile.trusted_person_match_bonus * 1.15, 4),
                max_trusted_bonus=round(profile.max_trusted_bonus * 1.1, 4),
            )
            signals.append("emotion_sad")
        elif label in {"calm", "neutral"}:
            profile = _with(
                profile,
                name=f"{profile.name}+calm",
                recency_bonus_multiplier=round(profile.recency_bonus_multiplier * 1.05, 4),
            )
            signals.append("emotion_calm")

    if not signals:
        signals.append("baseline")

    return _with(profile, signals=tuple(signals))


def _with(profile: ClinicalWeightProfile, **updates) -> ClinicalWeightProfile:
    data = {
        "name": profile.name,
        "context_weight_factor": profile.context_weight_factor,
        "affective_weight_factor": profile.affective_weight_factor,
        "trusted_person_match_bonus": profile.trusted_person_match_bonus,
        "max_trusted_bonus": profile.max_trusted_bonus,
        "anchor_match_bonus": profile.anchor_match_bonus,
        "max_anchor_bonus": profile.max_anchor_bonus,
        "routine_bonus_multiplier": profile.routine_bonus_multiplier,
        "recency_bonus_multiplier": profile.recency_bonus_multiplier,
        "staleness_penalty_multiplier": profile.staleness_penalty_multiplier,
        "signals": profile.signals,
    }
    data.update(updates)
    return ClinicalWeightProfile(**data)
