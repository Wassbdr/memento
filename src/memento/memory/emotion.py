"""Emotion context utilities for adaptive memory recall."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol


@dataclass(frozen=True)
class EmotionalState:
    """Lightweight emotional context used to adapt clinical ranking."""

    label: str
    intensity: float = 0.0
    confidence: float = 0.0
    source: str = "inferred"

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("label must not be empty")
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError("intensity must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")


class EmotionalStateDetector(Protocol):
    """Detect one emotional state from one user utterance or transcript."""

    def detect(self, text: str) -> EmotionalState:
        """Return one emotional context for one text input."""


class RuleBasedEmotionalStateDetector:
    """Simple default detector used when no acoustic model is configured.

    The interface is designed to be replaced by a voice-tone model later on.
    """

    _AGITATION_TERMS = {
        "perdu",
        "perdue",
        "angoisse",
        "angoissee",
        "agite",
        "agitee",
        "panique",
        "stress",
        "stresse",
        "urgent",
        "aide",
    }
    _SADNESS_TERMS = {
        "triste",
        "seule",
        "seul",
        "pleure",
        "deprime",
        "deprimee",
        "manque",
    }
    _CALM_TERMS = {
        "calme",
        "tranquille",
        "ca va",
        "merci",
    }

    def detect(self, text: str) -> EmotionalState:
        tokens = set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))
        if not tokens:
            return EmotionalState(label="neutre", intensity=0.0, confidence=0.2, source="rule-based")

        agitation_hits = len(tokens & self._AGITATION_TERMS)
        sadness_hits = len(tokens & self._SADNESS_TERMS)
        calm_hits = len(tokens & self._CALM_TERMS)

        if agitation_hits > 0:
            intensity = min(0.4 + agitation_hits * 0.15, 1.0)
            return EmotionalState(label="agite", intensity=intensity, confidence=0.65, source="rule-based")
        if sadness_hits > 0:
            intensity = min(0.35 + sadness_hits * 0.12, 1.0)
            return EmotionalState(label="triste", intensity=intensity, confidence=0.6, source="rule-based")
        if calm_hits > 0:
            intensity = min(0.25 + calm_hits * 0.1, 1.0)
            return EmotionalState(label="calme", intensity=intensity, confidence=0.55, source="rule-based")

        return EmotionalState(label="neutre", intensity=0.1, confidence=0.3, source="rule-based")
