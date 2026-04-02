"""End-to-end voice response orchestration and latency tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .playback import PlaybackResult, SpeakerPlayer
from .synthesis import SpeechSynthesizer, SynthesizedSpeech


@dataclass(frozen=True)
class VoiceExperienceTargets:
    """Target thresholds used to assess runtime voice quality."""

    max_synthesis_latency_ms: float = 1_500.0
    max_playback_dispatch_latency_ms: float = 200.0
    max_end_to_end_latency_ms: float = 1_700.0
    max_realtime_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.max_synthesis_latency_ms <= 0:
            raise ValueError("max_synthesis_latency_ms must be positive")
        if self.max_playback_dispatch_latency_ms <= 0:
            raise ValueError("max_playback_dispatch_latency_ms must be positive")
        if self.max_end_to_end_latency_ms <= 0:
            raise ValueError("max_end_to_end_latency_ms must be positive")
        if self.max_realtime_factor <= 0:
            raise ValueError("max_realtime_factor must be positive")


@dataclass(frozen=True)
class VoiceExperienceMetrics:
    """Latency indicators for one end-to-end spoken reply."""

    synthesis_latency_ms: float
    playback_dispatch_latency_ms: float | None
    playback_completion_latency_ms: float | None
    end_to_end_latency_ms: float | None
    end_to_end_completion_latency_ms: float | None
    audio_duration_ms: float
    realtime_factor: float | None


@dataclass(frozen=True)
class VoiceResponseResult:
    """Aggregated output of the voice response pipeline."""

    synthesis: SynthesizedSpeech
    playback: PlaybackResult
    metrics: VoiceExperienceMetrics
    targets: VoiceExperienceTargets
    meets_targets: bool


class VoiceResponsePipeline:
    """Synthesize text and immediately play it through the configured speaker."""

    def __init__(
        self,
        synthesizer: SpeechSynthesizer,
        player: SpeakerPlayer,
        targets: VoiceExperienceTargets | None = None,
    ) -> None:
        self._synthesizer = synthesizer
        self._player = player
        self._targets = targets or VoiceExperienceTargets()

    @property
    def targets(self) -> VoiceExperienceTargets:
        return self._targets

    def speak(
        self,
        text: str,
        *,
        voice_id: str | None = None,
        reference_audio: bytes | str | Path | None = None,
    ) -> VoiceResponseResult:
        synthesis = self._synthesizer.synthesize(
            text=text,
            voice_id=voice_id,
            reference_audio=reference_audio,
        )
        playback = self._player.play(synthesis)
        end_to_end_latency_ms = None
        if playback.dispatch_latency_ms is not None:
            end_to_end_latency_ms = synthesis.latency_ms + playback.dispatch_latency_ms

        end_to_end_completion_latency_ms = None
        if playback.completion_latency_ms is not None:
            end_to_end_completion_latency_ms = (
                synthesis.latency_ms + playback.completion_latency_ms
            )

        metrics = VoiceExperienceMetrics(
            synthesis_latency_ms=synthesis.latency_ms,
            playback_dispatch_latency_ms=playback.dispatch_latency_ms,
            playback_completion_latency_ms=playback.completion_latency_ms,
            end_to_end_latency_ms=end_to_end_latency_ms,
            end_to_end_completion_latency_ms=end_to_end_completion_latency_ms,
            audio_duration_ms=playback.duration_ms,
            realtime_factor=_compute_realtime_factor(
                synthesis_latency_ms=synthesis.latency_ms,
                playback_dispatch_latency_ms=playback.dispatch_latency_ms,
                audio_duration_ms=playback.duration_ms,
            ),
        )
        meets_targets = _meets_targets(metrics=metrics, targets=self._targets)
        return VoiceResponseResult(
            synthesis=synthesis,
            playback=playback,
            metrics=metrics,
            targets=self._targets,
            meets_targets=meets_targets,
        )


def _compute_realtime_factor(
    *,
    synthesis_latency_ms: float,
    playback_dispatch_latency_ms: float | None,
    audio_duration_ms: float,
) -> float | None:
    if playback_dispatch_latency_ms is None or audio_duration_ms <= 0:
        return None
    return (synthesis_latency_ms + playback_dispatch_latency_ms) / audio_duration_ms


def _meets_targets(metrics: VoiceExperienceMetrics, targets: VoiceExperienceTargets) -> bool:
    if metrics.synthesis_latency_ms > targets.max_synthesis_latency_ms:
        return False
    if (
        metrics.playback_dispatch_latency_ms is not None
        and metrics.playback_dispatch_latency_ms > targets.max_playback_dispatch_latency_ms
    ):
        return False
    if (
        metrics.end_to_end_latency_ms is not None
        and metrics.end_to_end_latency_ms > targets.max_end_to_end_latency_ms
    ):
        return False
    if metrics.realtime_factor is not None and metrics.realtime_factor > targets.max_realtime_factor:
        return False
    return True
