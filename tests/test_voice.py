from memento.audio import (
    PlaybackResult,
    SpeakerConfig,
    SpeakerPlayer,
    SpeechSynthesizer,
    SynthesizedSpeech,
    TextToSpeechConfig,
    VoiceExperienceTargets,
    VoiceResponsePipeline,
)


class FakeSynthesizerBackend:
    def synthesize(
        self,
        text: str,
        model_name: str,
        voice_id: str | None,
        response_format: str,
        reference_audio_base64: str | None,
    ):
        return type("Result", (), {
            "audio_bytes": b"wav-bytes",
            "response_format": "wav",
            "sample_rate_hz": 16_000,
            "channels": 1,
        })()


class FakeSpeakerDevice:
    def play(
        self,
        samples: tuple[float, ...],
        sample_rate_hz: int,
        channels: int,
        *,
        device_name: str,
        blocking: bool,
    ) -> None:
        return None

    def stop(self) -> None:
        return None


def test_voice_response_pipeline_aggregates_metrics(monkeypatch) -> None:
    synthesizer = SpeechSynthesizer(
        backend=FakeSynthesizerBackend(),
        config=TextToSpeechConfig(response_format="wav"),
    )
    player = SpeakerPlayer(device=FakeSpeakerDevice(), config=SpeakerConfig())
    pipeline = VoiceResponsePipeline(
        synthesizer=synthesizer,
        player=player,
        targets=VoiceExperienceTargets(
            max_synthesis_latency_ms=5_000,
            max_playback_dispatch_latency_ms=5_000,
            max_end_to_end_latency_ms=5_000,
            max_realtime_factor=50.0,
        ),
    )

    def fake_synthesize(*args, **kwargs) -> SynthesizedSpeech:
        return SynthesizedSpeech(
            text="Bonjour",
            audio_bytes=(
                b"RIFF&\x00\x00\x00WAVEfmt "
                b"\x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00"
                b"\x02\x00\x10\x00data\x02\x00\x00\x00\x00\x00"
            ),
            response_format="wav",
            sample_rate_hz=None,
            channels=1,
            latency_ms=120.0,
            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            voice_id=None,
            character_count=7,
        )

    def fake_play(_speech: SynthesizedSpeech) -> PlaybackResult:
        return PlaybackResult(
            duration_ms=500.0,
            dispatch_latency_ms=40.0,
            completion_latency_ms=None,
            sample_rate_hz=16_000,
            channels=1,
            interrupted_previous=True,
            blocking=False,
        )

    monkeypatch.setattr(synthesizer, "synthesize", fake_synthesize)
    monkeypatch.setattr(player, "play", fake_play)

    result = pipeline.speak("Bonjour")

    assert result.metrics.synthesis_latency_ms == 120.0
    assert result.metrics.playback_dispatch_latency_ms == 40.0
    assert result.metrics.playback_completion_latency_ms is None
    assert result.metrics.end_to_end_latency_ms == 160.0
    assert result.metrics.end_to_end_completion_latency_ms is None
    assert result.metrics.audio_duration_ms == 500.0
    assert result.metrics.realtime_factor == 0.32
    assert result.meets_targets is True


def test_voice_response_pipeline_skips_dispatch_targets_when_playback_is_blocking(monkeypatch) -> None:
    synthesizer = SpeechSynthesizer(
        backend=FakeSynthesizerBackend(),
        config=TextToSpeechConfig(response_format="wav"),
    )
    player = SpeakerPlayer(device=FakeSpeakerDevice(), config=SpeakerConfig(blocking=True))
    pipeline = VoiceResponsePipeline(
        synthesizer=synthesizer,
        player=player,
        targets=VoiceExperienceTargets(
            max_synthesis_latency_ms=5_000,
            max_playback_dispatch_latency_ms=1.0,
            max_end_to_end_latency_ms=1.0,
            max_realtime_factor=1.0,
        ),
    )

    def fake_synthesize(*args, **kwargs) -> SynthesizedSpeech:
        return SynthesizedSpeech(
            text="Bonjour",
            audio_bytes=(
                b"RIFF&\x00\x00\x00WAVEfmt "
                b"\x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00"
                b"\x02\x00\x10\x00data\x02\x00\x00\x00\x00\x00"
            ),
            response_format="wav",
            sample_rate_hz=None,
            channels=1,
            latency_ms=120.0,
            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            voice_id=None,
            character_count=7,
        )

    def fake_play(_speech: SynthesizedSpeech) -> PlaybackResult:
        return PlaybackResult(
            duration_ms=500.0,
            dispatch_latency_ms=None,
            completion_latency_ms=540.0,
            sample_rate_hz=16_000,
            channels=1,
            interrupted_previous=False,
            blocking=True,
        )

    monkeypatch.setattr(synthesizer, "synthesize", fake_synthesize)
    monkeypatch.setattr(player, "play", fake_play)

    result = pipeline.speak("Bonjour")

    assert result.metrics.playback_dispatch_latency_ms is None
    assert result.metrics.playback_completion_latency_ms == 540.0
    assert result.metrics.end_to_end_latency_ms is None
    assert result.metrics.end_to_end_completion_latency_ms == 660.0
    assert result.metrics.realtime_factor is None
    assert result.meets_targets is True
