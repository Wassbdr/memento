from memento.audio import AudioFrame, EnergyVAD, VoiceActivityConfig


def build_frame(frame_index: int, amplitude: float) -> AudioFrame:
    return AudioFrame(
        samples=(amplitude, amplitude, amplitude, amplitude),
        sample_rate_hz=100,
        frame_index=frame_index,
    )


def test_vad_ignores_silence_and_low_background_noise() -> None:
    vad = EnergyVAD(
        VoiceActivityConfig(
            speech_threshold=0.05,
            noise_floor=0.01,
            min_speech_frames=2,
            min_silence_frames=2,
        )
    )
    frames = [
        build_frame(0, 0.0),
        build_frame(1, 0.01),
        build_frame(2, 0.02),
        build_frame(3, 0.01),
    ]

    segments = vad.segment(frames)

    assert segments == ()


def test_vad_extracts_speech_segment_and_keeps_context_frame() -> None:
    vad = EnergyVAD(
        VoiceActivityConfig(
            speech_threshold=0.05,
            noise_floor=0.01,
            min_speech_frames=2,
            min_silence_frames=2,
            pre_roll_frames=1,
        )
    )
    frames = [
        build_frame(0, 0.01),
        build_frame(1, 0.08),
        build_frame(2, 0.08),
        build_frame(3, 0.09),
        build_frame(4, 0.0),
        build_frame(5, 0.0),
    ]

    segments = vad.segment(frames)

    assert len(segments) == 1
    assert segments[0].start_frame_index == 0
    assert segments[0].end_frame_index == 3
    assert segments[0].duration_ms == 160.0


def test_vad_classifies_speech_when_above_threshold() -> None:
    vad = EnergyVAD(VoiceActivityConfig(speech_threshold=0.05, noise_floor=0.01))

    speech_decision = vad.classify(build_frame(0, 0.06))
    silence_decision = vad.classify(build_frame(1, 0.01))

    assert speech_decision.is_speech is True
    assert silence_decision.is_speech is False
