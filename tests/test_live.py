from memento.audio import AudioFrame, StreamingSpeechSegmenter, VoiceActivityConfig


def build_frame(frame_index: int, amplitude: float) -> AudioFrame:
    return AudioFrame(
        samples=(amplitude, amplitude, amplitude, amplitude),
        sample_rate_hz=100,
        frame_index=frame_index,
    )


def test_streaming_segmenter_emits_segment_after_silence() -> None:
    segmenter = StreamingSpeechSegmenter(
        VoiceActivityConfig(
            speech_threshold=0.05,
            noise_floor=0.01,
            min_speech_frames=2,
            min_silence_frames=2,
            pre_roll_frames=1,
        )
    )

    results = [
        segmenter.push_frame(frame)
        for frame in (
            build_frame(0, 0.01),
            build_frame(1, 0.08),
            build_frame(2, 0.08),
            build_frame(3, 0.09),
            build_frame(4, 0.0),
            build_frame(5, 0.0),
        )
    ]

    emitted = [segment for segment in results if segment is not None]
    assert len(emitted) == 1
    assert emitted[0].start_frame_index == 0
    assert emitted[0].end_frame_index == 3


def test_streaming_segmenter_flushes_open_segment() -> None:
    segmenter = StreamingSpeechSegmenter(
        VoiceActivityConfig(
            speech_threshold=0.05,
            noise_floor=0.01,
            min_speech_frames=2,
            min_silence_frames=2,
            pre_roll_frames=0,
        )
    )

    assert segmenter.push_frame(build_frame(0, 0.07)) is None
    assert segmenter.push_frame(build_frame(1, 0.09)) is None

    trailing = segmenter.flush()

    assert trailing is not None
    assert trailing.start_frame_index == 0
    assert trailing.end_frame_index == 1
