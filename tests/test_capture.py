from dataclasses import dataclass

from memento.audio import MicrophoneConfig, RealTimeMicrophone


@dataclass
class FakeInputDevice:
    frames: list[tuple[float, ...]]

    def __post_init__(self) -> None:
        self.configured_with = None
        self.open_calls = 0
        self.close_calls = 0

    def open(self, config: MicrophoneConfig) -> None:
        self.configured_with = config
        self.open_calls += 1

    def read(self, sample_count: int) -> tuple[float, ...]:
        frame = self.frames.pop(0)
        assert len(frame) == sample_count
        return frame

    def close(self) -> None:
        self.close_calls += 1


def test_real_time_microphone_streams_frames_with_expected_metadata() -> None:
    config = MicrophoneConfig(
        device_name="fake-mic",
        sample_rate_hz=100,
        frame_duration_ms=40,
    )
    device = FakeInputDevice(
        frames=[
            (0.1, 0.2, 0.3, 0.4),
            (0.0, 0.0, 0.0, 0.0),
        ]
    )
    microphone = RealTimeMicrophone(device=device, config=config)

    frames = microphone.stream(max_frames=2)

    assert device.configured_with == config
    assert device.open_calls == 1
    assert len(frames) == 2
    assert frames[0].frame_index == 0
    assert frames[0].started_at_ms == 0.0
    assert frames[0].duration_ms == 40.0
    assert frames[1].frame_index == 1
    assert frames[1].started_at_ms == 40.0


def test_real_time_microphone_generates_health_report() -> None:
    config = MicrophoneConfig(
        device_name="fake-mic",
        sample_rate_hz=100,
        frame_duration_ms=40,
        silence_threshold=0.05,
        clipping_threshold=0.95,
    )
    device = FakeInputDevice(
        frames=[
            (0.99, 0.2, -0.1, 0.1),
            (0.0, 0.0, 0.0, 0.0),
        ]
    )
    microphone = RealTimeMicrophone(device=device, config=config)

    microphone.stream(max_frames=2)
    report = microphone.health_report()
    log = microphone.health_log()

    assert report.device_name == "fake-mic"
    assert report.total_frames == 2
    assert report.clipped_frames == 1
    assert report.silent_frames == 1
    assert len(log) == 2
    assert log[0].clipped is True
    assert log[1].silent is True


def test_real_time_microphone_stop_closes_device() -> None:
    config = MicrophoneConfig(device_name="fake-mic", sample_rate_hz=100, frame_duration_ms=40)
    device = FakeInputDevice(frames=[(0.1, 0.1, 0.1, 0.1)])
    microphone = RealTimeMicrophone(device=device, config=config)

    microphone.capture_frame()
    microphone.stop()

    assert device.close_calls == 1
