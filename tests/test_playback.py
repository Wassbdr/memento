import struct
from io import BytesIO
from time import sleep
import wave

from memento.audio import (
    SpeakerConfig,
    SpeakerPlayer,
    SynthesizedSpeech,
    decode_audio_bytes,
)


class FakeOutputDevice:
    def __init__(self, *, stop_result: bool | None = None, play_delay_s: float = 0.0) -> None:
        self.calls: list[dict[str, object]] = []
        self.stop_calls = 0
        self.stop_result = stop_result
        self.play_delay_s = play_delay_s

    def play(
        self,
        samples: tuple[float, ...],
        sample_rate_hz: int,
        channels: int,
        *,
        device_name: str,
        blocking: bool,
    ) -> None:
        self.calls.append(
            {
                "samples": samples,
                "sample_rate_hz": sample_rate_hz,
                "channels": channels,
                "device_name": device_name,
                "blocking": blocking,
            }
        )
        if self.play_delay_s:
            sleep(self.play_delay_s)

    def stop(self) -> bool | None:
        self.stop_calls += 1
        return self.stop_result


def build_wav_bytes() -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8_000)
        wav_file.writeframes(b"\x00\x00\xff\x3f\x01\xc0")
    return buffer.getvalue()


def test_decode_audio_bytes_supports_wav() -> None:
    decoded = decode_audio_bytes(build_wav_bytes(), "wav")

    assert decoded.sample_rate_hz == 8_000
    assert decoded.channels == 1
    assert len(decoded.samples) == 3
    assert decoded.duration_ms > 0


def test_decode_audio_bytes_supports_pcm() -> None:
    decoded = decode_audio_bytes(
        struct.pack("<ff", 0.25, -0.25),
        "pcm",
        sample_rate_hz=16_000,
        channels=1,
    )

    assert decoded.sample_rate_hz == 16_000
    assert decoded.samples == (0.25, -0.25)


def test_speaker_player_interrupts_previous_playback_and_dispatches_audio() -> None:
    device = FakeOutputDevice(stop_result=True)
    player = SpeakerPlayer(
        device=device,
        config=SpeakerConfig(device_name="kitchen-speaker", blocking=False, interrupt_current=True),
    )
    speech = SynthesizedSpeech(
        text="Bonjour",
        audio_bytes=build_wav_bytes(),
        response_format="wav",
        sample_rate_hz=None,
        channels=1,
        latency_ms=42.0,
        model_name="voxtral-tts-2603",
        voice_id=None,
        character_count=7,
    )

    result = player.play(speech)

    assert device.stop_calls == 1
    assert device.calls[0]["sample_rate_hz"] == 8_000
    assert device.calls[0]["device_name"] == "kitchen-speaker"
    assert result.duration_ms > 0
    assert result.dispatch_latency_ms is not None
    assert result.dispatch_latency_ms >= 0
    assert result.completion_latency_ms is None
    assert result.interrupted_previous is True


def test_speaker_player_does_not_claim_an_interruption_without_device_confirmation() -> None:
    device = FakeOutputDevice(stop_result=None)
    player = SpeakerPlayer(
        device=device,
        config=SpeakerConfig(device_name="bedroom-speaker", blocking=False, interrupt_current=True),
    )
    speech = SynthesizedSpeech(
        text="Bonjour",
        audio_bytes=build_wav_bytes(),
        response_format="wav",
        sample_rate_hz=None,
        channels=1,
        latency_ms=42.0,
        model_name="voxtral-tts-2603",
        voice_id=None,
        character_count=7,
    )

    result = player.play(speech)

    assert device.stop_calls == 1
    assert result.interrupted_previous is False


def test_blocking_playback_tracks_completion_latency_without_overloading_dispatch_metrics() -> None:
    device = FakeOutputDevice(play_delay_s=0.01)
    player = SpeakerPlayer(
        device=device,
        config=SpeakerConfig(device_name="office-speaker", blocking=True, interrupt_current=False),
    )
    speech = SynthesizedSpeech(
        text="Bonjour",
        audio_bytes=build_wav_bytes(),
        response_format="wav",
        sample_rate_hz=None,
        channels=1,
        latency_ms=42.0,
        model_name="voxtral-tts-2603",
        voice_id=None,
        character_count=7,
    )

    result = player.play(speech)

    assert result.dispatch_latency_ms is None
    assert result.completion_latency_ms is not None
    assert result.completion_latency_ms >= 10
    assert result.blocking is True
