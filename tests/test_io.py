from io import BytesIO
import wave

from memento.audio import load_wav_bytes, speech_segment_from_wav_bytes, write_wav_file


def test_write_wav_file_and_reload_roundtrip(tmp_path) -> None:
    wav_path = tmp_path / "sample.wav"

    write_wav_file(
        path=wav_path,
        samples=(0.0, 0.5, -0.5, 1.0),
        sample_rate_hz=16_000,
    )
    decoded = load_wav_bytes(wav_path.read_bytes())

    assert decoded.sample_rate_hz == 16_000
    assert decoded.channels == 1
    assert len(decoded.samples) == 4
    assert decoded.samples[1] > 0.49
    assert decoded.samples[2] < -0.49


def test_speech_segment_from_wav_bytes_builds_single_segment() -> None:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8_000)
        wav_file.writeframes(b"\x00\x00\xff\x3f\x01\xc0")

    segment = speech_segment_from_wav_bytes(buffer.getvalue())

    assert len(segment.frames) == 1
    assert segment.frames[0].sample_rate_hz == 8_000
    assert len(segment.samples) == 3
