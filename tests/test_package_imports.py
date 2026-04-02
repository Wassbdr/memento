import subprocess
import sys


def test_importing_audio_does_not_load_optional_memory_integrations() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import memento.audio; print('memento.memory.integrations' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "False"
