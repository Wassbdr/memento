#!/usr/bin/env bash
# Rebuild everything in showcase/ from a clean clone.
#
# Requirements: docker (compose v2), uv, ffmpeg, ~8 GB free disk for images and
# models, internet on first run (Docker images, Ollama qwen2.5:0.5b, Whisper
# small, Qwen3-TTS 0.6B, MiniLM embeddings, matplotlib). No GPU needed.
#
# Steps: start Neo4j + ChromaDB + Ollama, seed the fictitious patient (seed 7)
# into the real stores, synthesize the question WAVs with Qwen3-TTS, run the
# persistence check (ask, restart the stack, ask again), run the voice round
# trip WAV -> Whisper -> graph RAG -> LLM -> TTS, then render demo.mp4,
# demo.gif, poster.png and showcase.json from the recorded run.
#
# Env: SEED (default 7), SKIP_QUESTIONS=1 to reuse committed question WAVs,
# TEXT_ONLY=1 to skip answer TTS.
set -euo pipefail

cd "$(dirname "$0")/.."
SEED="${SEED:-7}"
export PYTHONHASHSEED=0

for tool in docker uv ffmpeg; do
  command -v "$tool" >/dev/null || { echo "missing required tool: $tool" >&2; exit 1; }
done

docker compose up -d
docker compose exec -T ollama ollama pull "${MEMENTO_LLM_MODEL:-qwen2.5:0.5b}"
uv sync --all-extras

demo() { uv run python -m memento.demo --seed "$SEED" "$@"; }

demo seed
if [[ "${SKIP_QUESTIONS:-0}" != "1" ]]; then
  demo questions --output-dir showcase/questions
fi
mkdir -p showcase/run
demo persistence --compose-dir . --output showcase/run/persistence.json
roundtrip_args=(--question-dir showcase/questions --output-dir showcase/run)
[[ "${TEXT_ONLY:-0}" == "1" ]] && roundtrip_args+=(--text-only)
demo roundtrip "${roundtrip_args[@]}"
uv run --with matplotlib==3.10.3 python showcase/render.py
echo "showcase rebuilt: $(ls showcase)"
