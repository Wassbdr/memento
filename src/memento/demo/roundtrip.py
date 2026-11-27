"""Microphone-free voice round trip: WAV question -> Whisper -> graph RAG -> LLM -> TTS."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter
import wave

from memento.audio import (
    OpenAIWhisperBackend,
    QwenTTSBackend,
    SpeechSynthesizer,
    TextToSpeechConfig,
    WhisperConfig,
)
from memento.memory import PatientMemorySnapshot

from .fictional import DEFAULT_DEMO_SEED, DemoQuestion, build_fictional_snapshot, demo_questions
from .stack import StackConfig, build_orchestrator, build_store_engine, hardware_summary

DEFAULT_DEMO_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_DEMO_WHISPER_MODEL = "small"
QUESTION_VOICE = "Serena"
ANSWER_VOICE = "Vivian"


@dataclass(frozen=True)
class VoiceModels:
    """Model choices for the CPU round trip (smallest variants the project supports)."""

    whisper_model: str = DEFAULT_DEMO_WHISPER_MODEL
    tts_model: str = DEFAULT_DEMO_TTS_MODEL


def build_tts(model_name: str, voice: str) -> SpeechSynthesizer:
    config = TextToSpeechConfig(
        model_name=model_name,
        voice_id=voice,
        language="French",
        device_map="cpu",
        dtype="float32",
        instruction="Parle lentement, d'une voix calme et chaleureuse.",
    )
    return SpeechSynthesizer(backend=QwenTTSBackend(config=config), config=config)


def seed_stores(*, seed: int = DEFAULT_DEMO_SEED, stack: StackConfig | None = None) -> dict[str, object]:
    """Write the fictitious patient into Neo4j and ChromaDB and return a sync summary."""

    snapshot = build_fictional_snapshot(seed)
    started = perf_counter()
    with build_store_engine(stack) as engine:
        report = engine.sync_snapshot(snapshot)
    return {
        "patient_id": report.patient_id,
        "graph_nodes": report.graph_nodes_written,
        "graph_relations": report.graph_relations_written,
        "indexed_documents": report.indexed_documents,
        "sync_seconds": round(perf_counter() - started, 2),
    }


def make_question_wavs(output_dir: Path, *, seed: int = DEFAULT_DEMO_SEED, models: VoiceModels | None = None) -> list[Path]:
    """Synthesize the demo questions with the project's own TTS (one-off, committed)."""

    models = models or VoiceModels()
    output_dir.mkdir(parents=True, exist_ok=True)
    synthesizer = build_tts(models.tts_model, QUESTION_VOICE)
    paths = []
    for question in demo_questions(build_fictional_snapshot(seed)):
        speech = synthesizer.synthesize(question.text)
        path = output_dir / f"{question.question_id}.wav"
        path.write_bytes(speech.audio_bytes)
        paths.append(path)
        print(f"[questions] {path.name}: {speech.latency_ms / 1000:.1f}s", flush=True)
    return paths


def run_roundtrip(
    question_dir: Path,
    output_dir: Path,
    *,
    seed: int = DEFAULT_DEMO_SEED,
    models: VoiceModels | None = None,
    stack: StackConfig | None = None,
    synthesize_answers: bool = True,
) -> dict[str, object]:
    """Run every question WAV through the real chain and save answers + timings.

    The memory is read from the stores as they are; nothing is re-synced here, so
    the answers prove that the graph persisted in Neo4j/ChromaDB.
    """

    models = models or VoiceModels()
    stack = stack or StackConfig()
    snapshot = build_fictional_snapshot(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_seconds: dict[str, float] = {}
    started = perf_counter()
    whisper = OpenAIWhisperBackend(WhisperConfig(model_name=models.whisper_model, device="cpu", language="fr"))
    whisper._get_model()
    load_seconds["whisper"] = round(perf_counter() - started, 2)

    tts = None
    if synthesize_answers:
        started = perf_counter()
        tts = build_tts(models.tts_model, ANSWER_VOICE)
        tts.synthesize("Bonjour.")
        load_seconds["tts_with_warmup"] = round(perf_counter() - started, 2)

    engine = build_store_engine(stack)
    orchestrator = build_orchestrator(engine, stack)
    orchestrator.respond(snapshot.patient.patient_id, "Bonjour")

    turns = []
    for question in demo_questions(snapshot):
        wav_path = question_dir / f"{question.question_id}.wav"
        turns.append(
            _run_one_turn(
                question=question,
                wav_path=wav_path,
                output_dir=output_dir,
                snapshot=snapshot,
                whisper=whisper,
                orchestrator=orchestrator,
                tts=tts,
            )
        )
        print(f"[roundtrip] {question.question_id}: {json.dumps(turns[-1]['timings_s'])}", flush=True)

    graph = engine.graph_store.graph_for_patient(snapshot.patient.patient_id)
    engine.close()
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "hardware": hardware_summary(),
        "python": platform.python_version(),
        "models": {
            "stt": f"openai-whisper {models.whisper_model} (CPU, fp32)",
            "embeddings": stack.embedding_model,
            "llm": f"{stack.llm_model} via Ollama (CPU, temperature 0)",
            "tts": f"{models.tts_model} (CPU, fp32)" if synthesize_answers else "skipped",
            "stores": "Neo4j 5.26 + ChromaDB 1.0 (docker compose)",
        },
        "load_seconds": load_seconds,
        "turns": turns,
        "graph": {
            "nodes": [
                {"id": node.node_id, "label": node.label, "name": _node_name(node.properties)}
                for node in (graph.nodes if graph else ())
            ],
            "edges": [
                {"source": rel.source_id, "target": rel.target_id, "type": rel.relation_type}
                for rel in (graph.relations if graph else ())
            ],
        },
    }
    (output_dir / "roundtrip.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _run_one_turn(*, question: DemoQuestion, wav_path: Path, output_dir: Path, snapshot: PatientMemorySnapshot,
                  whisper: OpenAIWhisperBackend, orchestrator, tts: SpeechSynthesizer | None) -> dict[str, object]:
    started = perf_counter()
    transcript = whisper.transcribe_file(wav_path, language="fr").text
    stt_s = perf_counter() - started

    started = perf_counter()
    response = orchestrator.respond(snapshot.patient.patient_id, transcript)
    rag_and_llm_s = perf_counter() - started
    llm_s = (response.generation.latency_ms or 0.0) / 1000
    retrieval_s = max(rag_and_llm_s - llm_s, 0.0)

    tts_s = None
    answer_wav = None
    answer_audio_s = None
    if tts is not None:
        started = perf_counter()
        speech = tts.synthesize(response.answer)
        tts_s = perf_counter() - started
        answer_path = output_dir / f"{question.question_id}_answer.wav"
        answer_path.write_bytes(speech.audio_bytes)
        answer_wav = answer_path.name
        answer_audio_s = _wav_seconds(answer_path)

    answer_lower = response.answer.lower()
    retrieved = [
        {"id": memory.source_node_id, "name": memory.source_display_name, "score": round(memory.ranking_score, 3)}
        for memory in response.trace.retrieved_memories
    ]
    return {
        "question_id": question.question_id,
        "question_text": question.text,
        "question_wav": wav_path.name,
        "question_audio_s": round(_wav_seconds(wav_path), 2),
        "transcript": transcript,
        "answer": response.answer,
        "answer_wav": answer_wav,
        "answer_audio_s": round(answer_audio_s, 2) if answer_audio_s else None,
        "guard_applied": response.trace.guard_applied,
        "grounded": all(term.lower() in answer_lower for term in question.expected_terms),
        "expected_terms": list(question.expected_terms),
        "retrieved": retrieved,
        "related_nodes": _related_node_ids(response),
        "timings_s": {
            "stt": round(stt_s, 2),
            "retrieval": round(retrieval_s, 2),
            "llm": round(llm_s, 2),
            "tts": round(tts_s, 2) if tts_s is not None else None,
        },
    }


def _related_node_ids(response) -> list[str]:
    ids = []
    for hit in response.context.memory_recall.hits:
        ids.append(hit.source_node_id)
    return ids


def _node_name(properties: dict[str, object]) -> str:
    for key in ("preferred_name", "name", "title", "label", "id"):
        value = properties.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "?"


def _wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())
