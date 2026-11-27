"""Ask the same question before and after restarting the stack."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from .fictional import DEFAULT_DEMO_SEED, build_fictional_snapshot, demo_questions
from .roundtrip import seed_stores
from .stack import StackConfig, build_orchestrator, build_store_engine, restart_stack, wait_for_stack


def ask_from_stores(question: str, *, seed: int = DEFAULT_DEMO_SEED, stack: StackConfig | None = None) -> dict[str, object]:
    """Answer one question with a fresh engine that only reads the persisted stores."""

    snapshot = build_fictional_snapshot(seed)
    with build_store_engine(stack) as engine:
        orchestrator = build_orchestrator(engine, stack)
        response = orchestrator.respond(snapshot.patient.patient_id, question)
    return {
        "answer": response.answer,
        "guard_applied": response.trace.guard_applied,
        "retrieved": [memory.source_node_id for memory in response.trace.retrieved_memories],
    }


def persistence_check(compose_dir: str | Path, *, seed: int = DEFAULT_DEMO_SEED,
                      stack: StackConfig | None = None) -> dict[str, object]:
    """Seed, ask, restart Neo4j/ChromaDB/Ollama, ask again from a fresh process state."""

    stack = stack or StackConfig()
    question = demo_questions(build_fictional_snapshot(seed))[0]
    wait_for_stack(stack)
    seed_stores(seed=seed, stack=stack)
    before = ask_from_stores(question.text, seed=seed, stack=stack)

    started = perf_counter()
    restart_stack(str(compose_dir))
    wait_for_stack(stack)
    restart_s = perf_counter() - started

    after = ask_from_stores(question.text, seed=seed, stack=stack)
    expected = [term.lower() for term in question.expected_terms]
    return {
        "question": question.text,
        "expected_terms": list(question.expected_terms),
        "before": before,
        "after": after,
        "restart_seconds": round(restart_s, 1),
        "passed": all(term in after["answer"].lower() for term in expected)
        and any(node in after["retrieved"] for node in question.expected_node_ids),
    }
