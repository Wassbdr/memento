"""`python -m memento.demo {seed,questions,roundtrip,ask,persistence}`."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .fictional import DEFAULT_DEMO_SEED, build_fictional_snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="memento-demo", description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_DEMO_SEED)
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot", help="print the fictitious patient snapshot as JSON")
    snap.add_argument("--output", type=Path)
    sub.add_parser("seed", help="load the fictitious patient into Neo4j + ChromaDB")
    questions = sub.add_parser("questions", help="synthesize the question WAVs with Qwen3-TTS")
    questions.add_argument("--output-dir", type=Path, default=Path("showcase/run/questions"))
    roundtrip = sub.add_parser("roundtrip", help="WAV -> Whisper -> graph RAG -> LLM -> TTS")
    roundtrip.add_argument("--question-dir", type=Path, default=Path("showcase/run/questions"))
    roundtrip.add_argument("--output-dir", type=Path, default=Path("showcase/run"))
    roundtrip.add_argument("--whisper-model", default="small")
    roundtrip.add_argument("--text-only", action="store_true", help="skip answer TTS")
    ask = sub.add_parser("ask", help="answer one text question from the stores")
    ask.add_argument("question")
    persist = sub.add_parser("persistence", help="ask, restart the stack, ask again")
    persist.add_argument("--compose-dir", type=Path, default=Path("."))
    persist.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.command == "snapshot":
        payload = json.dumps(asdict(build_fictional_snapshot(args.seed)), ensure_ascii=False, indent=2)
        if args.output:
            args.output.write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
        return 0
    if args.command == "seed":
        from .roundtrip import seed_stores
        from .stack import wait_for_stack

        wait_for_stack()
        print(json.dumps(seed_stores(seed=args.seed)))
        return 0
    if args.command == "questions":
        from .roundtrip import make_question_wavs

        make_question_wavs(args.output_dir, seed=args.seed)
        return 0
    if args.command == "roundtrip":
        from .roundtrip import VoiceModels, run_roundtrip

        result = run_roundtrip(
            args.question_dir,
            args.output_dir,
            seed=args.seed,
            models=VoiceModels(whisper_model=args.whisper_model),
            synthesize_answers=not args.text_only,
        )
        for turn in result["turns"]:
            print(f"Q: {turn['transcript']}\nA: {turn['answer']}\n   {turn['timings_s']}")
        return 0 if all(turn["grounded"] for turn in result["turns"]) else 1
    if args.command == "ask":
        from .persistence import ask_from_stores

        print(json.dumps(ask_from_stores(args.question, seed=args.seed), ensure_ascii=False, indent=2))
        return 0
    if args.command == "persistence":
        from .persistence import persistence_check

        result = persistence_check(args.compose_dir, seed=args.seed)
        payload = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return 0 if result["passed"] else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
