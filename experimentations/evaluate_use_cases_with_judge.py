"""Evaluate Memento use-cases with deterministic checks and optional LLM-as-judge.

This script executes scenarios from `experimentations/memento_use_cases.json`
against snapshots declared in `experimentations/poc_test_cases.json`.

For each scenario it:
1) Runs the conversation orchestrator to obtain one assistant response.
2) Applies deterministic checks (context and style heuristics).
3) Optionally calls a dedicated judge model to score response quality.
4) Writes JSONL details and a Markdown summary report.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from statistics import mean
from typing import Any

from memento.conversation import (
    ConversationConfig,
    ConversationMessage,
    ConversationOrchestrator,
    OpenAICompatibleBackendConfig,
    OpenAICompatibleConversationBackend,
)
from memento.memory import MemorySyncEngine, PatientMemorySnapshot
from memento.runtime.bootstrap import snapshot_from_dict


DEFAULT_USE_CASES_FILE = "experimentations/memento_use_cases.json"
DEFAULT_POC_CASES_FILE = "experimentations/poc_test_cases.json"
DEFAULT_RESULTS_JSONL = "experimentations/use_case_eval_results.jsonl"
DEFAULT_REPORT_MD = "experimentations/use_case_eval_report.md"
DEFAULT_TRACES_JSON = "experimentations/use_case_eval_traces.json"


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class JudgeResult:
    enabled: bool
    parsed: bool
    overall_pass: bool | None
    scores: dict[str, float]
    violations: tuple[str, ...]
    strengths: tuple[str, ...]
    improvement: str
    raw_text: str
    error: str = ""
    debug_trace: dict[str, Any] | None = None


@dataclass(frozen=True)
class ScenarioResult:
    case_id: str
    snapshot_ref: str
    layer: str
    goal: str
    answer: str
    generation_latency_ms: float | None
    deterministic_pass: bool
    deterministic_checks: tuple[CheckResult, ...]
    judge: JudgeResult
    final_pass: bool
    final_score: float
    score_breakdown: dict[str, float | None]
    trace: dict[str, Any]
    runtime_error: str = ""


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Memento use-cases with optional LLM judge.")
    parser.add_argument("--use-cases-file", default=DEFAULT_USE_CASES_FILE)
    parser.add_argument("--poc-cases-file", default=DEFAULT_POC_CASES_FILE)
    parser.add_argument("--output-jsonl", default=DEFAULT_RESULTS_JSONL)
    parser.add_argument("--output-report", default=DEFAULT_REPORT_MD)
    parser.add_argument("--output-traces-json", default=DEFAULT_TRACES_JSON)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--debug-traces", action="store_true")

    parser.add_argument(
        "--llm-base-url",
        default=os.getenv(
            "MEMENTO_LLM_BASE_URL",
            os.getenv("NIM_BASE_URL", "http://127.0.0.1:11434/v1"),
        ),
    )
    parser.add_argument(
        "--llm-api-key",
        default=os.getenv(
            "MEMENTO_LLM_API_KEY",
            os.getenv("NVIDIA_API_KEY", ""),
        ),
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv(
            "MEMENTO_LLM_MODEL",
            os.getenv("NIM_MODEL", "Ministral 3 8B"),
        ),
    )
    parser.add_argument("--llm-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-prompt-memories", type=int, default=3)

    parser.add_argument("--judge-enabled", action="store_true")
    parser.add_argument("--judge-base-url", default="")
    parser.add_argument(
        "--judge-api-key",
        default=os.getenv(
            "MEMENTO_JUDGE_API_KEY",
            os.getenv("NVIDIA_API_KEY", ""),
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv(
            "MEMENTO_JUDGE_MODEL",
            os.getenv("NIM_MODEL", ""),
        ),
    )
    parser.add_argument("--judge-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--judge-temperature", type=float, default=0.0)

    parser.add_argument("--strict-exit", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    _load_env_file_if_present(Path(".env"))
    args = build_argument_parser().parse_args(argv)

    use_cases_payload = _read_json_file(args.use_cases_file)
    poc_cases_payload = _read_json_file(args.poc_cases_file)
    snapshots_by_ref = _load_snapshots_from_poc(poc_cases_payload)

    scenarios = _extract_scenarios(use_cases_payload)
    if args.max_cases > 0:
        scenarios = scenarios[: args.max_cases]

    judge_backend = _build_judge_backend(args)

    results: list[ScenarioResult] = []
    for scenario in scenarios:
        try:
            result = _evaluate_scenario(
                scenario=scenario,
                snapshots_by_ref=snapshots_by_ref,
                args=args,
                judge_backend=judge_backend,
            )
        except Exception as error:
            result = _scenario_failure_result(scenario=scenario, error=error)
        results.append(result)

    _write_jsonl_results(args.output_jsonl, results)
    _write_markdown_report(args.output_report, results)
    _write_traces_json(args.output_traces_json, results)

    total = len(results)
    final_passes = sum(1 for item in results if item.final_pass)
    avg_final_score = mean(item.final_score for item in results) if results else 0.0
    print(f"Evaluated {total} scenario(s). Final pass: {final_passes}/{total}.")
    print(f"Average final score: {avg_final_score:.2f}/100")
    print(f"Detailed results: {args.output_jsonl}")
    print(f"Summary report: {args.output_report}")
    print(f"Trace bundle: {args.output_traces_json}")

    if args.strict_exit and final_passes != total:
        return 1
    return 0


def _read_json_file(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object as root")
    return payload


def _extract_scenarios(payload: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("use-cases file must define a non-empty `scenarios` array")
    normalized: list[dict[str, Any]] = []
    for item in scenarios:
        if not isinstance(item, dict):
            continue
        normalized.append(item)
    if not normalized:
        raise ValueError("use-cases file does not contain valid scenario objects")
    return normalized


def _load_snapshots_from_poc(payload: dict[str, Any]) -> dict[str, PatientMemorySnapshot]:
    snapshots_payload = payload.get("patient_snapshots")
    if not isinstance(snapshots_payload, dict):
        raise ValueError("poc cases file must define `patient_snapshots` as an object")

    snapshots_by_ref: dict[str, PatientMemorySnapshot] = {}
    for snapshot_ref, item in snapshots_payload.items():
        if not isinstance(snapshot_ref, str):
            continue
        if not isinstance(item, dict):
            continue
        snapshot_payload = item.get("snapshot")
        if not isinstance(snapshot_payload, dict):
            continue
        snapshots_by_ref[snapshot_ref] = snapshot_from_dict(snapshot_payload)

    if not snapshots_by_ref:
        raise ValueError("no valid snapshot found in `patient_snapshots`")
    return snapshots_by_ref


def _build_judge_backend(args: argparse.Namespace) -> OpenAICompatibleConversationBackend | None:
    if not args.judge_enabled:
        return None
    if not args.judge_model.strip():
        raise ValueError("--judge-model is required when --judge-enabled is used")

    base_url = args.judge_base_url.strip() or args.llm_base_url.strip()
    api_key = args.judge_api_key.strip() or args.llm_api_key.strip() or None

    return OpenAICompatibleConversationBackend(
        config=OpenAICompatibleBackendConfig(
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=args.judge_timeout_seconds,
        )
    )


def _evaluate_scenario(
    *,
    scenario: dict[str, Any],
    snapshots_by_ref: dict[str, PatientMemorySnapshot],
    args: argparse.Namespace,
    judge_backend: OpenAICompatibleConversationBackend | None,
) -> ScenarioResult:
    case_id = str(scenario.get("id", "")).strip() or "unknown"
    snapshot_ref = str(scenario.get("snapshot_ref", "")).strip()
    if snapshot_ref not in snapshots_by_ref:
        raise ValueError(f"scenario {case_id}: unknown snapshot_ref {snapshot_ref!r}")

    snapshot = snapshots_by_ref[snapshot_ref]
    layer = str(scenario.get("layer", "")).strip()
    goal = str(scenario.get("goal", "")).strip()

    scenario_input = scenario.get("input")
    if not isinstance(scenario_input, dict):
        raise ValueError(f"scenario {case_id}: missing object field `input`")

    patient_id = str(scenario_input.get("patient_id", snapshot.patient.patient_id)).strip()
    utterance = str(scenario_input.get("utterance", scenario_input.get("query", ""))).strip()
    if not utterance:
        raise ValueError(f"scenario {case_id}: missing `input.utterance` or `input.query`")

    top_k_value = int(scenario_input.get("top_k", 3) or 3)
    reference_datetime = _parse_optional_datetime(scenario_input.get("reference_datetime"))

    orchestrator = _build_orchestrator(
        snapshot=snapshot,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_timeout_seconds=args.llm_timeout_seconds,
        llm_model=args.llm_model,
        temperature=args.temperature,
        top_k=top_k_value,
        max_prompt_memories=args.max_prompt_memories,
    )

    response = orchestrator.respond(
        patient_id,
        utterance,
        reference_datetime=reference_datetime,
    )

    expected_context = scenario.get("expected_context")
    if not isinstance(expected_context, dict):
        expected_context = {}

    response_requirements = scenario.get("response_requirements")
    if not isinstance(response_requirements, dict):
        response_requirements = {}

    deterministic_checks = _run_deterministic_checks(
        expected_context=expected_context,
        response_requirements=response_requirements,
        answer=response.answer,
        response=response,
        snapshot=snapshot,
    )
    deterministic_pass = all(item.passed for item in deterministic_checks)

    judge_result = _run_judge(
        judge_backend=judge_backend,
        judge_model_name=args.judge_model,
        judge_temperature=args.judge_temperature,
        scenario=scenario,
        response=response,
        deterministic_checks=deterministic_checks,
        debug_traces=args.debug_traces,
    )

    if judge_result.enabled and judge_result.parsed and judge_result.overall_pass is not None:
        final_pass = deterministic_pass and bool(judge_result.overall_pass)
    else:
        final_pass = deterministic_pass

    final_score, score_breakdown = _compute_final_score(
        deterministic_checks=deterministic_checks,
        judge_result=judge_result,
    )
    trace_payload = _build_scenario_trace(
        response=response,
        debug_traces=args.debug_traces,
        judge_result=judge_result,
    )

    return ScenarioResult(
        case_id=case_id,
        snapshot_ref=snapshot_ref,
        layer=layer,
        goal=goal,
        answer=response.answer,
        generation_latency_ms=response.generation.latency_ms,
        deterministic_pass=deterministic_pass,
        deterministic_checks=tuple(deterministic_checks),
        judge=judge_result,
        final_pass=final_pass,
        final_score=final_score,
        score_breakdown=score_breakdown,
        trace=trace_payload,
    )


def _scenario_failure_result(*, scenario: dict[str, Any], error: Exception) -> ScenarioResult:
    return ScenarioResult(
        case_id=str(scenario.get("id", "unknown")).strip() or "unknown",
        snapshot_ref=str(scenario.get("snapshot_ref", "")).strip(),
        layer=str(scenario.get("layer", "")).strip(),
        goal=str(scenario.get("goal", "")).strip(),
        answer="",
        generation_latency_ms=None,
        deterministic_pass=False,
        deterministic_checks=(
            CheckResult(
                name="runtime.execution",
                passed=False,
                detail=str(error),
            ),
        ),
        judge=JudgeResult(
            enabled=False,
            parsed=False,
            overall_pass=None,
            scores={},
            violations=(),
            strengths=(),
            improvement="",
            raw_text="",
            error="",
            debug_trace=None,
        ),
        final_pass=False,
        final_score=0.0,
        score_breakdown={"deterministic_score": 0.0, "judge_score": None},
        trace={},
        runtime_error=str(error),
    )


def _build_orchestrator(
    *,
    snapshot: PatientMemorySnapshot,
    llm_base_url: str,
    llm_api_key: str,
    llm_timeout_seconds: float,
    llm_model: str,
    temperature: float,
    top_k: int,
    max_prompt_memories: int,
) -> ConversationOrchestrator:
    memory_engine = MemorySyncEngine()
    memory_engine.sync_snapshot(snapshot)

    backend = OpenAICompatibleConversationBackend(
        config=OpenAICompatibleBackendConfig(
            base_url=llm_base_url,
            api_key=llm_api_key.strip() or None,
            timeout_seconds=llm_timeout_seconds,
        )
    )
    config = ConversationConfig(
        model_name=llm_model,
        temperature=temperature,
        top_k=top_k,
        max_prompt_memories=max_prompt_memories,
    )
    return ConversationOrchestrator(memory_engine=memory_engine, backend=backend, config=config)


def _parse_optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return datetime.fromisoformat(normalized)


def _run_deterministic_checks(
    *,
    expected_context: dict[str, Any],
    response_requirements: dict[str, Any],
    answer: str,
    response,
    snapshot: PatientMemorySnapshot,
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    checks.extend(_run_expected_context_checks(expected_context=expected_context, response=response))
    checks.extend(
        _run_response_requirement_checks(
            response_requirements=response_requirements,
            answer=answer,
            response=response,
            snapshot=snapshot,
        )
    )
    return checks


def _run_expected_context_checks(*, expected_context: dict[str, Any], response) -> list[CheckResult]:
    anchors = list(response.context.anchors)
    trusted_people = [person.name for person in response.context.trusted_people]
    routines = [routine.title for routine in response.context.routines]

    hits = list(response.context.memory_recall.hits)
    episodes = []
    places = [routine.place_name for routine in response.context.routines if routine.place_name]
    emotions = []
    for hit in hits:
        episodes.extend(list(hit.related_episodes))
        if hit.source_label.strip().lower() == "episode":
            episodes.append(hit.source_display_name)
        places.extend(list(hit.related_places))
        emotions.extend(list(hit.related_emotions))

    mapping = {
        "anchors_any": anchors,
        "trusted_people_any": trusted_people,
        "routines_any": routines,
        "episodes_any": episodes,
        "places_any": places,
        "emotions_any": emotions,
    }

    checks: list[CheckResult] = []
    for expected_key, actual_values in mapping.items():
        expected_values = expected_context.get(expected_key)
        if not isinstance(expected_values, list) or not expected_values:
            continue
        matched_values = _find_matches(expected_values, actual_values)
        passed = bool(matched_values)
        checks.append(
            CheckResult(
                name=f"context.{expected_key}",
                passed=passed,
                detail=(
                    f"expected any={expected_values}; matched={matched_values}; "
                    f"actual={actual_values}"
                ),
            )
        )

    return checks


def _run_response_requirement_checks(
    *,
    response_requirements: dict[str, Any],
    answer: str,
    response,
    snapshot: PatientMemorySnapshot,
) -> list[CheckResult]:
    checks: list[CheckResult] = []

    known_people = {person.name for person in snapshot.people}
    known_people.add(snapshot.patient.display_name)
    if snapshot.patient.preferred_name:
        known_people.add(snapshot.patient.preferred_name)

    for key, value in response_requirements.items():
        if value is not True:
            continue

        if key == "must_start_with_reassurance":
            passed = _starts_with_reassurance(answer)
            detail = "first sentence should contain reassuring phrasing"
        elif key == "must_include_temporal_anchor":
            passed = _contains_temporal_anchor(answer)
            detail = "answer should include a near-term temporal anchor"
        elif key == "must_include_relationship_hint":
            passed = _contains_any_token(answer, ["fille", "petit-fils", "mari", "infirmiere", "aide"])
            detail = "answer should include relationship wording"
        elif key == "must_avoid_generic_answer":
            passed = _is_not_generic_answer(answer, response)
            detail = "answer should mention concrete memory anchors"
        elif key == "must_include_future_near_anchor":
            passed = _contains_temporal_anchor(answer)
            detail = "answer should project a near future milestone"
        elif key == "must_not_sound_uncertain_without_reason":
            passed = not _contains_any_token(answer, ["peut-etre", "je crois", "pas sur", "incertain", "je ne sais pas"])
            detail = "answer should avoid uncertainty markers"
        elif key == "must_avoid_hard_contradiction":
            passed = not _contains_any_token(answer, ["vous avez tort", "c'est faux", "non, vous", "arretez"])
            detail = "answer should avoid abrupt contradiction"
        elif key == "must_include_immediate_next_step":
            passed = _contains_any_token(answer, ["maintenant", "d'abord", "bientot", "on va", "vous pouvez"])
            detail = "answer should include an immediate next step"
        elif key == "must_include_identity_support":
            passed = _contains_any_token(answer, ["vous etiez", "vous etes", "bon mecanicien", "fierte", "competent"])
            detail = "answer should reinforce identity"
        elif key == "must_be_short":
            passed = _sentence_count(answer) <= 3
            detail = "answer should remain concise (<= 3 sentences)"
        elif key == "must_include_person_name":
            passed = any(_contains_phrase(answer, name) for name in known_people)
            detail = "answer should include a known person name"
        elif key == "must_keep_single_instruction_flow":
            step_markers = ["ensuite", "puis", "apres", "d'abord"]
            markers_count = sum(answer.lower().count(marker) for marker in step_markers)
            passed = markers_count <= 2
            detail = "answer should keep a simple one-flow instruction"
        elif key == "must_include_spousal_anchor":
            passed = _contains_any_token(answer, ["samir", "mari"])
            detail = "answer should anchor around spouse"
        elif key == "must_include_concrete_sensory_cue":
            passed = _contains_any_token(answer, ["salon", "plateau", "the", "tasse", "tapis", "table"])
            detail = "answer should include concrete sensory cue"
        elif key == "must_include_time_projection":
            passed = _contains_temporal_anchor(answer)
            detail = "answer should project a concrete time reference"
        elif key == "must_not_add_unknown_logistics":
            unknown_names = _extract_unknown_names(answer, known_people)
            passed = len(unknown_names) == 0
            detail = f"unexpected names detected={unknown_names}"
        elif key == "must_include_correct_schedule":
            passed = _contains_any_token(answer, ["samedi"])
            detail = "answer should mention expected schedule (samedi)"
        elif key == "must_keep_positive_reframe":
            has_positive = _contains_any_token(answer, ["tranquil", "calme", "bientot", "on ira", "vous pourrez", "avec samir"])
            has_harsh = _contains_any_token(answer, ["vous avez tort", "non, vous", "arretez"])
            passed = has_positive and not has_harsh
            detail = "answer should preserve a positive reframing"
        else:
            passed = True
            detail = f"no deterministic rule implemented for {key}; marked as pass-by-default"

        checks.append(CheckResult(name=f"response.{key}", passed=passed, detail=detail))

    return checks


def _run_judge(
    *,
    judge_backend: OpenAICompatibleConversationBackend | None,
    judge_model_name: str,
    judge_temperature: float,
    scenario: dict[str, Any],
    response,
    deterministic_checks: list[CheckResult],
    debug_traces: bool,
) -> JudgeResult:
    if judge_backend is None:
        return JudgeResult(
            enabled=False,
            parsed=False,
            overall_pass=None,
            scores={},
            violations=(),
            strengths=(),
            improvement="",
            raw_text="",
            debug_trace=None,
        )

    strict_json_instruction = (
        "Tu es un evaluateur clinique de qualite pour Memento. "
        "Tu dois retourner STRICTEMENT un objet JSON valide sans markdown. "
        "Schema obligatoire: "
        '{"overall_pass": bool, "scores": {"factual_grounding": number, '
        '"reassurance_tone": number, "clarity_concision": number, '
        '"cognitive_load": number, "clinical_usefulness": number}, '
        '"violations": string[], "strengths": string[], "improvement": string}. '
        "Echelle des scores: 0 a 5. overall_pass=true seulement si "
        "factual_grounding>=3, reassurance_tone>=3, clarity_concision>=3, "
        "cognitive_load>=3, clinical_usefulness>=3 et aucune violation critique. "
        "Violations critiques: hallucination factuelle, contradiction brutale, ton anxiogene."
    )

    judge_input = {
        "scenario": {
            "id": scenario.get("id"),
            "goal": scenario.get("goal"),
            "input": scenario.get("input"),
            "expected_context": scenario.get("expected_context"),
            "response_requirements": scenario.get("response_requirements"),
        },
        "assistant_answer": response.answer,
        "retrieved_memories": [
            {
                "source_label": hit.source_label,
                "source_display_name": hit.source_display_name,
                "summary": hit.summary,
                "related_people": list(hit.related_people),
                "related_places": list(hit.related_places),
                "related_emotions": list(hit.related_emotions),
                "related_routines": list(hit.related_routines),
                "related_episodes": list(hit.related_episodes),
            }
            for hit in response.context.memory_recall.hits
        ],
        "deterministic_checks": [asdict(item) for item in deterministic_checks],
    }

    messages = (
        ConversationMessage(role="system", content=strict_json_instruction),
        ConversationMessage(role="user", content=json.dumps(judge_input, ensure_ascii=True, indent=2)),
    )
    judge_debug_trace: dict[str, Any] | None = None
    if debug_traces:
        judge_debug_trace = {
            "judge_messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "judge_model": judge_model_name,
            "judge_temperature": judge_temperature,
        }

    try:
        generation = judge_backend.generate(
            messages=messages,
            model_name=judge_model_name,
            temperature=judge_temperature,
        )
    except Exception as error:  # pragma: no cover - runtime path
        return JudgeResult(
            enabled=True,
            parsed=False,
            overall_pass=None,
            scores={},
            violations=(),
            strengths=(),
            improvement="",
            raw_text="",
            error=str(error),
            debug_trace=judge_debug_trace,
        )

    raw_text = generation.text.strip()
    try:
        judge_payload = _parse_judge_payload(raw_text)
    except ValueError as error:
        return JudgeResult(
            enabled=True,
            parsed=False,
            overall_pass=None,
            scores={},
            violations=(),
            strengths=(),
            improvement="",
            raw_text=raw_text,
            error=str(error),
            debug_trace=judge_debug_trace,
        )

    scores = judge_payload.get("scores")
    if not isinstance(scores, dict):
        scores = {}
    normalized_scores: dict[str, float] = {}
    for key, value in scores.items():
        try:
            normalized_scores[str(key)] = float(value)
        except (TypeError, ValueError):
            continue

    violations = _to_string_tuple(judge_payload.get("violations"))
    strengths = _to_string_tuple(judge_payload.get("strengths"))
    improvement = str(judge_payload.get("improvement", "")).strip()
    overall_pass_raw = judge_payload.get("overall_pass")
    overall_pass = bool(overall_pass_raw) if isinstance(overall_pass_raw, bool) else None

    return JudgeResult(
        enabled=True,
        parsed=True,
        overall_pass=overall_pass,
        scores=normalized_scores,
        violations=violations,
        strengths=strengths,
        improvement=improvement,
        raw_text=raw_text,
        debug_trace=judge_debug_trace,
    )


def _parse_judge_payload(raw_text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("judge output is not valid JSON")
        parsed = json.loads(raw_text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("judge output must be a JSON object")
    return parsed


def _write_jsonl_results(path: str, results: list[ScenarioResult]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for item in results:
        payload = {
            "case_id": item.case_id,
            "snapshot_ref": item.snapshot_ref,
            "layer": item.layer,
            "goal": item.goal,
            "answer": item.answer,
            "generation_latency_ms": item.generation_latency_ms,
            "deterministic_pass": item.deterministic_pass,
            "deterministic_checks": [asdict(check) for check in item.deterministic_checks],
            "judge": {
                "enabled": item.judge.enabled,
                "parsed": item.judge.parsed,
                "overall_pass": item.judge.overall_pass,
                "scores": item.judge.scores,
                "violations": list(item.judge.violations),
                "strengths": list(item.judge.strengths),
                "improvement": item.judge.improvement,
                "raw_text": item.judge.raw_text,
                "error": item.judge.error,
                "debug_trace": item.judge.debug_trace,
            },
            "final_pass": item.final_pass,
            "final_score": item.final_score,
            "score_breakdown": item.score_breakdown,
            "trace": item.trace,
            "runtime_error": item.runtime_error,
        }
        lines.append(json.dumps(payload, ensure_ascii=True))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_traces_json(path: str, results: list[ScenarioResult]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(results)
    pass_count = sum(1 for item in results if item.final_pass)
    avg_final_score = mean(item.final_score for item in results) if results else 0.0

    payload = {
        "suite": "memento_use_case_evaluation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_scenarios": total,
            "final_pass_count": pass_count,
            "final_pass_rate": (pass_count / total) if total else 0.0,
            "average_final_score": avg_final_score,
        },
        "scenarios": [
            {
                "case_id": item.case_id,
                "snapshot_ref": item.snapshot_ref,
                "layer": item.layer,
                "goal": item.goal,
                "answer": item.answer,
                "generation_latency_ms": item.generation_latency_ms,
                "deterministic_pass": item.deterministic_pass,
                "deterministic_checks": [asdict(check) for check in item.deterministic_checks],
                "judge": {
                    "enabled": item.judge.enabled,
                    "parsed": item.judge.parsed,
                    "overall_pass": item.judge.overall_pass,
                    "scores": item.judge.scores,
                    "violations": list(item.judge.violations),
                    "strengths": list(item.judge.strengths),
                    "improvement": item.judge.improvement,
                    "raw_text": item.judge.raw_text,
                    "error": item.judge.error,
                    "debug_trace": item.judge.debug_trace,
                },
                "final_pass": item.final_pass,
                "final_score": item.final_score,
                "score_breakdown": item.score_breakdown,
                "trace": item.trace,
                "runtime_error": item.runtime_error,
            }
            for item in results
        ],
    }

    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_markdown_report(path: str, results: list[ScenarioResult]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(results)
    deterministic_passes = sum(1 for item in results if item.deterministic_pass)
    final_passes = sum(1 for item in results if item.final_pass)
    avg_final_score = mean(item.final_score for item in results) if results else 0.0

    judge_enabled_cases = [item for item in results if item.judge.enabled and item.judge.parsed]
    factual_scores = [
        item.judge.scores.get("factual_grounding")
        for item in judge_enabled_cases
        if "factual_grounding" in item.judge.scores
    ]
    reassurance_scores = [
        item.judge.scores.get("reassurance_tone")
        for item in judge_enabled_cases
        if "reassurance_tone" in item.judge.scores
    ]

    lines = [
        "# Memento Use-Case Evaluation Report",
        "",
        f"- Total scenarios: {total}",
        f"- Deterministic pass: {deterministic_passes}/{total}",
        f"- Final pass (deterministic + judge): {final_passes}/{total}",
        f"- Avg final score: {avg_final_score:.2f}/100",
    ]
    if factual_scores:
        lines.append(f"- Avg factual grounding (judge): {mean(factual_scores):.2f}/5")
    if reassurance_scores:
        lines.append(f"- Avg reassurance tone (judge): {mean(reassurance_scores):.2f}/5")

    lines.extend(
        [
            "",
            "## Per scenario",
            "",
            "| Case | Deterministic | Judge | Final | Score |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for item in results:
        judge_cell = "n/a"
        if item.judge.enabled:
            if item.judge.parsed and item.judge.overall_pass is not None:
                judge_cell = "PASS" if item.judge.overall_pass else "FAIL"
            else:
                judge_cell = "ERROR"
        lines.append(
            "| "
            + f"{item.case_id} | "
            + f"{'PASS' if item.deterministic_pass else 'FAIL'} | "
            + f"{judge_cell} | "
            + f"{'PASS' if item.final_pass else 'FAIL'} | "
            + f"{item.final_score:.2f} |"
        )

    lines.extend(["", "## Findings", ""])
    for item in results:
        lines.append(f"### {item.case_id} - {'PASS' if item.final_pass else 'FAIL'}")
        lines.append(f"- Goal: {item.goal}")
        lines.append(f"- Final score: {item.final_score:.2f}/100")
        lines.append(f"- Answer: {item.answer}")
        if item.runtime_error:
            lines.append(f"- Runtime error: {item.runtime_error}")
        failed_checks = [check for check in item.deterministic_checks if not check.passed]
        if failed_checks:
            for failed in failed_checks:
                lines.append(f"- Deterministic fail: {failed.name} -> {failed.detail}")
        else:
            lines.append("- Deterministic checks: all pass")

        if item.judge.enabled:
            if item.judge.parsed:
                lines.append(f"- Judge overall: {item.judge.overall_pass}")
                if item.judge.scores:
                    lines.append(f"- Judge scores: {json.dumps(item.judge.scores, ensure_ascii=True)}")
                if item.judge.violations:
                    lines.append(f"- Judge violations: {list(item.judge.violations)}")
                if item.judge.improvement:
                    lines.append(f"- Judge improvement: {item.judge.improvement}")
            else:
                lines.append(f"- Judge error: {item.judge.error}")
        lines.append("")

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _load_env_file_if_present(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", raw_line)
        if match is None:
            continue

        key = match.group(1)
        value = match.group(2).strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        else:
            hash_index = value.find("#")
            if hash_index >= 0:
                value = value[:hash_index].rstrip()

        if key not in os.environ:
            os.environ[key] = value


def _compute_final_score(
    *,
    deterministic_checks: list[CheckResult],
    judge_result: JudgeResult,
) -> tuple[float, dict[str, float | None]]:
    total_checks = len(deterministic_checks)
    passed_checks = sum(1 for item in deterministic_checks if item.passed)
    if total_checks == 0:
        deterministic_score = 100.0
    else:
        deterministic_score = (passed_checks / total_checks) * 100.0

    judge_score: float | None = None
    if judge_result.enabled and judge_result.parsed and judge_result.scores:
        judge_score = (mean(judge_result.scores.values()) / 5.0) * 100.0

    final_score = deterministic_score if judge_score is None else (deterministic_score + judge_score) / 2.0
    rounded_final_score = round(final_score, 2)
    return rounded_final_score, {
        "deterministic_score": round(deterministic_score, 2),
        "judge_score": round(judge_score, 2) if judge_score is not None else None,
    }


def _build_scenario_trace(
    *,
    response,
    debug_traces: bool,
    judge_result: JudgeResult,
) -> dict[str, Any]:
    trace = {
        "model": {
            "model_name": response.generation.model_name,
            "latency_ms": response.generation.latency_ms,
            "finish_reason": response.generation.finish_reason,
            "prompt_tokens": response.generation.prompt_tokens,
            "completion_tokens": response.generation.completion_tokens,
        },
        "assistant": {
            "raw_generation_text": response.generation.text,
            "final_answer": response.answer,
        },
        "orchestrator": {
            "guard_applied": response.trace.guard_applied,
            "guard_reason": response.trace.guard_reason,
            "dropped_hits": response.trace.dropped_hits,
            "total_semantic_hits": response.trace.total_semantic_hits,
        },
        "context": {
            "patient_display_name": response.context.patient_display_name,
            "preferred_name": response.context.preferred_name,
            "anchors": list(response.context.anchors),
            "care_notes": list(response.context.care_notes),
            "trusted_people": [person.name for person in response.context.trusted_people],
            "routines": [routine.title for routine in response.context.routines],
        },
        "retrieved_memories": [
            {
                "source_label": memory.source_label,
                "source_display_name": memory.source_display_name,
                "summary": memory.summary,
                "ranking_score": memory.ranking_score,
                "signals": list(memory.signals),
                "related_people": list(memory.related_people),
                "related_places": list(memory.related_places),
                "related_emotions": list(memory.related_emotions),
                "related_routines": list(memory.related_routines),
                "related_episodes": list(memory.related_episodes),
            }
            for memory in response.trace.retrieved_memories
        ],
    }
    if debug_traces:
        trace["prompt"] = {
            "system_prompt": response.trace.system_prompt,
            "user_prompt": response.trace.user_prompt,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in response.trace.messages
            ],
        }
        if judge_result.debug_trace is not None:
            trace["judge_debug_trace"] = judge_result.debug_trace
    return trace


def _find_matches(expected_values: list[Any], actual_values: list[str]) -> list[str]:
    matches: list[str] = []
    for expected in expected_values:
        expected_text = str(expected).strip()
        if not expected_text:
            continue
        for actual in actual_values:
            if _contains_phrase(actual, expected_text) or _contains_phrase(expected_text, actual):
                matches.append(actual)
                break
    return matches


def _contains_phrase(text: str, phrase: str) -> bool:
    return _normalize_text(phrase) in _normalize_text(text)


def _contains_any_token(text: str, tokens: list[str]) -> bool:
    normalized = _normalize_text(text)
    return any(_normalize_text(token) in normalized for token in tokens)


def _contains_temporal_anchor(text: str) -> bool:
    return _contains_any_token(
        text,
        [
            "maintenant",
            "bientot",
            "dans quelques",
            "ce matin",
            "ce soir",
            "aujourd",
            "tout a l'heure",
            "apres",
            "mardi",
            "samedi",
            "dimanche",
        ],
    )


def _starts_with_reassurance(text: str) -> bool:
    normalized = _normalize_text(text)
    first_part = normalized[:100]
    reassurance_markers = [
        "oui",
        "vous etes",
        "ne vous inquietez",
        "tout va bien",
        "vous n etes pas seule",
        "je suis la",
        "calmement",
    ]
    return any(marker in first_part for marker in reassurance_markers)


def _is_not_generic_answer(answer: str, response) -> bool:
    if len(answer.strip()) < 25:
        return False
    anchors = list(response.context.anchors)
    names = [person.name for person in response.context.trusted_people]
    routines = [routine.title for routine in response.context.routines]
    references = anchors + names + routines
    return any(_contains_phrase(answer, reference) or _contains_phrase(reference, answer) for reference in references)


def _sentence_count(text: str) -> int:
    parts = [item.strip() for item in re.split(r"[.!?]+", text) if item.strip()]
    return len(parts)


def _extract_unknown_names(answer: str, known_people: set[str]) -> list[str]:
    known_tokens = {_normalize_text(name) for name in known_people if name.strip()}
    discovered: list[str] = []
    candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", answer)
    ignored = {
        "Je",
        "Vous",
        "Oui",
        "Non",
        "Mardi",
        "Mercredi",
        "Jeudi",
        "Vendredi",
        "Samedi",
        "Dimanche",
        "Lundi",
    }
    for candidate in candidates:
        if candidate in ignored:
            continue
        normalized = _normalize_text(candidate)
        if normalized and normalized not in known_tokens:
            discovered.append(candidate)
    return sorted(set(discovered))


def _normalize_text(value: str) -> str:
    lowered = value.lower().strip()
    return re.sub(r"\s+", " ", lowered)


def _to_string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    normalized = [str(item).strip() for item in value if str(item).strip()]
    return tuple(normalized)


if __name__ == "__main__":
    raise SystemExit(main())
