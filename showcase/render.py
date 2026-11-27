"""Render demo.mp4 / demo.gif / poster.png from showcase/run/roundtrip.json.

Every frame is drawn from the recorded run (transcripts, retrieved node ids,
stage timings, hardware) and the audio is the WAV produced by that run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import textwrap
import wave

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

W, H, DPI = 12.8, 7.2, 100
BG = "#0f172a"
PANEL = "#1e293b"
FG = "#e2e8f0"
MUTED = "#94a3b8"
ACCENT = "#f59e0b"
COLORS = {
    "Patient": "#f472b6",
    "Person": "#60a5fa",
    "Place": "#34d399",
    "Routine": "#a78bfa",
    "Episode": "#fbbf24",
    "Emotion": "#f87171",
}
TITLE_S, GAP_S, END_S = 3.0, 0.6, 3.5


def _layout(nodes: list[dict]) -> dict[str, tuple[float, float]]:
    patient = [n for n in nodes if n["label"] == "Patient"]
    others = sorted((n for n in nodes if n["label"] != "Patient"), key=lambda n: (n["label"], n["id"]))
    pos = {n["id"]: (0.0, 0.0) for n in patient}
    for index, node in enumerate(others):
        angle = 2 * math.pi * index / max(len(others), 1) + math.pi / 2
        radius = 1.0 if index % 2 == 0 else 0.78
        pos[node["id"]] = (radius * math.cos(angle), radius * math.sin(angle))
    return pos


def _draw_graph(ax, graph: dict, highlight: set[str]) -> None:
    pos = _layout(graph["nodes"])
    ax.set_facecolor(BG)
    for edge in graph["edges"]:
        if edge["source"] in pos and edge["target"] in pos:
            (x1, y1), (x2, y2) = pos[edge["source"]], pos[edge["target"]]
            hot = edge["source"] in highlight and edge["target"] in highlight
            ax.plot([x1, x2], [y1, y2], color=ACCENT if hot else "#334155", lw=2.2 if hot else 0.8, zorder=1)
    for node in graph["nodes"]:
        x, y = pos[node["id"]]
        hot = node["id"] in highlight
        color = COLORS.get(node["label"], "#cbd5e1")
        ax.scatter([x], [y], s=900 if hot else 260, color=color, alpha=1.0 if hot else 0.35,
                   edgecolors=ACCENT if hot else "none", linewidths=3, zorder=3)
        name = textwrap.shorten(node["name"], 26, placeholder="…")
        ax.text(x, y - 0.13, name, ha="center", va="top", fontsize=9.5 if hot else 7.5,
                color=FG if hot else MUTED, weight="bold" if hot else "normal", zorder=4)
    for i, (label, color) in enumerate(COLORS.items()):
        ax.scatter([-1.3], [1.25 - i * 0.11], s=60, color=color)
        ax.text(-1.24, 1.25 - i * 0.11, label, va="center", fontsize=8, color=MUTED)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.3, 1.35)
    ax.axis("off")


def _wrap(text: str, width: int = 44) -> str:
    return "\n".join(textwrap.wrap(text, width))


def _frame(path: Path, run: dict, turn: dict | None, phase: str, index: int) -> None:
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=BG)
    fig.text(0.03, 0.94, "Memento", fontsize=22, color=FG, weight="bold")
    fig.text(0.175, 0.945, "assistant vocal de mémoire · patiente fictive · tout en local sur CPU",
             fontsize=12, color=MUTED)
    graph_ax = fig.add_axes([0.0, 0.07, 0.58, 0.84])
    highlight = set(turn["retrieved_ids"]) if turn and phase == "answer" else set()
    _draw_graph(graph_ax, run["graph"], highlight)

    panel = fig.add_axes([0.59, 0.1, 0.39, 0.8])
    panel.axis("off")
    panel.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.0,rounding_size=0.03",
                                   transform=panel.transAxes, color=PANEL))
    if turn:
        t = turn["timings_s"]
        panel.text(0.05, 0.93, f"Question {index + 1}/{len(run['turns'])}", fontsize=13, color=ACCENT, weight="bold")
        panel.text(0.05, 0.86, "Question parlée → Whisper a entendu :",
                   fontsize=10, color=MUTED)
        panel.text(0.05, 0.84, _wrap(f"« {turn['transcript']} »"), fontsize=13, color=FG, va="top")
        if phase == "answer":
            panel.text(0.05, 0.62, f"Souvenirs retrouvés (Neo4j + ChromaDB) :", fontsize=10, color=MUTED)
            names = ", ".join(r["name"] for r in turn["retrieved"])
            panel.text(0.05, 0.6, _wrap(names, 50), fontsize=10.5, color=ACCENT, va="top")
            panel.text(0.05, 0.46, f"Réponse ({run['llm_short']}) puis voix Qwen3-TTS :", fontsize=10, color=MUTED)
            panel.text(0.05, 0.44, _wrap(turn["answer"], 44), fontsize=12, color=FG, va="top")
        stages = [("STT", t["stt"]), ("RAG", t["retrieval"]), ("LLM", t["llm"]), ("TTS", t["tts"])]
        line = "   ".join(f"{name} {value:.2f}s" for name, value in stages if value is not None)
        panel.text(0.05, 0.05, line, fontsize=10, color=MUTED, family="monospace")
    fig.text(0.03, 0.03, f"Rejoue un run réel enregistré par showcase/make_showcase.sh · {run['hardware']}",
             fontsize=8.5, color=MUTED)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def _card(path: Path, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=BG)
    fig.text(0.5, 0.62, title, ha="center", fontsize=30, color=FG, weight="bold")
    for i, line in enumerate(lines):
        fig.text(0.5, 0.5 - i * 0.07, line, ha="center", fontsize=15, color=MUTED if i else ACCENT)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def _median(values: list[float]) -> float:
    values = sorted(values)
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def stage_medians(run: dict) -> dict[str, float | None]:
    medians: dict[str, float | None] = {}
    for stage in ("stt", "retrieval", "llm", "tts"):
        values = [turn["timings_s"][stage] for turn in run["turns"] if turn["timings_s"][stage] is not None]
        medians[stage] = _median(values) if values else None
    return medians


def poster(path: Path, run: dict) -> None:
    med = stage_medians(run)
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=BG)
    fig.text(0.04, 0.88, "Memento", fontsize=34, color=FG, weight="bold")
    fig.text(0.04, 0.82, "Un assistant vocal qui répond aux questions d'une personne atteinte d'Alzheimer "
             "à partir de son graphe de souvenirs.", fontsize=13, color=MUTED)
    stages = [
        ("Question\nparlée", "WAV 24 kHz", None, "#475569"),
        ("Whisper", run["models_short"]["stt"], med["stt"], "#60a5fa"),
        ("Graph RAG", "Neo4j + ChromaDB", med["retrieval"], "#34d399"),
        ("LLM local", run["llm_short"], med["llm"], "#a78bfa"),
        ("Qwen3-TTS", "0.6B CustomVoice", med["tts"], "#fbbf24"),
        ("Réponse\nparlée", "WAV 24 kHz", None, "#475569"),
    ]
    ax = fig.add_axes([0.02, 0.36, 0.96, 0.36])
    ax.set_xlim(0, len(stages))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (name, sub, seconds, color) in enumerate(stages):
        ax.add_patch(FancyBboxPatch((i + 0.08, 0.2), 0.84, 0.6, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    color=color, alpha=0.9))
        ax.text(i + 0.5, 0.6, name, ha="center", va="center", fontsize=14, color=BG, weight="bold")
        ax.text(i + 0.5, 0.33, sub, ha="center", va="center", fontsize=9, color=BG)
        if seconds is not None:
            ax.text(i + 0.5, 0.05, f"{seconds:.2f} s", ha="center", va="center", fontsize=18,
                    color=ACCENT, weight="bold")
        if i < len(stages) - 1:
            ax.annotate("", xy=(i + 1.1, 0.5), xytext=(i + 0.9, 0.5),
                        arrowprops=dict(arrowstyle="->", color=FG, lw=2))
    total = sum(v for v in med.values() if v is not None)
    grounded = sum(1 for turn in run["turns"] if turn["grounded"])
    lines = [
        f"Temps médian par étape sur {len(run['turns'])} questions · total ≈ {total:.1f} s par réponse, sans GPU",
        f"{grounded}/{len(run['turns'])} réponses citent le bon souvenir · mémoire relue après redémarrage de la stack : "
        + ("oui" if run.get("persistence_passed") else "non testé"),
        f"Matériel : {run['hardware']}",
        f"Patiente et proches fictifs (seed {run['seed']}) · rendu par showcase/make_showcase.sh",
    ]
    for i, line in enumerate(lines):
        fig.text(0.04, 0.27 - i * 0.055, line, fontsize=12.5 if i < 2 else 10.5, color=FG if i < 2 else MUTED)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def write_showcase_json(path: Path, run: dict, persistence: dict | None) -> None:
    med = stage_medians(run)
    setup = f"3 questions on a fictitious patient (seed {run['seed']}), {run['hardware']}"
    total = sum(v for v in med.values() if v is not None)
    grounded = sum(1 for turn in run["turns"] if turn["grounded"])
    results = [
        {"label": "Answers citing the expected memory (Whisper -> graph RAG -> LLM -> TTS)",
         "value": f"{grounded}/{len(run['turns'])}", "command": "bash showcase/make_showcase.sh", "setup": setup},
        {"label": "Median voice round trip on CPU (STT + retrieval + LLM + TTS)", "value": f"{total:.1f} s",
         "command": "bash showcase/make_showcase.sh", "setup": setup},
    ]
    for stage, name in (("stt", f"Whisper {run['models_short']['stt']} transcription"),
                        ("retrieval", "Graph RAG retrieval (Neo4j + ChromaDB)"),
                        ("llm", f"{run['llm_short']} answer via Ollama"),
                        ("tts", "Qwen3-TTS 0.6B answer synthesis")):
        if med[stage] is not None:
            results.append({"label": f"Median {name}", "value": f"{med[stage]:.2f} s",
                            "command": "bash showcase/make_showcase.sh", "setup": setup})
    if persistence:
        results.append({"label": "Same answer from stored memory after restarting Neo4j, ChromaDB and Ollama",
                        "value": "pass" if persistence["passed"] else "fail",
                        "command": "MEMENTO_STACK_TESTS=1 uv run pytest -m stack",
                        "setup": f"docker compose restart ({persistence['restart_seconds']} s), fresh clients, "
                                 f"seed {run['seed']}"})
    payload = {
        "slug": "memento",
        "title": "Memento",
        "tagline": "A local, screen-free voice assistant that answers a person with Alzheimer's disease from "
                   "their own memory graph.",
        "problem": "People with Alzheimer's disease ask the same orienting questions many times a day: who is "
                   "coming, where am I, who is this person. Generic assistants do not know their family or "
                   "routines, and a wrong answer is worse than none.",
        "approach": "Caregivers fill a personal knowledge graph (relatives, places, routines, dated memories) "
                    "stored in Neo4j and indexed in ChromaDB. A spoken question is transcribed by Whisper, the "
                    "graph and vector stores return the relevant memories with an explainable clinical score, a "
                    "small local LLM answers from that context only, a grounding guard rejects unsupported "
                    "names, and Qwen3-TTS speaks the answer.",
        "stack": ["Python", "Neo4j", "ChromaDB", "sentence-transformers", "Ollama", "Qwen2.5",
                  "openai-whisper", "Qwen3-TTS", "Docker Compose", "pytest"],
        "demo": {
            "gif": "showcase/demo.gif",
            "mp4": "showcase/demo.mp4",
            "poster": "showcase/poster.png",
            "kind": "replay",
            "caption": "Three spoken questions from a fictitious patient: the transcript, the memory graph nodes "
                       "retrieved from Neo4j/ChromaDB, then the spoken answer, with per-stage CPU timings. "
                       "Rendered from the run recorded by showcase/make_showcase.sh.",
        },
        "reproduce": [
            "docker compose up -d && docker compose exec ollama ollama pull qwen2.5:0.5b",
            "uv sync --all-extras",
            "bash showcase/make_showcase.sh",
        ],
        "results": results,
        "limits": [
            "CPU-only timings: answer TTS dominates the latency (about 10 s per short sentence).",
            "qwen2.5:0.5b sometimes adds loosely grounded details; the grounding guard tolerates one unknown "
            "token, so small slips (a wrong time) can pass.",
            "Demo data is a single fictitious patient; no clinical evaluation has been done.",
            "Live microphone mode and the React runtime are not exercised by the showcase run.",
        ],
        "role": "TODO",
        "repo": "https://github.com/Wassbdr/memento",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _silence(path: Path, seconds: float, rate: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(seconds * rate))


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def render(run_dir: Path, question_dir: Path, out_dir: Path) -> None:
    run = json.loads((run_dir / "roundtrip.json").read_text(encoding="utf-8"))
    persistence = run_dir / "persistence.json"
    if persistence.exists():
        run["persistence_passed"] = json.loads(persistence.read_text(encoding="utf-8"))["passed"]
    run["llm_short"] = run["models"]["llm"].split(" ")[0]
    run["models_short"] = {"stt": run["models"]["stt"].replace("openai-whisper ", "").split(" (")[0]}
    for turn in run["turns"]:
        turn["retrieved_ids"] = [r["id"] for r in turn["retrieved"]]

    work = out_dir / "_frames"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    rate = 24000
    segments: list[tuple[Path, Path, float]] = []

    title = work / "title.png"
    _card(title, "Memento", ["Question posée à voix haute → réponse tirée de son graphe de souvenirs",
                             "Whisper · Neo4j + ChromaDB · LLM local · Qwen3-TTS — tout sur CPU",
                             "Patiente et proches fictifs"])
    silence = work / "title.wav"
    _silence(silence, TITLE_S, rate)
    segments.append((title, silence, TITLE_S))

    for index, turn in enumerate(run["turns"]):
        question_png = work / f"{index}_q.png"
        answer_png = work / f"{index}_a.png"
        _frame(question_png, run, turn, "question", index)
        _frame(answer_png, run, turn, "answer", index)
        q_wav = question_dir / turn["question_wav"]
        q_audio = work / f"{index}_q.wav"
        _ffmpeg("-i", str(q_wav), "-af", f"apad=pad_dur={GAP_S}", "-ar", str(rate), "-ac", "1", str(q_audio))
        segments.append((question_png, q_audio, turn["question_audio_s"] + GAP_S))
        if turn.get("answer_wav"):
            a_audio = work / f"{index}_a.wav"
            _ffmpeg("-i", str(run_dir / turn["answer_wav"]), "-af", f"apad=pad_dur={GAP_S}", "-ar", str(rate),
                    "-ac", "1", str(a_audio))
            segments.append((answer_png, a_audio, turn["answer_audio_s"] + GAP_S))
        else:
            a_audio = work / f"{index}_a.wav"
            _silence(a_audio, 4.0, rate)
            segments.append((answer_png, a_audio, 4.0))

    end = work / "end.png"
    poster(end, run)
    end_audio = work / "end.wav"
    _silence(end_audio, END_S, rate)
    segments.append((end, end_audio, END_S))

    clips = []
    for i, (image, audio, seconds) in enumerate(segments):
        clip = work / f"clip_{i:02d}.mp4"
        _ffmpeg("-loop", "1", "-framerate", "10", "-i", str(image), "-i", str(audio), "-t", f"{seconds:.3f}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "10", "-c:a", "aac", "-ar", "48000", "-ac", "1",
                str(clip))
        clips.append(clip)
    listing = work / "clips.txt"
    listing.write_text("".join(f"file '{clip.name}'\n" for clip in clips), encoding="utf-8")
    _ffmpeg("-f", "concat", "-safe", "0", "-i", str(listing), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "23", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(out_dir / "demo.mp4"))
    _ffmpeg("-i", str(out_dir / "demo.mp4"), "-vf",
            "fps=6,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer",
            "-an", str(out_dir / "demo.gif"))
    poster(out_dir / "poster.png", run)
    shutil.rmtree(work)
    write_showcase_json(out_dir / "showcase.json", run, json.loads(persistence.read_text(encoding="utf-8"))
                        if persistence.exists() else None)
    total = sum(seconds for _, _, seconds in segments)
    print(f"rendered demo.mp4 / demo.gif ({total:.1f}s) and poster.png into {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("showcase/run"))
    parser.add_argument("--question-dir", type=Path, default=Path("showcase/questions"))
    parser.add_argument("--out-dir", type=Path, default=Path("showcase"))
    args = parser.parse_args()
    render(args.run_dir, args.question_dir, args.out_dir)


if __name__ == "__main__":
    main()
