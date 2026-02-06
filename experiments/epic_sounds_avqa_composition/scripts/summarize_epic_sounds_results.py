#!/usr/bin/env python3
"""
Summarize EPIC-SOUNDS AV-QA experiment results and optionally update RESEARCH_AGENDA.

Expected run dirs (defaults):
- checkpoints/epic_sounds_avqa/preffn
- checkpoints/epic_sounds_avqa/kv_augment

Each run directory should contain:
- history.json (epoch-wise metrics)
- results.json (high-level metadata)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


START_MARKER = "<!-- EPIC_AVQA_RESULTS_START -->"
END_MARKER = "<!-- EPIC_AVQA_RESULTS_END -->"


@dataclass
class RunSummary:
    run_name: str
    architecture: str
    path: Path
    best_epoch: int
    both_exact: float
    both_f1: float
    av_exact: float
    av_f1: float
    audio_exact: Optional[float]
    image_exact: Optional[float]



def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



def _best_epoch_entry(history: List[Dict[str, Any]], key_modality: str = "both") -> Tuple[int, Dict[str, Any]]:
    best_idx = -1
    best_score = float("-inf")
    best_entry: Dict[str, Any] = {}

    for i, entry in enumerate(history):
        eval_obj = entry.get("eval", {})
        score = eval_obj.get(key_modality, {}).get("exact_match", float("-inf"))
        if score > best_score:
            best_score = score
            best_idx = i
            best_entry = entry

    if best_idx < 0:
        raise ValueError("No valid eval entries found in history")
    return best_idx + 1, best_entry



def summarize_run(run_path: Path, run_name: str) -> RunSummary:
    history_path = run_path / "history.json"
    results_path = run_path / "results.json"

    if not history_path.exists():
        raise FileNotFoundError(f"Missing history file: {history_path}")

    history = _read_json(history_path)
    if not isinstance(history, list) or not history:
        raise ValueError(f"History is empty or malformed: {history_path}")

    best_epoch, best = _best_epoch_entry(history, key_modality="both")
    eval_obj = best.get("eval", {})

    architecture = "unknown"
    if results_path.exists():
        results = _read_json(results_path)
        architecture = str(results.get("architecture", architecture))

    both = eval_obj.get("both", {})
    audio = eval_obj.get("audio", {})
    image = eval_obj.get("image", {})
    both_by_type = both.get("by_question_type", {})
    av = both_by_type.get("av_composition", {})

    return RunSummary(
        run_name=run_name,
        architecture=architecture,
        path=run_path,
        best_epoch=best_epoch,
        both_exact=float(both.get("exact_match", 0.0)),
        both_f1=float(both.get("token_f1", 0.0)),
        av_exact=float(av.get("exact_match", 0.0)),
        av_f1=float(av.get("token_f1", 0.0)),
        audio_exact=float(audio.get("exact_match", 0.0)) if audio else None,
        image_exact=float(image.get("exact_match", 0.0)) if image else None,
    )



def build_markdown(summaries: List[RunSummary]) -> str:
    lines: List[str] = []
    today = date.today().isoformat()

    lines.append("### Phase 5A EPIC-SOUNDS Result Snapshot")
    lines.append("")
    lines.append(f"Updated: {today}")
    lines.append("")
    lines.append("| Run | Architecture | Best Epoch | EM (both) | F1 (both) | EM (av_composition) | F1 (av_composition) | EM (audio eval) | EM (image eval) | Artifact |")
    lines.append("|-----|--------------|------------|-----------|-----------|----------------------|----------------------|------------------|------------------|----------|")

    for s in summaries:
        audio_em = f"{s.audio_exact:.2f}" if s.audio_exact is not None else "N/A"
        image_em = f"{s.image_exact:.2f}" if s.image_exact is not None else "N/A"
        lines.append(
            "| "
            f"{s.run_name} | {s.architecture} | {s.best_epoch} | {s.both_exact:.2f} | {s.both_f1:.2f} | "
            f"{s.av_exact:.2f} | {s.av_f1:.2f} | {audio_em} | {image_em} | `{s.path}` |"
        )

    lines.append("")
    if len(summaries) >= 2:
        # compare top two by av EM
        ranked = sorted(summaries, key=lambda x: x.av_exact, reverse=True)
        best = ranked[0]
        second = ranked[1]
        delta = best.av_exact - second.av_exact
        lines.append("**Quick Read**:")
        lines.append(f"- Best AV composition EM: `{best.run_name}` ({best.av_exact:.2f})")
        lines.append(f"- Margin vs next run: {delta:.2f} EM")

    return "\n".join(lines)



def update_research_agenda(agenda_path: Path, markdown_block: str) -> None:
    text = agenda_path.read_text(encoding="utf-8")
    start_idx = text.find(START_MARKER)
    end_idx = text.find(END_MARKER)

    wrapped = f"{START_MARKER}\n{markdown_block}\n{END_MARKER}"

    if start_idx >= 0 and end_idx >= 0 and end_idx > start_idx:
        new_text = text[:start_idx] + wrapped + text[end_idx + len(END_MARKER):]
    else:
        insertion = (
            "\n\n"
            "### Phase 5A Auto Summary\n\n"
            f"{wrapped}\n"
        )
        new_text = text + insertion

    agenda_path.write_text(new_text, encoding="utf-8")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize EPIC-SOUNDS AV-QA run results")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run specification: name=path (repeatable). Example: preffn=checkpoints/epic_sounds_avqa/preffn",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("experiments/epic_sounds_avqa_composition/results/latest_summary.md"),
    )
    parser.add_argument(
        "--update-research-agenda",
        action="store_true",
        help="Update docs/RESEARCH_AGENDA.md between EPIC markers",
    )
    parser.add_argument(
        "--research-agenda-path",
        type=Path,
        default=Path("docs/RESEARCH_AGENDA.md"),
    )
    return parser.parse_args()



def parse_run_specs(specs: List[str]) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --run spec '{spec}'. Use name=path")
        name, path = spec.split("=", 1)
        name = name.strip()
        p = Path(path.strip())
        if not name:
            raise ValueError(f"Invalid empty run name in spec '{spec}'")
        out.append((name, p))
    return out



def main() -> None:
    args = parse_args()

    if args.run:
        run_specs = parse_run_specs(args.run)
    else:
        run_specs = [
            ("epic_preffn", Path("checkpoints/epic_sounds_avqa/preffn")),
            ("epic_kvaug", Path("checkpoints/epic_sounds_avqa/kv_augment")),
        ]

    summaries: List[RunSummary] = []
    for name, path in run_specs:
        if not path.exists():
            print(f"[warn] skipping missing run path: {path}")
            continue
        try:
            summaries.append(summarize_run(path, name))
        except Exception as exc:
            print(f"[warn] failed to summarize {name} ({path}): {exc}")

    if not summaries:
        raise RuntimeError("No runs could be summarized")

    markdown = build_markdown(summaries)

    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown + "\n", encoding="utf-8")
    print(f"[ok] wrote summary markdown: {args.output_markdown}")

    if args.update_research_agenda:
        update_research_agenda(args.research_agenda_path, markdown)
        print(f"[ok] updated research agenda: {args.research_agenda_path}")

    print("\n" + markdown)


if __name__ == "__main__":
    main()
