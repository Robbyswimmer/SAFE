from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_final_metrics(history: List[Dict[str, Any]]) -> Dict[str, float]:
    if not history:
        return {}
    row = history[-1]
    eval_block = row.get("eval", {})
    out: Dict[str, float] = {}
    for modality in ("text", "audio", "image", "both"):
        metrics = eval_block.get(modality, {})
        if metrics:
            out[f"{modality}_exact_match"] = float(metrics.get("exact_match", 0.0))
            out[f"{modality}_extracted_match"] = float(metrics.get("extracted_match", 0.0))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize composition calibration sweep runs into CSV")
    p.add_argument("--sweep-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(args.sweep_root.iterdir()):
        if not run_dir.is_dir():
            continue
        results_path = run_dir / "results.json"
        history_path = run_dir / "history.json"
        if not results_path.exists() or not history_path.exists():
            continue

        results = _load_json(results_path)
        history = _load_json(history_path)
        row: Dict[str, Any] = {
            "run_dir": str(run_dir),
            "train_max_samples": int(results.get("train_max_samples", 0)),
            "compose_calibration_trainable": str(results.get("compose_calibration_trainable", "")),
            "best_exact_match": float(results.get("best_exact_match", 0.0)),
        }
        row.update(_extract_final_metrics(history))
        rows.append(row)

    rows.sort(key=lambda r: (int(r.get("train_max_samples", 0)), str(r.get("compose_calibration_trainable", ""))))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_dir",
        "train_max_samples",
        "compose_calibration_trainable",
        "best_exact_match",
        "text_exact_match",
        "text_extracted_match",
        "audio_exact_match",
        "audio_extracted_match",
        "image_exact_match",
        "image_extracted_match",
        "both_exact_match",
        "both_extracted_match",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[summary] wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
