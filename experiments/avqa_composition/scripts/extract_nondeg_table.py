#!/usr/bin/env python3
"""Extract non-degradation table from training history.json files.

Reads history.json from one or more checkpoint directories and prints:
- text_em (frozen VLM baseline) at each epoch
- audio_em, image_em, both_em
- Composition gain and text_em stability

Usage:
    python experiments/avqa_composition/scripts/extract_nondeg_table.py \
        checkpoints/bn_sweep_8b/bn756_*/ \
        checkpoints/internvl_4b/ \
        checkpoints/internvl_1b/

Outputs LaTeX-ready table for paper Table 2 (non-degradation verification).
"""

import argparse
import json
import sys
from pathlib import Path


def load_history(path: Path):
    hf = path / "history.json"
    if not hf.exists():
        # Try parent
        hf = path.parent / "history.json"
    if not hf.exists():
        return None
    with open(hf) as f:
        return json.load(f)


def extract_metrics(history):
    rows = []
    for entry in history:
        epoch = entry.get("epoch", "?")
        ev = entry.get("eval", {})
        text_em = ev.get("text", {}).get("extracted_match", None)
        audio_em = ev.get("audio", {}).get("extracted_match", None)
        image_em = ev.get("image", {}).get("extracted_match", None)
        both_em = ev.get("both", {}).get("extracted_match", None)
        rows.append({
            "epoch": epoch,
            "text_em": text_em,
            "audio_em": audio_em,
            "image_em": image_em,
            "both_em": both_em,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Extract non-degradation table")
    parser.add_argument("dirs", nargs="+", help="Checkpoint directories with history.json")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX table")
    args = parser.parse_args()

    for d in args.dirs:
        p = Path(d)
        history = load_history(p)
        if history is None:
            print(f"[SKIP] No history.json in {p}", file=sys.stderr)
            continue

        rows = extract_metrics(history)
        if not rows:
            print(f"[SKIP] No epochs in {p}", file=sys.stderr)
            continue

        print(f"\n{'='*70}")
        print(f"  {p.name}")
        print(f"{'='*70}")

        # Check text_em stability
        text_vals = [r["text_em"] for r in rows if r["text_em"] is not None]
        if text_vals:
            text_range = max(text_vals) - min(text_vals)
            stable = text_range < 0.5  # <0.5pp drift = stable
            print(f"  Text EM range: {min(text_vals):.2f}% – {max(text_vals):.2f}%  "
                  f"(drift={text_range:.2f}pp, {'STABLE' if stable else 'DRIFT'})")
        print()

        # Table header
        print(f"  {'Epoch':>5}  {'Text':>7}  {'Audio':>7}  {'Image':>7}  {'Both':>7}  {'Gain':>7}")
        print(f"  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")

        for r in rows:
            def fmt(v):
                return f"{v:7.2f}" if v is not None else "    —  "
            gain = ""
            if r["both_em"] is not None and r["audio_em"] is not None and r["image_em"] is not None:
                g = r["both_em"] - max(r["audio_em"], r["image_em"])
                gain = f"{g:+7.2f}"
            else:
                gain = "    —  "
            print(f"  {r['epoch']:>5}  {fmt(r['text_em'])}  {fmt(r['audio_em'])}  "
                  f"{fmt(r['image_em'])}  {fmt(r['both_em'])}  {gain}")

        # Best epoch
        best = max(rows, key=lambda r: r["both_em"] or 0)
        if best["both_em"] is not None:
            print(f"\n  Best: epoch {best['epoch']}, both={best['both_em']:.2f}%")

    if args.latex:
        print("\n% LaTeX table (paste into paper)")
        print("\\begin{tabular}{lccccc}")
        print("\\toprule")
        print("Config & Epoch & Text & Audio & Image & Both \\\\")
        print("\\midrule")
        for d in args.dirs:
            p = Path(d)
            history = load_history(p)
            if history is None:
                continue
            rows = extract_metrics(history)
            best = max(rows, key=lambda r: r["both_em"] or 0)
            def fmt(v):
                return f"{v:.1f}" if v is not None else "—"
            print(f"{p.name} & {best['epoch']} & {fmt(best['text_em'])} & "
                  f"{fmt(best['audio_em'])} & {fmt(best['image_em'])} & "
                  f"{fmt(best['both_em'])} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")


if __name__ == "__main__":
    main()
