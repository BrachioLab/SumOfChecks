#!/usr/bin/env python3
"""Verify frozen predictions and export the paper notebook's calculations."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import os

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def verify_predictions():
    manifest = json.loads((ROOT / "docs/prediction_manifest.json").read_text())
    for entry in manifest:
        path = ROOT / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Prediction checksum mismatch: {path}")
    return manifest


def evaluate():
    verify_predictions()
    notebook = json.loads((ROOT / "notebooks/rubrics/compact_endoscapes_map.ipynb").read_text())
    namespace = {"display": lambda *args, **kwargs: None}
    previous = Path.cwd()
    try:
        os.chdir(ROOT)
        with contextlib.redirect_stdout(io.StringIO()):
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), f"paper_table_cell_{index}", "exec"), namespace)
    finally:
        os.chdir(previous)
    return namespace


def main():
    ns = evaluate()
    target = ROOT / "results/paper"
    target.mkdir(parents=True, exist_ok=True)
    report = {"units": "fraction", "table1": ns["table_stats"], "ablation": ns["abl_stats"],
              "notes": "See docs/reproduction.md for run counts, scoring differences, and the unresolved Opus SubQ+FS row."}
    (target / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (target / "table1.tex").write_text("\n".join(line.rstrip() for line in ns["latex_str"].splitlines()) + "\n")
    (target / "ablation.tex").write_text("\n".join(line.rstrip() for line in ns["abl_latex"].splitlines()) + "\n")
    filtered = pd.concat([df[df["model"] == model] for model, df in ns["all_df_f"].items()])
    counts = filtered.groupby(["model", "experiment_label", "criterion", "score_type", "copy_idx"])["file_name"].nunique()
    counts.rename("valid_frames").to_csv(target / "frame_counts.csv")
    for model, stats in ns["table_stats"].items():
        mean, std = stats["OURS"]["avg"]
        print(f"{model}: Sum-of-Checks mAP {100 * mean:.1f} +/- {100 * std:.1f}%")
    print(f"Results: {target}")
    print(report["notes"])


if __name__ == "__main__":
    main()
