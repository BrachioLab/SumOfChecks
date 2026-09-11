#!/usr/bin/env python3
"""Run the seven paper conditions and optional no-FS ablation."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["gpt-4.1-mini", "claude-haiku-4-5-20251001", "claude-opus-4-5-20251101"]
FS = ["--few_shot", "--few_shot_k", "4", "--few_shot_json_type", "fs_v3"]
METHODS = {
    "direct": ["--preset", "direct"],
    "direct_fs": ["--preset", "direct", *FS],
    "cot": ["--preset", "direct", "--include_rationale_direct_first"],
    "cot_fs": ["--preset", "direct", "--include_rationale_direct_first", *FS],
    "subq": ["--preset", "self_rubric", "--include_rationale_stage1"],
    "subq_fs": ["--preset", "self_rubric", "--include_rationale_stage1", *FS],
    "sum_of_checks": ["--preset", "main", *FS],
    "no_fs": ["--preset", "main"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS)[:-1])
    parser.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/generated")
    args = parser.parse_args()
    if any(run < 1 for run in args.runs):
        parser.error("Run numbers must be positive.")
    for model in args.models:
        for run in args.runs:
            for method in args.methods:
                command = [sys.executable, str(ROOT / "scripts/rubric_oracle_reasoning_frame.py"),
                           "--manifest_jsonl", str(ROOT / "data/manifests/endoscapes/test_dev.jsonl"),
                           "--use_manifest_rows", "--model", model, "--temperature", "0.1",
                           "--run", str(run), "--output_root", str(args.output_root.resolve()),
                           *METHODS[method]]
                if args.dryrun:
                    command += ["--dryrun"]
                limit = args.max_items if args.max_items is not None else (1 if args.dryrun else None)
                if limit is not None:
                    command += ["--max_items", str(limit)]
                print(f"{model}: {method}, run {run}", flush=True)
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
