import argparse
import json
import random
from pathlib import Path


def _load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _dev_count(dev_size: str, total: int) -> int:
    try:
        f = float(dev_size)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid --dev_size: {dev_size!r}") from e

    if 0 < f < 1:
        n = int(total * f)
        if total > 0 and n == 0:
            n = 1
        return n
    if f >= 1 and f.is_integer():
        return int(f)
    raise argparse.ArgumentTypeError("--dev_size must be an int >= 1 or a float in (0, 1)")


def main() -> int:
    p = argparse.ArgumentParser(description="Split a JSONL manifest into dev and rest subsets.")
    p.add_argument("--input", required=True, help="Input manifest JSONL (e.g., data/train_manifest.jsonl)")
    p.add_argument("--dev_size", required=True, help="Dev size: int count (>=1) or float fraction (0,1)")
    p.add_argument(
        "--dataset",
        choices=["cvs", "endoscapes"],
        default="cvs",
        help="Dataset type. Both datasets are split by video; Endoscapes also reports frame counts.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = p.parse_args()

    in_path = Path(args.input)
    records = _load_jsonl(in_path)
    rng = random.Random(args.seed)
    if args.dataset == "endoscapes":
        total = len(records)
        dev_n = min(_dev_count(args.dev_size, total), total)
        rng.shuffle(records)
        dev_records = records[:dev_n]
        rest_records = records[dev_n:]
    else:
        total = len(records)
        dev_n = min(_dev_count(args.dev_size, total), total)
        rng.shuffle(records)
        dev_records = records[:dev_n]
        rest_records = records[dev_n:]

    stem = in_path.stem
    prefix = stem[:-9] if stem.endswith("_manifest") else stem
    dev_path = in_path.parent / f"{prefix}_dev.jsonl"
    rest_path = in_path.parent / f"{prefix}_rest.jsonl"

    with dev_path.open("w", encoding="utf-8") as f:
        for r in dev_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with rest_path.open("w", encoding="utf-8") as f:
        for r in rest_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Total samples: {total}")
    if args.dataset == "endoscapes":
        dev_frames = sum(len(r.get("frame_ids") or []) for r in dev_records)
        rest_frames = sum(len(r.get("frame_ids") or []) for r in rest_records)
        print(f"Dev videos: {len(dev_records)} (frames: {dev_frames})")
        print(f"Train_rest videos: {len(rest_records)} (frames: {rest_frames})")
    else:
        print(f"Dev samples: {len(dev_records)}")
        print(f"Train_rest samples: {len(rest_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
