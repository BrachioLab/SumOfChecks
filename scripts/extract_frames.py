import argparse
import json
from pathlib import Path

import cv2
from tqdm import tqdm


def _iter_manifest(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _frame_outpath(out_root: Path, split: str, video_id: str, frame_id: int) -> Path:
    return out_root / split / video_id / f"frame_{frame_id:06d}.jpg"


def main() -> int:
    p = argparse.ArgumentParser(description="Extract sampled frames from videos listed in a manifest JSONL.")
    p.add_argument("--root", required=True, help="Dataset root (e.g., data/CVS_Challenge_SAGES_v1)")
    p.add_argument("--manifest", required=True, help="Manifest JSONL path")
    p.add_argument("--out", required=True, help="Output folder (e.g., data/frames)")
    args = p.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    manifest_path = Path(args.manifest)

    records = list(_iter_manifest(manifest_path))
    total_written = 0
    total_skipped = 0

    for rec in tqdm(records, total=len(records), unit="video", desc="Videos"):
        video_id = rec["id"]
        split = rec["split"]
        video_path = root / rec["video_relpath"]
        frame_ids = sorted({int(x) for x in rec.get("frame_ids", [])})

        pending = []
        for fid in frame_ids:
            out_path = _frame_outpath(out_root, split, video_id, fid)
            if out_path.exists():
                total_skipped += 1
            else:
                pending.append(fid)
        if not pending:
            continue
        if not video_path.exists():
            tqdm.write(f"{video_id}: missing video {video_path}")
            continue

        (out_root / split / video_id).mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            tqdm.write(f"{video_id}: failed to open video {video_path}")
            continue

        pending_set = set(pending)
        max_fid = max(pending)
        written = 0
        idx = 0
        while pending_set and idx <= max_fid:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in pending_set:
                out_path = _frame_outpath(out_root, split, video_id, idx)
                if not out_path.exists():
                    if cv2.imwrite(str(out_path), frame):
                        written += 1
                    else:
                        tqdm.write(f"{video_id}: failed to write frame {idx} -> {out_path}")
                pending_set.remove(idx)
            idx += 1
        cap.release()
        total_written += written

    print(f"Total videos processed: {len(records)}")
    print(f"Total frames written: {total_written}")
    print(f"Total frames skipped: {total_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
