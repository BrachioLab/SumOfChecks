import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd


def _relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _read_frame_ids(frame_csv: Path) -> list[int]:
    try:
        df = pd.read_csv(frame_csv)
    except Exception:
        return []
    if df.empty or "frame_id" not in df.columns:
        return []
    return [int(v) for v in df["frame_id"].dropna().astype(int).tolist()]


def _build_cvs_manifest(root: Path, split: str, out_path: Path) -> None:
    videos_dir = root / split / "videos"
    videos = sorted(videos_dir.glob("*.mp4"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first_id = None
    with out_path.open("w", encoding="utf-8") as f:
        for video_path in videos:
            video_id = video_path.stem
            first_id = first_id or video_id
            label_dir = root / split / "labels" / video_id
            frame_csv = label_dir / "frame.csv"
            rec = {
                "id": video_id,
                "split": split,
                "video_relpath": _relpath(video_path, root),
                "frame_ids": _read_frame_ids(frame_csv) if frame_csv.exists() else [],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(videos)} videos to {out_path}")
    print(f"First example id: {first_id}")


def _parse_endoscapes_frame_name(p: Path) -> tuple[Optional[str], Optional[int]]:
    stem = p.stem
    if "_" not in stem:
        return None, None
    vid_str, fid_str = stem.split("_", 1)
    try:
        fid = int(fid_str)
    except Exception:
        return None, None
    return vid_str, fid


def _build_endoscapes_manifest(root: Path, split: str, out_path: Path) -> None:
    split_dir = root / split
    ann_path = split_dir / "annotation_ds_coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {ann_path}")

    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    images = coco.get("images", [])
    by_video: dict[str, list[tuple[int, str]]] = {}
    for img in images:
        file_name = img.get("file_name") or ""
        file_path = Path(file_name)
        if file_path.is_absolute():
            rel = _relpath(file_path, root)
        else:
            if file_path.parts and file_path.parts[0] == split:
                rel = file_path.as_posix()
            else:
                rel = (Path(split) / file_path).as_posix()

        vid = img.get("video_id")
        frame_id = img.get("frame_id")
        if vid is None or frame_id is None:
            vid_str, fid = _parse_endoscapes_frame_name(Path(file_name))
            if vid_str is None or fid is None:
                continue
            vid = vid_str
            frame_id = fid
        vid = str(vid)
        try:
            fid_int = int(frame_id)
        except Exception:
            continue
        by_video.setdefault(vid, []).append((fid_int, rel))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    first_id = None
    with out_path.open("w", encoding="utf-8") as f:
        for vid in sorted(by_video.keys()):
            pairs = sorted(by_video[vid], key=lambda x: x[0])
            frame_ids = [fid for fid, _ in pairs]
            frame_paths = [p for _, p in pairs]
            first_id = first_id or vid
            rec = {
                "id": vid,
                "split": split,
                "video_relpath": "",
                "frame_ids": frame_ids,
                "video_frame_paths": frame_paths,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(by_video)} videos to {out_path}")
    print(f"First example id: {first_id}")


def main() -> int:
    p = argparse.ArgumentParser(description="Build JSONL manifests for CVS or Endoscapes datasets.")
    p.add_argument("--dataset", required=True, choices=["cvs", "endoscapes"])
    p.add_argument("--root", required=True, help="Dataset root (e.g., data/CVS_Challenge_SAGES_v1 or data/endoscapes)")
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True, help="Output JSONL path")
    args = p.parse_args()

    root = Path(args.root)
    out_path = Path(args.out)

    if args.dataset == "cvs":
        if args.split not in ("train", "test"):
            raise ValueError("CVS split must be 'train' or 'test'.")
        _build_cvs_manifest(root, args.split, out_path)
    else:
        _build_endoscapes_manifest(root, args.split, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
