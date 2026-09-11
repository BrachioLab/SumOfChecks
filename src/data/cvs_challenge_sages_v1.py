"""CVS Challenge SAGES v1 dataset loader.

Available manifest names in this repo (data/manifests/cvs_challenge_sages_v1/*.jsonl):
- train_manifest
- train_dev
- train_dev5
- train_rest
- test_manifest

You can also pass an explicit manifest JSONL path via manifest_jsonl.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def abs_posix(p: Path) -> str:
    return p.resolve().as_posix()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _frame_rows_from_dataset(dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "frames" in dataset and isinstance(dataset["frames"], list):
        return dataset["frames"]
    frames: List[Dict[str, Any]] = []
    videos = dataset.get("videos") or {}
    if isinstance(videos, dict):
        for vid in videos.values():
            if isinstance(vid, dict) and isinstance(vid.get("frames"), list):
                frames.extend(vid["frames"])
    return frames


def _load_video_csv(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
    if not row:
        return {}
    raw = {k: row.get(k) for k in row.keys()}

    def _score_for(prefix: str) -> Dict[str, Any]:
        votes: List[float] = []
        for i in (1, 2, 3):
            key = f"{prefix}_rater{i}"
            val = row.get(key)
            if val is None or str(val).strip() == "":
                continue
            try:
                votes.append(float(val))
            except Exception:
                continue
        cont = sum(votes) / 3.0 if votes else None
        binary = None
        if cont is not None:
            binary = 1 if cont > 0.5 else 0
        return {
            "raw_votes": votes,
            "continuous": cont,
            "binary": binary,
        }

    scores = {
        "c1": _score_for("c1"),
        "c2": _score_for("c2"),
        "c3": _score_for("c3"),
    }
    return {"raw": raw, "scores": scores}


def get_video_data(dataset: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    frame_rows = _frame_rows_from_dataset(dataset)
    labels_root = dataset.get("labels_root")
    videos: Dict[str, Dict[str, Any]] = {}

    for row in frame_rows:
        vid = str(row.get("video_id"))
        split = row.get("split")
        videos.setdefault(vid, {"video_id": vid, "split": split, "frames": []})
        videos[vid]["frames"].append(
            {
                "frame_id": row.get("frame_id"),
                "image_path": row.get("image_path"),
                "labels": row.get("labels", {}),
            }
        )

    if labels_root is not None:
        labels_root = Path(labels_root)
        for vid, info in videos.items():
            split = info.get("split")
            if not split:
                info["video_labels"] = {}
                continue
            csv_path = labels_root / str(split) / "labels" / str(vid) / "video.csv"
            info["video_labels"] = _load_video_csv(csv_path)
    else:
        for info in videos.values():
            info["video_labels"] = {}

    return videos


@dataclass(frozen=True)
class CvsChallengeSagesDataset:
    name: str = "cvs_challenge_sages_v1"
    manifests: Tuple[str, ...] = (
        "train_manifest",
        "train_dev",
        "train_dev5",
        "train_rest",
        "test_manifest",
    )

    def available_manifests(self) -> Tuple[str, ...]:
        return self.manifests

    def resolve_manifest_path(
        self,
        *,
        manifest: Optional[str],
        frames_root: Optional[Path],
        manifest_root: Optional[Path],
        manifest_jsonl: Optional[Path],
    ) -> Optional[Path]:
        if manifest_jsonl is not None:
            return manifest_jsonl
        if manifest is None:
            return None
        if manifest_root is not None:
            return (manifest_root / f"{manifest}.jsonl").resolve()
        if frames_root is None:
            raise ValueError("frames_root is required for cvs_challenge_sages_v1")
        repo_root = frames_root.parent
        return (repo_root / "manifests" / self.name / f"{manifest}.jsonl").resolve()

    def load(
        self,
        *,
        manifest_jsonl: Optional[Path] = None,
        manifest: Optional[str] = None,
        split: Optional[str] = None,
        frames_root: Optional[Path] = None,
        manifest_root: Optional[Path] = None,
        labels_root: Optional[Path] = None,
    ) -> Dict[str, Any]:
        manifest_path = self.resolve_manifest_path(
            manifest=manifest,
            frames_root=frames_root,
            manifest_root=manifest_root,
            manifest_jsonl=manifest_jsonl,
        )
        if manifest_path is None:
            raise ValueError("manifest_jsonl or manifest is required for cvs_challenge_sages_v1")
        if frames_root is None:
            raise ValueError("frames_root is required for cvs_challenge_sages_v1")
        return load_cvs_challenge_sages_v1(
            manifest_jsonl=manifest_path,
            frames_root=frames_root,
            labels_root=labels_root,
            split_override=split,
        )


def _majority_vote(votes: List[int]) -> Optional[int]:
    if not votes:
        return None
    threshold = (len(votes) + 1) // 2
    return 1 if sum(votes) >= threshold else 0


def _summarize_votes(votes: List[Any]) -> Dict[str, Any]:
    clean: List[float] = []
    for v in votes:
        try:
            if v is None or (isinstance(v, str) and not v.strip()):
                continue
            clean.append(float(v))
        except Exception:
            continue
    bin_votes = [int(v) for v in clean if v in (0.0, 1.0)]
    binary = _majority_vote(bin_votes)
    continuous = None
    if bin_votes:
        continuous = float(sum(bin_votes) / len(bin_votes))
    return {"binary": binary, "continuous": continuous, "votes": clean}


def _parse_cvs_row_labels(row: Dict[str, Any], *, criterion: str) -> Dict[str, Any]:
    rater_votes: List[Any] = []
    continuous_votes: List[Any] = []
    for k, v in row.items():
        if k.startswith(f"{criterion}_rater"):
            rater_votes.append(v)
        elif k in (f"{criterion}_score", f"{criterion}_continuous", f"{criterion}_prob"):
            continuous_votes.append(v)
    summary = _summarize_votes(rater_votes)
    if continuous_votes:
        cont_summary = _summarize_votes(continuous_votes)
        summary["continuous"] = cont_summary.get("continuous")
    return summary


def load_cvs_challenge_sages_v1(
    *,
    manifest_jsonl: Path,
    frames_root: Path,
    labels_root: Optional[Path] = None,
    split_override: Optional[str] = None,
) -> Dict[str, Any]:
    manifest_rows = list(iter_jsonl(manifest_jsonl))
    videos: Dict[str, Dict[str, Any]] = {}
    frame_rows: List[Dict[str, Any]] = []

    for rec in manifest_rows:
        split = rec.get("split") or split_override
        if not split:
            raise ValueError("split missing in manifest and no split_override provided")
        video_id = str(rec.get("id") or rec.get("video_id"))
        frame_ids = rec.get("frame_ids") or []
        if not video_id or not frame_ids:
            continue
        videos[video_id] = {"video_id": video_id, "frames": []}

        labels_by_frame: Dict[int, Dict[str, Any]] = {}
        if labels_root is not None:
            labels_path = (labels_root / split / "labels" / video_id / "frame.csv").resolve()
            if labels_path.exists():
                with labels_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        fid = int(row["frame_id"])
                        labels_by_frame[fid] = {
                            "c1": _parse_cvs_row_labels(row, criterion="c1"),
                            "c2": _parse_cvs_row_labels(row, criterion="c2"),
                            "c3": _parse_cvs_row_labels(row, criterion="c3"),
                        }

        for fid in frame_ids:
            try:
                fid_int = int(fid)
            except Exception:
                continue
            img_p = (frames_root / str(split) / str(video_id) / f"frame_{fid_int:06d}.jpg").resolve()
            frame_row = {
                "frame_id": fid_int,
                "image_path": abs_posix(img_p),
                "labels": labels_by_frame.get(fid_int, {}),
                "split": split,
                "video_id": video_id,
            }
            videos[video_id]["frames"].append(frame_row)
            frame_rows.append(frame_row)
        videos[video_id]["frames"].sort(key=lambda r: r.get("frame_id", -1))

    return {
        "name": "cvs_challenge_sages_v1",
        "split": split_override or "mixed",
        "frames_root": frames_root,
        "labels_root": labels_root,
        "videos": videos,
        "frames": frame_rows,
    }
