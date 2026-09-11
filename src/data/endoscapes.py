"""Endoscapes dataset loader.

Available manifest names in this repo (data/manifests/endoscapes/*.jsonl):
- train_manifest
- val_manifest
- val_dev
- val_rest
- test_manifest
- test_dev
- test_rest

You can also pass an explicit manifest JSONL path via manifest_jsonl.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_ANN_FILE = "annotation_ds_coco.json"


@dataclass(frozen=True)
class EndoscapesDataset:
    name: str = "endoscapes"
    manifests: Tuple[str, ...] = (
        "train_manifest",
        "val_manifest",
        "val_dev",
        "val_rest",
        "test_manifest",
        "test_dev",
        "test_rest",
    )

    def available_manifests(self) -> Tuple[str, ...]:
        return self.manifests

    def resolve_manifest_path(
        self,
        *,
        manifest: Optional[str],
        data_root: Optional[Path],
        manifest_root: Optional[Path],
        manifest_jsonl: Optional[Path],
    ) -> Optional[Path]:
        if manifest_jsonl is not None:
            return manifest_jsonl
        if manifest is None:
            return None
        if manifest_root is not None:
            return (manifest_root / f"{manifest}.jsonl").resolve()
        if data_root is None:
            raise ValueError("data_root is required for endoscapes")
        repo_root = data_root.parent
        return (repo_root / "manifests" / self.name / f"{manifest}.jsonl").resolve()

    def load(
        self,
        *,
        manifest_jsonl: Optional[Path] = None,
        manifest: Optional[str] = None,
        split: Optional[str] = None,
        data_root: Optional[Path] = None,
        manifest_root: Optional[Path] = None,
        ann_file: Optional[str] = None,
        manifest_only: bool = False,
        few_shot_jsonl: Optional[Path] = None,
        split_few_shot: bool = False,
    ) -> Dict[str, Any]:
        if data_root is None:
            raise ValueError("data_root is required for endoscapes")
        manifest_path = self.resolve_manifest_path(
            manifest=manifest,
            data_root=data_root,
            manifest_root=manifest_root,
            manifest_jsonl=manifest_jsonl,
        )
        ds = load_endoscapes(
            split=split,
            ann_file=ann_file or DEFAULT_ANN_FILE,
            manifest_jsonl=manifest_path,
            data_root=data_root,
            manifest_only=manifest_only,
        )
        if few_shot_jsonl is not None:
            few_shot_rows = load_endoscapes_few_shot(
                few_shot_jsonl=few_shot_jsonl,
                data_root=data_root,
                default_split=split or "val",
            )
            ds["few_shot_examples"] = few_shot_rows
            if split_few_shot:
                few_shot_frames, rest_frames = split_rows_by_few_shot(
                    frame_rows=ds.get("frames", []),
                    few_shot_rows=few_shot_rows,
                )
                ds["few_shot_frames"] = few_shot_frames
                ds["rest_frames"] = rest_frames
        return ds


def abs_posix(p: Path) -> str:
    return p.resolve().as_posix()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _endoscapes_frame_num_from_filename(file_name: str) -> int:
    stem = Path(file_name).stem
    if "_" in stem:
        return int(stem.split("_", 1)[1])
    return int(stem)


def _endoscapes_video_id_from_filename(file_name: str) -> Optional[str]:
    stem = Path(file_name).stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return None


def resolve_image_path(
    image_path: Optional[str],
    file_name: Optional[str],
    *,
    split: str,
    dataset_root: Path,
) -> Path:
    if image_path:
        p = Path(image_path)
        if p.is_absolute():
            return p
        if p.parts and p.parts[0] == split:
            return (dataset_root / p).resolve()
        return (dataset_root / split / p).resolve()
    if file_name:
        p = Path(file_name)
        if p.is_absolute():
            return p
        if p.parts and p.parts[0] == split:
            return (dataset_root / p).resolve()
        return (dataset_root / split / p).resolve()
    return Path()


def _label_binary(label: Any) -> Optional[int]:
    if label is None:
        return None
    if isinstance(label, dict):
        val = label.get("binary")
        if val is not None:
            try:
                return int(val)
            except Exception:
                return None
        cont = label.get("continuous")
        if cont is not None:
            try:
                return 1 if float(cont) > 0.5 else 0
            except Exception:
                return None
        return None
    try:
        return int(label)
    except Exception:
        return None


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


def get_video_data(dataset: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    frame_rows = _frame_rows_from_dataset(dataset)
    by_video_any: Dict[str, Dict[str, bool]] = {}
    videos: Dict[str, Dict[str, Any]] = {}

    for row in frame_rows:
        vid = str(row.get("video_id"))
        labels = row.get("labels") or {}
        videos.setdefault(
            vid,
            {
                "video_id": vid,
                "split": row.get("split"),
                "frames": [],
                "video_labels": {},
            },
        )
        videos[vid]["frames"].append(
            {
                "frame_id": row.get("frame_id"),
                "image_path": row.get("image_path"),
                "labels": labels,
            }
        )
        for crit in ("c1", "c2", "c3"):
            val = _label_binary(labels.get(crit))
            if val is None:
                continue
            by_video_any.setdefault(vid, {}).setdefault(crit, False)
            if val == 1:
                by_video_any[vid][crit] = True

    for vid, crits in by_video_any.items():
        out: Dict[str, int] = {}
        for crit in ("c1", "c2", "c3"):
            if crit in crits:
                out[crit] = 1 if crits[crit] else 0
        if vid in videos:
            videos[vid]["video_labels"] = out

    return videos


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
    if clean:
        # Endoscapes "ds" can already be averaged; keep mean as continuous.
        continuous = float(sum(clean) / len(clean))
    if binary is None and continuous is not None:
        binary = 1 if continuous > 0.5 else 0
    return {"binary": binary, "continuous": continuous, "votes": clean}


def _parse_endoscapes_ds_labels(ds: Any) -> Dict[str, Any]:
    if not isinstance(ds, (list, tuple)):
        return {}
    labels: Dict[str, Any] = {}
    for idx, criterion in enumerate(("c1", "c2", "c3")):
        if idx >= len(ds):
            continue
        val = ds[idx]
        if isinstance(val, (list, tuple)):
            labels[criterion] = _summarize_votes(list(val))
        else:
            labels[criterion] = _summarize_votes([val])
    return labels


def load_endoscapes_manifest_only(
    *,
    manifest_jsonl: Path,
    data_root: Path,
    ann_file: str = DEFAULT_ANN_FILE,
    include_labels: bool = True,
) -> List[Dict[str, Any]]:
    label_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if include_labels:
        splits: set[str] = set()
        for rec in iter_jsonl(manifest_jsonl):
            sp = rec.get("split")
            if sp:
                splits.add(str(sp))
        for sp in splits:
            ann_path = (data_root / sp / ann_file).resolve()
            coco = load_json(ann_path) if ann_path.exists() else {}
            images = coco.get("images", [])
            for img in images:
                file_name = img.get("file_name")
                video_id = img.get("video_id")
                if video_id is None and file_name:
                    video_id = _endoscapes_video_id_from_filename(str(file_name))
                if video_id is None:
                    continue
                frame_num = img.get("frame_id")
                if frame_num is None and file_name:
                    try:
                        frame_num = _endoscapes_frame_num_from_filename(str(file_name))
                    except Exception:
                        frame_num = None
                try:
                    frame_id = int(frame_num)
                except Exception:
                    continue
                labels = _parse_endoscapes_ds_labels(img.get("ds"))
                label_map[(str(video_id), frame_id)] = labels

    rows: List[Dict[str, Any]] = []
    for rec in iter_jsonl(manifest_jsonl):
        vid = rec.get("id") or rec.get("video_id")
        split = rec.get("split")
        if not vid or not split:
            continue
        frame_ids = rec.get("frame_ids") or []
        frame_paths = rec.get("video_frame_paths") or []
        if frame_paths and len(frame_paths) != len(frame_ids):
            frame_paths = []
        for idx, fid in enumerate(frame_ids):
            try:
                fid_int = int(fid)
            except Exception:
                continue
            rel_path = frame_paths[idx] if idx < len(frame_paths) else f"{vid}_{fid_int}.jpg"
            file_name = Path(rel_path).name
            img_path = resolve_image_path(
                rel_path,
                file_name,
                split=str(split),
                dataset_root=data_root,
            )
            rows.append(
                {
                    "video_id": str(vid),
                    "frame_id": fid_int,
                    "split": str(split),
                    "image_path": abs_posix(img_path),
                    "file_name": file_name,
                    "labels": label_map.get((str(vid), fid_int), {}),
                    "image": {
                        "video_id": str(vid),
                        "frame_id": fid_int,
                        "split": str(split),
                        "image_path": str(rel_path),
                        "file_name": file_name,
                    },
                }
            )
    return rows


def load_endoscapes(
    *,
    split: Optional[str] = None,
    ann_file: str = DEFAULT_ANN_FILE,
    manifest_jsonl: Optional[Path] = None,
    data_root: Optional[Path] = None,
    manifest_only: bool = False,
    include_labels: bool = True,
) -> Dict[str, Any]:
    if data_root is None:
        raise ValueError("data_root is required for endoscapes loader")

    if manifest_only:
        if manifest_jsonl is None:
            raise ValueError("manifest_jsonl is required when manifest_only=True")
        frame_rows = load_endoscapes_manifest_only(
            manifest_jsonl=manifest_jsonl,
            data_root=data_root,
            ann_file=ann_file,
            include_labels=include_labels,
        )
        videos: Dict[str, Dict[str, Any]] = {}
        for row in frame_rows:
            vid = str(row.get("video_id"))
            videos.setdefault(vid, {"video_id": vid, "frames": []})
            videos[vid]["frames"].append(
                {
                    "frame_id": row.get("frame_id"),
                    "file_name": row.get("file_name"),
                    "image_path": row.get("image_path"),
                    "labels": row.get("labels", {}),
                }
            )
        for vid in videos.values():
            vid["frames"].sort(key=lambda r: r.get("frame_id", -1))
        return {
            "name": "endoscapes",
            "split": split or "mixed",
            "frames_root": data_root,
            "videos": videos,
            "frames": frame_rows,
        }

    if split is None and manifest_jsonl is None:
        raise ValueError("split is required when manifest_jsonl is not provided.")

    videos: Dict[str, Dict[str, Any]] = {}
    frame_rows: List[Dict[str, Any]] = []

    def _image_path(file_name: str, *, split_name: str) -> Path:
        p = Path(file_name)
        if p.is_absolute():
            return p
        if p.parts and p.parts[0] == split_name:
            return (data_root / p).resolve()
        return (data_root / split_name / p).resolve()

    if manifest_jsonl is not None:
        manifest_rows = list(iter_jsonl(manifest_jsonl))
        needed: Dict[str, Dict[str, set[int]]] = {}
        for rec in manifest_rows:
            vid = str(rec.get("id") or rec.get("video_id") or "")
            sp = str(rec.get("split") or "")
            if not vid or not sp:
                continue
            frame_ids = rec.get("frame_ids") or []
            ids: set[int] = set()
            for fid in frame_ids:
                try:
                    ids.add(int(fid))
                except Exception:
                    continue
            needed.setdefault(sp, {}).setdefault(vid, set()).update(ids)

        for sp, vids in needed.items():
            ann_path = (data_root / sp / ann_file).resolve()
            coco = load_json(ann_path) if ann_path.exists() else {}
            images = coco.get("images", [])
            annotations = coco.get("annotations", [])

            anns_by_image: Dict[int, List[Dict[str, Any]]] = {}
            for ann in annotations:
                try:
                    anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)
                except Exception:
                    continue

            for img in images:
                try:
                    image_id = int(img["id"])
                except Exception:
                    continue
                video_id = img.get("video_id")
                if video_id is None:
                    try:
                        video_id = int(Path(img["file_name"]).stem.split("_", 1)[0])
                    except Exception:
                        continue
                video_id = str(int(video_id))
                if video_id not in vids:
                    continue

                frame_num = img.get("frame_id")
                if frame_num is None:
                    frame_num = _endoscapes_frame_num_from_filename(img["file_name"])
                try:
                    frame_id = int(frame_num)
                except Exception:
                    continue
                if frame_id not in vids[video_id]:
                    continue

                img_path = _image_path(img["file_name"], split_name=sp)
                frame_row = {
                    "image_id": image_id,
                    "file_name": img["file_name"],
                    "frame_id": frame_id,
                    "image_path": abs_posix(img_path),
                    "height": img.get("height"),
                    "width": img.get("width"),
                    "labels": _parse_endoscapes_ds_labels(img.get("ds")),
                    "annotations": anns_by_image.get(image_id, []),
                    "split": sp,
                    "video_id": video_id,
                }
                videos.setdefault(video_id, {"video_id": video_id, "frames": []})
                videos[video_id]["frames"].append(frame_row)
                frame_rows.append(frame_row)
    else:
        split_vids_path = data_root / f"{split}_vids.txt"
        split_video_set = (
            set(_read_endoscapes_video_ids(split_vids_path)) if split_vids_path.exists() else set()
        )

        ann_path = (data_root / split / ann_file).resolve()
        coco = load_json(ann_path) if ann_path.exists() else {}
        images = coco.get("images", [])
        annotations = coco.get("annotations", [])

        anns_by_image: Dict[int, List[Dict[str, Any]]] = {}
        for ann in annotations:
            try:
                anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)
            except Exception:
                continue

        for img in images:
            try:
                image_id = int(img["id"])
            except Exception:
                continue
            video_id = img.get("video_id")
            if video_id is None:
                try:
                    video_id = int(Path(img["file_name"]).stem.split("_", 1)[0])
                except Exception:
                    continue
            video_id = int(video_id)

            if split_video_set and video_id not in split_video_set:
                continue

            frame_num = img.get("frame_id")
            if frame_num is None:
                frame_num = _endoscapes_frame_num_from_filename(img["file_name"])

            vid_key = str(video_id)
            img_path = _image_path(img["file_name"], split_name=split)
            frame_row = {
                "image_id": image_id,
                "file_name": img["file_name"],
                "frame_id": int(frame_num),
                "image_path": abs_posix(img_path),
                "height": img.get("height"),
                "width": img.get("width"),
                "labels": _parse_endoscapes_ds_labels(img.get("ds")),
                "annotations": anns_by_image.get(image_id, []),
                "split": split,
                "video_id": vid_key,
            }
            videos.setdefault(vid_key, {"video_id": vid_key, "frames": []})
            videos[vid_key]["frames"].append(frame_row)
            frame_rows.append(frame_row)

    for vid in videos.values():
        vid["frames"].sort(key=lambda r: r.get("frame_id", -1))

    return {
        "name": "endoscapes",
        "split": split or "mixed",
        "frames_root": data_root,
        "videos": videos,
        "frames": frame_rows,
    }


def _read_endoscapes_video_ids(path: Path) -> List[int]:
    ids: List[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.append(int(line))
        except Exception:
            continue
    return ids


def extract_identifiers(row: Dict[str, Any], *, default_split: str) -> Tuple[str, str, int]:
    image = row.get("image") or {}
    if not isinstance(image, dict):
        image = {}
    split = str(row.get("split") or image.get("split") or default_split)
    video_id = row.get("video_id") or row.get("video") or image.get("video_id")
    frame_id = row.get("frame_id") or row.get("frame") or image.get("frame_id")
    file_name = image.get("file_name") or row.get("file_name")
    image_id = image.get("image_id") or row.get("image_id")

    if video_id is None and file_name:
        video_id = _endoscapes_video_id_from_filename(str(file_name))
    if video_id is None and image_id is not None:
        try:
            video_id = int(image_id) // 1_000_000
        except Exception:
            video_id = None
    if frame_id is None and file_name:
        try:
            frame_id = _endoscapes_frame_num_from_filename(str(file_name))
        except Exception:
            frame_id = None
    if frame_id is None and image_id is not None:
        try:
            frame_id = int(image_id) % 1_000_000
        except Exception:
            frame_id = None

    vid_str = str(video_id) if video_id is not None else "unknown"
    try:
        frame_int = int(frame_id) if frame_id is not None else -1
    except Exception:
        frame_int = -1
    return vid_str, split, frame_int


def extract_gt(row: Dict[str, Any]) -> Dict[str, Optional[float]]:
    gt = row.get("gt") or row.get("labels") or {}
    if not isinstance(gt, dict):
        return {}
    out: Dict[str, Optional[float]] = {}
    for key in ("c1", "c2", "c3"):
        try:
            val = gt.get(key)
            out[key] = None if val is None else float(val)
        except Exception:
            out[key] = None
    return out


def extract_oracle_rubrics(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
    rubrics = row.get("rubrics") or row.get("rubric_labels")
    if isinstance(rubrics, dict):
        return {str(k): str(v) for k, v in rubrics.items()}
    return None


def load_endoscapes_few_shot(
    *,
    few_shot_jsonl: Path,
    data_root: Path,
    default_split: str = "val",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in iter_jsonl(few_shot_jsonl):
        video_id, split, frame_id = extract_identifiers(row, default_split=default_split)
        file_name = row.get("file_name") or (row.get("image") or {}).get("file_name")
        image_path = row.get("saved_image") or row.get("image_path")
        resolved = resolve_image_path(
            str(image_path) if image_path else None,
            str(file_name) if file_name else None,
            split=split,
            dataset_root=data_root,
        )
        rows.append(
            {
                "video_id": video_id,
                "split": split,
                "frame_id": frame_id,
                "file_name": file_name,
                "image_path": abs_posix(resolved),
                "gt": extract_gt(row),
                "oracle_rubrics": extract_oracle_rubrics(row) or {},
                "raw": row,
            }
        )
    return rows


def split_rows_by_few_shot(
    *,
    frame_rows: List[Dict[str, Any]],
    few_shot_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    key_set = {
        (str(r.get("video_id")), int(r.get("frame_id", -1))) for r in few_shot_rows
    }
    few_shot_out: List[Dict[str, Any]] = []
    rest_out: List[Dict[str, Any]] = []
    for row in frame_rows:
        key = (str(row.get("video_id")), int(row.get("frame_id", -1)))
        if key in key_set:
            few_shot_out.append(row)
        else:
            rest_out.append(row)
    return few_shot_out, rest_out
