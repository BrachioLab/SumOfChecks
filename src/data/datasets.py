from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cvs_challenge_sages_v1 import CvsChallengeSagesDataset, get_video_data as sages_video_data
from .endoscapes import EndoscapesDataset, split_rows_by_few_shot, get_video_data as endoscapes_video_data


class DatasetBundle(dict):
    def get_frame_data(self) -> List[Dict[str, Any]]:
        return get_frame_rows(self)

    def get_video_data(self) -> Dict[str, Dict[str, Any]]:
        name = str(self.get("name") or "").strip().lower()
        if name == "endoscapes":
            return endoscapes_video_data(self)
        if name == "cvs_challenge_sages_v1":
            return sages_video_data(self)
        raise ValueError(f"Unknown dataset for video data: {name}")

def get_dataset(
    dataset_name: str,
    *,
    manifest_jsonl: Optional[Path] = None,
    manifest: Optional[str] = None,
    split: Optional[str] = None,
    data_root: Optional[Path] = None,
    manifest_root: Optional[Path] = None,
    ann_file: Optional[str] = None,
    manifest_only: bool = False,
    frames_root: Optional[Path] = None,
    labels_root: Optional[Path] = None,
    few_shot_jsonl: Optional[Path] = None,
    split_few_shot: bool = False,
) -> Dict[str, Any]:
    name = (dataset_name or "").strip().lower()
    if name == "endoscapes":
        ds = EndoscapesDataset().load(
            manifest_jsonl=manifest_jsonl,
            manifest=manifest,
            split=split,
            data_root=data_root,
            manifest_root=manifest_root,
            ann_file=ann_file,
            manifest_only=manifest_only,
            few_shot_jsonl=few_shot_jsonl,
            split_few_shot=split_few_shot,
        )
        return DatasetBundle(ds)
    if name == "cvs_challenge_sages_v1":
        ds = CvsChallengeSagesDataset().load(
            manifest_jsonl=manifest_jsonl,
            manifest=manifest,
            split=split,
            frames_root=frames_root,
            manifest_root=manifest_root,
            labels_root=labels_root,
        )
        return DatasetBundle(ds)

    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_frame_rows(dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "frames" in dataset and isinstance(dataset["frames"], list):
        return dataset["frames"]
    frames: List[Dict[str, Any]] = []
    videos = dataset.get("videos") or {}
    if isinstance(videos, dict):
        for vid in videos.values():
            if isinstance(vid, dict) and isinstance(vid.get("frames"), list):
                frames.extend(vid["frames"])
    return frames


def split_frames_by_few_shot(
    frame_rows: List[Dict[str, Any]],
    few_shot_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return split_rows_by_few_shot(frame_rows=frame_rows, few_shot_rows=few_shot_rows)
