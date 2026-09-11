#!/usr/bin/env python3
"""
Rubric-oracle reasoning pipeline.

EXAMPLES
  python scripts/rubric_oracle_reasoning_frame.py --preset debug --dryrun
  python scripts/rubric_oracle_reasoning_frame.py --preset main --max_items 50
  python scripts/rubric_oracle_reasoning_frame.py --preset upper_bound --max_items 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image
from tqdm.auto import tqdm

from data import get_dataset, get_frame_rows  # repo-local
import models  # repo-local

# ============================================================
# CONFIG (edit only this block for paths / prompts / defaults)
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
print("REPO_ROOT", REPO_ROOT)

# Provider SDKs read OPENAI_API_KEY and ANTHROPIC_API_KEY from the environment.

# Default model
MODEL_ID = "gpt-4.1-mini"

# Dataset roots
ENDOSCAPES_ROOT = (REPO_ROOT / "data" / "endoscapes").resolve()
ENDOSCAPES_MANIFESTS_ROOT = (REPO_ROOT / "data" / "manifests" / "endoscapes").resolve()
SAGES_FRAMES_ROOT = (REPO_ROOT / "data" / "frames").resolve()
SAGES_FRAMES_FULL_ROOT = (REPO_ROOT / "data" / "frames_full").resolve()
SAGES_LABELS_ROOT = (REPO_ROOT / "data" / "CVS_Challenge_SAGES_v1").resolve()
ENDOSCAPES_VID_ANN_FILE = "annotation_coco_vid.json"
SAGES_MANIFESTS_ROOT = (
    REPO_ROOT / "data" / "manifests" / "cvs_challenge_sages_v1"
).resolve()

# Default rubric + annotation paths
DEFAULT_RUBRIC_PATH = (REPO_ROOT / "rubrics" / "cvs_rubrics_v3.json").resolve()
DEFAULT_ANNOTATION_PATH = (
    REPO_ROOT
    / "annotations"
    / "rubric_labels"
    / "endoscapes_val__rubrics_v1__seed13__batch0__filtered_no_all_uncertain.jsonl"
).resolve()

DEFAULT_ANNOTATION_PATH_FIX_ERROR = (
    REPO_ROOT
    / "annotations"
    / "rubric_labels"
    / "endoscapes_val__rubrics_v1__seed13__batch0__filtered_no_all_uncertain_fix_error.jsonl"
).resolve()

DEFAULT_FEW_SHOT_JSONL_ENDO_V3 = (
    REPO_ROOT / "few_shot_examples" / "endoscapes" / "rubrics"
    / "filtered" / "selected_examples_v3.jsonl"
)

DEFAULT_ANNOTATION_PATH_SAGES = (
    REPO_ROOT
    / "annotations"
    / "rubric_labels"
    / "sages__rubrics_v3__seed13__batch0__20260219_203800.jsonl"
).resolve()

# Annotation versions registry for output tags / future variants.
# Keyed by (dataset, version). Falls back to ("endoscapes", version) for backwards compat.
ANNOTATION_VERSION_REGISTRY: Dict[str, List[Path]] = {
    "v1": [DEFAULT_ANNOTATION_PATH],
    "v2": [DEFAULT_ANNOTATION_PATH_FIX_ERROR],
}

ANNOTATION_VERSION_REGISTRY_SAGES: Dict[str, List[Path]] = {
    "v2": [DEFAULT_ANNOTATION_PATH_SAGES],
}

# Prompts
SYSTEM_PROMPT = (
    "You are an expert surgical assistant. "
    "Follow the output format exactly."
)
DIRECT_SYSTEM_PROMPT = (
    "You are a surgical vision assistant. "
    "Follow the output format exactly."
)

# Optional criterion definitions (ground truth semantics).
CRITERION_DEFINITIONS = {
    "C1": "Two and only two tubular structures are visible entering the gallbladder.",
    "C2": "The hepatocystic triangle is cleared of fat and fibrous tissue.",
    "C3": "The lower third of the gallbladder is detached from the liver bed.",
}

# Pipeline modes:
# - one_call: image -> rubric labels -> reasoning -> prediction in one model call.
# - two_call: Stage 1 predicts rubric labels; Stage 2 reasons using fixed labels.

# ----------------- helpers -----------------
def abs_posix(p: Path) -> str:
    return p.resolve().as_posix()

def resolve_dataset_roots(dataset_name: str) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    name = (dataset_name or "").strip().lower()
    if name == "endoscapes":
        return ENDOSCAPES_ROOT, None, None, ENDOSCAPES_MANIFESTS_ROOT
    if name == "cvs_challenge_sages_v1":
        labels_root = SAGES_LABELS_ROOT if SAGES_LABELS_ROOT.exists() else None
        return None, SAGES_FRAMES_ROOT, labels_root, SAGES_MANIFESTS_ROOT
    raise ValueError(f"Unknown dataset: {dataset_name}")

def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

_NORMALIZE_SIDE: Optional[int] = None  # set by --normalize_side flag


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as im:
        img = im.convert("RGB")
    if _NORMALIZE_SIDE is not None:
        w, h = img.size
        short_side = min(w, h)
        if short_side != _NORMALIZE_SIDE:
            scale = _NORMALIZE_SIDE / short_side
            new_w = round(w * scale)
            new_h = round(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
    return img

def _strip_code_fences(text: str) -> str:
    """Strip leading/trailing markdown code fences (```json ... ```)."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _repair_truncated_json(text: str) -> Dict[str, Any]:
    """Try to repair truncated JSON by closing open brackets/braces.

    Walks the string tracking nesting depth (respecting JSON string literals)
    then appends the required closing characters.  Falls back to raising
    ``json.JSONDecodeError`` if the result still cannot be parsed.
    """
    # Trim trailing comma / whitespace so closing chars are valid
    t = text.rstrip()
    if t.endswith(","):
        t = t[:-1]

    # Truncation may have cut a string literal mid-way.  If the number of
    # unescaped quotes is odd we are inside a string — close it first.
    in_string = False
    i = 0
    while i < len(t):
        ch = t[i]
        if ch == "\\" and in_string:
            i += 2  # skip escaped char
            continue
        if ch == '"':
            in_string = not in_string
        i += 1
    if in_string:
        t += '"'

    # Now count unmatched openers
    stack: list[str] = []
    in_str = False
    i = 0
    while i < len(t):
        ch = t[i]
        if ch == "\\" and in_str:
            i += 2
            continue
        if ch == '"':
            in_str = not in_str
            i += 1
            continue
        if in_str:
            i += 1
            continue
        if ch in ("{", "["):
            stack.append("}" if ch == "{" else "]")
        elif ch in ("}", "]"):
            if stack:
                stack.pop()
        i += 1

    # Append required closers in reverse order
    t += "".join(reversed(stack))

    return json.loads(t)


def extract_json_obj(text: str) -> Dict[str, Any]:
    t = _strip_code_fences(text or "")
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return _repair_truncated_json(t)

def sanitize_model_tag(model_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", (model_id or "").strip())
    return safe or "model"

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

def parse_annotation_paths(raw: Optional[str], *, version: str, dataset: str = "endoscapes") -> List[Path]:
    if raw:
        paths = [Path(p.strip()) for p in raw.split(",") if p.strip()]
        return [p.resolve() for p in paths]
    if dataset == "cvs_challenge_sages_v1":
        registry_paths = ANNOTATION_VERSION_REGISTRY_SAGES.get(version, [])
    else:
        registry_paths = ANNOTATION_VERSION_REGISTRY.get(version, [])
    if not registry_paths:
        raise ValueError(f"No annotation paths registered for dataset='{dataset}', annotation_version='{version}'")
    return [p.resolve() for p in registry_paths]

def parse_few_shot_path(args: argparse.Namespace) -> Path:
    raw = getattr(args, "few_shot_path", "") or ""
    if raw:
        return Path(raw).resolve()
    dataset = getattr(args, "few_shot_dataset", None) or args.dataset
    if dataset != "endoscapes":
        raise ValueError("Only Endoscapes paper exemplars are bundled; provide --few_shot_path for SAGES.")
    return DEFAULT_FEW_SHOT_JSONL_ENDO_V3


def normalize_few_shot_json_type(args: argparse.Namespace) -> str:
    return "fs_v3"


def load_few_shot_rows(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))

def select_few_shot_rows(
    rows: List[Dict[str, Any]],
    k: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if not rows:
        return []
    if k <= 0 or k >= len(rows):
        return rows
    rng = random.Random(seed)
    rows_copy = list(rows)
    rng.shuffle(rows_copy)
    return rows_copy[:k]

def rubric_tag_from_path(path: Path) -> str:
    stem = path.stem
    m = re.search(r"rubrics[_-]?v\\d+", stem)
    if m:
        return m.group(0).replace("_", "").replace("-", "")
    return re.sub(r"[^A-Za-z0-9]+", "", stem) or "rubrics"

def build_manifest_filter(manifest_jsonl: Path) -> Dict[str, set[int]]:
    allowed: Dict[str, set[int]] = {}
    for rec in iter_jsonl(manifest_jsonl):
        vid = rec.get("id") or rec.get("video_id")
        split = rec.get("split")
        if not vid or not split:
            continue
        frame_ids = rec.get("frame_ids") or []
        for fid in frame_ids:
            try:
                fid_int = int(fid)
            except Exception:
                continue
            allowed.setdefault(str(split), {}).setdefault(str(vid), set()).add(fid_int)
    return allowed

def load_manifest_rows(manifest_jsonl: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in iter_jsonl(manifest_jsonl):
        vid = rec.get("id") or rec.get("video_id")
        split = rec.get("split")
        frame_ids = rec.get("frame_ids") or []
        frame_paths = rec.get("video_frame_paths") or []
        if not vid or not split:
            continue
        if frame_paths and len(frame_paths) != len(frame_ids):
            frame_paths = []
        for idx, fid in enumerate(frame_ids):
            try:
                fid_int = int(fid)
            except Exception:
                continue
            rel_path = frame_paths[idx] if idx < len(frame_paths) else f"{vid}_{fid_int}.jpg"
            file_name = Path(rel_path).name
            rows.append(
                {
                    "video_id": str(vid),
                    "frame_id": fid_int,
                    "split": str(split),
                    "image_path": rel_path,
                    "file_name": file_name,
                    "image": {
                        "video_id": str(vid),
                        "frame_id": fid_int,
                        "split": str(split),
                        "image_path": rel_path,
                        "file_name": file_name,
                    },
                }
            )
    return rows

def in_manifest_filter(
    allowed: Dict[str, Dict[str, set[int]]],
    *,
    video_id: str,
    split: str,
    frame_id: int,
) -> bool:
    return frame_id in allowed.get(str(split), {}).get(str(video_id), set())

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
            if isinstance(val, dict):
                val = val.get("continuous", val.get("binary"))
            out[key] = None if val is None else float(val)
        except Exception:
            out[key] = None
    return out

def extract_oracle_rubrics(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
    rubrics = row.get("rubrics") or row.get("rubric_labels")
    if isinstance(rubrics, dict):
        return {str(k): str(v) for k, v in rubrics.items()}
    return None

def load_rubrics(
    rubric_path: Path,
) -> Tuple[
    List[str],
    Dict[str, Dict[str, List[Dict[str, Any]]]],
    Dict[str, Dict[str, Any]],
]:
    data = load_json(rubric_path)
    criteria_blocks = data.get("criteria")
    if not isinstance(criteria_blocks, dict):
        raise KeyError("Rubrics missing criteria blocks")
    criterion_order = ["C1", "C2", "C3"]
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    rubric_items_by_id: Dict[str, Dict[str, Any]] = {}
    for criterion in criterion_order:
        block = criteria_blocks.get(criterion)
        if not block or "items" not in block:
            raise KeyError(f"Rubrics missing items for {criterion}")
        items = list(block["items"])
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for item in items:
            label_type_raw = str(item.get("label_type") or "evidence").lower()
            label_type_group = label_type_raw if label_type_raw in ("precondition", "evidence") else "evidence"
            item_out = dict(item)
            item_out["criterion"] = criterion
            item_out["label_type"] = label_type_raw
            by_type.setdefault(label_type_group, []).append(item_out)
            item_id = item_out.get("id")
            if item_id:
                rubric_items_by_id[str(item_id)] = item_out
        grouped[criterion] = by_type
    return criterion_order, grouped, rubric_items_by_id

def render_rubric_items(
    criterion_order: List[str],
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> str:
    lines: List[str] = []
    for criterion in criterion_order:
        lines.append(f"{criterion}:")
        by_type = grouped.get(criterion, {})
        for label_type in ("precondition", "evidence"):
            items = by_type.get(label_type, [])
            for item in items:
                item_id = item.get("id", "")
                text = item.get("text", "")
                lines.append(f"  - {item_id}: {text}")
    return "\n".join(lines)

def render_rubrics_block(
    rubric_items_by_id: Dict[str, Dict[str, Any]],
    rubric_labels: Dict[str, str],
    include_weights: bool,
) -> List[str]:
    lines: List[str] = []
    for item_id, item in rubric_items_by_id.items():
        text = item.get("text", "")
        label = str(rubric_labels.get(item_id, "uncertain")).lower()
        if include_weights:
            weight = item.get("weight")
            lines.append(f"{item_id} (weight={weight}): {text} Answer: {label}")
        else:
            lines.append(f"{item_id}: {text} Answer: {label}")
    return lines

def compute_weighted_score(
    rubric_items_by_id: Dict[str, Dict[str, Any]],
    rubric_labels: Dict[str, str],
    include_types: str = "all",
) -> Dict[str, float]:
    label_map = {"yes": 1.0, "no": 0.0, "uncertain": 0.0}
    include_types = (include_types or "all").lower()
    scores = {"c1": 0.0, "c2": 0.0, "c3": 0.0}

    def _include_item(label_type: str) -> bool:
        if include_types == "all":
            return True
        if include_types == "evidence_only":
            return label_type == "evidence"
        if include_types == "preconditions_only":
            return label_type == "precondition"
        if include_types == "modifiers_only":
            return label_type in ("modifier", "modifiers")
        return True

    for item_id, item in rubric_items_by_id.items():
        label_type = str(item.get("label_type") or "evidence").lower()
        if not _include_item(label_type):
            continue
        raw_label = rubric_labels.get(item_id)
        if raw_label is None:
            print(f"[WARN] Missing rubric label for weighted score: {item_id}")
            raw_label = "uncertain"
        label_val = label_map.get(str(raw_label).lower(), 0.0)
        try:
            weight = float(item.get("weight", 0.0))
        except Exception:
            weight = 0.0
        criterion = str(item.get("criterion", "")).lower()
        if criterion in scores:
            scores[criterion] += weight * label_val
    return scores

def render_oracle_labels_block(oracle: Dict[str, str]) -> str:
    parts = [f"{k}={v}" for k, v in sorted(oracle.items())]
    return ", ".join(parts)

def normalize_criteria(raw: Optional[str]) -> List[str]:
    if not raw:
        return ["c1", "c2", "c3"]
    parts = re.split(r"[,\s]+", raw.strip())
    normalized = []
    for part in parts:
        if not part:
            continue
        key = part.strip().lower()
        if key in ("c1", "c2", "c3"):
            normalized.append(key)
    return normalized or ["c1", "c2", "c3"]

def resolve_few_shot_image_path(row: Dict[str, Any], *, default_split: str, dataset_root: Optional[Path] = None) -> Path:
    image_path = row.get("saved_image") or row.get("image_path")
    file_name = row.get("file_name") or (row.get("image") or {}).get("file_name")
    if image_path:
        p = Path(str(image_path))
        if p.is_absolute():
            return p
        if (REPO_ROOT / p).is_file():
            return REPO_ROOT / p
    split = str(row.get("split") or (row.get("image") or {}).get("split") or default_split)
    root = dataset_root if dataset_root is not None else ENDOSCAPES_ROOT
    resolved = resolve_image_path(
        str(image_path) if image_path else None,
        str(file_name) if file_name else None,
        split=split,
        dataset_root=root,
    )
    return resolved

def prepare_few_shot_examples(
    rows: List[Dict[str, Any]],
    *,
    default_split: str,
    require_oracle_rubrics: bool,
    dataset_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for row in rows:
        video_id, split, frame_id = extract_identifiers(row, default_split=default_split)
        resolved_path = resolve_few_shot_image_path(row, default_split=default_split, dataset_root=dataset_root)
        if not resolved_path.exists():
            print(f"[WARN] Missing few-shot image: {resolved_path}")
            continue
        oracle_rubrics = extract_oracle_rubrics(row)
        if require_oracle_rubrics and not oracle_rubrics:
            print(f"[WARN] Missing few-shot rubrics for frame: video_id={video_id}, frame_id={frame_id}")
            continue
        examples.append(
            {
                "video_id": video_id,
                "split": split,
                "frame_id": frame_id,
                "file_name": row.get("file_name"),
                "image_path": abs_posix(resolved_path),
                "img": load_image(resolved_path),
                "gt": extract_gt(row),
                "oracle_rubrics": oracle_rubrics or {},
            }
        )
    return examples

def format_prompt_parts_for_print(parts: Tuple[Any, ...]) -> str:
    lines: List[str] = []
    for idx, part in enumerate(parts):
        if isinstance(part, Image.Image):
            lines.append(f"[{idx}] <PIL.Image size={part.size}>")
        else:
            text = str(part)
            lines.append(f"[{idx}] {text}")
    return "\n".join(lines)

def sanitize_prompt_parts(parts: List[Any]) -> Tuple[Any, ...]:
    sanitized: List[Any] = []
    for part in parts:
        if isinstance(part, str) and not part.strip():
            continue
        sanitized.append(part)
    return tuple(sanitized)

def build_direct_prompt(
    *,
    criteria: List[str],
    include_criterion_definitions: bool,
    include_rationale: bool,
    rationale_first: bool = False,
    video_id: str,
    split: str,
    frame_id: int,
) -> Tuple[str, str]:
    criteria_upper = [c.upper() for c in criteria]
    definition_lines: List[str] = []
    if include_criterion_definitions:
        for crit in criteria_upper:
            definition = CRITERION_DEFINITIONS.get(crit)
            if definition:
                definition_lines.append(f"- {crit}: {definition}")

    pred_schema = ", ".join(f'"{c}": float' for c in criteria)
    rationale_schema_part = '"rationale": { ' + ", ".join(f'"{c}": string' for c in criteria) + " }"
    pred_schema_part = '"pred": { ' + pred_schema + " }"
    if include_rationale:
        if rationale_first:
            schema_block = "{\n  " + rationale_schema_part + ",\n  " + pred_schema_part + "\n}"
        else:
            schema_block = "{\n  " + pred_schema_part + ",\n  " + rationale_schema_part + "\n}"
    else:
        schema_block = "{\n  " + pred_schema_part + "\n}"

    lines = [
        "Direct CVS criterion prediction task.",
        "",
        f"Predict confidence scores for: {', '.join(criteria_upper)}.",
        "Return a probability in [0,1] for each requested criterion.",
    ]
    if definition_lines:
        lines.extend(["", "Criterion definitions:", *definition_lines])
    if include_rationale:
        if rationale_first:
            lines.extend(
                [
                    "",
                    "Provide a short rationale per criterion.",
                    "Output rationale first, then pred.",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "Provide a short rationale per criterion.",
                    "Output rationale before pred.",
                ]
            )
    lines.extend(
        [
            "",
            "Return ONLY valid JSON (no markdown, no extra keys) with this schema:",
            schema_block,
        ]
    )
    prompt = "\n".join(lines)
    return prompt, schema_block

def build_few_shot_header() -> str:
    return (
        "You are an expert gallbladder surgeon experienced in laparoscopic cholecystectomy. "
        "You are in a training setting reviewing prior labeled examples. "
        "Follow the output format exactly."
    )

def build_direct_prompt_few_shot(
    *,
    criteria: List[str],
    include_criterion_definitions: bool,
    include_rationale: bool,
    rationale_first: bool = False,
    examples: List[Dict[str, Any]],
    video_id: str,
    split: str,
    frame_id: int,
    img: Image.Image,
) -> Tuple[Tuple[Any, ...], str]:
    prompt_text, schema_block = build_direct_prompt(
        criteria=criteria,
        include_criterion_definitions=include_criterion_definitions,
        include_rationale=include_rationale,
        rationale_first=rationale_first,
        video_id=video_id,
        split=split,
        frame_id=frame_id,
    )
    parts: List[Any] = [build_few_shot_header(), ""]
    for idx, ex in enumerate(examples, start=1):
        parts.append(
            "\n".join(
                [
                    f"Example {idx} (training):",
                    "Image:",
                ]
            )
        )
        parts.append(ex["img"])
        gt_pred = {k: ex["gt"].get(k) for k in criteria if ex["gt"].get(k) is not None}
        parts.append("Ground-truth pred:")
        parts.append(json.dumps({"pred": gt_pred}, ensure_ascii=False))
        parts.append("")
    parts.append("Now analyze the next image.")
    parts.append("Image:")
    parts.append(img)
    parts.append(prompt_text)
    return sanitize_prompt_parts(parts), schema_block

def build_prompt(
    *,
    criterion_order: List[str],
    rubric_grouped: Dict[str, Dict[str, List[Dict[str, Any]]]],
    annotation_mode: str,
    include_rationale: bool,
    include_criterion_definitions: bool,
    video_id: str,
    split: str,
    frame_id: int,
    gt: Dict[str, Optional[float]],
    oracle_rubrics: Optional[Dict[str, str]],
) -> Tuple[str, str]:
    rubric_items_block = render_rubric_items(criterion_order, rubric_grouped)
    oracle_block = ""
    if oracle_rubrics and annotation_mode in ("oracle_only", "oracle_plus_predicted"):
        oracle_block = f"Oracle rubric labels: {render_oracle_labels_block(oracle_rubrics)}\n"

    needs_rubric_labels = annotation_mode in ("predicted_only", "oracle_plus_predicted")
    rubric_label_instruction = (
        "First, label EACH rubric item as yes/no/uncertain based only on the image.\n"
        if needs_rubric_labels
        else "Use the oracle rubric labels provided.\n"
    )

    rationale_note = ""
    if include_rationale:
        rationale_note = (
            "Provide a short rationale per criterion that references rubric item IDs.\n"
        )

    rubric_labels_schema = ""
    if needs_rubric_labels:
        rubric_labels_schema = '"rubric_labels": { "<item_id>": "yes"|"no"|"uncertain", ... },\n  '

    rationale_schema = ""
    if include_rationale:
        rationale_schema = '"rationale": { "c1": string, "c2": string, "c3": string },\n  '

    schema_block = (
        "{\n  "
        + rubric_labels_schema
        + '"rubrics_block": [ "C1-1 ... Answer: yes", ... ],\n  '
        + rationale_schema
        + '"pred": { "c1": float, "c2": float, "c3": float }\n'
        + "}"
    )

    lines = [
        "You are evaluating Critical View of Safety (CVS).",
    ]
    if include_criterion_definitions:
        definition_lines = []
        for criterion in criterion_order:
            definition = CRITERION_DEFINITIONS.get(criterion)
            if definition:
                definition_lines.append(f"- {criterion}: {definition}")
        if definition_lines:
            lines.extend(
                [
                    "",
                    "Criterion definitions (for reference):",
                    *definition_lines,
                ]
            )
    lines.extend(
        [
            "",
            "Rubric items (grouped by criterion and label type):",
            rubric_items_block,
            "",
        ]
    )
    if oracle_block:
        lines.append(oracle_block.rstrip())
    lines.extend(
        [
            "Instructions:",
            f"- {rubric_label_instruction.rstrip()}",
        ]
    )
    if annotation_mode == "oracle_plus_predicted":
        lines.append(
            "- Oracle labels are provided for comparison; still predict labels from the image."
        )
    lines.extend(
        [
            "- Then create a rubrics_block list using the SAME rubric statement text (including the item ID), "
            "appending \"Answer: <yes/no/uncertain>\" to each statement.",
            "- Use rubric labels to justify confidence changes.",
            "- If preconditions fail, the criterion confidence should not be high.",
            "- Finally, output belief scores for C1/C2/C3 as floats in [0,1].",
        ]
    )
    if rationale_note:
        lines.append(f"- {rationale_note.rstrip()}")
        lines.append("- If rationale is requested, output rationale before pred.")

    prompt = "\n".join(lines)

    prompt += (
        "\nReturn ONLY valid JSON (no markdown, no extra keys) with this schema:\n"
        f"{schema_block}\n"
    )
    return prompt, schema_block

def build_prompt_few_shot(
    *,
    criterion_order: List[str],
    rubric_grouped: Dict[str, Dict[str, List[Dict[str, Any]]]],
    rubric_items_by_id: Dict[str, Dict[str, Any]],
    annotation_mode: str,
    include_rationale: bool,
    include_criterion_definitions: bool,
    examples: List[Dict[str, Any]],
    include_weights: bool,
    few_shot_short: bool,
    video_id: str,
    split: str,
    frame_id: int,
    gt: Dict[str, Optional[float]],
    oracle_rubrics: Optional[Dict[str, str]],
    img: Image.Image,
) -> Tuple[Tuple[Any, ...], str]:
    prompt_text, schema_block = build_prompt(
        criterion_order=criterion_order,
        rubric_grouped=rubric_grouped,
        annotation_mode=annotation_mode,
        include_rationale=include_rationale,
        include_criterion_definitions=include_criterion_definitions,
        video_id=video_id,
        split=split,
        frame_id=frame_id,
        gt=gt,
        oracle_rubrics=oracle_rubrics,
    )
    parts: List[Any] = [build_few_shot_header(), ""]
    if few_shot_short:
        rubric_items_block = render_rubric_items(criterion_order, rubric_grouped)
        parts.append("Rubric items:")
        parts.append(rubric_items_block)
        parts.append("")
    for idx, ex in enumerate(examples, start=1):
        parts.append(
            "\n".join(
                [
                    f"Example {idx} (training):",
                    "Image:",
                ]
            )
        )
        parts.append(ex["img"])
        oracle_block = render_oracle_labels_block(ex["oracle_rubrics"])
        parts.append(f"Oracle rubric labels: {oracle_block}")
        if not few_shot_short:
            rubrics_block = render_rubrics_block(
                rubric_items_by_id,
                ex["oracle_rubrics"],
                include_weights,
            )
            parts.append("rubrics_block:")
            parts.append("\n".join(f"- {line}" for line in rubrics_block))
        gt_pred = {k: ex["gt"].get(k) for k in ("c1", "c2", "c3") if ex["gt"].get(k) is not None}
        if gt_pred:
            parts.append("Ground-truth pred:")
            parts.append(json.dumps({"pred": gt_pred}, ensure_ascii=False))
        parts.append("")
    parts.append("Now analyze the next image.")
    parts.append("Image:")
    parts.append(img)
    parts.append(prompt_text)
    return sanitize_prompt_parts(parts), schema_block

def build_stage1_prompt(
    *,
    rubric_items_block: str,
    include_rationale: bool,
    include_rubric_items: bool = True,
    video_id: str,
    split: str,
    frame_id: int,
) -> Tuple[str, str]:
    rationale_schema = ""
    if include_rationale:
        rationale_schema = '"rationale": { "<item_id>": string, ... }, '
    schema_block = "{ " + rationale_schema + '"rubric_labels": { "<item_id>": "yes"|"no"|"uncertain", ... } }'
    lines = [
        "Rubric labeling task.",
        "",
    ]
    if include_rubric_items:
        lines.extend(
            [
                "Rubric items:",
                rubric_items_block,
                "",
            ]
        )
    lines.append("Label EACH rubric item as yes/no/uncertain based only on the image.")
    if include_rationale:
        lines.append("Provide a brief rationale for each rubric item.")
        lines.append("Output rationale before rubric_labels.")
    lines.extend(
        [
            "Return ONLY valid JSON (no markdown, no extra keys) with this schema:",
            schema_block,
        ]
    )
    prompt = "\n".join(lines) + "\n"
    return prompt, schema_block

def build_stage1_prompt_few_shot(
    *,
    rubric_items_block: str,
    rubric_items_by_id: Dict[str, Dict[str, Any]],
    include_rationale: bool,
    include_weights: bool,
    examples: List[Dict[str, Any]],
    few_shot_short: bool,
    video_id: str,
    split: str,
    frame_id: int,
    img: Image.Image,
) -> Tuple[Tuple[Any, ...], str]:
    prompt_text, schema_block = build_stage1_prompt(
        rubric_items_block=rubric_items_block,
        include_rationale=include_rationale,
        include_rubric_items=not few_shot_short,
        video_id=video_id,
        split=split,
        frame_id=frame_id,
    )
    parts: List[Any] = [build_few_shot_header(), ""]
    if few_shot_short:
        parts.append("Rubric items:")
        parts.append(rubric_items_block)
        parts.append("")
    for idx, ex in enumerate(examples, start=1):
        parts.append(
            "\n".join(
                [
                    f"Example {idx} (training):",
                    "Image:",
                ]
            )
        )
        parts.append(ex["img"])
        oracle_block = render_oracle_labels_block(ex["oracle_rubrics"])
        parts.append(f"Oracle rubric labels: {oracle_block}")
        if not few_shot_short:
            rubrics_block = render_rubrics_block(
                rubric_items_by_id,
                ex["oracle_rubrics"],
                include_weights,
            )
            parts.append("rubrics_block:")
            parts.append("\n".join(f"- {line}" for line in rubrics_block))
        parts.append("")
    parts.append("Now analyze the next image.")
    parts.append("Image:")
    parts.append(img)
    parts.append(prompt_text)
    return sanitize_prompt_parts(parts), schema_block

def build_stage1_prompt_with_self_rubric(
    *,
    rubric_items_block: str,
    self_rubric_block: str,
    include_rationale: bool,
    video_id: str,
    split: str,
    frame_id: int,
) -> Tuple[str, str]:
    """Stage 1 prompt for sr_guided: predict predefined rubric labels with self-rubric checklist as context."""
    standard_prompt, schema_block = build_stage1_prompt(
        rubric_items_block=rubric_items_block,
        include_rationale=include_rationale,
        video_id=video_id,
        split=split,
        frame_id=frame_id,
    )
    context_block = (
        "You previously performed a free-form assessment of this image. Here are your findings:\n"
        "\n"
        "<self_assessment>\n"
        f"{self_rubric_block}\n"
        "</self_assessment>\n"
        "\n"
        "Now, using this assessment as context, evaluate the following predefined rubric items.\n"
        "\n"
    )
    prompt = context_block + standard_prompt
    return prompt, schema_block

def build_stage1_prompt_with_self_rubric_few_shot(
    *,
    rubric_items_block: str,
    rubric_items_by_id: Dict[str, Dict[str, Any]],
    self_rubric_block: str,
    include_rationale: bool,
    include_weights: bool,
    examples: List[Dict[str, Any]],
    few_shot_short: bool,
    video_id: str,
    split: str,
    frame_id: int,
    img: Image.Image,
) -> Tuple[Tuple[Any, ...], str]:
    """Stage 1 few-shot for sr_guided: few-shot examples shown normally, then self-rubric context before query."""
    prompt_text, schema_block = build_stage1_prompt(
        rubric_items_block=rubric_items_block,
        include_rationale=include_rationale,
        include_rubric_items=not few_shot_short,
        video_id=video_id,
        split=split,
        frame_id=frame_id,
    )
    parts: List[Any] = [build_few_shot_header(), ""]
    if few_shot_short:
        parts.append("Rubric items:")
        parts.append(rubric_items_block)
        parts.append("")
    for idx, ex in enumerate(examples, start=1):
        parts.append(
            "\n".join(
                [
                    f"Example {idx} (training):",
                    "Image:",
                ]
            )
        )
        parts.append(ex["img"])
        oracle_block = render_oracle_labels_block(ex["oracle_rubrics"])
        parts.append(f"Oracle rubric labels: {oracle_block}")
        if not few_shot_short:
            rubrics_block = render_rubrics_block(
                rubric_items_by_id,
                ex["oracle_rubrics"],
                include_weights,
            )
            parts.append("rubrics_block:")
            parts.append("\n".join(f"- {line}" for line in rubrics_block))
        parts.append("")
    parts.append("Now analyze the next image.")
    parts.append("Image:")
    parts.append(img)
    # Insert self-rubric context before the standard stage 1 prompt
    context_block = (
        "You previously performed a free-form assessment of this image. Here are your findings:\n"
        "\n"
        "<self_assessment>\n"
        f"{self_rubric_block}\n"
        "</self_assessment>\n"
        "\n"
        "Now, using this assessment as context, evaluate the following predefined rubric items.\n"
    )
    parts.append(context_block)
    parts.append(prompt_text)
    return sanitize_prompt_parts(parts), schema_block

def build_stage2_prompt(
    *,
    criterion_order: List[str],
    rubric_items_block: str,
    rubric_labels: Dict[str, str],
    rubrics_block: List[str],
    include_rationale: bool,
    include_criterion_definitions: bool,
    stage1_short: bool,
    video_id: str,
    split: str,
    frame_id: int,
    gt: Dict[str, Optional[float]],
) -> Tuple[str, str]:
    rationale_schema = ""
    if include_rationale:
        rationale_schema = ',\n  "rationale": { "c1": string, "c2": string, "c3": string }'

    if stage1_short:
        schema_block = (
            "{\n  "
            + (rationale_schema if include_rationale else "")
            + '\n  "pred": { "c1": float, "c2": float, "c3": float }'
            + "\n}"
        )
    else:
        schema_block = (
            "{\n  "
            + '"rubrics_block": [ "C1-1 ... Answer: yes", ... ],\n  '
            + (rationale_schema if include_rationale else "")
            + '\n  "pred": { "c1": float, "c2": float, "c3": float }'
            + "\n}"
        )

    oracle_block = render_oracle_labels_block(rubric_labels)

    lines = [
        "Rubric-guided reasoning task.",
    ]
    if include_criterion_definitions:
        definition_lines = []
        for criterion in criterion_order:
            definition = CRITERION_DEFINITIONS.get(criterion)
            if definition:
                definition_lines.append(f"- {criterion}: {definition}")
        if definition_lines:
            lines.extend(
                [
                    "",
                    "Criterion definitions (for reference):",
                    *definition_lines,
                ]
            )
    label_header = (
        "Rubric labels for the current image:"
        if stage1_short
        else "Rubric labels (fixed facts for this step):"
    )
    lines.extend(
        [
            label_header,
            oracle_block,
            "",
            "Instructions:",
            "- Treat rubric labels as ground truth for this step.",
            "- Do NOT re-predict rubric labels.",
            "- Use rubric labels to justify confidence changes.",
            "- If preconditions fail, the criterion confidence should not be high.",
            "- Output belief scores for C1/C2/C3 as floats in [0,1].",
        ]
    )
    if include_rationale:
        lines.append("- Provide a short rationale per criterion that references rubric item IDs.")
        lines.append("- Output rationale before pred.")
    if not stage1_short:
        lines.extend(
            [
                "",
                "rubrics_block (use verbatim in output):",
                "\n".join(f"- {line}" for line in rubrics_block),
            ]
        )

    prompt = "\n".join(lines)
    prompt += (
        "\n\nReturn ONLY valid JSON (no markdown, no extra keys) with this schema:\n"
        f"{schema_block}\n"
    )
    return prompt, schema_block

def build_stage2_prompt_few_shot(
    *,
    criterion_order: List[str],
    rubric_items_block: str,
    rubric_items_by_id: Dict[str, Dict[str, Any]],
    rubric_labels: Dict[str, str],
    rubrics_block: List[str],
    include_rationale: bool,
    include_criterion_definitions: bool,
    include_weights: bool,
    examples: List[Dict[str, Any]],
    few_shot_short: bool,
    stage1_short: bool,
    video_id: str,
    split: str,
    frame_id: int,
    gt: Dict[str, Optional[float]],
    img: Image.Image,
) -> Tuple[Tuple[Any, ...], str]:
    prompt_text, schema_block = build_stage2_prompt(
        criterion_order=criterion_order,
        rubric_items_block=rubric_items_block,
        rubric_labels=rubric_labels,
        rubrics_block=rubrics_block,
        include_rationale=include_rationale,
        include_criterion_definitions=include_criterion_definitions,
        stage1_short=stage1_short,
        video_id=video_id,
        split=split,
        frame_id=frame_id,
        gt=gt,
    )
    parts: List[Any] = [build_few_shot_header(), ""]
    if few_shot_short:
        parts.append("Rubric items:")
        parts.append(rubric_items_block)
        parts.append("")
    for idx, ex in enumerate(examples, start=1):
        parts.append(
            "\n".join(
                [
                    f"Example {idx} (training):",
                    "Image:",
                ]
            )
        )
        parts.append(ex["img"])
        oracle_block = render_oracle_labels_block(ex["oracle_rubrics"])
        parts.append(f"Oracle rubric labels: {oracle_block}")
        if not few_shot_short and not stage1_short:
            ex_rubrics_block = render_rubrics_block(
                rubric_items_by_id,
                ex["oracle_rubrics"],
                include_weights,
            )
            parts.append("rubrics_block:")
            parts.append("\n".join(f"- {line}" for line in ex_rubrics_block))
        gt_pred = {k: ex["gt"].get(k) for k in ("c1", "c2", "c3") if ex["gt"].get(k) is not None}
        if gt_pred:
            parts.append("Ground-truth pred:")
            parts.append(json.dumps({"pred": gt_pred}, ensure_ascii=False))
        parts.append("")
    parts.append("Now analyze the next image.")
    parts.append("Image:")
    parts.append(img)
    parts.append(prompt_text)
    return sanitize_prompt_parts(parts), schema_block

def validate_output(
    obj: Dict[str, Any],
    *,
    expect_rubric_labels: bool,
    include_rationale: bool,
) -> List[str]:
    errs: List[str] = []
    if not isinstance(obj, dict):
        return ["not_a_dict"]
    if expect_rubric_labels and "rubric_labels" not in obj:
        errs.append("missing_rubric_labels")
    if "rubrics_block" not in obj:
        errs.append("missing_rubrics_block")
    if "pred" not in obj:
        errs.append("missing_pred")
    else:
        pred = obj.get("pred")
        if not isinstance(pred, dict):
            errs.append("pred_not_dict")
        else:
            for k in ("c1", "c2", "c3"):
                if k not in pred:
                    errs.append(f"pred_missing_{k}")
    if include_rationale and "rationale" not in obj:
        errs.append("missing_rationale")
    return errs

def validate_stage1(obj: Dict[str, Any]) -> List[str]:
    if not isinstance(obj, dict):
        return ["not_a_dict"]
    if "rubric_labels" not in obj:
        return ["missing_rubric_labels"]
    labels = obj.get("rubric_labels")
    if not isinstance(labels, dict):
        return ["rubric_labels_not_dict"]
    return []

def validate_stage1_with_rationale(obj: Dict[str, Any], *, include_rationale: bool) -> List[str]:
    errs = validate_stage1(obj)
    if include_rationale and "rationale" not in obj:
        errs.append("missing_rationale")
    return errs

def validate_stage2(
    obj: Dict[str, Any],
    *,
    include_rationale: bool,
    expect_rubrics_block: bool,
) -> List[str]:
    if not isinstance(obj, dict):
        return ["not_a_dict"]
    if expect_rubrics_block and "rubrics_block" not in obj:
        return ["missing_rubrics_block"]
    if "pred" not in obj:
        return ["missing_pred"]
    pred = obj.get("pred")
    if not isinstance(pred, dict):
        return ["pred_not_dict"]
    errs: List[str] = []
    for k in ("c1", "c2", "c3"):
        if k not in pred:
            errs.append(f"pred_missing_{k}")
    if include_rationale and "rationale" not in obj:
        errs.append("missing_rationale")
    return errs

def validate_direct_output(
    obj: Dict[str, Any],
    *,
    criteria: List[str],
    include_rationale: bool,
) -> List[str]:
    errs: List[str] = []
    if not isinstance(obj, dict):
        return ["not_a_dict"]
    allowed_keys = {"pred", "rationale"} if include_rationale else {"pred"}
    if not set(obj.keys()).issubset(allowed_keys):
        errs.append("unexpected_keys")
    if include_rationale and "rationale" not in obj:
        errs.append("missing_rationale")
    pred = obj.get("pred")
    if not isinstance(pred, dict):
        return ["pred_not_dict"]
    expected = set(criteria)
    if set(pred.keys()) != expected:
        errs.append("pred_keys_mismatch")
    for key in criteria:
        val = pred.get(key)
        try:
            val_f = float(val)
        except Exception:
            errs.append(f"pred_not_float_{key}")
            continue
        if not (0.0 <= val_f <= 1.0):
            errs.append(f"pred_out_of_range_{key}")
    if include_rationale:
        rationale = obj.get("rationale")
        if not isinstance(rationale, dict):
            errs.append("rationale_not_dict")
        else:
            for key in criteria:
                if key not in rationale:
                    errs.append(f"rationale_missing_{key}")
    return errs

# --------------- per-criterion helpers ---------------

def merge_per_criterion_results(
    per_crit_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge per-criterion prediction results into a single combined dict."""
    merged_pred: Dict[str, float] = {}
    merged_rationale: Dict[str, str] = {}
    all_parse_ok = True
    all_parse_errors: List[str] = []
    per_crit_metadata: Dict[str, Dict[str, Any]] = {}

    for result in per_crit_results:
        crit = result["criterion"]
        if not result.get("parse_ok", False):
            all_parse_ok = False
        errors = result.get("parse_errors", [])
        all_parse_errors.extend(f"{crit}:{e}" for e in errors)

        pred = result.get("pred", {})
        if isinstance(pred, dict):
            merged_pred.update(pred)

        rationale = result.get("rationale", {})
        if isinstance(rationale, dict):
            merged_rationale.update(rationale)

        per_crit_metadata[crit] = {
            "raw_response": result.get("raw_response", ""),
            "parsed": result.get("parsed", {}),
            "parse_ok": result.get("parse_ok", False),
            "parse_errors": errors,
        }

    return {
        "pred": merged_pred,
        "rationale": merged_rationale if merged_rationale else None,
        "parse_ok": all_parse_ok,
        "parse_errors": all_parse_errors,
        "per_criterion_metadata": per_crit_metadata,
    }


def validate_stage2_single_criterion(
    obj: Dict[str, Any],
    *,
    criterion: str,
    include_rationale: bool,
) -> List[str]:
    """Validate stage 2 output for a single criterion."""
    if not isinstance(obj, dict):
        return ["not_a_dict"]
    if "pred" not in obj:
        return ["missing_pred"]
    pred = obj.get("pred")
    if not isinstance(pred, dict):
        return ["pred_not_dict"]
    crit_lower = criterion.lower()
    errs: List[str] = []
    if crit_lower not in pred:
        errs.append(f"pred_missing_{crit_lower}")
    else:
        try:
            val = float(pred[crit_lower])
            if not (0.0 <= val <= 1.0):
                errs.append(f"pred_out_of_range_{crit_lower}")
        except (TypeError, ValueError):
            errs.append(f"pred_not_float_{crit_lower}")
    if include_rationale and "rationale" not in obj:
        errs.append("missing_rationale")
    return errs


def filter_checklist_for_criterion(
    checklist: Dict[str, List[Dict[str, Any]]],
    criterion: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return a checklist dict containing only items for the given criterion."""
    crit_upper = criterion.upper()
    return {crit_upper: checklist.get(crit_upper, [])}


# --------------- self_rubric helpers ---------------

def render_oracle_as_checklist(
    rubric_items_by_id: Dict[str, Dict[str, Any]],
    oracle_rubrics: Dict[str, str],
    include_rationale: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """Convert pre-defined rubric items + oracle labels into checklist format for few-shot."""
    checklist: Dict[str, List[Dict[str, Any]]] = {}
    for item_id, item in rubric_items_by_id.items():
        criterion = item.get("criterion", "")
        text = item.get("text", "")
        label = str(oracle_rubrics.get(item_id, "uncertain")).lower()
        entry: Dict[str, Any] = {"question": text, "answer": label}
        if include_rationale:
            entry["rationale"] = ""
        checklist.setdefault(criterion, []).append(entry)
    return checklist


def render_checklist_block(checklist: Dict[str, List[Dict[str, Any]]]) -> str:
    """Render a checklist dict to text for stage 2 prompt."""
    lines: List[str] = []
    for criterion in ("C1", "C2", "C3"):
        items = checklist.get(criterion, [])
        if not items:
            continue
        lines.append(f"{criterion}:")
        for item in items:
            q = item.get("question", "")
            a = item.get("answer", "uncertain")
            lines.append(f"  - Q: {q} A: {a}")
    return "\n".join(lines)


def build_self_rubric_stage1_prompt(
    *,
    criterion_order: List[str],
    include_criterion_definitions: bool,
    include_rationale: bool,
) -> Tuple[str, str]:
    """Stage 1 prompt: generate + answer checklist questions."""
    rationale_item = ', "rationale": string' if include_rationale else ""
    schema_block = (
        '{\n'
        '  "checklist": {\n'
        '    "C1": [ {"question": string, "answer": "yes"|"no"|"uncertain"'
        + rationale_item + '}, ... ],\n'
        '    "C2": [ ... ],\n'
        '    "C3": [ ... ]\n'
        '  }\n'
        '}'
    )

    lines = [
        "You are evaluating Critical View of Safety (CVS).",
        "",
    ]
    if include_criterion_definitions:
        definition_lines = []
        for criterion in criterion_order:
            definition = CRITERION_DEFINITIONS.get(criterion)
            if definition:
                definition_lines.append(f"- {criterion}: {definition}")
        if definition_lines:
            lines.extend([
                "Criterion definitions (for reference):",
                *definition_lines,
                "",
            ])
    lines.extend([
        "Instructions:",
        "- For each criterion (C1, C2, C3), generate a checklist of diagnostic questions that help assess whether the criterion is met.",
        "- Each question should be answerable by looking at the image.",
        "- Answer each question as yes/no/uncertain based only on the image.",
    ])
    if include_rationale:
        lines.append("- Provide a brief rationale for each answer.")
    lines.extend([
        "",
        "Return ONLY valid JSON (no markdown, no extra keys) with this schema:",
        schema_block,
    ])
    prompt = "\n".join(lines) + "\n"
    return prompt, schema_block


def build_self_rubric_stage1_prompt_few_shot(
    *,
    criterion_order: List[str],
    include_criterion_definitions: bool,
    include_rationale: bool,
    examples: List[Dict[str, Any]],
    rubric_items_by_id: Dict[str, Dict[str, Any]],
    img: Image.Image,
) -> Tuple[Tuple[Any, ...], str]:
    """Stage 1 few-shot: show example images + GT scores (no oracle checklists)."""
    prompt_text, schema_block = build_self_rubric_stage1_prompt(
        criterion_order=criterion_order,
        include_criterion_definitions=include_criterion_definitions,
        include_rationale=include_rationale,
    )
    parts: List[Any] = [build_few_shot_header(), ""]
    for idx, ex in enumerate(examples, start=1):
        parts.append(f"Example {idx} (training):\nImage:")
        parts.append(ex["img"])
        gt_pred = {k: ex["gt"].get(k) for k in ("c1", "c2", "c3") if ex["gt"].get(k) is not None}
        if gt_pred:
            parts.append("Ground-truth CVS scores:")
            parts.append(json.dumps({"pred": gt_pred}, ensure_ascii=False))
        parts.append("")
    parts.append("Now analyze the next image.")
    parts.append("Image:")
    parts.append(img)
    parts.append(prompt_text)
    return sanitize_prompt_parts(parts), schema_block


def build_self_rubric_stage2_prompt(
    *,
    criterion_order: List[str],
    checklist_block: str,
    include_rationale: bool,
    include_criterion_definitions: bool,
) -> Tuple[str, str]:
    """Stage 2 prompt: reason from self-generated checklist to predict scores."""
    rationale_schema = ""
    if include_rationale:
        rationale_schema = '\n  "rationale": { "c1": string, "c2": string, "c3": string },'
    schema_block = (
        "{"
        + rationale_schema
        + '\n  "pred": { "c1": float, "c2": float, "c3": float }\n'
        + "}"
    )

    lines = [
        "Checklist-guided reasoning task.",
    ]
    if include_criterion_definitions:
        definition_lines = []
        for criterion in criterion_order:
            definition = CRITERION_DEFINITIONS.get(criterion)
            if definition:
                definition_lines.append(f"- {criterion}: {definition}")
        if definition_lines:
            lines.extend([
                "",
                "Criterion definitions (for reference):",
                *definition_lines,
            ])
    lines.extend([
        "",
        "Checklist answers for the current image:",
        checklist_block,
        "",
        "Instructions:",
        "- Treat the checklist answers as evidence for this step.",
        "- Use checklist answers to justify confidence changes.",
        "- Output belief scores for C1/C2/C3 as floats in [0,1].",
    ])
    if include_rationale:
        lines.append("- Provide a short rationale per criterion.")
        lines.append("- Output rationale before pred.")
    lines.extend([
        "",
        "Return ONLY valid JSON (no markdown, no extra keys) with this schema:",
        schema_block,
    ])
    prompt = "\n".join(lines) + "\n"
    return prompt, schema_block


def build_self_rubric_stage2_prompt_few_shot(
    *,
    criterion_order: List[str],
    checklist_block: str,
    include_rationale: bool,
    include_criterion_definitions: bool,
    examples: List[Dict[str, Any]],
    rubric_items_by_id: Dict[str, Dict[str, Any]],
    img: Image.Image,
) -> Tuple[Tuple[Any, ...], str]:
    """Stage 2 few-shot: show example images + ground-truth preds (no oracle checklists)."""
    prompt_text, schema_block = build_self_rubric_stage2_prompt(
        criterion_order=criterion_order,
        checklist_block=checklist_block,
        include_rationale=include_rationale,
        include_criterion_definitions=include_criterion_definitions,
    )
    parts: List[Any] = [build_few_shot_header(), ""]
    for idx, ex in enumerate(examples, start=1):
        parts.append(f"Example {idx} (training):\nImage:")
        parts.append(ex["img"])
        gt_pred = {k: ex["gt"].get(k) for k in ("c1", "c2", "c3") if ex["gt"].get(k) is not None}
        if gt_pred:
            parts.append("Ground-truth pred:")
            parts.append(json.dumps({"pred": gt_pred}, ensure_ascii=False))
        parts.append("")
    parts.append("Now analyze the next image.")
    parts.append("Image:")
    parts.append(img)
    parts.append(prompt_text)
    return sanitize_prompt_parts(parts), schema_block


def validate_self_rubric_stage1(
    obj: Dict[str, Any],
    *,
    include_rationale: bool,
) -> List[str]:
    """Validate self_rubric stage 1 output."""
    errs: List[str] = []
    if not isinstance(obj, dict):
        return ["not_a_dict"]
    if "checklist" not in obj:
        return ["missing_checklist"]
    checklist = obj["checklist"]
    if not isinstance(checklist, dict):
        return ["checklist_not_dict"]
    for criterion in ("C1", "C2", "C3"):
        items = checklist.get(criterion)
        if items is None:
            errs.append(f"checklist_missing_{criterion}")
            continue
        if not isinstance(items, list):
            errs.append(f"checklist_{criterion}_not_list")
            continue
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                errs.append(f"checklist_{criterion}_{i}_not_dict")
                continue
            if "question" not in item:
                errs.append(f"checklist_{criterion}_{i}_missing_question")
            if "answer" not in item:
                errs.append(f"checklist_{criterion}_{i}_missing_answer")
            if include_rationale and "rationale" not in item:
                errs.append(f"checklist_{criterion}_{i}_missing_rationale")
    return errs


# --------------- per-criterion self_rubric stage 2 ---------------

def build_self_rubric_stage2_prompt_single_criterion(
    *,
    criterion: str,
    checklist_block: str,
    include_rationale: bool,
    include_criterion_definitions: bool,
) -> Tuple[str, str]:
    """Stage 2 prompt for predicting a single criterion from its checklist."""
    crit_lower = criterion.lower()
    crit_upper = criterion.upper()
    rationale_schema = ""
    if include_rationale:
        rationale_schema = f'\n  "rationale": {{ "{crit_lower}": string }},'
    schema_block = (
        "{"
        + rationale_schema
        + f'\n  "pred": {{ "{crit_lower}": float }}\n'
        + "}"
    )

    lines = [
        "Checklist-guided reasoning task.",
    ]
    if include_criterion_definitions:
        definition = CRITERION_DEFINITIONS.get(crit_upper)
        if definition:
            lines.extend([
                "",
                "Criterion definition (for reference):",
                f"- {crit_upper}: {definition}",
            ])
    lines.extend([
        "",
        f"Checklist answers for criterion {crit_upper}:",
        checklist_block,
        "",
        "Instructions:",
        "- Treat the checklist answers as evidence for this step.",
        "- Use checklist answers to justify confidence changes.",
        f"- Output a belief score for {crit_upper} as a float in [0,1].",
    ])
    if include_rationale:
        lines.append(f"- Provide a short rationale for {crit_upper}.")
        lines.append("- Output rationale before pred.")
    lines.extend([
        "",
        "Return ONLY valid JSON (no markdown, no extra keys) with this schema:",
        schema_block,
    ])
    prompt = "\n".join(lines) + "\n"
    return prompt, schema_block


def build_self_rubric_stage2_prompt_single_criterion_few_shot(
    *,
    criterion: str,
    checklist_block: str,
    include_rationale: bool,
    include_criterion_definitions: bool,
    examples: List[Dict[str, Any]],
    rubric_items_by_id: Dict[str, Dict[str, Any]],
    img: "Image.Image",
) -> Tuple[Tuple[Any, ...], str]:
    """Stage 2 few-shot for predicting a single criterion (no oracle checklists)."""
    crit_lower = criterion.lower()
    crit_upper = criterion.upper()
    prompt_text, schema_block = build_self_rubric_stage2_prompt_single_criterion(
        criterion=criterion,
        checklist_block=checklist_block,
        include_rationale=include_rationale,
        include_criterion_definitions=include_criterion_definitions,
    )
    parts: List[Any] = [build_few_shot_header(), ""]
    for idx, ex in enumerate(examples, start=1):
        parts.append(f"Example {idx} (training):\nImage:")
        parts.append(ex["img"])
        gt_val = ex["gt"].get(crit_lower)
        if gt_val is not None:
            parts.append("Ground-truth pred:")
            parts.append(json.dumps({"pred": {crit_lower: gt_val}}, ensure_ascii=False))
        parts.append("")
    parts.append("Now analyze the next image.")
    parts.append("Image:")
    parts.append(img)
    parts.append(prompt_text)
    return sanitize_prompt_parts(parts), schema_block


# --------------- previous-frames helpers ---------------

def build_video_frame_index(
    filtered_rows: List[Dict[str, Any]],
    *,
    mode: str,
    dataset: str,
    default_split: str,
    image_root: Path,
) -> Dict[str, List[Tuple[int, Path]]]:
    """Build video_id -> sorted [(frame_id, image_path)] index.

    mode="key"        – use only the frames already present in filtered_rows.
    mode="continuous"  – scan denser frame sources (annotation_coco_vid.json
                         for endoscapes, data/frames_full/ for SAGES).
    """
    index: Dict[str, List[Tuple[int, Path]]] = {}

    if mode == "key":
        for row in filtered_rows:
            video_id, split, frame_id = extract_identifiers(row, default_split=default_split)
            if frame_id < 0:
                continue
            image = row.get("image") or {}
            file_name = image.get("file_name") or row.get("file_name")
            image_path = image.get("image_path") or row.get("image_path")
            resolved = resolve_image_path(
                image_path if isinstance(image_path, str) else None,
                file_name if isinstance(file_name, str) else None,
                split=split,
                dataset_root=image_root,
            )
            index.setdefault(video_id, []).append((frame_id, resolved))

    elif mode == "continuous" and dataset == "endoscapes":
        # Collect splits present in filtered_rows
        splits_needed: set[str] = set()
        for row in filtered_rows:
            _, split, _ = extract_identifiers(row, default_split=default_split)
            splits_needed.add(split)
        for split in splits_needed:
            ann_path = ENDOSCAPES_ROOT / split / ENDOSCAPES_VID_ANN_FILE
            if not ann_path.exists():
                print(f"[WARN] Missing continuous annotation file: {ann_path}")
                continue
            ann_data = load_json(ann_path)
            for img_entry in ann_data.get("images", []):
                file_name = img_entry.get("file_name", "")
                vid = img_entry.get("video_id")
                if vid is None or not file_name:
                    continue
                vid_str = str(vid)
                frame_num = _endoscapes_frame_num_from_filename(file_name)
                img_path = (ENDOSCAPES_ROOT / split / file_name).resolve()
                index.setdefault(vid_str, []).append((frame_num, img_path))

    elif mode == "continuous" and dataset == "cvs_challenge_sages_v1":
        # Collect (video_id, split) pairs from filtered_rows
        vid_splits: set[Tuple[str, str]] = set()
        for row in filtered_rows:
            video_id, split, _ = extract_identifiers(row, default_split=default_split)
            vid_splits.add((video_id, split))
        for video_id, split in vid_splits:
            vid_dir = SAGES_FRAMES_FULL_ROOT / split / video_id
            if not vid_dir.is_dir():
                continue
            for fp in vid_dir.iterdir():
                if fp.suffix.lower() not in (".jpg", ".png"):
                    continue
                stem = fp.stem  # e.g. frame_000030
                try:
                    fid = int(stem.split("_", 1)[1])
                except (IndexError, ValueError):
                    continue
                index.setdefault(video_id, []).append((fid, fp.resolve()))
    else:
        print(f"[WARN] Unsupported prev_frames_mode={mode} for dataset={dataset}")

    # Sort each video's frames by frame_id and deduplicate
    for vid in index:
        index[vid] = sorted(set(index[vid]), key=lambda x: x[0])
    return index


def get_previous_frames(
    video_id: str,
    frame_id: int,
    k: int,
    video_frame_index: Dict[str, List[Tuple[int, Path]]],
) -> List[Tuple[int, Path]]:
    """Return the k closest predecessor frames (fid < frame_id) for a video."""
    frames = video_frame_index.get(video_id, [])
    if not frames or k <= 0:
        return []
    # Binary search for insertion point
    import bisect
    fids = [f[0] for f in frames]
    pos = bisect.bisect_left(fids, frame_id)
    # Take up to k entries before pos
    start = max(0, pos - k)
    return frames[start:pos]


def inject_previous_frames(
    prompt_or_parts,
    prev_frame_images: List[Tuple[int, "Image.Image"]],
) -> Any:
    """Insert previous-frame images into the prompt before the current frame.

    Handles two input shapes:
    - tuple of parts (multi-part prompt with text + images)
    - (text, img) pair (simple prompt)

    Returns the same shape with previous frames injected.
    """
    if not prev_frame_images:
        return prompt_or_parts

    prev_block: List[Any] = [
        "Previous frames from the same video (for temporal context — reason about the current frame only):"
    ]
    for fid, img in prev_frame_images:
        prev_block.append(f"Frame {fid}:")
        prev_block.append(img)
    prev_block.append("Current frame to analyze:")

    # Case 1: tuple of parts (few-shot or multi-part prompt)
    if isinstance(prompt_or_parts, tuple):
        parts = list(prompt_or_parts)
        # Find the last image in parts — that's the current frame image.
        # We insert prev_block before it.
        # Pattern: ... "Now analyze the next image." / "Image:" / <current_img> ...
        # OR:      ... "Image:" / <current_img> / <prompt_text>
        last_img_idx = None
        for i in range(len(parts) - 1, -1, -1):
            if isinstance(parts[i], Image.Image):
                last_img_idx = i
                break
        if last_img_idx is None:
            return prompt_or_parts  # no image found, return unchanged

        # Find the text part(s) just before the current image that say "Image:" or
        # "Now analyze the next image."
        insert_idx = last_img_idx
        # Walk back over text parts that introduce the current image
        while insert_idx > 0:
            prev_part = parts[insert_idx - 1]
            if isinstance(prev_part, str) and prev_part.strip() in (
                "Image:", "Now analyze the next image.",
                "Now analyze the next image.\nImage:",
            ):
                insert_idx -= 1
            else:
                break

        new_parts = parts[:insert_idx] + prev_block + parts[last_img_idx:]
        return sanitize_prompt_parts(new_parts)

    # Case 2: (prompt_text, img) pair — simple single-image prompt
    if isinstance(prompt_or_parts, (list, tuple)) and len(prompt_or_parts) == 2:
        prompt_text, current_img = prompt_or_parts
        if isinstance(prompt_text, str) and isinstance(current_img, Image.Image):
            parts: List[Any] = [prompt_text] + prev_block + [current_img]
            return sanitize_prompt_parts(parts)

    return prompt_or_parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Rubric-oracle reasoning for CVS.")
    parser.add_argument("--preset", type=str, default="main",
                        choices=["debug", "main", "upper_bound", "direct", "self_rubric", "sr_guided"])
    parser.add_argument("--dataset", type=str, default="endoscapes",
                        choices=["endoscapes", "cvs_challenge_sages_v1"])
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--manifest_jsonl", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--manifest_root", type=str, default=None)
    parser.add_argument("--use_manifest_rows", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--rubric_path", type=str, default=str(DEFAULT_RUBRIC_PATH))
    parser.add_argument("--annotation_paths", type=str, default=None)
    parser.add_argument("--annotation_version", type=str, default="v2")
    parser.add_argument("--annotation_mode", type=str, default=argparse.SUPPRESS,
                        choices=["predicted_only", "oracle_only", "oracle_plus_predicted", "direct", "self_rubric", "sr_guided"])
    parser.add_argument("--pipeline_mode", type=str, default=argparse.SUPPRESS,
                        choices=["one_call", "two_call", "direct", "self_rubric", "sr_guided"])
    parser.add_argument("--model", type=str, default=MODEL_ID,
                        choices=["gpt-4.1-mini", "claude-haiku-4-5-20251001", "claude-opus-4-5-20251101"])
    parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "outputs" / "generated")
    parser.add_argument("--include_rationale_stage1", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--include_rationale_stage2", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--include_rationale_direct", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--include_rationale_direct_first", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--include_criterion_definitions", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--rubrics_block_include_weights", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--compute_weighted_rubric_score", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument(
        "--weighted_score_include_types",
        type=str,
        default=argparse.SUPPRESS,
        choices=["all", "evidence_only", "preconditions_only", "modifiers_only"],
    )
    parser.add_argument("--few_shot", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--few_shot_k", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--few_shot_path", type=str, default=argparse.SUPPRESS)
    parser.add_argument(
        "--few_shot_json_type",
        type=str,
        default=argparse.SUPPRESS,
        choices=["fs_v3", "v3"],
    )
    parser.add_argument(
        "--few_shot_dataset",
        type=str,
        default=None,
        choices=["endoscapes", "cvs_challenge_sages_v1"],
        help="Override which dataset's few-shot examples to use. "
             "Default: same as --dataset.",
    )
    parser.add_argument("--few_shot_short", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--stage1_short", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--few_shot_seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--criteria", type=str, default=None)
    parser.add_argument("--stage0_few_shot", action="store_true", default=False,
                        help="Enable few-shot for stage 0 (self-rubric) in sr_guided mode. "
                             "Uses the same few-shot examples as stage 1.")
    parser.add_argument("--per_criterion", action="store_true", default=False)
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--use_cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prev_frames_k", type=int, default=0)
    parser.add_argument("--prev_frames_mode", type=str, default="key", choices=["key", "continuous"])
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--normalize_side", type=int, default=None,
                        help="Resize all images (input + few-shot) so the shorter side "
                             "equals this value in pixels (e.g. 480). Preserves aspect ratio.")
    parser.add_argument("--run", type=int, default=None,
                        help="Run number (e.g. 1, 2, 3). If provided, uses _runXXX "
                             "suffix instead of timestamp. Padded to 3 digits.")

    args = parser.parse_args()
    if args.output_root.resolve().is_relative_to((REPO_ROOT / "outputs/rubric_oracle").resolve()):
        parser.error("The frozen paper output directory is read-only; choose another --output_root.")

    # Set global image normalization
    global _NORMALIZE_SIDE
    _NORMALIZE_SIDE = args.normalize_side
    if _NORMALIZE_SIDE is not None:
        print(f"[INFO] Image normalization enabled: shorter side → {_NORMALIZE_SIDE}px")

    presets = {
        "debug": {
            "pipeline_mode": "one_call",
            "annotation_mode": "predicted_only",
            "include_rationale_stage1": False,
            "include_rationale_stage2": False,
            "include_rationale_direct": False,
            "include_rationale_direct_first": False,
            "include_criterion_definitions": False,
            "rubrics_block_include_weights": False,
            "compute_weighted_rubric_score": False,
            "weighted_score_include_types": "all",
            "few_shot": False,
            "few_shot_k": 0,
            "few_shot_path": "",
            "few_shot_seed": 13,
            "few_shot_short": False,
            "stage1_short": False,
            "use_manifest_rows": False,
        },
        "main": {
            "pipeline_mode": "two_call",
            "annotation_mode": "predicted_only",
            "include_rationale_stage1": False,
            "include_rationale_stage2": False,
            "include_rationale_direct": False,
            "include_rationale_direct_first": False,
            "include_criterion_definitions": True,
            "rubrics_block_include_weights": False,
            "compute_weighted_rubric_score": True,
            "weighted_score_include_types": "all",
            "few_shot": False,
            "few_shot_k": 0,
            "few_shot_json_type": "fs_v3",
            "few_shot_path": "",
            "few_shot_seed": 13,
            "few_shot_short": False,
            "stage1_short": False,
            "use_manifest_rows": False,
        },
        "upper_bound": {
            "pipeline_mode": "two_call",
            "annotation_mode": "oracle_only",
            "include_rationale_stage1": False,
            "include_rationale_stage2": False,
            "include_rationale_direct": False,
            "include_rationale_direct_first": False,
            "include_criterion_definitions": True,
            "rubrics_block_include_weights": False,
            "compute_weighted_rubric_score": True,
            "weighted_score_include_types": "all",
            "few_shot": False,
            "few_shot_k": 0,
            "few_shot_path": "",
            "few_shot_seed": 13,
            "few_shot_short": False,
            "stage1_short": False,
            "use_manifest_rows": False,
        },
        "direct": {
            "pipeline_mode": "direct",
            "annotation_mode": "direct",
            "include_rationale_stage1": False,
            "include_rationale_stage2": False,
            "include_rationale_direct": False,
            "include_rationale_direct_first": False,
            "include_criterion_definitions": True,
            "rubrics_block_include_weights": False,
            "compute_weighted_rubric_score": False,
            "weighted_score_include_types": "all",
            "few_shot": False,
            "few_shot_k": 0,
            "few_shot_path": "",
            "few_shot_seed": 13,
            "few_shot_short": False,
            "stage1_short": False,
            "use_manifest_rows": False,
        },
        "self_rubric": {
            "pipeline_mode": "self_rubric",
            "annotation_mode": "self_rubric",
            "include_rationale_stage1": False,
            "include_rationale_stage2": False,
            "include_rationale_direct": False,
            "include_rationale_direct_first": False,
            "include_criterion_definitions": True,
            "rubrics_block_include_weights": False,
            "compute_weighted_rubric_score": False,
            "weighted_score_include_types": "all",
            "few_shot": False,
            "few_shot_k": 0,
            "few_shot_json_type": "fs_v3",
            "few_shot_path": "",
            "few_shot_seed": 13,
            "few_shot_short": False,
            "stage1_short": True,
            "use_manifest_rows": False,
        },
        "sr_guided": {
            "pipeline_mode": "sr_guided",
            "annotation_mode": "sr_guided",
            "include_rationale_stage1": False,
            "include_rationale_stage2": False,
            "include_rationale_direct": False,
            "include_rationale_direct_first": False,
            "include_criterion_definitions": True,
            "rubrics_block_include_weights": False,
            "compute_weighted_rubric_score": True,
            "weighted_score_include_types": "all",
            "few_shot": False,
            "few_shot_k": 0,
            "few_shot_json_type": "fs_v3",
            "few_shot_path": "",
            "few_shot_seed": 13,
            "few_shot_short": False,
            "stage1_short": False,
            "use_manifest_rows": False,
        },
    }
    preset = presets.get(args.preset, {})
    for key, value in preset.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    if args.pipeline_mode == "direct" and not hasattr(args, "annotation_mode"):
        args.annotation_mode = "direct"
    if args.pipeline_mode == "self_rubric" and not hasattr(args, "annotation_mode"):
        args.annotation_mode = "self_rubric"
    if args.pipeline_mode == "sr_guided" and not hasattr(args, "annotation_mode"):
        args.annotation_mode = "sr_guided"
    data_root, frames_root, labels_root, default_manifest_root = resolve_dataset_roots(args.dataset)
    manifest_root = Path(args.manifest_root).resolve() if args.manifest_root else default_manifest_root
    image_root = data_root if data_root is not None else frames_root
    if image_root is None:
        raise ValueError("Missing dataset image root (data_root or frames_root).")

    rubric_path = Path(args.rubric_path).resolve()
    if args.pipeline_mode not in ("direct", "self_rubric") and not rubric_path.exists():
        raise FileNotFoundError(f"Missing rubric file: {rubric_path}")

    annotation_paths: List[Path] = []
    if not bool(args.use_manifest_rows):
        annotation_paths = parse_annotation_paths(args.annotation_paths, version=args.annotation_version, dataset=args.dataset)
        for p in annotation_paths:
            if not p.exists():
                raise FileNotFoundError(f"Missing annotation file: {p}")

    manifest_jsonl = Path(args.manifest_jsonl).resolve() if args.manifest_jsonl else None
    if manifest_jsonl and not manifest_jsonl.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_jsonl}")

    criterion_order: List[str] = []
    rubric_grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    rubric_items_by_id: Dict[str, Dict[str, Any]] = {}
    if args.pipeline_mode not in ("direct", "self_rubric"):
        criterion_order, rubric_grouped, rubric_items_by_id = load_rubrics(rubric_path)
    elif args.pipeline_mode == "self_rubric":
        criterion_order = ["C1", "C2", "C3"]
        if rubric_path.exists():
            _, rubric_grouped, rubric_items_by_id = load_rubrics(rubric_path)
    annotation_rows: List[Dict[str, Any]] = []
    if not bool(args.use_manifest_rows):
        for p in annotation_paths:
            annotation_rows.extend(list(iter_jsonl(p)))

    manifest_frame_set: Optional[set[Tuple[str, int]]] = None
    if args.manifest_jsonl or args.manifest:
        ds_manifest = get_dataset(
            args.dataset,
            manifest_jsonl=manifest_jsonl,
            manifest=args.manifest,
            split=args.split,
            data_root=data_root,
            frames_root=frames_root,
            labels_root=labels_root,
            manifest_root=manifest_root,
            manifest_only=(args.dataset == "endoscapes"),
        )
        manifest_rows = get_frame_rows(ds_manifest)
        manifest_frame_set = {
            (str(r.get("video_id")), int(r.get("frame_id")))
            for r in manifest_rows
            if r.get("frame_id") is not None
        }

    filtered_rows: List[Dict[str, Any]] = []
    if bool(args.use_manifest_rows):
        if args.annotation_mode in ("oracle_only", "oracle_plus_predicted"):
            raise ValueError("Oracle rubric labels are unavailable when using manifest rows.")
        if not (args.manifest_jsonl or args.manifest):
            raise ValueError("--use_manifest_rows requires --manifest_jsonl or --manifest")
        ds = get_dataset(
            args.dataset,
            manifest_jsonl=manifest_jsonl,
            manifest=args.manifest,
            split=args.split,
            data_root=data_root,
            frames_root=frames_root,
            labels_root=labels_root,
            manifest_root=manifest_root,
            manifest_only=(args.dataset == "endoscapes"),
        )
        filtered_rows = get_frame_rows(ds)
    else:
        for row in annotation_rows:
            video_id, split, frame_id = extract_identifiers(row, default_split=args.split)
            if frame_id < 0:
                print(f"[WARN] Skip row with invalid frame_id (video_id={video_id}, split={split})")
                continue
            if manifest_frame_set and (video_id, frame_id) not in manifest_frame_set:
                continue
            filtered_rows.append(row)

    rng = random.Random(args.seed)
    if args.max_items is not None and len(filtered_rows) > args.max_items:
        rng.shuffle(filtered_rows)
        filtered_rows = filtered_rows[: args.max_items]

    if not filtered_rows:
        raise RuntimeError("No annotations to process after filtering.")

    model = None
    if not args.dryrun:
        model = models.create_model(args.model, use_cache=args.use_cache, dataset=args.dataset, temperature=args.temperature)

    few_shot_examples: List[Dict[str, Any]] = []
    few_shot_k_actual = 0
    if bool(args.few_shot):
        few_shot_path = parse_few_shot_path(args)
        if not few_shot_path.exists():
            raise FileNotFoundError(f"Missing few-shot file: {few_shot_path}")
        few_shot_rows = load_few_shot_rows(few_shot_path)
        few_shot_rows = select_few_shot_rows(
            few_shot_rows,
            int(args.few_shot_k),
            int(args.few_shot_seed),
        )
        require_oracle = args.pipeline_mode not in ("direct",)
        few_shot_examples = prepare_few_shot_examples(
            few_shot_rows,
            default_split=args.split,
            require_oracle_rubrics=require_oracle,
            dataset_root=image_root,
        )
        few_shot_k_actual = len(few_shot_examples)
        if few_shot_k_actual == 0:
            raise RuntimeError("No usable few-shot examples found.")

    rubric_tag = rubric_tag_from_path(rubric_path)
    model_tag = sanitize_model_tag(args.model)
    output_path = (
        args.output_root.resolve()
        / f"{args.dataset}_{args.split}__{rubric_tag}__ann{args.annotation_version}"
        f"__{args.annotation_mode}__seed{args.seed}__{model_tag}.jsonl"
    )
    if manifest_jsonl:
        manifest_tag = manifest_jsonl.stem
        output_path = output_path.with_name(
            output_path.name.replace(
                f"__seed{args.seed}__",
                f"__{manifest_tag}__seed{args.seed}__",
            )
        )
        output_path = (
            output_path.parent
            / "manifests"
            / manifest_tag
            / output_path.name
        )
    if args.pipeline_mode == "direct":
        output_path = output_path.with_name(
            output_path.name.replace(
                f"__{args.annotation_mode}__",
                f"__{args.annotation_mode}__preset-direct__",
            )
        )
    if args.pipeline_mode == "sr_guided":
        output_path = output_path.with_name(
            output_path.name.replace(
                f"__{args.annotation_mode}__",
                f"__sr_guided__",
            )
        )
        if args.stage0_few_shot:
            output_path = output_path.with_name(
                output_path.name.replace(
                    "__sr_guided__",
                    "__sr_guided__s0fs__",
                )
            )
    if bool(args.include_criterion_definitions):
        output_path = output_path.with_name(
            output_path.name.replace(
                f"__seed{args.seed}__",
                f"__critdef__seed{args.seed}__",
            )
        )
    rationale_tag = ""
    if bool(args.include_rationale_stage1):
        rationale_tag += "__rat1"
    if bool(args.include_rationale_stage2):
        rationale_tag += "__rat2"
    if bool(getattr(args, "include_rationale_direct_first", False)):
        rationale_tag += "__ratdf"
    elif bool(args.include_rationale_direct):
        rationale_tag += "__ratd"
    if bool(args.stage1_short):
        rationale_tag += "__s1short"
    if rationale_tag:
        output_path = output_path.with_name(
            output_path.name.replace(
                f"__seed{args.seed}__",
                f"{rationale_tag}__seed{args.seed}__",
            )
        )
    if bool(args.few_shot):
        few_shot_tag = f"__fs{few_shot_k_actual}__"
        fs_type = normalize_few_shot_json_type(args)
        if fs_type and fs_type.startswith("fs_v") and fs_type != "fs_v1":
            # e.g. fs_v2 -> __fsv2__, fs_v3 -> __fsv3__, ... fs_v8 -> __fsv8__
            ver = fs_type.replace("fs_v", "fsv")
            few_shot_tag = f"__{ver}__fs{few_shot_k_actual}__"
        # Cross-dataset FS: tag with source dataset when different from eval dataset
        fs_dataset_override = getattr(args, "few_shot_dataset", None)
        eval_dataset = getattr(args, "dataset", "endoscapes") or "endoscapes"
        if fs_dataset_override and fs_dataset_override != eval_dataset:
            # e.g. __fsd-sages__fsv3__fs4__ or __fsd-endo__fsv3__fs4__
            short_name = "sages" if fs_dataset_override == "cvs_challenge_sages_v1" else "endo"
            few_shot_tag = f"__fsd-{short_name}{few_shot_tag}"
        output_path = output_path.with_name(
            output_path.name.replace(
                f"__seed{args.seed}__",
                f"{few_shot_tag}__seed{args.seed}__",
            )
        )

    if args.prev_frames_k > 0:
        mode_char = "k" if args.prev_frames_mode == "key" else "c"
        pf_tag = f"__pf{args.prev_frames_k}{mode_char}__"
        output_path = output_path.with_name(
            output_path.name.replace(
                f"__seed{args.seed}__",
                f"{pf_tag}__seed{args.seed}__",
            )
        )

    if args.per_criterion:
        output_path = output_path.with_name(
            output_path.name.replace(
                f"__seed{args.seed}__",
                f"__percrit__seed{args.seed}__",
            )
        )

    if args.temperature != 0.1:
        temp_tag = f"__temp{args.temperature:g}"
        output_path = output_path.with_name(
            output_path.name.replace(
                f"__seed{args.seed}__",
                f"{temp_tag}__seed{args.seed}__",
            )
        )

    if args.normalize_side is not None:
        norm_tag = f"__normside{args.normalize_side}"
        output_path = output_path.with_name(
            output_path.name.replace(
                f"__seed{args.seed}__",
                f"{norm_tag}__seed{args.seed}__",
            )
        )

    if args.run is not None:
        run_tag = f"_run{args.run:03d}"
        output_path = output_path.with_name(
            output_path.stem + run_tag + output_path.suffix
        )
    elif not args.use_cache:
        dt_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path.with_name(
            output_path.stem + f"__{dt_tag}" + output_path.suffix
        )

    print(f"[INFO] Output path: {output_path}")

    # When using --run, support resuming: load existing results, keep only good
    # entries (non-empty raw_response and parse_ok), skip them during processing.
    _completed_keys: set = set()
    if not args.dryrun and args.run is not None and output_path.exists():
        good_lines = []
        n_total = 0
        n_bad = 0
        with output_path.open() as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                n_total += 1
                try:
                    _obj = json.loads(_line)
                except json.JSONDecodeError:
                    n_bad += 1
                    continue
                raw = _obj.get("raw_response", "")
                parse_ok = _obj.get("parse_ok", False)
                # For per_criterion mode, raw_response is "per_criterion_mode" — check per_criterion_metadata
                if raw == "per_criterion_mode":
                    # Check if any sub-criterion had empty response
                    pcm = _obj.get("per_criterion_metadata", {})
                    has_empty = any(
                        not v.get("raw_response", "")
                        for v in pcm.values()
                        if isinstance(v, dict)
                    )
                    if has_empty or not parse_ok:
                        n_bad += 1
                        continue
                elif not raw or not parse_ok:
                    n_bad += 1
                    continue
                good_lines.append(_line)
                _key = f"{_obj.get('video_id', '')}##{_obj.get('frame_id', '')}"
                _completed_keys.add(_key)
        # Rewrite the file with only good entries
        with output_path.open("w", encoding="utf-8") as _f:
            for _line in good_lines:
                _f.write(_line + "\n")
        print(f"[INFO] Resuming run: {n_total} existing entries, "
              f"{n_bad} bad (empty/parse-fail) removed, "
              f"{len(_completed_keys)} good entries kept")

    # Try to reuse valid entries from a related manifest to avoid redundant
    # API calls.  Supports two directions:
    #   1. Running the full set (e.g. test_dev_dev) → reuse from _small subset
    #   2. Running the _small subset → reuse from the full set
    # The manifest tag is embedded in both the directory name and the filename,
    # so we translate both when looking for the donor file.
    _reuse_cache: Dict[str, str] = {}  # key -> json line
    if args.run is not None and manifest_jsonl is not None:
        _manifest_stem = manifest_jsonl.stem  # e.g. "test_dev_dev" or "test_dev_dev_small"
        _donor_stems: list[str] = []
        # Direction 1: full → small
        _small_manifest = _manifest_stem + "_small"
        if _manifest_stem != _small_manifest:
            _donor_stems.append(_small_manifest)
        # Direction 2: small → full
        if _manifest_stem.endswith("_small"):
            _full_manifest = _manifest_stem[: -len("_small")]
            _donor_stems.append(_full_manifest)

        for _donor_stem in _donor_stems:
            _donor_dir = output_path.parent.parent / _donor_stem
            if not _donor_dir.is_dir():
                continue
            # Translate filename: replace current manifest tag with donor tag
            _donor_filename = output_path.name.replace(
                f"__{_manifest_stem}__", f"__{_donor_stem}__"
            )
            _donor_file = _donor_dir / _donor_filename
            if not _donor_file.exists():
                continue
            _n_loaded = 0
            with _donor_file.open() as _df:
                for _dline in _df:
                    _dline = _dline.strip()
                    if not _dline:
                        continue
                    try:
                        _dobj = json.loads(_dline)
                    except json.JSONDecodeError:
                        continue
                    _draw = _dobj.get("raw_response", "")
                    _dparse_ok = _dobj.get("parse_ok", False)
                    if _draw == "per_criterion_mode":
                        _dpcm = _dobj.get("per_criterion_metadata", {})
                        _d_has_empty = any(
                            not v.get("raw_response", "")
                            for v in _dpcm.values()
                            if isinstance(v, dict)
                        )
                        if _d_has_empty or not _dparse_ok:
                            continue
                    elif not _draw or not _dparse_ok:
                        continue
                    _dkey = f"{_dobj.get('video_id', '')}##{_dobj.get('frame_id', '')}"
                    if _dkey not in _completed_keys and _dkey not in _reuse_cache:
                        _reuse_cache[_dkey] = _dline
                        _n_loaded += 1
            print(f"[INFO] Loaded {_n_loaded} reusable entries from {_donor_file}")

    # For sr_guided: try to reuse stage 0 (self-rubric) results from an existing
    # self_rubric run with the same model, dataset/manifest, and run number (no FS).
    # Skip caching when stage0_few_shot is enabled (cached results are from no-FS runs).
    _sr_stage0_cache: Dict[str, Dict[str, Any]] = {}  # key -> {"checklist": ..., "stage0_raw_response": ..., ...}
    if args.pipeline_mode == "sr_guided" and args.run is not None and not args.stage0_few_shot:
        # Construct the expected self_rubric donor path by replacing __sr_guided__
        # with __self_rubric__ in the output path, and also try with __s1short__ tag
        _sr_output_name = output_path.name
        _sr_candidates: list[Path] = []
        # Try exact replacement: sr_guided -> self_rubric
        _sr_name_base = _sr_output_name.replace("__sr_guided__", "__self_rubric__")
        _sr_candidates.append(output_path.parent / _sr_name_base)
        # Also try with __s1short__ tag (default for self_rubric preset)
        _sr_name_s1short = _sr_name_base.replace(
            f"__seed{args.seed}__",
            f"__s1short__seed{args.seed}__",
        )
        _sr_candidates.append(output_path.parent / _sr_name_s1short)
        # Also try with __rat1__s1short__ (self_rubric with rationale)
        _sr_name_rat1 = _sr_name_base.replace(
            f"__seed{args.seed}__",
            f"__rat1__s1short__seed{args.seed}__",
        )
        _sr_candidates.append(output_path.parent / _sr_name_rat1)
        # Also glob for any self_rubric file matching the model and run
        _sr_glob_pattern = _sr_name_base.replace(
            f"__critdef__",
            f"__critdef__*",
        )
        import glob as _glob_mod
        _sr_glob_matches = sorted(_glob_mod.glob(str(output_path.parent / _sr_glob_pattern)))
        for _gm in _sr_glob_matches:
            _gm_path = Path(_gm)
            if _gm_path not in _sr_candidates:
                _sr_candidates.append(_gm_path)

        for _sr_donor in _sr_candidates:
            if _sr_donor.exists():
                _n_sr_loaded = 0
                with _sr_donor.open() as _f:
                    for _line in _f:
                        _line = _line.strip()
                        if not _line:
                            continue
                        try:
                            _obj = json.loads(_line)
                        except json.JSONDecodeError:
                            continue
                        # Only reuse entries with a valid checklist
                        _checklist = _obj.get("checklist", {})
                        if not _checklist:
                            continue
                        _skey = f"{_obj.get('video_id', '')}##{_obj.get('frame_id', '')}"
                        if _skey not in _sr_stage0_cache:
                            _sr_stage0_cache[_skey] = {
                                "checklist": _checklist,
                                "stage0_raw_response": _obj.get("stage1_raw_response", ""),
                                "stage0_parsed": _obj.get("stage1_parsed", {}),
                                "stage0_parse_ok": _obj.get("stage1_parse_ok", False),
                                "stage0_parse_errors": _obj.get("stage1_parse_errors", []),
                            }
                            _n_sr_loaded += 1
                print(f"[INFO] Loaded {_n_sr_loaded} self-rubric stage 0 results from {_sr_donor}")
                break  # use first matching donor file

    # Build previous-frames index if requested
    video_frame_index: Dict[str, List[Tuple[int, Path]]] = {}
    if args.prev_frames_k > 0:
        video_frame_index = build_video_frame_index(
            filtered_rows,
            mode=args.prev_frames_mode,
            dataset=args.dataset,
            default_split=args.split,
            image_root=image_root,
        )
        total_indexed = sum(len(v) for v in video_frame_index.values())
        print(f"[INFO] Built prev-frames index: {len(video_frame_index)} videos, {total_indexed} frames (mode={args.prev_frames_mode})")

    for idx, row in enumerate(tqdm(filtered_rows, desc="rubric_oracle")):
        image = row.get("image") or {}
        file_name = image.get("file_name") or row.get("file_name")
        image_path = image.get("image_path") or row.get("image_path")
        video_id, split, frame_id = extract_identifiers(row, default_split=args.split)
        gt = extract_gt(row)

        # Skip already-completed entries when resuming a run
        _entry_key = f"{video_id}##{frame_id}"
        if _completed_keys and _entry_key in _completed_keys:
            continue

        # Reuse valid entry from a related manifest if available
        if _reuse_cache and _entry_key in _reuse_cache:
            append_jsonl(output_path, json.loads(_reuse_cache[_entry_key]))
            _completed_keys.add(_entry_key)
            continue

        resolved_path = resolve_image_path(
            image_path if isinstance(image_path, str) else None,
            file_name if isinstance(file_name, str) else None,
            split=split,
            dataset_root=image_root,
        )
        if not resolved_path.exists():
            print(f"[WARN] Missing image: {resolved_path}")
            continue

        # Load previous frame images if requested
        prev_frame_images: List[Tuple[int, Image.Image]] = []
        prev_frame_ids: List[int] = []
        if args.prev_frames_k > 0:
            prev_entries = get_previous_frames(video_id, frame_id, args.prev_frames_k, video_frame_index)
            for pfid, ppath in prev_entries:
                if ppath.exists():
                    prev_frame_images.append((pfid, load_image(ppath)))
                    prev_frame_ids.append(pfid)

        if args.pipeline_mode == "self_rubric":
            # Self-rubric pipeline: stage 1 generates checklist, stage 2 predicts scores.
            stage1_prompt = ""
            stage1_schema = ""
            stage1_parts: Optional[Tuple[Any, ...]] = None
            if bool(args.few_shot) and rubric_items_by_id:
                img = load_image(resolved_path)
                stage1_parts, stage1_schema = build_self_rubric_stage1_prompt_few_shot(
                    criterion_order=criterion_order,
                    include_criterion_definitions=bool(args.include_criterion_definitions),
                    include_rationale=bool(args.include_rationale_stage1),
                    examples=few_shot_examples,
                    rubric_items_by_id=rubric_items_by_id,
                    img=img,
                )
            else:
                stage1_prompt, stage1_schema = build_self_rubric_stage1_prompt(
                    criterion_order=criterion_order,
                    include_criterion_definitions=bool(args.include_criterion_definitions),
                    include_rationale=bool(args.include_rationale_stage1),
                )

            # Inject previous frames into stage 1
            if prev_frame_images:
                if stage1_parts is not None:
                    stage1_parts = inject_previous_frames(stage1_parts, prev_frame_images)
                else:
                    img = load_image(resolved_path)
                    stage1_parts = inject_previous_frames((stage1_prompt, img), prev_frame_images)

            if args.dryrun:
                print("===== SYSTEM PROMPT =====\n", SYSTEM_PROMPT)
                if stage1_parts is not None:
                    print("\n===== STAGE 1 PROMPT PARTS =====\n", format_prompt_parts_for_print(stage1_parts))
                else:
                    print("\n===== STAGE 1 PROMPT =====\n", stage1_prompt)
                print("\n===== STAGE 1 OUTPUT SCHEMA =====\n", stage1_schema)
                # Show example stage 2 prompt with dummy checklist
                dummy_checklist: Dict[str, List[Dict[str, Any]]] = {
                    crit: [{"question": "Example question?", "answer": "uncertain"}]
                    for crit in criterion_order
                }
                if args.per_criterion:
                    for crit in ["c1", "c2", "c3"]:
                        crit_filtered = filter_checklist_for_criterion(dummy_checklist, crit)
                        crit_block = render_checklist_block(crit_filtered)
                        s2_prompt, s2_schema = build_self_rubric_stage2_prompt_single_criterion(
                            criterion=crit,
                            checklist_block=crit_block,
                            include_rationale=bool(args.include_rationale_stage2),
                            include_criterion_definitions=bool(args.include_criterion_definitions),
                        )
                        if bool(args.few_shot) and rubric_items_by_id:
                            img_dry = load_image(resolved_path) if stage1_parts is None else img
                            s2_parts_dry, _ = build_self_rubric_stage2_prompt_single_criterion_few_shot(
                                criterion=crit,
                                checklist_block=crit_block,
                                include_rationale=bool(args.include_rationale_stage2),
                                include_criterion_definitions=bool(args.include_criterion_definitions),
                                examples=few_shot_examples,
                                rubric_items_by_id=rubric_items_by_id,
                                img=img_dry,
                            )
                            print(f"\n===== STAGE 2 PROMPT PARTS ({crit.upper()}) =====\n", format_prompt_parts_for_print(s2_parts_dry))
                        else:
                            print(f"\n===== STAGE 2 PROMPT ({crit.upper()}) =====\n", s2_prompt)
                        print(f"\n===== STAGE 2 OUTPUT SCHEMA ({crit.upper()}) =====\n", s2_schema)
                else:
                    dummy_block = render_checklist_block(dummy_checklist)
                    stage2_prompt, stage2_schema = build_self_rubric_stage2_prompt(
                        criterion_order=criterion_order,
                        checklist_block=dummy_block,
                        include_rationale=bool(args.include_rationale_stage2),
                        include_criterion_definitions=bool(args.include_criterion_definitions),
                    )
                    if bool(args.few_shot) and rubric_items_by_id:
                        img = load_image(resolved_path) if stage1_parts is None else img
                        stage2_parts_dry, _ = build_self_rubric_stage2_prompt_few_shot(
                            criterion_order=criterion_order,
                            checklist_block=dummy_block,
                            include_rationale=bool(args.include_rationale_stage2),
                            include_criterion_definitions=bool(args.include_criterion_definitions),
                            examples=few_shot_examples,
                            rubric_items_by_id=rubric_items_by_id,
                            img=img,
                        )
                        print("\n===== STAGE 2 PROMPT PARTS =====\n", format_prompt_parts_for_print(stage2_parts_dry))
                    else:
                        print("\n===== STAGE 2 PROMPT =====\n", stage2_prompt)
                    print("\n===== STAGE 2 OUTPUT SCHEMA =====\n", stage2_schema)
                break

            # Stage 1: generate + answer checklist
            stage1_raw = ""
            stage1_parsed: Dict[str, Any] = {}
            stage1_parse_errors: List[str] = []
            stage1_parse_ok = False
            try:
                if stage1_parts is not None:
                    stage1_raw = model([stage1_parts], system_prompt=SYSTEM_PROMPT)[0]
                else:
                    img = load_image(resolved_path)
                    stage1_raw = model([(stage1_prompt, img)], system_prompt=SYSTEM_PROMPT)[0]
                stage1_parsed = extract_json_obj(stage1_raw)
                stage1_parse_errors = validate_self_rubric_stage1(
                    stage1_parsed,
                    include_rationale=bool(args.include_rationale_stage1),
                )
                stage1_parse_ok = len(stage1_parse_errors) == 0
            except Exception as exc:
                stage1_parse_errors = [f"exception:{type(exc).__name__}"]

            checklist = stage1_parsed.get("checklist", {}) if isinstance(stage1_parsed, dict) else {}
            if not checklist:
                print(f"[WARN] Missing checklist for stage 2: video_id={video_id}, frame_id={frame_id}")
                # Save stage 1 results with error — skip stage 2
                out_row = {
                    "video_id": video_id,
                    "split": split,
                    "frame_id": frame_id,
                    "dataset": args.dataset,
                    "image_path": abs_posix(resolved_path),
                    "gt": gt,
                    "annotation_mode": args.annotation_mode,
                    "annotation_version": args.annotation_version,
                    "pipeline_mode": args.pipeline_mode,
                    "checklist": checklist,
                    "stage1_prompt": stage1_prompt if stage1_parts is None else str(stage1_parts),
                    "stage1_schema": stage1_schema,
                    "stage1_raw_response": stage1_raw,
                    "stage1_parsed": stage1_parsed,
                    "stage1_parse_ok": stage1_parse_ok,
                    "stage1_parse_errors": stage1_parse_errors,
                    "prompt": "",
                    "system_prompt": SYSTEM_PROMPT,
                    "parse_ok": False,
                    "parse_errors": ["stage1_missing_checklist"],
                    "raw_response": "",
                    "parsed": {},
                    "error": "Missing checklist from stage 1; stage 2 skipped.",
                }
                if args.prev_frames_k > 0:
                    out_row["prev_frames_k"] = args.prev_frames_k
                    out_row["prev_frames_mode"] = args.prev_frames_mode
                    out_row["prev_frame_ids"] = prev_frame_ids
                append_jsonl(output_path, out_row)
                continue

            if args.per_criterion:
                # --- Per-criterion mode: one stage 2 call per criterion ---
                per_crit_results: List[Dict[str, Any]] = []
                for crit in ["c1", "c2", "c3"]:
                    crit_filtered = filter_checklist_for_criterion(checklist, crit)
                    crit_block = render_checklist_block(crit_filtered)

                    crit_parts: Optional[Tuple[Any, ...]] = None
                    crit_prompt, crit_schema = build_self_rubric_stage2_prompt_single_criterion(
                        criterion=crit,
                        checklist_block=crit_block,
                        include_rationale=bool(args.include_rationale_stage2),
                        include_criterion_definitions=bool(args.include_criterion_definitions),
                    )
                    if bool(args.few_shot) and rubric_items_by_id:
                        img = load_image(resolved_path)
                        crit_parts, _ = build_self_rubric_stage2_prompt_single_criterion_few_shot(
                            criterion=crit,
                            checklist_block=crit_block,
                            include_rationale=bool(args.include_rationale_stage2),
                            include_criterion_definitions=bool(args.include_criterion_definitions),
                            examples=few_shot_examples,
                            rubric_items_by_id=rubric_items_by_id,
                            img=img,
                        )

                    # Inject previous frames
                    if prev_frame_images:
                        if crit_parts is not None:
                            crit_parts = inject_previous_frames(crit_parts, prev_frame_images)
                        else:
                            img = load_image(resolved_path)
                            crit_parts = inject_previous_frames((crit_prompt, img), prev_frame_images)

                    crit_response = ""
                    crit_parsed: Dict[str, Any] = {}
                    crit_parse_ok = False
                    crit_parse_errors: List[str] = []
                    try:
                        if crit_parts is not None:
                            crit_response = model([crit_parts], system_prompt=SYSTEM_PROMPT)[0]
                        else:
                            img = load_image(resolved_path)
                            crit_response = model([(crit_prompt, img)], system_prompt=SYSTEM_PROMPT)[0]
                        crit_parsed = extract_json_obj(crit_response)
                        crit_parse_errors = validate_stage2_single_criterion(
                            crit_parsed,
                            criterion=crit,
                            include_rationale=bool(args.include_rationale_stage2),
                        )
                        crit_parse_ok = len(crit_parse_errors) == 0
                    except Exception as exc:
                        crit_parse_errors = [f"exception:{type(exc).__name__}"]

                    crit_pred = crit_parsed.get("pred", {}) if isinstance(crit_parsed, dict) else {}
                    crit_rationale = crit_parsed.get("rationale", {}) if isinstance(crit_parsed, dict) else {}
                    per_crit_results.append({
                        "criterion": crit,
                        "pred": crit_pred,
                        "rationale": crit_rationale,
                        "raw_response": crit_response,
                        "parsed": crit_parsed,
                        "parse_ok": crit_parse_ok,
                        "parse_errors": crit_parse_errors,
                    })

                merged = merge_per_criterion_results(per_crit_results)
                out_row = {
                    "video_id": video_id,
                    "split": split,
                    "frame_id": frame_id,
                    "dataset": args.dataset,
                    "image_path": abs_posix(resolved_path),
                    "gt": gt,
                    "annotation_mode": args.annotation_mode,
                    "annotation_version": args.annotation_version,
                    "pipeline_mode": args.pipeline_mode,
                    "per_criterion": True,
                    "checklist": checklist,
                    "stage1_prompt": stage1_prompt if stage1_parts is None else str(stage1_parts),
                    "stage1_schema": stage1_schema,
                    "stage1_raw_response": stage1_raw,
                    "stage1_parsed": stage1_parsed,
                    "stage1_parse_ok": stage1_parse_ok,
                    "stage1_parse_errors": stage1_parse_errors,
                    "prompt": "per_criterion_mode",
                    "system_prompt": SYSTEM_PROMPT,
                    "parse_ok": merged["parse_ok"],
                    "parse_errors": merged["parse_errors"],
                    "raw_response": "per_criterion_mode",
                    "parsed": {"pred": merged["pred"]},
                    "pred": merged["pred"],
                    "per_criterion_metadata": merged["per_criterion_metadata"],
                }
                if merged["rationale"]:
                    out_row["rationale"] = merged["rationale"]
                if args.prev_frames_k > 0:
                    out_row["prev_frames_k"] = args.prev_frames_k
                    out_row["prev_frames_mode"] = args.prev_frames_mode
                    out_row["prev_frame_ids"] = prev_frame_ids

                append_jsonl(output_path, out_row)
                continue

            # --- Joint mode (default): single stage 2 call for all criteria ---
            checklist_block = render_checklist_block(checklist)

            # Stage 2: reason from checklist to predict scores
            stage2_parts: Optional[Tuple[Any, ...]] = None
            stage2_prompt, stage2_schema = build_self_rubric_stage2_prompt(
                criterion_order=criterion_order,
                checklist_block=checklist_block,
                include_rationale=bool(args.include_rationale_stage2),
                include_criterion_definitions=bool(args.include_criterion_definitions),
            )
            if bool(args.few_shot) and rubric_items_by_id:
                img = load_image(resolved_path)
                stage2_parts, _ = build_self_rubric_stage2_prompt_few_shot(
                    criterion_order=criterion_order,
                    checklist_block=checklist_block,
                    include_rationale=bool(args.include_rationale_stage2),
                    include_criterion_definitions=bool(args.include_criterion_definitions),
                    examples=few_shot_examples,
                    rubric_items_by_id=rubric_items_by_id,
                    img=img,
                )

            # Inject previous frames into stage 2
            if prev_frame_images:
                if stage2_parts is not None:
                    stage2_parts = inject_previous_frames(stage2_parts, prev_frame_images)
                else:
                    img = load_image(resolved_path)
                    stage2_parts = inject_previous_frames((stage2_prompt, img), prev_frame_images)

            response_text = ""
            parsed: Dict[str, Any] = {}
            parse_ok = False
            parse_errors: List[str] = []
            try:
                if stage2_parts is not None:
                    response_text = model([stage2_parts], system_prompt=SYSTEM_PROMPT)[0]
                else:
                    img = load_image(resolved_path)
                    response_text = model([(stage2_prompt, img)], system_prompt=SYSTEM_PROMPT)[0]
                parsed = extract_json_obj(response_text)
                parse_errors = validate_stage2(
                    parsed,
                    include_rationale=bool(args.include_rationale_stage2),
                    expect_rubrics_block=False,
                )
                parse_ok = len(parse_errors) == 0
            except Exception as exc:
                parse_errors = [f"exception:{type(exc).__name__}"]

            out_row = {
                "video_id": video_id,
                "split": split,
                "frame_id": frame_id,
                "dataset": args.dataset,
                "image_path": abs_posix(resolved_path),
                "gt": gt,
                "annotation_mode": args.annotation_mode,
                "annotation_version": args.annotation_version,
                "pipeline_mode": args.pipeline_mode,
                "checklist": checklist,
                "checklist_block": checklist_block,
                "stage1_prompt": stage1_prompt if stage1_parts is None else str(stage1_parts),
                "stage1_schema": stage1_schema,
                "stage1_raw_response": stage1_raw,
                "stage1_parsed": stage1_parsed,
                "stage1_parse_ok": stage1_parse_ok,
                "stage1_parse_errors": stage1_parse_errors,
                "prompt": stage2_prompt,
                "system_prompt": SYSTEM_PROMPT,
                "parse_ok": parse_ok,
                "parse_errors": parse_errors,
                "raw_response": response_text,
                "parsed": parsed,
            }
            if isinstance(parsed, dict) and "pred" in parsed:
                out_row["pred"] = parsed.get("pred")
            if isinstance(parsed, dict) and "rationale" in parsed:
                out_row["rationale"] = parsed.get("rationale")
            if args.prev_frames_k > 0:
                out_row["prev_frames_k"] = args.prev_frames_k
                out_row["prev_frames_mode"] = args.prev_frames_mode
                out_row["prev_frame_ids"] = prev_frame_ids

            append_jsonl(output_path, out_row)
            continue

        if args.pipeline_mode == "sr_guided":
            # SR-guided pipeline: 3 stages
            # Stage 0: self-rubric checklist (FS per --stage0_few_shot)
            # Stage 1: predict predefined rubric labels with self-rubric context (FS per args)
            # Final pred: weighted average from stage 1 rubric labels

            # ── Stage 0: Self-rubric checklist ──
            stage0_prompt = ""
            stage0_schema = ""
            stage0_parts: Optional[Tuple[Any, ...]] = None
            if args.stage0_few_shot and bool(args.few_shot) and rubric_items_by_id:
                # Stage 0 with few-shot (same as self_rubric FS stage 1)
                img = load_image(resolved_path)
                stage0_parts, stage0_schema = build_self_rubric_stage1_prompt_few_shot(
                    criterion_order=criterion_order,
                    include_criterion_definitions=bool(args.include_criterion_definitions),
                    include_rationale=False,  # stage 0 never includes rationale
                    examples=few_shot_examples,
                    rubric_items_by_id=rubric_items_by_id,
                    img=img,
                )
            else:
                # Stage 0 without few-shot (original behavior)
                stage0_prompt, stage0_schema = build_self_rubric_stage1_prompt(
                    criterion_order=criterion_order,
                    include_criterion_definitions=bool(args.include_criterion_definitions),
                    include_rationale=False,  # stage 0 never includes rationale
                )

            # Inject previous frames into stage 0
            if prev_frame_images:
                if stage0_parts is not None:
                    stage0_parts = inject_previous_frames(stage0_parts, prev_frame_images)
                else:
                    img = load_image(resolved_path)
                    stage0_parts = inject_previous_frames((stage0_prompt, img), prev_frame_images)

            if args.dryrun:
                print("===== SYSTEM PROMPT =====\n", SYSTEM_PROMPT)
                print(f"\n[Stage 0 few-shot: {args.stage0_few_shot and bool(args.few_shot)}]")
                if stage0_parts is not None:
                    print("\n===== STAGE 0 PROMPT PARTS (self-rubric) =====\n", format_prompt_parts_for_print(stage0_parts))
                else:
                    print("\n===== STAGE 0 PROMPT (self-rubric) =====\n", stage0_prompt)
                print("\n===== STAGE 0 OUTPUT SCHEMA =====\n", stage0_schema)
                # Show example stage 1 prompt with dummy checklist
                dummy_checklist: Dict[str, List[Dict[str, Any]]] = {
                    crit: [{"question": "Example question?", "answer": "uncertain"}]
                    for crit in criterion_order
                }
                dummy_block = render_checklist_block(dummy_checklist)
                rubric_items_block_dry = render_rubric_items(criterion_order, rubric_grouped)
                if bool(args.few_shot):
                    img = load_image(resolved_path)
                    s1_parts_dry, s1_schema_dry = build_stage1_prompt_with_self_rubric_few_shot(
                        rubric_items_block=rubric_items_block_dry,
                        rubric_items_by_id=rubric_items_by_id,
                        self_rubric_block=dummy_block,
                        include_rationale=bool(args.include_rationale_stage1),
                        include_weights=bool(args.rubrics_block_include_weights),
                        examples=few_shot_examples,
                        few_shot_short=bool(args.few_shot_short),
                        video_id=video_id, split=split, frame_id=frame_id, img=img,
                    )
                    print("\n===== STAGE 1 PROMPT PARTS (predefined rubrics + self-rubric context) =====\n", format_prompt_parts_for_print(s1_parts_dry))
                else:
                    s1_prompt_dry, s1_schema_dry = build_stage1_prompt_with_self_rubric(
                        rubric_items_block=rubric_items_block_dry,
                        self_rubric_block=dummy_block,
                        include_rationale=bool(args.include_rationale_stage1),
                        video_id=video_id, split=split, frame_id=frame_id,
                    )
                    print("\n===== STAGE 1 PROMPT (predefined rubrics + self-rubric context) =====\n", s1_prompt_dry)
                print("\n===== STAGE 1 OUTPUT SCHEMA =====\n", s1_schema_dry)
                print("\n===== FINAL PRED =====\nComputed via weighted average of stage 1 rubric labels (no stage 2 model call)")
                break

            stage0_raw = ""
            stage0_parsed: Dict[str, Any] = {}
            stage0_parse_ok = False
            stage0_parse_errors: List[str] = []
            checklist: Dict[str, List[Dict[str, Any]]] = {}
            _sr_cached = _sr_stage0_cache.get(_entry_key)
            if _sr_cached:
                # Reuse stage 0 from existing self_rubric results
                checklist = _sr_cached["checklist"]
                stage0_raw = _sr_cached["stage0_raw_response"]
                stage0_parsed = _sr_cached.get("stage0_parsed", {"checklist": checklist})
                stage0_parse_ok = _sr_cached.get("stage0_parse_ok", True)
                stage0_parse_errors = _sr_cached.get("stage0_parse_errors", [])
            else:
                try:
                    if stage0_parts is not None:
                        stage0_raw = model([stage0_parts], system_prompt=SYSTEM_PROMPT)[0]
                    else:
                        img = load_image(resolved_path)
                        stage0_raw = model([(stage0_prompt, img)], system_prompt=SYSTEM_PROMPT)[0]
                    stage0_parsed = extract_json_obj(stage0_raw)
                    stage0_parse_errors = validate_self_rubric_stage1(
                        stage0_parsed,
                        include_rationale=False,
                    )
                    stage0_parse_ok = len(stage0_parse_errors) == 0
                    if stage0_parse_ok:
                        checklist = stage0_parsed.get("checklist", {})
                except Exception as exc:
                    stage0_parse_errors = [f"exception:{type(exc).__name__}"]

            if not checklist:
                print(f"[WARN] Missing checklist from stage 0: video_id={video_id}, frame_id={frame_id}")
                out_row = {
                    "video_id": video_id,
                    "split": split,
                    "frame_id": frame_id,
                    "dataset": args.dataset,
                    "image_path": abs_posix(resolved_path),
                    "gt": gt,
                    "annotation_mode": args.annotation_mode,
                    "annotation_version": args.annotation_version,
                    "pipeline_mode": args.pipeline_mode,
                    "self_rubric_checklist": checklist,
                    "stage0_raw_response": stage0_raw,
                    "stage0_parse_ok": stage0_parse_ok,
                    "stage0_parse_errors": stage0_parse_errors,
                    "rubric_labels_stage1": {},
                    "stage1_raw_response": "",
                    "stage1_parse_ok": False,
                    "stage1_parse_errors": ["stage0_missing_checklist"],
                    "rubrics_block": [],
                    "raw_response": "",
                    "parsed": {},
                    "parse_ok": False,
                    "parse_errors": ["stage0_missing_checklist"],
                    "system_prompt": SYSTEM_PROMPT,
                    "error": "Missing checklist from stage 0; stages 1-2 skipped.",
                }
                if args.prev_frames_k > 0:
                    out_row["prev_frames_k"] = args.prev_frames_k
                    out_row["prev_frames_mode"] = args.prev_frames_mode
                    out_row["prev_frame_ids"] = prev_frame_ids
                append_jsonl(output_path, out_row)
                continue

            # ── Stage 1: Predict predefined rubric labels (with self-rubric context) ──
            self_rubric_block = render_checklist_block(checklist)
            rubric_items_block = render_rubric_items(criterion_order, rubric_grouped)

            stage1_prompt = ""
            stage1_schema = ""
            stage1_parts: Optional[Tuple[Any, ...]] = None
            if bool(args.few_shot):
                img = load_image(resolved_path)
                stage1_parts, stage1_schema = build_stage1_prompt_with_self_rubric_few_shot(
                    rubric_items_block=rubric_items_block,
                    rubric_items_by_id=rubric_items_by_id,
                    self_rubric_block=self_rubric_block,
                    include_rationale=bool(args.include_rationale_stage1),
                    include_weights=bool(args.rubrics_block_include_weights),
                    examples=few_shot_examples,
                    few_shot_short=bool(args.few_shot_short),
                    video_id=video_id,
                    split=split,
                    frame_id=frame_id,
                    img=img,
                )
            else:
                stage1_prompt, stage1_schema = build_stage1_prompt_with_self_rubric(
                    rubric_items_block=rubric_items_block,
                    self_rubric_block=self_rubric_block,
                    include_rationale=bool(args.include_rationale_stage1),
                    video_id=video_id,
                    split=split,
                    frame_id=frame_id,
                )

            # Inject previous frames into stage 1
            if prev_frame_images:
                if stage1_parts is not None:
                    stage1_parts = inject_previous_frames(stage1_parts, prev_frame_images)
                else:
                    img = load_image(resolved_path)
                    stage1_parts = inject_previous_frames((stage1_prompt, img), prev_frame_images)

            rubric_labels_stage1: Dict[str, str] = {}
            stage1_raw = ""
            stage1_parsed: Dict[str, Any] = {}
            stage1_parse_errors: List[str] = []
            stage1_parse_ok = False
            try:
                if stage1_parts is not None:
                    stage1_raw = model([stage1_parts], system_prompt=SYSTEM_PROMPT)[0]
                else:
                    img = load_image(resolved_path)
                    stage1_raw = model([(stage1_prompt, img)], system_prompt=SYSTEM_PROMPT)[0]
                stage1_parsed = extract_json_obj(stage1_raw)
                stage1_parse_errors = validate_stage1_with_rationale(
                    stage1_parsed,
                    include_rationale=bool(args.include_rationale_stage1),
                )
                stage1_parse_ok = len(stage1_parse_errors) == 0
                if stage1_parse_ok:
                    rubric_labels_stage1 = stage1_parsed.get("rubric_labels", {})
            except Exception as exc:
                stage1_parse_errors = [f"exception:{type(exc).__name__}"]

            if not rubric_labels_stage1:
                print(f"[WARN] Missing rubric labels from stage 1: video_id={video_id}, frame_id={frame_id}")
                out_row = {
                    "video_id": video_id,
                    "split": split,
                    "frame_id": frame_id,
                    "dataset": args.dataset,
                    "image_path": abs_posix(resolved_path),
                    "gt": gt,
                    "annotation_mode": args.annotation_mode,
                    "annotation_version": args.annotation_version,
                    "pipeline_mode": args.pipeline_mode,
                    "self_rubric_checklist": checklist,
                    "stage0_raw_response": stage0_raw,
                    "stage0_parse_ok": stage0_parse_ok,
                    "stage0_parse_errors": stage0_parse_errors,
                    "rubric_labels_stage1": rubric_labels_stage1,
                    "stage1_raw_response": stage1_raw,
                    "stage1_parsed": stage1_parsed,
                    "stage1_parse_ok": stage1_parse_ok,
                    "stage1_parse_errors": stage1_parse_errors,
                    "rubrics_block": [],
                    "raw_response": "",
                    "parsed": {},
                    "parse_ok": False,
                    "parse_errors": ["stage1_missing_rubric_labels"],
                    "system_prompt": SYSTEM_PROMPT,
                    "error": "Missing rubric labels from stage 1; stage 2 skipped.",
                }
                if args.prev_frames_k > 0:
                    out_row["prev_frames_k"] = args.prev_frames_k
                    out_row["prev_frames_mode"] = args.prev_frames_mode
                    out_row["prev_frame_ids"] = prev_frame_ids
                append_jsonl(output_path, out_row)
                continue

            # ── Final pred: weighted average from stage 1 rubric labels (no stage 2 model call) ──
            rubrics_block = render_rubrics_block(
                rubric_items_by_id,
                rubric_labels_stage1,
                bool(args.rubrics_block_include_weights),
            )
            pred = compute_weighted_score(
                rubric_items_by_id,
                rubric_labels_stage1,
                include_types=args.weighted_score_include_types,
            )
            parse_ok = stage1_parse_ok
            parse_errors = stage1_parse_errors

            out_row = {
                "video_id": video_id,
                "split": split,
                "frame_id": frame_id,
                "dataset": args.dataset,
                "image_path": abs_posix(resolved_path),
                "gt": gt,
                "pred": pred,
                "annotation_mode": args.annotation_mode,
                "annotation_version": args.annotation_version,
                "pipeline_mode": args.pipeline_mode,
                # Stage 0 (self-rubric)
                "self_rubric_checklist": checklist,
                "stage0_raw_response": stage0_raw,
                "stage0_parse_ok": stage0_parse_ok,
                "stage0_parse_errors": stage0_parse_errors,
                # Stage 1 (predefined rubrics with self-rubric context)
                "rubric_labels_stage1": rubric_labels_stage1,
                "stage1_raw_response": stage1_raw,
                "stage1_parsed": stage1_parsed,
                "stage1_parse_ok": stage1_parse_ok,
                "stage1_parse_errors": stage1_parse_errors,
                # Final prediction (weighted average of stage 1)
                "rubrics_block": rubrics_block,
                "parse_ok": parse_ok,
                "parse_errors": parse_errors,
                "system_prompt": SYSTEM_PROMPT,
            }
            if args.prev_frames_k > 0:
                out_row["prev_frames_k"] = args.prev_frames_k
                out_row["prev_frames_mode"] = args.prev_frames_mode
                out_row["prev_frame_ids"] = prev_frame_ids

            append_jsonl(output_path, out_row)
            continue

        if args.pipeline_mode == "direct":
            # Direct pipeline: single call without rubric generation or labels.
            criteria = normalize_criteria(args.criteria)
            per_criterion_mode = args.per_criterion and len(criteria) > 1
            _direct_rationale = bool(args.include_rationale_direct) or bool(getattr(args, "include_rationale_direct_first", False))
            _direct_rationale_first = bool(getattr(args, "include_rationale_direct_first", False))

            if per_criterion_mode:
                # --- Per-criterion mode: one model call per criterion ---
                if args.dryrun:
                    print("===== SYSTEM PROMPT =====\n", DIRECT_SYSTEM_PROMPT)
                    for crit in criteria:
                        crit_criteria = [crit]
                        if bool(args.few_shot):
                            img = load_image(resolved_path)
                            crit_parts, crit_schema = build_direct_prompt_few_shot(
                                criteria=crit_criteria,
                                include_criterion_definitions=bool(args.include_criterion_definitions),
                                include_rationale=_direct_rationale,
                                rationale_first=_direct_rationale_first,
                                examples=few_shot_examples,
                                video_id=video_id, split=split, frame_id=frame_id, img=img,
                            )
                            if prev_frame_images:
                                crit_parts = inject_previous_frames(crit_parts, prev_frame_images)
                            print(f"\n===== PROMPT PARTS ({crit.upper()}) =====\n", format_prompt_parts_for_print(crit_parts))
                        else:
                            crit_prompt, crit_schema = build_direct_prompt(
                                criteria=crit_criteria,
                                include_criterion_definitions=bool(args.include_criterion_definitions),
                                include_rationale=_direct_rationale,
                                rationale_first=_direct_rationale_first,
                                video_id=video_id, split=split, frame_id=frame_id,
                            )
                            if prev_frame_images:
                                img = load_image(resolved_path)
                                crit_parts_dry = inject_previous_frames((crit_prompt, img), prev_frame_images)
                                print(f"\n===== PROMPT PARTS ({crit.upper()}) =====\n", format_prompt_parts_for_print(crit_parts_dry))
                            else:
                                print(f"\n===== PROMPT ({crit.upper()}) =====\n", crit_prompt)
                        print(f"\n===== OUTPUT SCHEMA ({crit.upper()}) =====\n", crit_schema)
                    break

                per_crit_results: List[Dict[str, Any]] = []
                for crit in criteria:
                    crit_criteria = [crit]
                    crit_prompt_parts: Optional[Tuple[Any, ...]] = None
                    crit_prompt = ""
                    if bool(args.few_shot):
                        img = load_image(resolved_path)
                        crit_prompt_parts, _ = build_direct_prompt_few_shot(
                            criteria=crit_criteria,
                            include_criterion_definitions=bool(args.include_criterion_definitions),
                            include_rationale=_direct_rationale,
                            rationale_first=_direct_rationale_first,
                            examples=few_shot_examples,
                            video_id=video_id, split=split, frame_id=frame_id, img=img,
                        )
                    else:
                        crit_prompt, _ = build_direct_prompt(
                            criteria=crit_criteria,
                            include_criterion_definitions=bool(args.include_criterion_definitions),
                            include_rationale=_direct_rationale,
                            rationale_first=_direct_rationale_first,
                            video_id=video_id, split=split, frame_id=frame_id,
                        )
                    # Inject previous frames
                    if prev_frame_images:
                        if crit_prompt_parts is not None:
                            crit_prompt_parts = inject_previous_frames(crit_prompt_parts, prev_frame_images)
                        else:
                            img = load_image(resolved_path)
                            crit_prompt_parts = inject_previous_frames((crit_prompt, img), prev_frame_images)

                    crit_response = ""
                    crit_parsed: Dict[str, Any] = {}
                    crit_parse_ok = False
                    crit_parse_errors: List[str] = []
                    try:
                        if crit_prompt_parts is not None:
                            crit_response = model([crit_prompt_parts], system_prompt=DIRECT_SYSTEM_PROMPT)[0]
                        else:
                            img = load_image(resolved_path)
                            crit_response = model([(crit_prompt, img)], system_prompt=DIRECT_SYSTEM_PROMPT)[0]
                        crit_parsed = extract_json_obj(crit_response)
                        crit_parse_errors = validate_direct_output(
                            crit_parsed, criteria=crit_criteria,
                            include_rationale=_direct_rationale,
                        )
                        crit_parse_ok = len(crit_parse_errors) == 0
                    except Exception as exc:
                        crit_parse_errors = [f"exception:{type(exc).__name__}"]

                    crit_pred = crit_parsed.get("pred", {}) if isinstance(crit_parsed, dict) else {}
                    crit_rationale = crit_parsed.get("rationale", {}) if isinstance(crit_parsed, dict) else {}
                    per_crit_results.append({
                        "criterion": crit,
                        "pred": crit_pred,
                        "rationale": crit_rationale,
                        "raw_response": crit_response,
                        "parsed": crit_parsed,
                        "parse_ok": crit_parse_ok,
                        "parse_errors": crit_parse_errors,
                    })

                merged = merge_per_criterion_results(per_crit_results)
                out_row = {
                    "video_id": video_id,
                    "split": split,
                    "frame_id": frame_id,
                    "dataset": args.dataset,
                    "image_path": abs_posix(resolved_path),
                    "gt": gt,
                    "annotation_paths": [abs_posix(p) for p in annotation_paths],
                    "annotation_version": args.annotation_version,
                    "pipeline_mode": args.pipeline_mode,
                    "criteria": criteria,
                    "per_criterion": True,
                    "prompt": "per_criterion_mode",
                    "system_prompt": DIRECT_SYSTEM_PROMPT,
                    "parse_ok": merged["parse_ok"],
                    "parse_errors": merged["parse_errors"],
                    "raw_response": "per_criterion_mode",
                    "parsed": {"pred": merged["pred"]},
                    "pred": merged["pred"],
                    "per_criterion_metadata": merged["per_criterion_metadata"],
                }
                if merged["rationale"]:
                    out_row["rationale"] = merged["rationale"]
                if args.prev_frames_k > 0:
                    out_row["prev_frames_k"] = args.prev_frames_k
                    out_row["prev_frames_mode"] = args.prev_frames_mode
                    out_row["prev_frame_ids"] = prev_frame_ids

                append_jsonl(output_path, out_row)
                continue

            # --- Joint mode (default): single call for all criteria ---
            prompt_parts: Optional[Tuple[Any, ...]] = None
            prompt = ""
            schema_block = ""
            if bool(args.few_shot):
                img = load_image(resolved_path)
                prompt_parts, schema_block = build_direct_prompt_few_shot(
                    criteria=criteria,
                    include_criterion_definitions=bool(args.include_criterion_definitions),
                    include_rationale=_direct_rationale,
                    rationale_first=_direct_rationale_first,
                    examples=few_shot_examples,
                    video_id=video_id,
                    split=split,
                    frame_id=frame_id,
                    img=img,
                )
            else:
                prompt, schema_block = build_direct_prompt(
                    criteria=criteria,
                    include_criterion_definitions=bool(args.include_criterion_definitions),
                    include_rationale=_direct_rationale,
                    rationale_first=_direct_rationale_first,
                    video_id=video_id,
                    split=split,
                    frame_id=frame_id,
                )

            # Inject previous frames
            if prev_frame_images:
                if prompt_parts is not None:
                    prompt_parts = inject_previous_frames(prompt_parts, prev_frame_images)
                else:
                    img = load_image(resolved_path)
                    prompt_parts = inject_previous_frames((prompt, img), prev_frame_images)

            if args.dryrun:
                print("===== SYSTEM PROMPT =====\n", DIRECT_SYSTEM_PROMPT)
                if prompt_parts is not None:
                    print("\n===== PROMPT PARTS =====\n", format_prompt_parts_for_print(prompt_parts))
                else:
                    print("\n===== PROMPT =====\n", prompt)
                print("\n===== OUTPUT SCHEMA =====\n", schema_block)
                break

            if prompt_parts is None:
                img = load_image(resolved_path)
            response_text = ""
            parsed: Dict[str, Any] = {}
            parse_ok = False
            parse_errors: List[str] = []
            try:
                if prompt_parts is not None:
                    response_text = model([prompt_parts], system_prompt=DIRECT_SYSTEM_PROMPT)[0]
                else:
                    response_text = model([(prompt, img)], system_prompt=DIRECT_SYSTEM_PROMPT)[0]
                parsed = extract_json_obj(response_text)
                parse_errors = validate_direct_output(
                    parsed,
                    criteria=criteria,
                    include_rationale=_direct_rationale,
                )
                parse_ok = len(parse_errors) == 0
            except Exception as exc:
                parse_errors = [f"exception:{type(exc).__name__}"]

            out_row = {
                "video_id": video_id,
                "split": split,
                "frame_id": frame_id,
                "dataset": args.dataset,
                "image_path": abs_posix(resolved_path),
                "gt": gt,
                "annotation_paths": [abs_posix(p) for p in annotation_paths],
                "annotation_version": args.annotation_version,
                "pipeline_mode": args.pipeline_mode,
                "criteria": criteria,
                "prompt": prompt,
                "system_prompt": DIRECT_SYSTEM_PROMPT,
                "parse_ok": parse_ok,
                "parse_errors": parse_errors,
                "raw_response": response_text,
                "parsed": parsed,
            }
            if isinstance(parsed, dict) and "pred" in parsed:
                out_row["pred"] = parsed.get("pred")
            if isinstance(parsed, dict) and "rationale" in parsed:
                out_row["rationale"] = parsed.get("rationale")
            if args.prev_frames_k > 0:
                out_row["prev_frames_k"] = args.prev_frames_k
                out_row["prev_frames_mode"] = args.prev_frames_mode
                out_row["prev_frame_ids"] = prev_frame_ids

            append_jsonl(output_path, out_row)
            continue

        oracle_rubrics = extract_oracle_rubrics(row)
        if args.pipeline_mode == "one_call":
            if args.annotation_mode in ("oracle_only", "oracle_plus_predicted") and not oracle_rubrics:
                print(f"[WARN] Skip frame without oracle rubrics: video_id={video_id}, frame_id={frame_id}")
                continue

            prompt_parts: Optional[Tuple[Any, ...]] = None
            prompt = ""
            schema_block = ""
            if bool(args.few_shot):
                img = load_image(resolved_path)
                prompt_parts, schema_block = build_prompt_few_shot(
                    criterion_order=criterion_order,
                    rubric_grouped=rubric_grouped,
                    rubric_items_by_id=rubric_items_by_id,
                    annotation_mode=args.annotation_mode,
                    include_rationale=bool(args.include_rationale_stage2),
                    include_criterion_definitions=bool(args.include_criterion_definitions),
                    examples=few_shot_examples,
                    include_weights=bool(args.rubrics_block_include_weights),
                    few_shot_short=bool(args.few_shot_short),
                    video_id=video_id,
                    split=split,
                    frame_id=frame_id,
                    gt=gt,
                    oracle_rubrics=oracle_rubrics,
                    img=img,
                )
            else:
                prompt, schema_block = build_prompt(
                    criterion_order=criterion_order,
                    rubric_grouped=rubric_grouped,
                    annotation_mode=args.annotation_mode,
                    include_rationale=bool(args.include_rationale_stage2),
                    include_criterion_definitions=bool(args.include_criterion_definitions),
                    video_id=video_id,
                    split=split,
                    frame_id=frame_id,
                    gt=gt,
                    oracle_rubrics=oracle_rubrics,
                )

            # Inject previous frames
            if prev_frame_images:
                if prompt_parts is not None:
                    prompt_parts = inject_previous_frames(prompt_parts, prev_frame_images)
                else:
                    img = load_image(resolved_path)
                    prompt_parts = inject_previous_frames((prompt, img), prev_frame_images)

            if args.dryrun:
                print("===== SYSTEM PROMPT =====\n", SYSTEM_PROMPT)
                if prompt_parts is not None:
                    print("\n===== PROMPT PARTS =====\n", format_prompt_parts_for_print(prompt_parts))
                else:
                    print("\n===== PROMPT =====\n", prompt)
                print("\n===== OUTPUT SCHEMA =====\n", schema_block)
                example_labels = oracle_rubrics or {}
                if args.annotation_mode != "oracle_only":
                    example_labels = {}
                    for item_id in rubric_items_by_id.keys():
                        example_labels[item_id] = "uncertain"
                rubrics_block = render_rubrics_block(
                    rubric_items_by_id,
                    example_labels,
                    bool(args.rubrics_block_include_weights),
                )
                example_block = "\n".join(f"- {line}" for line in rubrics_block)
                print(f"\n===== RUBRICS_BLOCK EXAMPLE =====\n{example_block}")
                break

            response_text = ""
            parsed: Dict[str, Any] = {}
            parse_ok = False
            parse_errors: List[str] = []
            try:
                if prompt_parts is not None:
                    response_text = model([prompt_parts], system_prompt=SYSTEM_PROMPT)[0]
                else:
                    img = load_image(resolved_path)
                    response_text = model([(prompt, img)], system_prompt=SYSTEM_PROMPT)[0]
                parsed = extract_json_obj(response_text)
                parse_errors = validate_output(
                    parsed,
                    expect_rubric_labels=args.annotation_mode in ("predicted_only", "oracle_plus_predicted"),
                    include_rationale=bool(args.include_rationale_stage2),
                )
                parse_ok = len(parse_errors) == 0
            except Exception as exc:
                parse_errors = [f"exception:{type(exc).__name__}"]

            rubric_labels_stage1 = {}
            rubric_labels_source = "predicted"
            if args.annotation_mode == "oracle_only" and oracle_rubrics:
                rubric_labels_stage1 = oracle_rubrics
                rubric_labels_source = "oracle"
            else:
                rubric_labels_stage1 = parsed.get("rubric_labels") if isinstance(parsed, dict) else {}
                rubric_labels_source = "predicted"

            rubrics_block = render_rubrics_block(
                rubric_items_by_id,
                rubric_labels_stage1 or {},
                bool(args.rubrics_block_include_weights),
            )
            weighted_score = None
            if args.compute_weighted_rubric_score:
                weighted_score = compute_weighted_score(
                    rubric_items_by_id,
                    rubric_labels_stage1 or {},
                    include_types=args.weighted_score_include_types,
                )

            out_row = {
                "video_id": video_id,
                "split": split,
                "frame_id": frame_id,
                "dataset": args.dataset,
                "image_path": abs_posix(resolved_path),
                "gt": gt,
                "rubric_path": abs_posix(rubric_path),
                "annotation_paths": [abs_posix(p) for p in annotation_paths],
                "annotation_mode": args.annotation_mode,
                "annotation_version": args.annotation_version,
                "pipeline_mode": args.pipeline_mode,
                "rubric_labels_source": rubric_labels_source,
                "rubric_labels_stage1": rubric_labels_stage1,
                "rubrics_block": rubrics_block,
                "prompt": prompt,
                "system_prompt": SYSTEM_PROMPT,
                "parse_ok": parse_ok,
                "parse_errors": parse_errors,
                "raw_response": response_text,
                "parsed": parsed,
            }
            if oracle_rubrics:
                out_row["oracle_rubrics"] = oracle_rubrics
            if isinstance(parsed, dict) and "pred" in parsed:
                out_row["pred"] = parsed.get("pred")
            if isinstance(parsed, dict) and "rationale" in parsed:
                out_row["rationale"] = parsed.get("rationale")
            if weighted_score is not None:
                out_row["weighted_score"] = weighted_score
            if args.prev_frames_k > 0:
                out_row["prev_frames_k"] = args.prev_frames_k
                out_row["prev_frames_mode"] = args.prev_frames_mode
                out_row["prev_frame_ids"] = prev_frame_ids

            append_jsonl(output_path, out_row)
            continue

        # two_call pipeline
        if args.annotation_mode == "oracle_only" and not oracle_rubrics:
            print(f"[WARN] Skip frame without oracle rubrics: video_id={video_id}, frame_id={frame_id}")
            continue

        rubric_items_block = render_rubric_items(criterion_order, rubric_grouped)
        stage1_prompt = ""
        stage1_schema = ""
        stage1_parts: Optional[Tuple[Any, ...]] = None
        if args.annotation_mode != "oracle_only":
            if bool(args.few_shot):
                img = load_image(resolved_path)
                stage1_parts, stage1_schema = build_stage1_prompt_few_shot(
                    rubric_items_block=rubric_items_block,
                    rubric_items_by_id=rubric_items_by_id,
                    include_rationale=bool(args.include_rationale_stage1),
                    include_weights=bool(args.rubrics_block_include_weights),
                    examples=few_shot_examples,
                    few_shot_short=bool(args.few_shot_short),
                    video_id=video_id,
                    split=split,
                    frame_id=frame_id,
                    img=img,
                )
            else:
                stage1_prompt, stage1_schema = build_stage1_prompt(
                    rubric_items_block=rubric_items_block,
                    include_rationale=bool(args.include_rationale_stage1),
                    video_id=video_id,
                    split=split,
                    frame_id=frame_id,
                )

            # Inject previous frames into stage 1
            if prev_frame_images:
                if stage1_parts is not None:
                    stage1_parts = inject_previous_frames(stage1_parts, prev_frame_images)
                else:
                    img = load_image(resolved_path)
                    stage1_parts = inject_previous_frames((stage1_prompt, img), prev_frame_images)

        if args.dryrun:
            print("===== SYSTEM PROMPT =====\n", SYSTEM_PROMPT)
            rubric_labels_stage1 = oracle_rubrics or {}
            if args.annotation_mode != "oracle_only":
                if stage1_parts is not None:
                    print("\n===== STAGE 1 PROMPT PARTS =====\n", format_prompt_parts_for_print(stage1_parts))
                else:
                    print("\n===== STAGE 1 PROMPT =====\n", stage1_prompt)
                rubric_labels_stage1 = {}
                for criterion in criterion_order:
                    by_type = rubric_grouped.get(criterion, {})
                    items: List[Dict[str, Any]] = []
                    for label_type in ("precondition", "evidence"):
                        items.extend(by_type.get(label_type, []))
                    for item in items:
                        item_id = item.get("id", "")
                        if item_id:
                            rubric_labels_stage1[item_id] = "uncertain"
            if args.annotation_mode == "oracle_only" and not rubric_labels_stage1:
                print(f"[WARN] Skip frame without oracle rubrics: video_id={video_id}, frame_id={frame_id}")
                break
            rubrics_block = render_rubrics_block(
                rubric_items_by_id,
                rubric_labels_stage1,
                bool(args.rubrics_block_include_weights),
            )
            example_block = "\n".join(f"- {line}" for line in rubrics_block)
            print(f"\n===== RUBRICS_BLOCK EXAMPLE =====\n{example_block}")
            stage2_prompt, stage2_schema = build_stage2_prompt(
                criterion_order=criterion_order,
                rubric_items_block=rubric_items_block,
                rubric_labels=rubric_labels_stage1,
                rubrics_block=rubrics_block,
                include_rationale=bool(args.include_rationale_stage2),
                include_criterion_definitions=bool(args.include_criterion_definitions),
                stage1_short=bool(args.stage1_short),
                video_id=video_id,
                split=split,
                frame_id=frame_id,
                gt=gt,
            )
            if bool(args.few_shot):
                img = load_image(resolved_path)
                stage2_parts, _ = build_stage2_prompt_few_shot(
                    criterion_order=criterion_order,
                    rubric_items_block=rubric_items_block,
                    rubric_items_by_id=rubric_items_by_id,
                    rubric_labels=rubric_labels_stage1,
                    rubrics_block=rubrics_block,
                    include_rationale=bool(args.include_rationale_stage2),
                    include_criterion_definitions=bool(args.include_criterion_definitions),
                    include_weights=bool(args.rubrics_block_include_weights),
                    examples=few_shot_examples,
                    few_shot_short=bool(args.few_shot_short),
                    stage1_short=bool(args.stage1_short),
                    video_id=video_id,
                    split=split,
                    frame_id=frame_id,
                    gt=gt,
                    img=img,
                )
                print("\n===== STAGE 2 PROMPT PARTS =====\n", format_prompt_parts_for_print(stage2_parts))
            else:
                print("\n===== STAGE 2 PROMPT =====\n", stage2_prompt)
            print("\n===== STAGE 2 OUTPUT SCHEMA =====\n", stage2_schema)
            break

        rubric_labels_stage1: Dict[str, str] = {}
        stage1_raw = ""
        stage1_parsed: Dict[str, Any] = {}
        stage1_parse_errors: List[str] = []
        stage1_parse_ok = False
        rubric_labels_source = "predicted"

        if args.annotation_mode == "oracle_only":
            rubric_labels_stage1 = oracle_rubrics or {}
            rubric_labels_source = "oracle"
            stage1_parsed = {"rubric_labels": rubric_labels_stage1}
            stage1_parse_ok = bool(rubric_labels_stage1)
        else:
            try:
                if stage1_parts is not None:
                    stage1_raw = model([stage1_parts], system_prompt=SYSTEM_PROMPT)[0]
                else:
                    img = load_image(resolved_path)
                    stage1_raw = model([(stage1_prompt, img)], system_prompt=SYSTEM_PROMPT)[0]
                stage1_parsed = extract_json_obj(stage1_raw)
                stage1_parse_errors = validate_stage1_with_rationale(
                    stage1_parsed,
                    include_rationale=bool(args.include_rationale_stage1),
                )
                stage1_parse_ok = len(stage1_parse_errors) == 0
                if stage1_parse_ok:
                    rubric_labels_stage1 = stage1_parsed.get("rubric_labels", {})
            except Exception as exc:
                stage1_parse_errors = [f"exception:{type(exc).__name__}"]

        if not rubric_labels_stage1:
            print(f"[WARN] Missing rubric labels for stage 2: video_id={video_id}, frame_id={frame_id}")
            # Save stage 1 results with error — skip stage 2
            out_row = {
                "video_id": video_id,
                "split": split,
                "frame_id": frame_id,
                "dataset": args.dataset,
                "image_path": abs_posix(resolved_path),
                "gt": gt,
                "rubric_path": abs_posix(rubric_path),
                "annotation_paths": [abs_posix(p) for p in annotation_paths],
                "annotation_mode": args.annotation_mode,
                "annotation_version": args.annotation_version,
                "pipeline_mode": args.pipeline_mode,
                "rubric_labels_source": rubric_labels_source,
                "rubric_labels_stage1": rubric_labels_stage1,
                "stage1_prompt": stage1_prompt if stage1_parts is None else str(stage1_parts),
                "stage1_schema": stage1_schema,
                "stage1_raw_response": stage1_raw,
                "stage1_parsed": stage1_parsed,
                "stage1_parse_ok": stage1_parse_ok,
                "stage1_parse_errors": stage1_parse_errors,
                "parse_ok": False,
                "parse_errors": ["stage1_missing_rubric_labels"],
                "raw_response": "",
                "parsed": {},
                "error": "Missing rubric labels from stage 1; stage 2 skipped.",
            }
            if oracle_rubrics:
                out_row["oracle_rubrics"] = oracle_rubrics
            if isinstance(stage1_parsed, dict) and "rationale" in stage1_parsed:
                out_row["stage1_rationale"] = stage1_parsed.get("rationale")
            if args.prev_frames_k > 0:
                out_row["prev_frames_k"] = args.prev_frames_k
                out_row["prev_frames_mode"] = args.prev_frames_mode
                out_row["prev_frame_ids"] = prev_frame_ids
            append_jsonl(output_path, out_row)
            continue

        rubrics_block = render_rubrics_block(
            rubric_items_by_id,
            rubric_labels_stage1,
            bool(args.rubrics_block_include_weights),
        )

        stage2_parts: Optional[Tuple[Any, ...]] = None
        stage2_prompt, stage2_schema = build_stage2_prompt(
            criterion_order=criterion_order,
            rubric_items_block=rubric_items_block,
            rubric_labels=rubric_labels_stage1,
            rubrics_block=rubrics_block,
            include_rationale=bool(args.include_rationale_stage2),
            include_criterion_definitions=bool(args.include_criterion_definitions),
            stage1_short=bool(args.stage1_short),
            video_id=video_id,
            split=split,
            frame_id=frame_id,
            gt=gt,
        )
        if bool(args.few_shot):
            img = load_image(resolved_path)
            stage2_parts, _ = build_stage2_prompt_few_shot(
                criterion_order=criterion_order,
                rubric_items_block=rubric_items_block,
                rubric_items_by_id=rubric_items_by_id,
                rubric_labels=rubric_labels_stage1,
                rubrics_block=rubrics_block,
                include_rationale=bool(args.include_rationale_stage2),
                include_criterion_definitions=bool(args.include_criterion_definitions),
                include_weights=bool(args.rubrics_block_include_weights),
                examples=few_shot_examples,
                few_shot_short=bool(args.few_shot_short),
                stage1_short=bool(args.stage1_short),
                video_id=video_id,
                split=split,
                frame_id=frame_id,
                gt=gt,
                img=img,
            )

        # Inject previous frames into stage 2
        if prev_frame_images:
            if stage2_parts is not None:
                stage2_parts = inject_previous_frames(stage2_parts, prev_frame_images)
            else:
                img = load_image(resolved_path)
                stage2_parts = inject_previous_frames((stage2_prompt, img), prev_frame_images)

        response_text = ""
        parsed: Dict[str, Any] = {}
        parse_ok = False
        parse_errors: List[str] = []
        try:
            if stage2_parts is not None:
                response_text = model([stage2_parts], system_prompt=SYSTEM_PROMPT)[0]
            else:
                img = load_image(resolved_path)
                response_text = model([(stage2_prompt, img)], system_prompt=SYSTEM_PROMPT)[0]
            parsed = extract_json_obj(response_text)
            parse_errors = validate_stage2(
                parsed,
                include_rationale=bool(args.include_rationale_stage2),
                expect_rubrics_block=not bool(args.stage1_short),
            )
            parse_ok = len(parse_errors) == 0
        except Exception as exc:
            parse_errors = [f"exception:{type(exc).__name__}"]

        weighted_score = None
        if args.compute_weighted_rubric_score:
            weighted_score = compute_weighted_score(
                rubric_items_by_id,
                rubric_labels_stage1,
                include_types=args.weighted_score_include_types,
            )

        out_row = {
            "video_id": video_id,
            "split": split,
            "frame_id": frame_id,
            "dataset": args.dataset,
            "image_path": abs_posix(resolved_path),
            "gt": gt,
            "rubric_path": abs_posix(rubric_path),
            "annotation_paths": [abs_posix(p) for p in annotation_paths],
            "annotation_mode": args.annotation_mode,
            "annotation_version": args.annotation_version,
            "pipeline_mode": args.pipeline_mode,
            "rubric_labels_source": rubric_labels_source,
            "rubric_labels_stage1": rubric_labels_stage1,
            "rubrics_block": rubrics_block,
            "rubrics_block_stage1": rubrics_block,
            "rubrics_block_stage2": rubrics_block,
            "stage1_prompt": stage1_prompt if stage1_parts is None else str(stage1_parts),
            "stage1_schema": stage1_schema,
            "stage1_raw_response": stage1_raw,
            "stage1_parsed": stage1_parsed,
            "stage1_parse_ok": stage1_parse_ok,
            "stage1_parse_errors": stage1_parse_errors,
            "prompt": stage2_prompt,
            "system_prompt": SYSTEM_PROMPT,
            "parse_ok": parse_ok,
            "parse_errors": parse_errors,
            "raw_response": response_text,
            "parsed": parsed,
        }
        if oracle_rubrics:
            out_row["oracle_rubrics"] = oracle_rubrics
        if isinstance(stage1_parsed, dict) and "rationale" in stage1_parsed:
            out_row["stage1_rationale"] = stage1_parsed.get("rationale")
        if isinstance(parsed, dict) and "pred" in parsed:
            out_row["pred"] = parsed.get("pred")
        if isinstance(parsed, dict) and "rationale" in parsed:
            out_row["rationale"] = parsed.get("rationale")
        if weighted_score is not None:
            out_row["weighted_score"] = weighted_score
        if args.prev_frames_k > 0:
            out_row["prev_frames_k"] = args.prev_frames_k
            out_row["prev_frames_mode"] = args.prev_frames_mode
            out_row["prev_frame_ids"] = prev_frame_ids

        append_jsonl(output_path, out_row)

if __name__ == "__main__":
    main()
