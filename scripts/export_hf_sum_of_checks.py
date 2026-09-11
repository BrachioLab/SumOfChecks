#!/usr/bin/env python3
"""Export, validate, and optionally publish the human-labeled rubric subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_XET_CACHE", str(ROOT / ".cache/huggingface/xet"))

from datasets import Dataset, Features, Image, List, Value, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image as PILImage

REPO_ID = "BrachioLab/sum-of-checks"
COLLECTION = "BrachioLab/laparoscopic-cholecystectomy-6a13cd06fa87aa9d2603e2ac"
RUBRIC = ROOT / "rubrics/cvs_rubrics_v3.json"
FEW_SHOT_DIR = Path("few_shot_examples/endoscapes/rubrics/filtered")
FEW_SHOT_MANIFEST = FEW_SHOT_DIR / "selected_examples_v3.jsonl"
SAGES_FEW_SHOT_MANIFEST = Path("few_shot_examples/cvs_challenge_sages_v1/rubrics/filtered/selected_examples_v5.jsonl")
SELECTION_MANIFESTS = {"endoscapes": FEW_SHOT_MANIFEST, "sages": SAGES_FEW_SHOT_MANIFEST}
SOURCES = {
    "endoscapes": ("validation", ROOT / "annotations/rubric_labels/endoscapes_val__rubrics_v1__seed13__batch0__filtered_no_all_uncertain_fix_error.jsonl"),
    "sages": ("train", ROOT / "annotations/rubric_labels/sages__rubrics_v3__seed13__batch0__20260219_203800.jsonl"),
}
CRITERIA = ("c1", "c2", "c3")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def selected_examples(config: str) -> list[dict]:
    rows = read_rows(ROOT / SELECTION_MANIFESTS[config])
    if config == "sages":
        random.Random(13).shuffle(rows)
    return rows[:4]


def selection_key(example: dict) -> tuple[str, int]:
    frame = example.get("frame_id")
    if frame is None:
        frame = int(Path(example["file_name"]).stem.rsplit("_", 1)[1])
    return str(example["video_id"]), int(frame)


def partition_rows(rows: dict) -> dict:
    result = {}
    for config, records in rows.items():
        by_key = {(r["video_id"], r["frame_id"]): r for r in records}
        selected = [by_key[selection_key(r)] for r in selected_examples(config)]
        dev = [r for r in records if not r["is_few_shot"]]
        if len(selected) != 4 or len(dev) + len(selected) != len(records):
            raise ValueError(f"Invalid split sizes: {config}")
        for field in ("example_id", "image_sha256"):
            if {r[field] for r in selected} & {r[field] for r in dev}:
                raise ValueError(f"Few-shot/dev overlap in {field}: {config}")
        result[config] = {"few_shot": selected, "dev": dev}
    return result


def features(item_ids: list[str]) -> Features:
    return Features({
        "example_id": Value("string"), "image": Image(), "image_sha256": Value("string"),
        "image_filename": Value("string"), "width": Value("int32"), "height": Value("int32"),
        "source_dataset": Value("string"), "source_split": Value("string"),
        "source_subset": Value("string"), "video_id": Value("string"), "frame_id": Value("int64"),
        "cvs_labels": {c: Value("float64") for c in CRITERIA},
        "cvs_binary_labels": {c: Value("int8") for c in CRITERIA},
        "cvs_rater_labels": {c: List(Value("int8")) for c in CRITERIA},
        "original_cvs_annotation_json": Value("string"),
        "annotation_cvs_labels": {c: Value("float64") for c in CRITERIA},
        "rubric_labels": {key: Value("string") for key in item_ids},
        "rubric_version": Value("string"), "annotation_rubric_version": Value("string"),
        "source_annotation_rubric_version": Value("string"),
        "annotation_source": Value("string"), "annotation_row": Value("int32"),
        "annotation_notes": Value("string"), "annotation_timestamp": Value("string"),
        "sample_group_json": Value("string"), "is_paper_few_shot": Value("bool"),
        "is_few_shot": Value("bool"), "few_shot_order": Value("int8"),
    })


def build_rows(endo_root: Path, sages_root: Path, frames_root: Path) -> dict[str, list[dict]]:
    rubric = json.loads(RUBRIC.read_text())
    item_ids = [i["id"] for block in rubric["criteria"].values() for i in block["items"]]
    images = json.loads((endo_root / "val/annotation_ds_coco.json").read_text())["images"]
    endo_annotations = {r["file_name"]: r for r in images}
    selections = {config: {selection_key(r): i for i, r in enumerate(selected_examples(config))}
                  for config in SOURCES}
    sages_annotations = {}
    result = {}
    for config, (_, source) in SOURCES.items():
        rows = []
        seen = set()
        for index, annotation in enumerate(read_rows(source), start=1):
            image = annotation["image"]
            video = str(image["video_id"])
            if config == "endoscapes":
                filename = image["file_name"]
                frame = int(Path(filename).stem.rsplit("_", 1)[1])
                split = annotation["split"]
                image_path = endo_root / split / filename
                original = endo_annotations[filename]
                cvs = dict(zip(CRITERIA, map(float, original["ds"])))
                raters = {c: [] for c in CRITERIA}
                subset = "corrected_rubric_annotations"
            else:
                frame = int(image["frame_id"])
                split = image["split"]
                filename = f"frame_{frame:06d}.jpg"
                image_path = frames_root / split / video / filename
                key = (split, video)
                if key not in sages_annotations:
                    with (sages_root / split / "labels" / video / "frame.csv").open() as handle:
                        sages_annotations[key] = {int(r["frame_id"]): r for r in csv.DictReader(handle)}
                original = sages_annotations[key][frame]
                raters = {c: [int(original[f"{c}_rater{i}"]) for i in (1, 2, 3)] for c in CRITERIA}
                if any(v not in (0, 1) for values in raters.values() for v in values):
                    raise ValueError(f"Unexpected SAGES rater value: {video}/{frame}")
                cvs = {c: float(sum(raters[c]) >= 2) for c in CRITERIA}
                subset = annotation["source_manifest"]
            identity = f"{config}/{split}/{video}/{frame}"
            if identity in seen:
                raise ValueError(f"Duplicate image annotation: {identity}")
            seen.add(identity)
            if set(annotation["rubrics"]) != set(item_ids):
                raise ValueError(f"Incomplete rubric labels: {identity}")
            if not set(annotation["rubrics"].values()) <= {"yes", "no", "uncertain"}:
                raise ValueError(f"Invalid rubric labels: {identity}")
            image_bytes = image_path.read_bytes()
            with PILImage.open(io.BytesIO(image_bytes)) as im:
                width, height = im.size
                im.verify()
            rows.append({
                "example_id": identity, "image": {"bytes": image_bytes, "path": filename},
                "image_sha256": sha256(image_bytes), "image_filename": filename,
                "width": width, "height": height,
                "source_dataset": "Endoscapes2023" if config == "endoscapes" else "SAGES_CVS_Challenge_2024",
                "source_split": split, "source_subset": subset, "video_id": video, "frame_id": frame,
                "cvs_labels": cvs, "cvs_binary_labels": {c: int(cvs[c] > .5) for c in CRITERIA},
                "cvs_rater_labels": raters, "original_cvs_annotation_json": json.dumps(original, sort_keys=True),
                "annotation_cvs_labels": {c: float(annotation["gt"][c]) for c in CRITERIA},
                "rubric_labels": annotation["rubrics"], "rubric_version": "cvs_rubrics_v3",
                "annotation_rubric_version": "cvs_rubrics_v3",
                "source_annotation_rubric_version": annotation["rubric_version"],
                "annotation_source": source.name, "annotation_row": index,
                "annotation_notes": annotation.get("notes", ""), "annotation_timestamp": annotation.get("timestamp", ""),
                "sample_group_json": json.dumps(annotation.get("sample_group", {}), sort_keys=True),
                "is_paper_few_shot": config == "endoscapes" and (video, frame) in selections[config],
                "is_few_shot": (video, frame) in selections[config],
                "few_shot_order": selections[config].get((video, frame), -1),
            })
        result[config] = rows
    return result


def few_shot_assets(rows: dict) -> list[Path]:
    partition_rows(rows)
    assets = []
    for config, manifest in SELECTION_MANIFESTS.items():
        assets.append(manifest)
        by_key = {(r["video_id"], r["frame_id"]): r for r in rows[config]}
        for example in read_rows(ROOT / manifest):
            row = by_key[selection_key(example)]
            path = Path(example["saved_image"])
            if path.parent != manifest.parent / "images":
                raise ValueError("Unexpected few-shot image path")
            binary_targets = {c: int(row["annotation_cvs_labels"][c] > .5) for c in CRITERIA}
            if example["rubrics"] != row["rubric_labels"] or example["gt"] != binary_targets:
                raise ValueError(f"Few-shot labels differ: {example['file_name']}")
            if sha256((ROOT / path).read_bytes()) != row["image_sha256"]:
                raise ValueError(f"Few-shot image differs: {example['file_name']}")
            assets.append(path)
    return assets


def few_shot_reference() -> str:
    lines = [
        "## Few-shot Video and Frame Reference", "",
        "Rows follow the exact few-shot prompt order for each source.", "",
        "| Source | Original video filename / ID | Frame ID | Image filename | CVS targets |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for config in SOURCES:
        for example in selected_examples(config):
            video, frame = selection_key(example)
            video_label = f"`{video}.mp4`" if config == "sages" else f"ID `{video}`"
            lines.append(f"| {config} | {video_label} | {frame} | `{example['file_name']}` | `{example['gt_pattern']}` |")
    lines.extend([
        "", "SAGES filenames are relative to the original dataset's `train/videos/` directory; frame IDs are the original video frame indices used to extract the JPGs.",
        "Endoscapes source metadata supplies numeric video IDs, not original video filenames. Its frame IDs above are the original frame numbers encoded in `<video_id>_<frame_id>.jpg`, not the reindexed `frame_id` in `annotation_coco_vid.json`.",
        "Original videos are not included in this image-and-label upload.",
    ])
    return "\n".join(lines)


def dataset_card(rows: dict, repo_id: str) -> str:
    return f"""---
pretty_name: Sum-of-Checks
license: other
license_name: source-specific-noncommercial-licenses
license_link: LICENSE.md
language:
- en
task_categories:
- image-classification
size_categories:
- n<1K
tags:
- surgery
- critical-view-of-safety
- rubric-v3
configs:
- config_name: endoscapes
  data_files:
  - split: few_shot
    path: endoscapes/few_shot.parquet
  - split: dev
    path: endoscapes/dev.parquet
- config_name: sages
  data_files:
  - split: few_shot
    path: sages/few_shot.parquet
  - split: dev
    path: sages/dev.parquet
---

# Sum-of-Checks

Human rubric labels, original laparoscopic images, and original Critical View of Safety (CVS) labels accompanying
[Sum-of-Checks: Structured Reasoning for Surgical Safety with Large Vision-Language Models](https://github.com/BrachioLab/SumOfChecks).

## Subsets

| Config / split | Images | Source |
| --- | ---: | --- |
| endoscapes / few_shot | 4 | Paper exemplars, in manifest order |
| endoscapes / dev | {len(rows['endoscapes']) - 4} | Remaining corrected Endoscapes annotations |
| sages / few_shot | 4 | v5 candidates sampled with k=4, seed=13, in prompt order |
| sages / dev | {len(rows['sages']) - 4} | Remaining SAGES annotations |

These are the human-labeled rubric subsets, not the paper's 791-frame evaluation set.
The splits are disjoint by image identity and image checksum, with all 113 images retained exactly once.
They are not video-disjoint. `source_split` retains the original val/train provenance.
`is_few_shot` marks both sources; `few_shot_order` is zero-based prompt order, or -1 for dev.
`is_paper_few_shot` retains its historical Endoscapes-only meaning.
SAGES is an additional annotation subset, not an experiment reported in the Endoscapes paper.

The rubric annotator was an ML PhD student trained by a surgeon. These rubric annotations are
development data, not expert surgical ground truth. Original source CVS labels are retained separately.

## Few-shot examples

The exact four exemplars (CVS patterns 000, 111, 110, 001) are also available as a standalone
[selection manifest]({FEW_SHOT_MANIFEST.as_posix()}) and original JPGs in
`{FEW_SHOT_DIR.as_posix()}/images/`. These duplicate only the corresponding Parquet images, not additional examples.
The manifest preserves the paper selection order, CVS targets, and all 19 rubric answers.
Its `saved_image` paths resolve from the downloaded repository root.

SAGES retains the [six-candidate v5 manifest]({SAGES_FEW_SHOT_MANIFEST.as_posix()}) and its original JPGs.
Only four candidates enter `few_shot`: patterns 001, 000, 101, 111, selected by Python Random(13).shuffle
followed by taking the first four. The unused candidates (110, 100) remain in `dev`.
SAGES manifest paths were made repository-relative; all other values and source image bytes are unchanged.
The selection version v5 is independent of the rubric specification v3.

```python
from datasets import load_dataset
few_shot = load_dataset("{repo_id}", "endoscapes", split="few_shot")
sages_few_shot = load_dataset("{repo_id}", "sages", split="few_shot")
```

For saved image examples and loading in paper order, see the
[example notebook](https://github.com/BrachioLab/SumOfChecks/blob/main/notebooks/load_hf_sum_of_checks.ipynb).

{few_shot_reference()}

## Fields

- `image`: original image-file bytes embedded in Parquet; no resizing or re-encoding. SAGES images are the existing JPG video frames used during labeling.
- `cvs_labels`: original Endoscapes continuous `ds` values, or majority-vote SAGES criterion labels. `cvs_binary_labels` thresholds at greater than 0.5.
- `cvs_rater_labels`: all three original SAGES votes for each criterion, in rater1/rater2/rater3 order; empty for Endoscapes, whose source here supplies continuous `ds` labels.
- `original_cvs_annotation_json`: complete source Endoscapes image-annotation object or SAGES frame.csv row, preserving the source fields and values.
- `annotation_cvs_labels`: CVS labels recorded alongside the manual rubric annotation, separately retained from the original dataset labels.
- `rubric_labels`: answers to all 19 checks, keyed by item IDs such as C1-1; values are yes, no, or uncertain.
- `rubric_version`: bundled specification `cvs_rubrics_v3`. See [the rubric JSON](rubrics/cvs_rubrics_v3.json) for check text, types, and weights.
- `annotation_rubric_version`: `cvs_rubrics_v3` for both Endoscapes and SAGES, following the dataset maintainer's version attribution. This metadata update does not change any rubric answers.
- `source_annotation_rubric_version`: verbatim metadata from the archived annotation file (`cvs_rubrics_v1` for Endoscapes, `cvs_rubrics_v3` for SAGES), retained only for source traceability. Legacy Endoscapes filenames are unchanged.
- `source_dataset`, `source_split`, `source_subset`, `video_id`, `frame_id`, and annotation source/row identify provenance. Notes and timestamps retain the manual annotation context.
- `image_sha256`, `width`, and `height` support image integrity checks.

The three criteria are: C1, exactly two structures entering the gallbladder; C2, clearance of the hepatocystic triangle; C3, lower-third gallbladder detachment from the liver bed.
Sum-of-Checks aggregation uses the v3 weights with yes=1 and no/uncertain=0.

## Load

```python
from datasets import load_dataset
endo = load_dataset("{repo_id}", "endoscapes", split="dev")
sages = load_dataset("{repo_id}", "sages", split="dev")
image = endo[0]["image"]  # PIL image
labels = endo[0]["rubric_labels"]
```

The rubric JSON is a separate repository file and can be downloaded with `huggingface_hub.hf_hub_download`.
`export_manifest.json` records image checksums, annotation-source checksums, and the rubric checksum.

## Sources and terms

- Endoscapes: [CAMMA dataset](https://github.com/CAMMA-public/Endoscapes), CC BY-NC-SA 4.0; see [license](licenses/endoscapes.txt).
- SAGES: [CAMMA SAGES CVS Challenge 2024](https://huggingface.co/datasets/CAMMA-public/SAGES_CVS_Challenge_2024), CC BY-NC 4.0; see [license](licenses/sages.txt) and [dataset paper](https://arxiv.org/abs/2509.17100).
- Rubric annotations and v3 specification: [BrachioLab Sum-of-Checks](https://github.com/BrachioLab/SumOfChecks).

Source-specific image/label licenses continue to apply; this export does not replace them with a blanket permissive license.
This small, selected annotation set is intended for research and does not establish clinical performance.
"""


def export(out: Path, endo_root: Path, sages_root: Path, frames_root: Path, repo_id: str) -> dict:
    rows = build_rows(endo_root, sages_root, frames_root)
    ids = list(rows["endoscapes"][0]["rubric_labels"])
    manifest = {"rubric_sha256": sha256(RUBRIC.read_bytes()), "configs": {}}
    manifest["few_shot_selection"] = {
        config: {"manifest": path.as_posix(), "k": 4,
                 "seed": 13 if config == "sages" else None,
                 "policy": "shuffle_then_first_k" if config == "sages" else "manifest_order",
                 "selected": [{"video_id": r["video_id"], "frame_id": selection_key(r)[1],
                               "gt_pattern": r["gt_pattern"]} for r in selected_examples(config)]}
        for config, path in SELECTION_MANIFESTS.items()
    }
    manifest["few_shot_assets"] = {}
    for relative in few_shot_assets(rows):
        destination = out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
        manifest["few_shot_assets"][relative.as_posix()] = sha256(destination.read_bytes())
    for config, splits in partition_rows(rows).items():
        legacy_split, source = SOURCES[config]
        (out / config / f"{legacy_split}.parquet").unlink(missing_ok=True)
        manifest["configs"][config] = {"annotation_source": source.name,
            "annotation_sha256": sha256(source.read_bytes()), "splits": {}}
        for split, records in splits.items():
            path = out / config / f"{split}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            Dataset.from_list(records, features=features(ids)).to_parquet(str(path))
            manifest["configs"][config]["splits"][split] = {"rows": len(records),
                "images": [{"example_id": r["example_id"], "sha256": r["image_sha256"]} for r in records]}
    (out / "rubrics").mkdir(exist_ok=True)
    shutil.copy2(RUBRIC, out / "rubrics/cvs_rubrics_v3.json")
    (out / "licenses").mkdir(exist_ok=True)
    shutil.copy2(endo_root / "LICENSE", out / "licenses/endoscapes.txt")
    sages_license = hf_hub_download("CAMMA-public/SAGES_CVS_Challenge_2024", "LICENSE_CC_BY_NC_4.0.txt", repo_type="dataset", token=False)
    shutil.copy2(sages_license, out / "licenses/sages.txt")
    (out / "LICENSE.md").write_text("# Source-specific licenses\n\nEndoscapes images and original labels: CC BY-NC-SA 4.0, reproduced in licenses/endoscapes.txt.\n\nSAGES images and original labels: CC BY-NC 4.0, reproduced in licenses/sages.txt.\n\nRubric annotations and specification originate from BrachioLab/SumOfChecks; no additional license grant is asserted by this packaging script.\n")
    (out / "README.md").write_text(dataset_card(rows, repo_id))
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def validate(location: str, expected: dict, *, token=None, revision=None) -> None:
    # Rebuilt local exports can retain the same cache key despite schema changes.
    with tempfile.TemporaryDirectory(prefix="sumofchecks-validation-") as cache_dir:
        _validate(location, expected, token=token, revision=revision, cache_dir=cache_dir)


def _validate(location: str, expected: dict, *, token=None, revision=None, cache_dir: str) -> None:
    for config, splits in partition_rows(expected).items():
        loaded = load_dataset(location, config, token=token, revision=revision,
                              cache_dir=cache_dir)
        if set(loaded) != {"few_shot", "dev"}:
            raise ValueError(f"Unexpected splits: {config}")
        for field in ("example_id", "image_sha256"):
            if set(loaded["few_shot"][field]) & set(loaded["dev"][field]):
                raise ValueError(f"Remote few-shot/dev overlap: {config}/{field}")
        for split, records in splits.items():
            ds = loaded[split]
            if len(ds) != len(records):
                raise ValueError(f"Row count mismatch for {config}")
            raw = ds.cast_column("image", Image(decode=False))
            for index, (got, want) in enumerate(zip(raw, records)):
                if sha256(got["image"]["bytes"]) != want["image_sha256"]:
                    raise ValueError(f"Changed image: {want['example_id']}")
                for key in want.keys() - {"image"}:
                    if got[key] != want[key]:
                        raise ValueError(f"Changed {key}: {want['example_id']}")
                if ds[index]["image"].size != (want["width"], want["height"]):
                    raise ValueError(f"Image decode mismatch: {want['example_id']}")
            print(f"Validated {config}/{split}: {len(ds)} images, original CVS labels, and 19 rubric labels each.")
    for relative in few_shot_assets(expected):
        path = Path(location) / relative if Path(location).is_dir() else Path(hf_hub_download(
            location, relative.as_posix(), repo_type="dataset", token=token, revision=revision,
        ))
        if sha256(path.read_bytes()) != sha256((ROOT / relative).read_bytes()):
            raise ValueError(f"Changed few-shot asset: {relative}")
    print("Validated both selection manifests and all original candidate images.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "hf_repos/sum-of-checks")
    parser.add_argument("--endoscapes-root", type=Path, default=ROOT / "data/endoscapes")
    parser.add_argument("--sages-root", type=Path, default=ROOT / "data/CVS_Challenge_SAGES_v1")
    parser.add_argument("--frames-root", type=Path, default=ROOT / "data/frames")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--env-file", type=Path, help="Optional dotenv file containing HF_TOKEN; otherwise use normal Hugging Face authentication.")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    export(out, args.endoscapes_root, args.sages_root, args.frames_root, args.repo_id)
    expected = build_rows(args.endoscapes_root, args.sages_root, args.frames_root)
    if args.validate or args.upload:
        validate(str(out), expected)
    if args.upload:
        token = None
        if args.env_file:
            from dotenv import dotenv_values
            token = dotenv_values(args.env_file).get("HF_TOKEN")
            if not token:
                raise ValueError("The specified env file does not contain HF_TOKEN")
        api = HfApi(token=token)
        api.whoami()
        api.get_collection(args.collection)
        api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
        commit = api.upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=str(out),
            allow_patterns=["README.md", "LICENSE.md", "export_manifest.json", "rubrics/*.json", "licenses/*.txt", "endoscapes/*.parquet", "sages/*.parquet", "few_shot_examples/**/*.jsonl", "few_shot_examples/**/*.jpg"],
            delete_patterns=["endoscapes/validation.parquet", "sages/train.parquet"],
            commit_message="Separate Endoscapes and SAGES v5 few-shot examples from dev")
        validate(args.repo_id, expected, token=token, revision=commit.oid)
        rubric_download = hf_hub_download(args.repo_id, "rubrics/cvs_rubrics_v3.json", repo_type="dataset", revision=commit.oid, token=token)
        if sha256(Path(rubric_download).read_bytes()) != sha256(RUBRIC.read_bytes()):
            raise ValueError("Remote rubric checksum mismatch")
        api.add_collection_item(args.collection, item_id=args.repo_id, item_type="dataset", exists_ok=True)
        print(f"Uploaded and verified: https://huggingface.co/datasets/{args.repo_id}")
        print(f"Commit: {commit.oid}")
        print(f"Collection: https://huggingface.co/collections/{args.collection}")


if __name__ == "__main__":
    main()
