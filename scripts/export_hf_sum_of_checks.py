#!/usr/bin/env python3
"""Export, validate, and optionally publish the human-labeled rubric subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_XET_CACHE", str(ROOT / ".cache/huggingface/xet"))

from datasets import Dataset, Features, Image, List, Value, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image as PILImage

REPO_ID = "BrachioLab/sum-of-checks"
COLLECTION = "BrachioLab/laparoscopic-cholecystectomy-6a13cd06fa87aa9d2603e2ac"
RUBRIC = ROOT / "rubrics/cvs_rubrics_v3.json"
SOURCES = {
    "endoscapes": ("validation", ROOT / "annotations/rubric_labels/endoscapes_val__rubrics_v1__seed13__batch0__filtered_no_all_uncertain_fix_error.jsonl"),
    "sages": ("train", ROOT / "annotations/rubric_labels/sages__rubrics_v3__seed13__batch0__20260219_203800.jsonl"),
}
CRITERIA = ("c1", "c2", "c3")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
        "annotation_source": Value("string"), "annotation_row": Value("int32"),
        "annotation_notes": Value("string"), "annotation_timestamp": Value("string"),
        "sample_group_json": Value("string"), "is_paper_few_shot": Value("bool"),
    })


def build_rows(endo_root: Path, sages_root: Path, frames_root: Path) -> dict[str, list[dict]]:
    rubric = json.loads(RUBRIC.read_text())
    item_ids = [i["id"] for block in rubric["criteria"].values() for i in block["items"]]
    images = json.loads((endo_root / "val/annotation_ds_coco.json").read_text())["images"]
    endo_annotations = {r["file_name"]: r for r in images}
    few_shot = {r["file_name"] for r in read_rows(ROOT / "few_shot_examples/endoscapes/rubrics/filtered/selected_examples_v3.jsonl")}
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
                "annotation_rubric_version": annotation["rubric_version"],
                "annotation_source": source.name, "annotation_row": index,
                "annotation_notes": annotation.get("notes", ""), "annotation_timestamp": annotation.get("timestamp", ""),
                "sample_group_json": json.dumps(annotation.get("sample_group", {}), sort_keys=True),
                "is_paper_few_shot": config == "endoscapes" and filename in few_shot,
            })
        result[config] = rows
    return result


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
  - split: validation
    path: endoscapes/validation.parquet
- config_name: sages
  data_files:
  - split: train
    path: sages/train.parquet
---

# Sum-of-Checks

Human rubric labels, original laparoscopic images, and original Critical View of Safety (CVS) labels accompanying
[Sum-of-Checks: Structured Reasoning for Surgical Safety with Large Vision-Language Models](https://github.com/BrachioLab/SumOfChecks).

## Subsets

| Config / split | Images | Source |
| --- | ---: | --- |
| endoscapes / validation | {len(rows['endoscapes'])} | Corrected, filtered Endoscapes validation annotations |
| sages / train | {len(rows['sages'])} | SAGES rubric annotations from train_dev and train_rest |

These are the human-labeled rubric subsets, not the paper's 791-frame evaluation set.
The four Endoscapes paper exemplars are marked `is_paper_few_shot`.
SAGES is an additional annotation subset, not an experiment reported in the Endoscapes paper.

## Fields

- `image`: original image-file bytes embedded in Parquet; no resizing or re-encoding. SAGES images are the existing JPG video frames used during labeling.
- `cvs_labels`: original Endoscapes continuous `ds` values, or majority-vote SAGES criterion labels. `cvs_binary_labels` thresholds at greater than 0.5.
- `cvs_rater_labels`: all three original SAGES votes for each criterion, in rater1/rater2/rater3 order; empty for Endoscapes, whose source here supplies continuous `ds` labels.
- `original_cvs_annotation_json`: complete source Endoscapes image-annotation object or SAGES frame.csv row, preserving the source fields and values.
- `annotation_cvs_labels`: CVS labels recorded alongside the manual rubric annotation, separately retained from the original dataset labels.
- `rubric_labels`: answers to all 19 checks, keyed by item IDs such as C1-1; values are yes, no, or uncertain.
- `rubric_version`: bundled specification `cvs_rubrics_v3`. See [the rubric JSON](rubrics/cvs_rubrics_v3.json) for check text, types, and weights.
- `annotation_rubric_version`: historical labeling version. Endoscapes retains `cvs_rubrics_v1`; SAGES uses `cvs_rubrics_v3`. Endoscapes labels were not newly relabeled under v3. Their item IDs and weights are shared with v3, but wording was revised.
- `source_dataset`, `source_split`, `source_subset`, `video_id`, `frame_id`, and annotation source/row identify provenance. Notes and timestamps retain the manual annotation context.
- `image_sha256`, `width`, and `height` support image integrity checks.

The three criteria are: C1, exactly two structures entering the gallbladder; C2, clearance of the hepatocystic triangle; C3, lower-third gallbladder detachment from the liver bed.
Sum-of-Checks aggregation uses the v3 weights with yes=1 and no/uncertain=0.

## Load

```python
from datasets import load_dataset
endo = load_dataset("{repo_id}", "endoscapes", split="validation")
sages = load_dataset("{repo_id}", "sages", split="train")
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
    for config, records in rows.items():
        split, source = SOURCES[config]
        path = out / config / f"{split}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        Dataset.from_list(records, features=features(ids)).to_parquet(str(path))
        manifest["configs"][config] = {"split": split, "rows": len(records),
            "annotation_source": source.name, "annotation_sha256": sha256(source.read_bytes()),
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
    for config, records in expected.items():
        split = SOURCES[config][0]
        ds = load_dataset(location, config, split=split, token=token, revision=revision)
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
            allow_patterns=["README.md", "LICENSE.md", "export_manifest.json", "rubrics/*.json", "licenses/*.txt", "endoscapes/*.parquet", "sages/*.parquet"],
            commit_message="Add rubric v3, 113 labeled images, and original CVS annotations")
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
