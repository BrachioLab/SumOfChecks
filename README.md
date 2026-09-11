# Sum-of-Checks

Code and artifacts for [Sum-of-Checks: Structured Reasoning for Surgical Safety with Large Vision-Language Models](Sum-of-Checks-paper.pdf).

## Pointers

| Workflow | Files |
| --- | --- |
| Checks and weights | [CVS rubric v3](rubrics/cvs_rubrics_v3.json) |
| Human labels | [Original, filtered, corrected Endoscapes labels and SAGES labels](annotations/rubric_labels/) |
| Paper exemplars | [Four Endoscapes v3 examples](few_shot_examples/endoscapes/rubrics/filtered/selected_examples_v3.jsonl) |
| Inference | [Paper launcher](scripts/run_paper.py), [frame runner and prompts](scripts/rubric_oracle_reasoning_frame.py) |
| Tables | [Evaluation notebook](notebooks/rubrics/compact_endoscapes_map.ipynb), [exported results](results/paper/) |
| Figure 2 | [Example viewer](notebooks/rubrics/view_endoscapes_examples.ipynb), video 184 / frame 36750 |
| Provenance | [Frozen prediction manifest](docs/prediction_manifest.json), [reproduction notes](docs/reproduction.md) |

## Setup

Python 3.10 or newer, with venv support:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Alternatively, create a conda environment with `conda create -n sumofchecks python=3.10`, activate it, and install the requirements. [requirements-lock.txt](requirements-lock.txt) records the exact tested Python 3.10 environment.

The local `.venv` is already installed. Run commands from the repository root. Inference reads `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` from the environment; evaluation and dry runs need no keys.

## Data

The paper evaluates [20 Endoscapes test videos / 791 frames](data/manifests/endoscapes/test_dev.jsonl). `test_dev` is a subset of the official test split. Four validation-set exemplars have label combinations 000, 111, 110, and 001.

Raw data remains external. On the lab server, these local links are already configured:

```bash
ln -s /mnt/md0/weiqiuy/datasets/endoscapes data/endoscapes
ln -s /mnt/md0/weiqiuy/datasets/CVS_Challenge_SAGES_v1 data/CVS_Challenge_SAGES_v1
ln -s /mnt/md0/weiqiuy/lvlm_cvs_reasoning/data/frames data/frames
```

Elsewhere, place or link Endoscapes under `data/endoscapes/`, including `val/`, `test/`, their images, and `annotation_ds_coco.json` files. SAGES annotation requires its raw dataset and sampled frames. The copied predictions and four exemplar images are sufficient for offline table evaluation without raw datasets.

## Run Experiments

Inspect all seven prompts and their images without API calls:

```bash
python scripts/run_paper.py --models gpt-4.1-mini --runs 1 --dryrun
```

Run all seven methods, three models, and three repetitions:

```bash
python scripts/run_paper.py
```

Models are GPT-4.1-mini, Claude Haiku 4.5, and Claude Opus 4.5. The launcher uses temperature 0.1, rubric v3, and four fsv3 exemplars. Jobs run sequentially and resume by run number. New predictions go to ignored `outputs/generated/`; the frozen artifacts are protected.

Run only Sum-of-Checks or the no-FS ablation:

```bash
python scripts/run_paper.py --models gpt-4.1-mini --methods sum_of_checks
python scripts/run_paper.py --models gpt-4.1-mini --methods no_fs
```

Sum-of-Checks uses `weighted_score`: yes=1, no/uncertain=0. `pred` is the separate LLM-aggregation output.

## Reproduce Results

```bash
python scripts/evaluate_paper.py
python -m pytest -q
jupyter lab
```

Evaluation verifies the 70 frozen prediction checksums and exports metrics, frame counts, and LaTeX tables to [results/paper/](results/paper/). Headline average mAP reproduces as 36.9%, 34.0%, and 37.0% for GPT, Haiku, and Opus respectively.

There are documented run-count and no-FS scoring discrepancies, plus an unresolved Opus SubQ+FS row. See [reproduction notes](docs/reproduction.md) before interpreting the exported tables as an exact reproduction of every PDF entry.

## Annotation

- [Endoscapes labeling](notebooks/rubrics/metadata_rubric_label_endoscapes.ipynb) and [SAGES labeling](notebooks/rubrics/metadata_rubric_label_sages.ipynb) use rubric v3; new labels go to ignored `annotations/new/`.
- [Endoscapes exemplar selection](notebooks/rubrics/select_few_shot_examples.ipynb) shows the corrected labels and the four paper examples. Export is disabled by default.
- Historical Endoscapes filenames and label metadata retain `rubrics_v1`; the corrected file is the final exemplar-selection input. SAGES labels are retained as additional annotation assets, not as results reported in this Endoscapes paper.

To prepare SAGES frames on another machine, run [extract_frames.py](scripts/extract_frames.py) for both `train_dev.jsonl` and `train_rest.jsonl` under `data/manifests/cvs_challenge_sages_v1/`, using `--root data/CVS_Challenge_SAGES_v1 --manifest <manifest> --out data/frames`.

## Hugging Face

[BrachioLab/sum-of-checks](https://huggingface.co/datasets/BrachioLab/sum-of-checks) packages the 53 corrected Endoscapes and 60 SAGES rubric-labeled images, their original CVS labels, all SAGES rater votes, and [rubric v3](rubrics/cvs_rubrics_v3.json). It belongs to the [Laparoscopic Cholecystectomy collection](https://huggingface.co/collections/BrachioLab/laparoscopic-cholecystectomy-6a13cd06fa87aa9d2603e2ac).

Build and validate the portable image dataset:

```bash
pip install -r requirements-hf.txt
python scripts/export_hf_sum_of_checks.py --validate
```

Publish with `python scripts/export_hf_sum_of_checks.py --upload`. Authentication uses `HF_TOKEN` or the Hugging Face login cache; `--env-file <path>` can supply an existing dotenv credential file. The script validates every image and label before upload and again from the uploaded revision, then adds the dataset to the collection. Export files under `hf_repos/` are ignored by Git.

```python
from datasets import load_dataset
endo = load_dataset("BrachioLab/sum-of-checks", "endoscapes", split="validation")
sages = load_dataset("BrachioLab/sum-of-checks", "sages", split="train")
```
