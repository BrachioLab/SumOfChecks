# Reproduction and provenance

The extraction comes from `../lvlm_cvs_reasoning`. [source_manifest.json](source_manifest.json) records the original file hashes; [prediction_manifest.json](prediction_manifest.json) pins the 70 copied prediction files and their hashes, models, methods, runs, row counts, and table associations. Predictions and all four historical annotation files are unchanged.

The selected predictions occupy about 482 MiB. They include the seven Endoscapes methods for the three paper models, GPT Direct's extra runs, and GPT's four no-FS runs. No SAGES predictions, alternative few-shot sets, segmentation models, schemas, credentials, or caches were copied.

## Reproduced results

`python scripts/evaluate_paper.py` executes the curated table notebook against the explicit manifest. The code computes per-run AP, averages across criteria, and reports sample standard deviations. It retains the source notebook's per-model common-frame filtering and missing-score handling. `results/paper/frame_counts.csv` records valid frame counts after the main table's common-frame filtering; ablation counts are printed in the notebook.

| Model | Sum-of-Checks average mAP (%) |
| --- | --- |
| GPT-4.1-mini | 36.9 +/- 0.5 |
| Claude Haiku 4.5 | 34.0 +/- 0.2 |
| Claude Opus 4.5 | 37.0 +/- 0.1 |

Figure 2 uses Opus run001, video 184, frame 36750. The six baseline predictions and weighted score (1.0, 0.6, 0.9) match the PDF. The underlying LLM aggregation predicts C2=0, which demonstrates why `pred` must not be used as the Sum-of-Checks score.

## Reporting discrepancies

1. **GPT Direct:** the PDF's 29.7 +/- 0.9 matches all six available runs, not the blanket three-run description. The first three yield 29.6 +/- 1.3. All six are retained and counted explicitly.
2. **GPT no-FS ablation:** the PDF's (23.4, 23.9, 44.1), average 30.5 +/- 0.4, matches LLM aggregation over four runs. Weighted aggregation over those runs gives (24.6, 24.6, 44.5), average 31.2 +/- 0.7. The exported ablation includes both, with explicit labels. The original notebook's comment said LLM aggregation while its calculation used weighted aggregation.
3. **Opus SubQ+FS:** the copied fsv3 runs give (20.7, 21.9, 43.7), average 28.8 +/- 0.2. The PDF reports (19.2, 19.7, 43.9), average 27.6 +/- 0.9. The source for that PDF row remains unresolved; these files are marked `unresolved_paper_row` in the manifest.

The new launcher defaults to three runs per condition, as described by the paper. It does not reproduce the historical six-/four-run reporting exceptions automatically. Re-running stochastic hosted models also need not reproduce historical numerical results.

## Portability changes

- Copied the frame runner, dataset loaders, and only OpenAI/Anthropic model adapters. Removed eager imports of unrelated model and organ stacks. The frame runner retains its original internal pipeline helpers to preserve prompt behavior; the supported paper workflow is exposed through `run_paper.py`.
- Removed local credential-file loading. Provider keys come from environment variables. No live API requests were used in verification.
- Removed torch/torchvision from model image utilities: this workflow passes PIL images, whose PNG encoding is unchanged. Optional array inputs now support uint8 arrays.
- Fixed code/notebook repository discovery and exemplar image paths. Historical paths embedded in original prediction and annotation records remain intact for provenance; viewers resolve current dataset paths.
- Restricted bundled few-shot versions to Endoscapes fsv3. SAGES inference with few-shot examples requires an explicit `--few_shot_path`.
- Corrected the runner's `weighted_score` to the paper's binary aggregation: uncertain contributes zero. The source runner used 0.5 for this auxiliary field, while the paper evaluation notebook recomputed binary scores from `rubric_labels_stage1`. Frozen predictions remain unchanged; evaluation continues to recompute their scores.
- Isolated new inference outputs from frozen artifacts. The runner rejects `--output_root` inside the frozen artifact directory.
- Retained the table and viewer notebooks, trimmed the exemplar notebook to the final selection, and updated both labeling notebooks to v3. Labeling writes new timestamped records under `annotations/new/`; historical annotation files are not overwritten.
- Added a CLI table exporter and focused verification tests. The original extraction analysis remains in [COPY_PLAN.md](../COPY_PLAN.md).

## Local data and validation

The local `.venv` was built with Python 3.10 because the system Python installation lacked `ensurepip`. `requirements.txt` is the portable dependency list; `requirements-lock.txt` records the installed versions used in verification.

Ignored local symlinks provide Endoscapes, the SAGES raw dataset, and existing sampled SAGES frames. These are dataset dependencies, not imports from the old codebase. To reconstruct SAGES frames elsewhere, use the copied `scripts/extract_frames.py` with the two bundled SAGES manifests.

Verification completed on 2026-09-11:

- `python -m pytest -q`: 13 passed. Coverage includes frozen checksums, original label files, all 791 evaluation images, exemplar decoding, binary weighted aggregation, Figure 2, mocked provider request formatting, inference/parsing/resume, and frozen-output protection.
- All five notebooks executed successfully from their notebook directory. Executed verification copies were written under `/tmp`, keeping committed notebooks free of stale outputs.
- Both annotation Save callbacks were exercised with test records under ignored `.cache/`; no human labels were submitted or overwritten. Exemplar export into `.cache/` reproduced the four frozen filenames, criterion labels, and check labels exactly.
- All eight paper/ablation conditions passed image-backed prompt dry runs. Evaluation regenerated all table artifacts and reproduced the three headline results.
- Rebuilding and splitting the Endoscapes test manifest reproduced the frozen 20 videos and 791 frames. The SAGES frame extractor's resume path verified 900 existing train-dev frames without rewriting them.
- Python compilation, README links, dependency checks, and `git diff --check` passed.

Live provider availability and billing credentials are not tested. No model API requests were made.
