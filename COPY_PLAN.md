# Sum-of-Checks extraction plan

Analysis date: 2026-09-10. Source: `../lvlm_cvs_reasoning/`.
Scope checked against `Sum-of-Checks-paper.pdf`; README style reference: `../CVSAct/README.md`.
Historical analysis before extraction. The scoped copy is now implemented; see [README.md](README.md) and [docs/reproduction.md](docs/reproduction.md) for the current workflow, verification, and remaining reporting discrepancies. Source files were not modified.

## Confirmed paper workflow

- Endoscapes `data/manifests/endoscapes/test_dev.jsonl`: 20 videos, 791 frames from the official test split. The name `test_dev` does not mean validation data.
- Models: `gpt-4.1-mini`, `claude-haiku-4-5-20251001`, `claude-opus-4-5-20251101`; temperature 0.1.
- Active rubric: `rubrics/cvs_rubrics_v3.json`, with 19 checks (6/6/7) and fixed weights.
- Four Endoscapes exemplars: `few_shot_examples/endoscapes/rubrics/filtered/selected_examples_v3.jsonl`.
- Inference: `scripts/rubric_oracle_reasoning_frame.py`.
- Tables 1 and 2: `notebooks/rubrics/compact_endoscapes_map.ipynb`.
- Qualitative viewer: `notebooks/rubrics/view_endoscapes_examples.ipynb`.
- The PDF leaves SAGES evaluation to future work. Preserve the explicitly requested SAGES labels and annotation workflow as additional annotation assets, without presenting SAGES experiments as paper results.

## Copy unchanged

All paths in this section are relative to the source repository; retain those paths initially.

| Files | Reason |
| --- | --- |
| `rubrics/cvs_rubrics_v3.json` | Authoritative checks and weights. |
| `annotations/rubric_labels/endoscapes_val__rubrics_v1__seed13__batch0__20260114_033331.jsonl` | Original annotations, 60 records. |
| `annotations/rubric_labels/endoscapes_val__rubrics_v1__seed13__batch0__filtered_no_all_uncertain.jsonl` | Filtered annotations, 53 records. |
| `annotations/rubric_labels/endoscapes_val__rubrics_v1__seed13__batch0__filtered_no_all_uncertain_fix_error.jsonl` | Corrected annotations, 53 records; input to Endoscapes exemplar selection and annotation version v2. |
| `annotations/rubric_labels/sages__rubrics_v3__seed13__batch0__20260219_203800.jsonl` | Requested SAGES annotations, 60 records. |
| `data/manifests/endoscapes/test_dev.jsonl` | Exact evaluation split. |
| `data/manifests/endoscapes/test_manifest.jsonl`, `test_rest.jsonl`, `val_manifest.jsonl` | Split provenance and validation annotation/exemplar context. |

Keep annotation filenames and their original `rubric_version` metadata. The Endoscapes `rubrics_v1` names describe annotation provenance, not the rubric used for final inference. v1 and v3 share item IDs and weights but have different wording; do not silently relabel historical annotations as v3.

## Copy with focused edits

| Source | Extraction work |
| --- | --- |
| `scripts/rubric_oracle_reasoning_frame.py` | Retain the paper presets, prompts, parsing, and output semantics. Set explicit v3/fsv3 defaults; remove references to omitted example versions from the supported interface. Use environment variables for API keys. |
| `src/data/{__init__,datasets,endoscapes,cvs_challenge_sages_v1}.py` | Dataset dependency closure. Remove organ-store exports from `__init__.py` so unrelated organ code is unnecessary. |
| `src/models/{__init__,base,utils,openai_gpt,anthropic_claude}.py` | Model dependency closure. Limit the factory/imports to the paper's OpenAI and Anthropic adapters. |
| `few_shot_examples/endoscapes/rubrics/filtered/selected_examples_v3.jsonl` | Preserve labels/order; replace old absolute image paths with portable repository-relative references and verify resolution. |
| `notebooks/rubrics/select_few_shot_examples.ipynb` | Retain the corrected-label input, shared helpers, and v3 selection/export cells; drop the other version exports. |
| `notebooks/rubrics/metadata_rubric_label_sages.ipynb` | Already uses rubric v3. Fix repository paths; retain only manifests actually required by its sampling configuration. |
| `notebooks/rubrics/metadata_rubric_label_endoscapes.ipynb` | Relevant original annotation UI, but currently uses rubric v1. Adapt future labeling to v3 with truthful new output metadata; document that historical labels came from the earlier UI. |
| `notebooks/rubrics/compact_endoscapes_map.ipynb` | Keep paper table calculations, remove alternative scoring plots, replace the hard-coded root, and pin input files instead of recursively discovering runs. Resolve the provenance issues below first. |
| `notebooks/rubrics/view_endoscapes_examples.ipynb` | Keep the three-model/seven-method comparison and rubric details; fix paths and expose the paper frame directly. |
| `scripts/rubrics/short_communication/final/run_all_methods_fsv3.sh` | Derive one concise Endoscapes runner. Current script also launches SAGES jobs. |
| `scripts/rubrics/short_communication/final/gpt-4.1-mini_b1_rubric_ablation.sh` | Extract the Endoscapes no-FS condition. The paper's LLM-aggregation ablation reuses the main FS outputs; it needs no separate inference run. |
| `scripts/build_manifest.py`, `scripts/split_manifest.py` | Retain relevant Endoscapes preparation/split functionality as provenance utilities. Frozen manifests remain authoritative. |

The four required exemplar images already exist under `few_shot_examples/endoscapes/rubrics/filtered/images/`:

| Image | C1/C2/C3 |
| --- | --- |
| `129_69650.jpg` | 000 |
| `154_33500.jpg` | 111 |
| `152_41050.jpg` | 110 |
| `148_21675.jpg` | 001 |

Copy only these four exemplar images, subject to the dataset's redistribution terms, or resolve them from the external dataset. Keep raw Endoscapes/SAGES data external. The loaders expect Endoscapes annotations/images under `data/endoscapes/`; SAGES uses `data/CVS_Challenge_SAGES_v1/` and `data/frames/`.

Do not copy the existing `requirements.txt`: it contains agent-framework dependencies and omits the paper pipeline's dependencies. Build a small dependency file from the retained imports. Besides the provider SDKs, notebooks need NumPy, pandas, scikit-learn, matplotlib, Pillow, tqdm, and notebook/widget support. Model utilities also import diskcache, torch, and torchvision; either retain those dependencies or make a separate, validated reduction of tensor-image support.

## Prediction artifacts and method mapping

Candidate directory: `outputs/rubric_oracle/manifests/test_dev/` only.
Filenames use prefix `endoscapes_val__cvsrubricsv3__annv2__`, then the experiment label, then `__seed13__<model>_runNNN.jsonl`. FS filenames have extra underscores before `seed13`; use exact paths in the eventual manifest.

| Paper method | Experiment label, ignoring trailing underscores | Score |
| --- | --- | --- |
| Direct | `direct__preset-direct__test_dev__critdef` | `pred` |
| Direct+FS | `direct__preset-direct__test_dev__critdef__fsv3__fs4` | `pred` |
| CoT | `direct__preset-direct__test_dev__critdef__ratdf` | `pred` |
| CoT+FS | `direct__preset-direct__test_dev__critdef__ratdf__fsv3__fs4` | `pred` |
| SubQ | `self_rubric__test_dev__critdef__rat1__s1short` | `pred` |
| SubQ+FS | `self_rubric__test_dev__critdef__rat1__s1short__fsv3__fs4` | `pred` |
| Sum-of-Checks | `predicted_only__test_dev__critdef__fsv3__fs4` | Weighted `rubric_labels_stage1`, yes=1, no/uncertain=0 |
| No-FS ablation | `predicted_only__test_dev__critdef` | See discrepancy below. |
| LLM-aggregation ablation | Same files as Sum-of-Checks | `pred` |

The final method is the weighted calculation, not simply the runner's `pred` field. For Figure 2, the raw LLM aggregation has C2=0 while the weighted score is C2=0.6.

Before copying predictions, create an explicit manifest with source path, model, method, run, aggregation, row/unique-frame counts, checksum, and associated paper table. Preserve original files. Do not copy the entire output tree or choose runs solely because their names contain `final`.

## Verification and unresolved provenance

I reran the calculation cells 1, 2, 3, 5, and 6 of `compact_endoscapes_map.ipynb` in memory using the existing `py310` environment, changing only its old repository-root path. Plot-writing cells were not executed. No model API calls were made.

- All three headline weighted results reproduce: GPT 36.9 +/- 0.5, Haiku 34.0 +/- 0.2, Opus 37.0 +/- 0.1 average mAP (%).
- Figure 2 is **video 184, frame 36750**, from Opus run001. All six baseline predictions and weighted scores (1.0, 0.6, 0.9) match the PDF. `view_endoscapes_examples.ipynb` is the appropriate viewer; the final composed figure source has not been established.
- **GPT Direct uses six discovered runs**, run001-run006. All six reproduce the PDF's 29.7 +/- 0.9; the first three give 29.6 +/- 1.3. This conflicts with the paper's blanket three-run description. Preserve all six as provenance candidates until the intended reporting is settled.
- **The no-FS ablation's PDF numbers match LLM aggregation over four runs**, not weighted aggregation: `pred` gives (23.4, 23.9, 44.1), average 30.5 +/- 0.4 over run001-run004. The current notebook uses weighted scoring and gives (24.6, 24.6, 44.5), average 31.2 +/- 0.7. This is both a scoring and run-count discrepancy; do not quietly change either to force agreement.
- **Opus SubQ+FS still differs**: current fsv3 run001-run003 give (20.7, 21.9, 43.7), average 28.8 +/- 0.2; the PDF gives (19.2, 19.7, 43.9), average 27.6 +/- 0.9. The intended files or calculation remain unidentified.
- Some saved notebook outputs are stale: saved Opus results retain a single run, whereas current files produce three-run results for those conditions.
- The notebook computes common-frame filtering per model and handles missing predictions; the nominal 791-frame split is not an assertion that every method contributes 791 valid predictions in every run. Preserve and report actual evaluated counts.

These issues do not prevent copying the verified code, rubric, annotations, exemplars, and split files. They do prevent labeling a guessed subset of prediction files as an exact reproduction of every paper row.

## Exclude or defer

- Exclude `schemas/` entirely, as requested. The retained frame runner embeds its prompt/output structures and does not reference that directory.
- Exclude other active rubric versions, example-version sweeps, `legacy/`, segmentation/model training, organ preprocessing, video-level/sequential experiments, surgical agents, and unrelated prompt sweeps.
- Exclude caches, logs, checkpoints, raw datasets, credentials (`.env`, `API_KEYS*.json`, cloud credential JSON), and copied notebook checkpoints.
- Do not use `scripts/rubric_oracle_reasoning.py` as the primary runner: it is the older annotation-driven counterpart. The confirmed paper launchers use the `_frame.py` runner.
- Do not copy `rubric_oracle_reasoning_frame_level.ipynb` as a current inference notebook: it defaults to rubric v2. `visualize_results.ipynb` also points to rubric v2 and older outputs.
- Defer `eval_all_models_summary*.ipynb`, `all_models_failure_diagnosis.ipynb`, `rubric_label_distributions.ipynb`, cross-dataset notebooks, and most `eval_rubric_oracle_*` notebooks. They contain broader SAGES/Gemini/version/resolution analyses, rather than being necessary for the two paper tables.
- `select_few_shot_examples_sages.ipynb` directly consumes the requested SAGES labels, but exports v1-v8 example sets. Keep it only if preserving a SAGES exemplar workflow is desired; no final SAGES FS version is established by this PDF.
- The paper discusses check reliability qualitatively, without a separate check-accuracy table/figure. A specific final quantitative artifact for that claim has not been established; do not promote a distribution plot to ground-truth accuracy evidence.

## README plan

Follow CVSAct's short introduction, linked workflow pointers, and explicit final-version labels. Aim for roughly 80-120 lines, with no historical experiment inventory or roadmap.

1. **Title and paper:** full paper title, one-sentence description, PDF link.
2. **Key pointers:** rubric v3; corrected Endoscapes annotations; SAGES annotations; four FS examples; paper table notebook; example viewer. A small table is enough.
3. **Setup:** one tested environment/install recipe, environment-variable API keys, repository-root execution convention. Avoid claiming editable-install support until packaging exists.
4. **Data:** external dataset locations; exact 20-video/791-frame manifest; where historical versus corrected labels live. State that SAGES assets are supplementary to the Endoscapes paper evaluation.
5. **Run experiments:** one curated Endoscapes command, the three model IDs, temperature 0.1, fsv3 with k=4, and a small paper-name-to-preset mapping.
6. **Reproduce results:** frozen prediction manifest, table notebook, weighted versus LLM aggregation, and the Figure 2 frame. Link detailed provenance rather than putting long output filenames in the README.
7. **Annotation:** one pointer per retained labeling/selection workflow. Explicitly distinguish historical Endoscapes label provenance from the active v3 rubric.

Keep long filenames, hashes, frame counts, and reporting discrepancies in `docs/reproduction.md` and a machine-readable artifact manifest. The public README should point to those details and avoid promising exact reproduction until the discrepancies are resolved.

## Implementation order

1. Copy the rubric, four annotation files, frozen manifests, and four exemplars.
2. Copy the frame runner and minimal data/model modules; fix imports, paths, and dependencies.
3. Extract the relevant notebook cells and two concise launchers; retain original source paths in provenance documentation.
4. Freeze prediction candidates and resolve the three reporting issues above before presenting them as final paper artifacts.
5. Verify data loading and prompt construction without API calls, regenerate tables from frozen outputs, check the Figure 2 example, and validate README links.
