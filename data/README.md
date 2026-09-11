# Sum-of-Checks Data

Open [Load and Explore Sum-of-Checks](../notebooks/load_hf_sum_of_checks.ipynb) to load the public dataset and inspect examples. The notebook includes saved PNG figures and plain-text rubric labels that render on GitHub; it does not use HTML or widgets.

The [Hugging Face dataset](https://huggingface.co/datasets/BrachioLab/sum-of-checks) includes:

- `endoscapes`: `few_shot` (4 paper exemplars) and `dev` (49 remaining images).
- `sages`: `few_shot` (all 6 v5 examples) and `dev` (54 remaining images).
- The splits have no shared image IDs or checksums; they are not video-disjoint. `source_split` preserves original val/train provenance.
- Both selection manifests and original exemplar JPGs are bundled under `few_shot_examples/`. SAGES includes the full v5 manifest in order: 000, 111, 110, 001, 101, 100. None of these six images appear in `dev`; this is not the four-example subsample from one SAGES experiment. The notebook displays both few-shot splits.
- Original image bytes, original CVS labels, individual SAGES rater votes, and the 19-check v3 rubric.

The notebook loads directly from Hugging Face, with no local dataset links or login required. Install `requirements-hf.txt` from the repository root to run it.

Rubric labels were provided by an ML PhD student trained by a surgeon. Treat them as development data, not expert surgical ground truth; original source CVS labels are retained separately.

Both Endoscapes and SAGES use rubric v3 (`rubric_version` and `annotation_rubric_version`), following the dataset maintainer's version attribution. Archived source-file metadata is retained separately as `source_annotation_rubric_version`; the legacy Endoscapes v1 filenames are unchanged. No rubric answers were changed by this metadata update. These annotation subsets are separate from the 791-frame paper evaluation split under [manifests/endoscapes/](manifests/endoscapes/).

## Few-shot Video and Frame Reference

Rows follow the full few-shot manifest order for each source.

| Source | Original video filename / ID | Frame ID | Image filename | CVS targets |
| --- | --- | ---: | --- | --- |
| endoscapes | ID `129` | 69650 | `129_69650.jpg` | `000` |
| endoscapes | ID `154` | 33500 | `154_33500.jpg` | `111` |
| endoscapes | ID `152` | 41050 | `152_41050.jpg` | `110` |
| endoscapes | ID `148` | 21675 | `148_21675.jpg` | `001` |
| sages | `24b701a0-f22d-4cba-9871-fab47bd18347.mp4` | 2400 | `frame_002400.jpg` | `000` |
| sages | `79b1e745-0fc7-45e0-8159-bcc027cc6db8.mp4` | 1800 | `frame_001800.jpg` | `111` |
| sages | `d171f114-19ac-4e30-b24e-ea8015e90bab.mp4` | 1050 | `frame_001050.jpg` | `110` |
| sages | `7a208039-f40e-4f03-b73d-555ec9b23119.mp4` | 900 | `frame_000900.jpg` | `001` |
| sages | `15c0e57a-97d2-4add-8404-063e3747cd47.mp4` | 150 | `frame_000150.jpg` | `101` |
| sages | `3b333490-f68d-4a7c-9e38-14c621da1c85.mp4` | 1050 | `frame_001050.jpg` | `100` |

SAGES filenames are relative to the original dataset's `train/videos/` directory; frame IDs are the original video frame indices used to extract the JPGs.
Endoscapes source metadata supplies numeric video IDs, not original video filenames. Its frame IDs above are the original frame numbers encoded in `<video_id>_<frame_id>.jpg`, not the reindexed `frame_id` in `annotation_coco_vid.json`.
Original videos are not included in this image-and-label upload.
