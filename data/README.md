# Sum-of-Checks Data

Open [Load and Explore Sum-of-Checks](../notebooks/load_hf_sum_of_checks.ipynb) to load the public dataset and inspect examples. The notebook includes saved PNG figures and plain-text rubric labels that render on GitHub; it does not use HTML or widgets.

The [Hugging Face dataset](https://huggingface.co/datasets/BrachioLab/sum-of-checks) includes:

- `endoscapes`: `few_shot` (4 paper exemplars) and `dev` (49 remaining images).
- `sages`: `few_shot` (4 examples selected from v5 with k=4, seed=13) and `dev` (56 remaining images).
- The splits have no shared image IDs or checksums; they are not video-disjoint. `source_split` preserves original val/train provenance.
- Both selection manifests and original exemplar JPGs are bundled under `few_shot_examples/`. SAGES retains the six-candidate v5 manifest; its unused candidates (110, 100) remain in `dev`. The selected prompt order is 001, 000, 101, 111. The notebook displays both few-shot splits.
- Original image bytes, original CVS labels, individual SAGES rater votes, and the 19-check v3 rubric.

The notebook loads directly from Hugging Face, with no local dataset links or login required. Install `requirements-hf.txt` from the repository root to run it.

Rubric labels were provided by an ML PhD student trained by a surgeon. Treat them as development data, not expert surgical ground truth; original source CVS labels are retained separately.

Both Endoscapes and SAGES use rubric v3 (`rubric_version` and `annotation_rubric_version`), following the dataset maintainer's version attribution. Archived source-file metadata is retained separately as `source_annotation_rubric_version`; the legacy Endoscapes v1 filenames are unchanged. No rubric answers were changed by this metadata update. These annotation subsets are separate from the 791-frame paper evaluation split under [manifests/endoscapes/](manifests/endoscapes/).

## Few-shot Video and Frame Reference

Rows follow the exact few-shot prompt order for each source.

| Source | Original video filename / ID | Frame ID | Image filename | CVS targets |
| --- | --- | ---: | --- | --- |
| endoscapes | ID `129` | 69650 | `129_69650.jpg` | `000` |
| endoscapes | ID `154` | 33500 | `154_33500.jpg` | `111` |
| endoscapes | ID `152` | 41050 | `152_41050.jpg` | `110` |
| endoscapes | ID `148` | 21675 | `148_21675.jpg` | `001` |
| sages | `7a208039-f40e-4f03-b73d-555ec9b23119.mp4` | 900 | `frame_000900.jpg` | `001` |
| sages | `24b701a0-f22d-4cba-9871-fab47bd18347.mp4` | 2400 | `frame_002400.jpg` | `000` |
| sages | `15c0e57a-97d2-4add-8404-063e3747cd47.mp4` | 150 | `frame_000150.jpg` | `101` |
| sages | `79b1e745-0fc7-45e0-8159-bcc027cc6db8.mp4` | 1800 | `frame_001800.jpg` | `111` |

SAGES filenames are relative to the original dataset's `train/videos/` directory; frame IDs are the original video frame indices used to extract the JPGs.
Endoscapes source metadata supplies numeric video IDs, not original video filenames. Its frame IDs above are the original frame numbers encoded in `<video_id>_<frame_id>.jpg`, not the reindexed `frame_id` in `annotation_coco_vid.json`.
Original videos are not included in this image-and-label upload.
