# Sum-of-Checks Data

Open [Load and Explore Sum-of-Checks](../notebooks/load_hf_sum_of_checks.ipynb) to load the public dataset and inspect examples. The notebook includes saved PNG figures and plain-text rubric labels that render on GitHub; it does not use HTML or widgets.

The [Hugging Face dataset](https://huggingface.co/datasets/BrachioLab/sum-of-checks) includes:

- `endoscapes/validation`: 53 corrected rubric-labeled images.
- `sages/train`: 60 rubric-labeled images.
- Original image bytes, original CVS labels, individual SAGES rater votes, and the 19-check v3 rubric.

The notebook loads directly from Hugging Face, with no local dataset links or login required. Install `requirements-hf.txt` from the repository root to run it.

Historical annotation versions are preserved. Endoscapes labels originated under rubric v1; the bundled active specification is v3. These annotation subsets are separate from the 791-frame paper evaluation split under [manifests/endoscapes/](manifests/endoscapes/).
