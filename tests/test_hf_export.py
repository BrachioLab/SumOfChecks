import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("datasets")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hf_export", ROOT / "scripts/export_hf_sum_of_checks.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


@pytest.fixture(scope="module")
def rows():
    if not (ROOT / "data/endoscapes/val").exists() or not (ROOT / "data/frames").exists():
        pytest.skip("External source images are not linked")
    return exporter.build_rows(ROOT / "data/endoscapes", ROOT / "data/CVS_Challenge_SAGES_v1", ROOT / "data/frames")


def test_subset_and_provenance(rows):
    assert {k: len(v) for k, v in rows.items()} == {"endoscapes": 53, "sages": 60}
    assert sum(r["is_paper_few_shot"] for rs in rows.values() for r in rs) == 4
    assert len({r["example_id"] for rs in rows.values() for r in rs}) == 113
    assert {r["annotation_rubric_version"] for r in rows["endoscapes"]} == {"cvs_rubrics_v1"}
    assert {r["annotation_rubric_version"] for r in rows["sages"]} == {"cvs_rubrics_v3"}
    for config, records in rows.items():
        for r in records:
            assert r["rubric_version"] == "cvs_rubrics_v3"
            assert len(r["rubric_labels"]) == 19
            assert not Path(r["image"]["path"]).is_absolute()
            original = json.loads(r["original_cvs_annotation_json"])
            if config == "endoscapes":
                assert list(r["cvs_labels"].values()) == original["ds"]
            else:
                for c in exporter.CRITERIA:
                    assert r["cvs_rater_labels"][c] == [int(original[f"{c}_rater{i}"]) for i in (1, 2, 3)]
                    assert r["cvs_labels"][c] == int(sum(r["cvs_rater_labels"][c]) >= 2)


def test_export_round_trip_and_corruption_detection(rows, tmp_path):
    from datasets import Dataset
    for config, records in rows.items():
        split = exporter.SOURCES[config][0]
        (tmp_path / config).mkdir()
        ds = Dataset.from_list(records, features=exporter.features(list(records[0]["rubric_labels"])))
        ds.to_parquet(str(tmp_path / config / f"{split}.parquet"))
    (tmp_path / "README.md").write_text(exporter.dataset_card(rows, exporter.REPO_ID))
    exporter.validate(str(tmp_path), rows)
    changed = {k: [dict(r) for r in v] for k, v in rows.items()}
    changed["endoscapes"][0]["image_sha256"] = "invalid"
    with pytest.raises(ValueError, match="Changed image"):
        exporter.validate(str(tmp_path), changed)
