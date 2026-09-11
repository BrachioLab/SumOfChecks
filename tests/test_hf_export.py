import importlib.util
import json
import shutil
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
    for config, splits in exporter.partition_rows(rows).items():
        (tmp_path / config).mkdir()
        for split, records in splits.items():
            ds = Dataset.from_list(records, features=exporter.features(list(records[0]["rubric_labels"])))
            ds.to_parquet(str(tmp_path / config / f"{split}.parquet"))
    (tmp_path / "README.md").write_text(exporter.dataset_card(rows, exporter.REPO_ID))
    for relative in exporter.few_shot_assets(rows):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, tmp_path / relative)
    exporter.validate(str(tmp_path), rows)
    changed = {k: [dict(r) for r in v] for k, v in rows.items()}
    changed["endoscapes"][0]["image_sha256"] = "invalid"
    with pytest.raises(ValueError, match="Changed image"):
        exporter.validate(str(tmp_path), changed)
    (tmp_path / exporter.FEW_SHOT_MANIFEST).write_text("corrupted")
    with pytest.raises(ValueError, match="Changed few-shot asset"):
        exporter.validate(str(tmp_path), rows)


def test_few_shot_assets_match_paper(rows):
    assets = exporter.few_shot_assets(rows)
    assert len(assets) == 12
    selection = exporter.read_rows(ROOT / exporter.FEW_SHOT_MANIFEST)
    assert [r["gt_pattern"] for r in selection] == ["000", "111", "110", "001"]
    changed = {k: [dict(r) for r in v] for k, v in rows.items()}
    next(r for r in changed["endoscapes"] if r["is_paper_few_shot"])["rubric_labels"] = {}
    with pytest.raises(ValueError, match="Few-shot labels differ"):
        exporter.few_shot_assets(changed)


def test_disjoint_splits_and_v5_selection(rows):
    splits = exporter.partition_rows(rows)
    assert {c: {s: len(rs) for s, rs in ss.items()} for c, ss in splits.items()} == {
        "endoscapes": {"few_shot": 4, "dev": 49}, "sages": {"few_shot": 4, "dev": 56},
    }
    assert [r["gt_pattern"] for r in exporter.selected_examples("sages")] == ["001", "000", "101", "111"]
    for config, ss in splits.items():
        assert [r["few_shot_order"] for r in ss["few_shot"]] == list(range(4))
        assert all(r["is_few_shot"] for r in ss["few_shot"])
        assert all(not r["is_few_shot"] and r["few_shot_order"] == -1 for r in ss["dev"])
        for field in ("example_id", "image_sha256"):
            assert not {r[field] for r in ss["few_shot"]} & {r[field] for r in ss["dev"]}
        assert {r["example_id"] for rs in ss.values() for r in rs} == {r["example_id"] for r in rows[config]}
