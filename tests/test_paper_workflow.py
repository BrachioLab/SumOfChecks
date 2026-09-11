import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("paper_runner", ROOT / "scripts/rubric_oracle_reasoning_frame.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_frozen_predictions_and_annotations():
    entries = json.loads((ROOT / "docs/prediction_manifest.json").read_text())
    assert len(entries) == 70
    for entry in entries:
        path = ROOT / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == entry["rows"]
        assert len({(r["video_id"], r["frame_id"]) for r in rows}) == entry["unique_frames"]
    sources = json.loads((ROOT / "docs/source_manifest.json").read_text())["files"]
    for entry in sources:
        if entry["path"].startswith("annotations/"):
            assert hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() == entry["source_sha256"]


def test_paper_exemplars_and_binary_aggregation():
    _, _, items = runner.load_rubrics(runner.DEFAULT_RUBRIC_PATH)
    assert len(items) == 19
    assert runner.compute_weighted_score(items, dict.fromkeys(items, "uncertain")) == dict.fromkeys(["c1", "c2", "c3"], 0)
    assert runner.compute_weighted_score(items, dict.fromkeys(items, "yes")) == pytest.approx(dict.fromkeys(["c1", "c2", "c3"], 1))
    rows = runner.load_few_shot_rows(runner.DEFAULT_FEW_SHOT_JSONL_ENDO_V3)
    assert [r["gt_pattern"] for r in rows] == ["000", "111", "110", "001"]
    for row in rows:
        with Image.open(runner.resolve_few_shot_image_path(row, default_split="val")) as image:
            image.verify()
    manifest = json.loads((ROOT / "docs/prediction_manifest.json").read_text())
    file = next(e for e in manifest if e["method"] == "Sum-of-Checks" and "opus" in e["model"] and e["run"] == 1)
    row = next(r for r in runner.iter_jsonl(ROOT / file["path"]) if (r["video_id"], r["frame_id"]) == ("184", 36750))
    assert runner.compute_weighted_score(items, row["rubric_labels_stage1"]) == pytest.approx({"c1": 1, "c2": .6, "c3": .9})
    assert row["pred"]["c2"] == 0


def test_all_evaluation_images():
    if not (ROOT / "data/endoscapes/test").exists():
        pytest.skip("External Endoscapes dataset is not linked")
    ds = runner.get_dataset("endoscapes", data_root=ROOT / "data/endoscapes", manifest_jsonl=ROOT / "data/manifests/endoscapes/test_dev.jsonl", manifest_only=True)
    rows = runner.get_frame_rows(ds)
    assert len(rows) == 791
    assert len({r["video_id"] for r in rows}) == 20
    for row in rows:
        with Image.open(ROOT / "data/endoscapes/test" / f"{row['video_id']}_{row['frame_id']}.jpg") as image:
            image.verify()


def test_frozen_output_directory_is_protected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["runner", "--output_root", str(ROOT / "outputs/rubric_oracle")])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


@pytest.mark.parametrize("model", ["gpt-4.1-mini", "claude-haiku-4-5-20251001", "claude-opus-4-5-20251101"])
def test_provider_request_format(monkeypatch, model):
    import models
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-placeholder")
    adapter = models.create_model(model, use_cache=False)
    calls = []
    def respond(**kwargs):
        calls.append(kwargs)
        if model.startswith("gpt"):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"pred":{}}'))])
        return SimpleNamespace(content=[SimpleNamespace(text='{"pred":{}}')])
    endpoint = adapter.client.chat.completions if model.startswith("gpt") else adapter.client.messages
    monkeypatch.setattr(endpoint, "create", respond)
    image = Image.new("RGB", (16, 12), (10, 20, 30))
    assert adapter([("Assess CVS", image)], system_prompt="Surgical assessment") == ['{"pred":{}}']
    assert calls[0]["model"] == model
    assert calls[0]["temperature"] == .1
    content = calls[0]["messages"][-1]["content"]
    encoded = content[1]["image_url"]["url"].split(",", 1)[1] if model.startswith("gpt") else content[1]["source"]["data"]
    decoded = Image.open(io.BytesIO(base64.b64decode(encoded)))
    np.testing.assert_array_equal(np.asarray(decoded), np.asarray(image))
    adapter.client.close()


@pytest.mark.parametrize("preset,fs", [(p,fs) for p in ["direct", "main", "self_rubric"] for fs in [False,True]])
def test_inference_and_resume(monkeypatch, tmp_path, preset, fs):
    if not (ROOT / "data/endoscapes/test").exists():
        pytest.skip("External Endoscapes dataset is not linked")
    calls = []
    _, _, items = runner.load_rubrics(runner.DEFAULT_RUBRIC_PATH)
    labels = dict.fromkeys(items, "uncertain")
    labels["C1-1"] = "yes"
    response = {"pred": {"c1": .2, "c2": .3, "c3": .4}, "rubrics_block": [],
                "rubric_labels": labels,
                "checklist": {c: [{"question": "Visible?", "answer": "yes", "rationale": "Visible"}] for c in ["C1", "C2", "C3"]}}
    if preset == "direct":
        response = {"pred": response["pred"]}
    def fake_model(prompts, **kwargs):
        calls.append(prompts)
        return [json.dumps(response)]
    monkeypatch.setattr(runner.models, "create_model", lambda *a, **kw: fake_model)
    argv = ["runner", "--preset", preset, "--manifest_jsonl", str(ROOT / "data/manifests/endoscapes/test_dev.jsonl"),
            "--use_manifest_rows", "--max_items", "1", "--run", "1", "--output_root", str(tmp_path)]
    if fs:
        argv += ["--few_shot", "--few_shot_k", "4"]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    files = list(tmp_path.rglob("*.jsonl"))
    assert len(files) == 1
    rows = list(runner.iter_jsonl(files[0]))
    assert len(rows) == 1 and rows[0]["parse_ok"]
    if preset == "main":
        assert rows[0]["weighted_score"] == pytest.approx({"c1": .1, "c2": 0, "c3": 0})
    n_calls = len(calls)
    runner.main()
    assert len(calls) == n_calls
    assert len(list(runner.iter_jsonl(files[0]))) == 1
