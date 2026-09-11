import copy
import json
from contextlib import closing
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from vloop.cli import parse_args
from vloop.evaluation_data import comparison_id, freeze_inputs, serialize_prediction, settings
from vloop.evaluation_view import detections
from vloop.labels import decode_mask
from vloop.release_data import capture, connect, export_coco

CLASSES = [{"id": 7, "name": "test-object", "model_index": 0}]
METADATA = {
    "postprocessing": {
        "eval_confidence": 0.001,
        "display_confidence": 0.5,
        "eval_max_detections": 100,
        "num_select": 100,
    }
}


def output():
    mask = np.zeros((3, 13, 23), dtype=bool)
    mask[0, 1:9, 1:10] = True
    mask[0, 3:7, 3:8] = False
    mask[0, 10:12, 18:21] = True
    return SimpleNamespace(
        xyxy=np.array([[0.5, 0.5, 1.5, 1.5]] * 3),
        confidence=np.array([0.9, 0.001, 0.8]),
        class_id=np.array([0, 0, 0]),
        mask=mask,
    )


def serialize(pred, **kwargs):
    return serialize_prediction(
        pred, width=23, height=13, classes=CLASSES, parameters=settings(METADATA, **kwargs)
    )


def test_preserves_full_mask_float_box_holes_and_empty_mask():
    pred = output()
    result = serialize(pred)
    assert len(result["instances"]) == 2  # Strict threshold, and preserve the empty mask.
    assert result["instances"][0]["bbox_xywh"] == [0.5, 0.5, 1, 1]
    np.testing.assert_array_equal(
        decode_mask(result["instances"][0]["segmentation"], 23, 13), pred.mask[0]
    )
    fo = SimpleNamespace(
        Detection=lambda **kw: SimpleNamespace(**kw), Detections=lambda **kw: SimpleNamespace(**kw)
    )
    boxes = detections(fo, result, masks=False).detections
    masks = detections(fo, result, masks=True).detections
    assert boxes[0].bounding_box == [0.5 / 23, 0.5 / 13, 1 / 23, 1 / 13]
    assert masks[0].bounding_box == [1 / 23, 1 / 13, 20 / 23, 11 / 13]
    assert not masks[1].mask.any()
    assert serialize(pred, max_detections=1)["instances"] == result["instances"][:1]


@pytest.mark.parametrize(
    "field,value",
    [
        ("class_id", np.array([0, 0, 1])),
        ("confidence", np.array([np.nan, 0.2, 0.3])),
        ("mask", np.zeros((3, 23, 13), dtype=bool)),
        ("mask", None),
        ("xyxy", np.array([[3, 1, 2, 4]] * 3)),
    ],
)
def test_invalid_model_outputs_fail(field, value):
    pred = output()
    setattr(pred, field, value)
    with pytest.raises(ValueError):
        serialize(pred)


def test_empty_predictions_and_invalid_settings():
    pred = SimpleNamespace(
        xyxy=np.empty((0, 4)), confidence=np.empty(0), class_id=np.array([], dtype=int), mask=None
    )
    assert serialize(pred)["instances"] == []
    for kwargs in (
        {"confidence": float("nan")},
        {"confidence": 0.8},
        {"max_detections": 101},
        {"max_detections": 0},
    ):
        with pytest.raises(ValueError):
            settings(METADATA, **kwargs)


def test_frozen_selection_and_comparison(project, tmp_path):
    from test_release import make_sample

    snapshot = tmp_path / "release/metadata/snapshot.sqlite3"
    capture(project, iter([make_sample(project, 1), make_sample(project, 2, empty=True)]), snapshot)
    with closing(connect(snapshot)) as db:
        db.execute("UPDATE records SET split='val'")
        db.commit()
    info = export_coco(project, snapshot, snapshot.parent.parent)
    args = {"split": "val", "classes": info["classes"]}
    first = freeze_inputs(snapshot, tmp_path / "one.sqlite3", **args)
    second = freeze_inputs(snapshot, tmp_path / "two.sqlite3", **args)
    limited = freeze_inputs(snapshot, tmp_path / "small.sqlite3", limit=1, **args)
    params = settings(METADATA)
    assert comparison_id(first, params) == comparison_id(
        second, {**params, "display_confidence": 0.9}
    )
    assert comparison_id(first, params) != comparison_id(limited, params)
    assert comparison_id(first, params) != comparison_id(first, {**params, "max_detections": 1})
    changed = copy.deepcopy(first)
    changed["classes"][0]["name"] = "different"
    assert comparison_id(first, params) != comparison_id(changed, params)
    with closing(connect(tmp_path / "small.sqlite3")) as db:
        assert db.execute("SELECT count(*) FROM samples").fetchone()[0] == 1
        assert json.loads(db.execute("SELECT annotation FROM samples").fetchone()[0])["width"] == 23
    with pytest.raises(ValueError, match="no images"):
        freeze_inputs(snapshot, tmp_path / "empty.sqlite3", split="test", classes=CLASSES)


def test_cli_and_explicit_test_selection(project):
    from vloop.evaluate import evaluate

    args = parse_args(["evaluate", "--job-id", "train_id"])
    assert args.split is None and args.limit is None
    assert parse_args(["evaluate", "--view", "evaluate_id", "--no-browser"]).no_browser
    for arguments in (
        ["--view", "id", "--limit", "3"],
        ["--job-id", "id", "--no-browser"],
        ["--job-id", "id", "--view", "id"],
    ):
        with pytest.raises(SystemExit):
            parse_args(["evaluate", *arguments])
    with pytest.raises(ValueError, match="explicit --split test"):
        evaluate(replace(project, eval_split="test"), job_id="train_id")


def test_mlflow_initialization_failure_keeps_job(project, monkeypatch):
    import vloop.evaluate as module

    def fail(*_args, **_kwargs):
        raise RuntimeError("offline tracking")

    monkeypatch.setattr(module, "create_run", fail)
    report = module.evaluate(project, job_id="train_id")
    assert report["status"] == "failed" and "offline tracking" in report["error"]
    assert (project.storage_dir / "runs" / report["job_id"] / "error.txt").is_file()
