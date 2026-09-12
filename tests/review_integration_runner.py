"""Run with an isolated database so FiftyOne's process-wide configuration is contained."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from PIL import Image

from vloop.approval import approved_annotation
from vloop.autolabel import PredictionStore
from vloop.config import load_config
from vloop.ingest import ingest
from vloop.labels import encode_mask
from vloop.review import (
    _save_change,
    change_reviews,
    load_review_dataset,
    prepare_review,
)
from vloop.review_audit import review_audit
from vloop.runtime import sha256_file, write_json


def run(config_path):
    path = Path(config_path)
    data = yaml.safe_load(path.read_text())
    data["classes"].append({"id": 42, "name": "other-object", "prompts": ["other object"]})
    path.write_text(yaml.safe_dump(data))
    cfg = load_config(path)
    for index, color in enumerate(("red", "blue", "green", "yellow")):
        Image.new("RGB", (13, 7), color).save(cfg.image_dir / f"{index}.png")
    assert ingest(cfg)["status"] == "completed"
    dataset = load_review_dataset(cfg)
    ids = dataset.values("id")
    job_id = "autolabel_20260906T000000_12345678"
    field = f"pred_{job_id}"
    store = PredictionStore(cfg, field, job_id, {"schema_version": 1})
    for index, sample in enumerate(dataset):
        if index >= 2:
            continue
        mask = np.zeros((7, 13), dtype=bool)
        mask[1:5, 2:8] = True
        mask[2:4, 4:6] = False
        instances = (
            [
                {
                    "class_id": 7,
                    "class_name": "test-object",
                    "prompt": "test object",
                    "confidence": 0.9,
                    "bbox_xywh": [2, 1, 6, 4],
                    "segmentation": encode_mask(mask),
                    "area": int(mask.sum()),
                }
            ]
            if index == 0
            else []
        )
        prediction = {
            "image_id": sample["image_id"],
            "image_path": sample.filepath,
            "width": 13,
            "height": 7,
            "instances": instances,
        }
        result_path = (
            cfg.storage_dir / "runs" / job_id / "predictions" / f"{sample['image_id']}.json"
        )
        write_json(result_path, prediction)
        store.put(prediction, sha256_file(result_path))
    original = dataset[ids[0]][field].to_json()
    existing = dataset[ids[3]]
    existing["ground_truth"] = dataset[ids[0]][field].copy()
    existing.save()
    report = prepare_review(cfg)
    assert report["status"] == "completed", report
    assert (report["initialized"], report["unavailable"]) == (2, 0), report
    dataset.reload()
    first, empty, missing, existing = [dataset[sample_id] for sample_id in ids]
    assert existing["review_initialized"]
    existing["ground_truth"] = None
    existing.save()
    assert first["ground_truth"].detections[0].id != first[field].detections[0].id
    assert first["review_status"] == "unreviewed"
    assert first["review_approved_hash"] is None
    assert empty["ground_truth"].detections == []
    assert missing["ground_truth"] is None
    assert set(dataset.active_label_schemas) >= {"ground_truth", "review_status"}
    assert dataset.label_schemas[field]["read_only"]
    assert len(dataset.list_saved_views()) == 8

    first["ground_truth"].detections[0].label = "other-object"
    first.save()
    result = change_reviews(cfg, [ids[0]], "complete", "integration-test")
    assert result["changed"] == 1, result
    first.reload()
    assert approved_annotation(cfg, first)["instances"][0]["class_id"] == 42
    old_approval = first["review_approval_id"]
    assert change_reviews(cfg, [ids[1]], "complete", "test")["failed"] == 1
    assert change_reviews(cfg, [ids[1]], "complete", "test", confirm_empty=True)["changed"] == 1
    empty.reload()
    assert approved_annotation(cfg, empty)["instances"] == []
    assert len(dataset.load_saved_view("vloop-completed")) == 2

    edited_mask = first["ground_truth"].detections[0].mask.copy()
    edited_mask[0, 0] = False
    first["ground_truth"].detections[0].mask = edited_mask
    first.save()
    try:
        approved_annotation(cfg, first)
        raise AssertionError("Changed mask was incorrectly accepted")
    except ValueError as exc:
        assert "changed" in str(exc)
    audit = review_audit(cfg)
    assert audit["status"] == "completed" and audit["invalidated"] == 1, audit
    first.reload()
    assert first["review_status"] == "in_progress"
    assert first["review_approved_hash"] is None
    assert (cfg.storage_dir / "reviews" / first["image_id"] / f"{old_approval}.json").is_file()
    assert change_reviews(cfg, [ids[0]], "complete", "test")["changed"] == 1
    first.reload()
    assert first["review_approval_id"] != old_approval
    assert len(approved_annotation(cfg, first)["instances"]) == 1

    assert change_reviews(cfg, [ids[2]], "complete", "test", confirm_empty=True)["failed"] == 1
    assert change_reviews(cfg, [ids[2]], "start", "test")["changed"] == 1
    assert change_reviews(cfg, [ids[2]], "complete", "test", confirm_empty=True)["changed"] == 1
    assert change_reviews(cfg, [ids[2]], "exclude", "test", note="unusable")["changed"] == 1
    missing.reload()
    assert missing["review_status"] == "excluded"
    assert missing["ground_truth"].detections == []
    assert missing["review_approved_hash"] is None
    again = prepare_review(cfg, job_id=job_id)
    assert again["initialized"] == 0, again
    existing.reload()
    assert existing["ground_truth"] is None
    first.reload()
    assert first[field].to_json() == original
    assert first["ground_truth"].detections[0].label == "other-object"

    # Native annotation edits that race an approval cannot be overwritten by it.
    raw = first.to_mongo_dict(include_id=True)
    dataset._sample_collection.update_one(
        {"_id": raw["_id"]}, {"$set": {"ground_truth.detections.0.label": "test-object"}}
    )
    try:
        _save_change(dataset, first, {"review_status": "completed"}, {"action": "stale"})
        raise AssertionError("Stale change was incorrectly accepted")
    except RuntimeError as exc:
        assert "changed during" in str(exc)
    audit = review_audit(cfg)
    assert audit["status"] == "completed" and audit["invalidated"] == 1, audit

    from fiftyone.plugins.context import build_plugin_contexts

    from vloop.review_operators import CompleteReview, selected_samples

    contexts = build_plugin_contexts()
    plugin = next(context for context in contexts if context.name == "@vloop/review")
    assert not plugin.errors, plugin.errors
    assert len(plugin.instances) == 3
    operator = CompleteReview()
    context = SimpleNamespace(
        dataset=dataset, current_sample=ids[0], selected=[ids[1]], params={"confirm": False}
    )
    assert selected_samples(context) == [ids[0]]
    assert operator.resolve_input(context).to_json()
    try:
        operator.execute(context)
        raise AssertionError("Approval without confirmation was accepted")
    except ValueError:
        pass
    print(
        json.dumps(
            {
                "status": "passed",
                "checks": [
                    "schema",
                    "copy_preservation",
                    "empty_vs_missing",
                    "approve",
                    "mask_edit_invalidation",
                    "reapprove",
                    "exclude",
                    "concurrent_edit",
                    "plugin_registration",
                ],
            }
        )
    )
    dataset.delete()


if __name__ == "__main__":
    run(sys.argv[1])
