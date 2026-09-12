import json
import sqlite3
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from vloop.approval import approved_annotation
from vloop.autolabel import PredictionStore
from vloop.config import load_config
from vloop.ingest import ingest
from vloop.labels import encode_mask
from vloop.review import (
    audit_changed_reviews,
    change_reviews,
    ensure_review_schema,
    initialize_sample,
    load_review_dataset,
    prepare_review,
)
from vloop.review_audit import review_audit
from vloop.review_batch import _apply_one, _inspect, policy_decision, review_batch
from vloop.runtime import sha256_file, write_json


def run(path):
    cfg = load_config(path)
    for index in range(12):
        Image.new("RGB", (13, 7), (index * 20, 50, 100)).save(cfg.image_dir / f"{index}.png")
    assert ingest(cfg)["status"] == "completed"
    dataset = load_review_dataset(cfg)
    job = "autolabel_20260906T000000_87654321"
    field = f"pred_{job}"
    store = PredictionStore(cfg, field, job, {"schema_version": 1})
    ids = dataset.values("id")
    checksums = {}
    for index, sample in enumerate(dataset):
        mask = np.zeros((7, 13), dtype=bool)
        mask[1:6, 2:9] = True
        instances = (
            []
            if index == 2
            else [
                {
                    "class_id": 7,
                    "class_name": "unknown" if index == 7 else "test-object",
                    "prompt": "test object",
                    "confidence": 0.6 if index == 1 else 0.99,
                    "bbox_xywh": [2, 1, 7, 5],
                    "segmentation": encode_mask(mask),
                    "area": int(mask.sum()),
                }
            ]
        )
        prediction = {
            "image_id": sample["image_id"],
            "image_path": sample.filepath,
            "width": 13,
            "height": 7,
            "instances": instances,
        }
        result = cfg.storage_dir / "runs" / job / "predictions" / f"{sample['image_id']}.json"
        write_json(result, prediction)
        checksums[sample["image_id"]] = sha256_file(result)
        store.put(prediction, checksums[sample["image_id"]])
    ensure_review_schema(dataset, cfg)
    import fiftyone as fo

    dataset.add_sample_field("scene", fo.StringField)
    selected_group = next(
        str(i)
        for i in range(1000)
        if policy_decision(f"group:{i}", [], minimum=0.9, sample_rate=0.2, seed=cfg.seed)
        == "sample"
    )
    other_group = next(
        str(i)
        for i in range(1000)
        if policy_decision(f"group:{i}", [], minimum=0.9, sample_rate=0.2, seed=cfg.seed)
        != "sample"
    )
    for index, sample_id in enumerate(ids):
        sample = dataset[sample_id]
        sample["scene"] = selected_group if index < 3 else other_group
        sample.save()
    grouped = review_batch(
        replace(cfg, release_group_field="scene"),
        job_id=job,
        minimum=0.9,
        sample_rate=0.2,
        actor="group-test",
    )
    assert grouped["status"] == "ready", grouped
    assert grouped["policy"]["sample_group_field"] == "scene"
    assert grouped["decisions"]["sample"] == 3  # High, low and empty predictions together.
    assert grouped["decisions"]["accept"] == 8
    with sqlite3.connect(Path(grouped["result_dir"]) / "samples.sqlite3") as connection:
        assert all(
            connection.execute("SELECT decision FROM inputs WHERE sample_id=?", (sid,)).fetchone()[
                0
            ]
            == "sample"
            for sid in ids[:3]
        )
    manual = dataset[ids[3]]
    manual["ground_truth"] = manual[field].copy()
    manual.save()
    edited = dataset[ids[4]]
    assert initialize_sample(cfg, dataset, edited, job)
    changed = edited["ground_truth"].detections[0].mask.copy()
    changed[0, 0] = False
    edited["ground_truth"].detections[0].mask = changed
    edited.save()
    inspected = 0

    def interrupt_preview(*args):
        nonlocal inspected
        inspected += 1
        if inspected == 2:
            raise KeyboardInterrupt
        return _inspect(*args)

    with patch("vloop.review_batch._inspect", interrupt_preview):
        partial = review_batch(
            cfg, job_id=job, minimum=0.9, sample_rate=0, actor="batch-test", limit=8, batch_size=2
        )
    assert partial["status"] == "interrupted", partial
    blocked = review_batch(cfg, resume=partial["job_id"], apply=True)
    assert blocked["status"] == "failed" and "Finish the preview" in blocked["error"]
    preview = review_batch(cfg, resume=partial["job_id"], batch_size=2)
    assert preview["status"] == "ready", preview
    assert preview["decisions"] == {
        "accept": 3,
        "sample": 0,
        "empty": 1,
        "low_confidence": 1,
        "preserve": 2,
        "error": 1,
    }, preview
    assert all(sample["review_status"] == "unreviewed" for sample in dataset)
    assert dataset[ids[0]]["ground_truth"] is None
    later = dataset[ids[5]]
    later["ground_truth"] = later[field].copy()
    later.save()
    later_truth = later["ground_truth"].to_json()
    raised = False

    def crash_after_db(*args, **kwargs):
        nonlocal raised
        result = _apply_one(*args, **kwargs)
        if not raised:
            raised = True
            raise KeyboardInterrupt
        return result

    with patch("vloop.review_batch._apply_one", crash_after_db):
        interrupted = review_batch(cfg, resume=preview["job_id"], apply=True, batch_size=2)
    assert interrupted["status"] == "interrupted", interrupted
    # Resume in a fresh process; DB success before progress commit must not cause re-adoption.
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "vloop",
            "review-batch",
            "--config",
            str(path),
            "--resume",
            preview["job_id"],
            "--apply",
            "--batch-size",
            "2",
        ],
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    report = json.loads((Path(preview["result_dir"]) / "report.json").read_text())
    assert report["applied"] == 8 and report["outcomes"]["recovered"] == 1, report
    later.reload()
    assert later["ground_truth"].to_json() == later_truth
    accepted = dataset[ids[0]]
    accepted.reload()
    assert accepted["review_status"] == "auto_accepted"
    assert accepted["review_kind"] == "automatic"
    assert len([e for e in accepted["review_history"] if e["action"] == "auto_accept"]) == 1
    try:
        approved_annotation(cfg, accepted)
        raise AssertionError("Automatic label was accepted as manually reviewed")
    except ValueError:
        pass
    assert approved_annotation(cfg, accepted, allow_auto=True)["instances"]
    assert (cfg.storage_dir / "reviews" / "records.sqlite3").is_file()
    assert not list((cfg.storage_dir / "reviews").glob("*/*.json"))
    assert dataset[ids[1]]["review_queue_reason"] == "low_confidence"
    assert dataset[ids[2]]["review_queue_reason"] == "empty"
    assert dataset[ids[1]]["ground_truth"] is None
    prepared = prepare_review(cfg, job_id=job, limit=1, queue="low_confidence")
    assert prepared["initialized"] == 1, prepared
    assert dataset[ids[1]]["ground_truth"] is not None
    assert dataset[ids[2]]["ground_truth"] is None
    assert change_reviews(cfg, [ids[1]], "exclude", "batch-test")["changed"] == 1
    assert len(dataset.load_saved_view("vloop-queue-low_confidence")) == 0
    heldout = review_batch(
        cfg, job_id=job, minimum=0.9, sample_rate=1, actor="batch-test", batch_size=2
    )
    assert heldout["decisions"]["sample"] >= 4, heldout
    applied = review_batch(cfg, resume=heldout["job_id"], apply=True, batch_size=2)
    assert applied["status"] == "completed", applied
    assert len(dataset.load_saved_view("vloop-queue-sample")) >= 4

    # Unchanged approvals are skipped using only indexed metadata; one changed image is rehashed.
    cfg = replace(cfg, review_audit_batch_size=1)

    class AuditClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz) + timedelta(seconds=10)

    with (
        patch("vloop.review.datetime", AuditClock),
        patch("vloop.review.approved_annotation", wraps=approved_annotation) as verify,
    ):
        assert audit_changed_reviews(cfg, dataset) == 0
        assert audit_changed_reviews(cfg, dataset) == 0
        assert verify.call_count == 0
    accepted.reload()
    mask = accepted["ground_truth"].detections[0].mask.copy()
    mask[0, 0] = False
    accepted["ground_truth"].detections[0].mask = mask
    accepted.save()
    with (
        patch("vloop.review.datetime", AuditClock),
        patch("vloop.review.approved_annotation", wraps=approved_annotation) as verify,
    ):
        assert audit_changed_reviews(cfg, dataset) == 1
        assert verify.call_count == 1
    accepted.reload()
    assert accepted["review_status"] == "in_progress"

    other = dataset[ids[6]]
    other.reload()
    assert other["review_status"] == "auto_accepted"
    Path(other.filepath).write_bytes(b"changed outside the database")
    audited = review_audit(cfg)
    assert audited["status"] == "completed" and audited["invalidated"] == 1, audited
    assert review_audit(cfg, resume=audited["job_id"])["checked"] == audited["checked"]
    for image_id, checksum in checksums.items():
        assert (
            sha256_file(cfg.storage_dir / "runs" / job / "predictions" / f"{image_id}.json")
            == checksum
        )
    manifest_path = Path(heldout["result_dir"]) / "manifest.json"
    tampered = json.loads(manifest_path.read_text())
    tampered["policy"]["minimum"] = 0
    write_json(manifest_path, tampered)
    rejected = review_batch(cfg, resume=heldout["job_id"], apply=True)
    assert rejected["status"] == "failed" and "policy was modified" in rejected["error"]
    with sqlite3.connect(Path(preview["result_dir"]) / "samples.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM inputs WHERE applied = 0").fetchone()[0] == 0
    print(
        json.dumps(
            {
                "status": "passed",
                "checks": [
                    "preview",
                    "policy_queue",
                    "manual_preservation",
                    "fresh_process_resume_after_db_commit",
                    "explicit_auto_opt_in",
                    "bounded_preparation",
                    "unchanged_approval_no_hashing",
                    "incremental_edit_audit",
                    "full_image_audit",
                    "policy_tampering",
                ],
            }
        )
    )
    dataset.delete()


if __name__ == "__main__":
    run(Path(sys.argv[1]))
