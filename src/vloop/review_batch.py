"""Disk-backed, resumable pseudo-label adoption with bounded memory."""

import hashlib
import json
import math
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from .approval import annotation_content, content_hash
from .config import config_from_dict
from .labels import to_detections
from .review import _save_change, ensure_review_schema, load_review_dataset
from .review_store import connect_records, save_record
from .runtime import finish_run, project_lock, sha256_file, start_run, write_json

DECISIONS = ("accept", "sample", "low_confidence", "empty", "preserve", "error")


def _implementation():
    root = Path(__file__).parent
    return {
        name: sha256_file(root / name)
        for name in ("review_batch.py", "approval.py", "labels.py", "review.py", "review_store.py")
    }


def policy_decision(image_id, confidences, *, minimum, sample_rate, seed):
    # Use a separate hash domain from release split assignment, including for groups.
    value = int(hashlib.sha256(f"review-sample-v2:{seed}:{image_id}".encode()).hexdigest()[:16], 16)
    if value / 2**64 < sample_rate:
        return "sample"
    if not confidences:
        return "empty"
    if any(
        value is None or not math.isfinite(value) or not 0 <= value <= 1 or value < minimum
        for value in confidences
    ):
        return "low_confidence"
    return "accept"


def sampling_key(cfg, sample):
    if cfg.release_group_field:
        group = sample[cfg.release_group_field]
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Configured group field must be a nonempty string")
        return "group:" + group
    return "image:" + sample["image_id"]


def _connect(path):
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _digest(connection):
    digest = hashlib.sha256()
    for row in connection.execute(
        "SELECT sample_id, image_id, prediction_sha256, revision FROM inputs ORDER BY sample_id"
    ):
        digest.update((json.dumps(list(row)) + "\n").encode())
    return digest.hexdigest()


def _snapshot(dataset, connection, field, limit):
    connection.executescript("""
        CREATE TABLE inputs (
            sample_id TEXT PRIMARY KEY, image_id TEXT NOT NULL,
            prediction_sha256 TEXT NOT NULL, revision TEXT NOT NULL,
            decision TEXT, detail TEXT, applied INTEGER NOT NULL DEFAULT 0,
            outcome TEXT, record_id TEXT
        );
        CREATE INDEX pending_preview ON inputs(decision, sample_id);
        CREATE INDEX pending_apply ON inputs(applied, sample_id);
    """)
    query = {f"{field}_sha256": {"$ne": None}, "review_status": {"$in": [None, "unreviewed"]}}
    upper = dataset._sample_collection.find_one(query, {"_id": 1}, sort=[("_id", -1)])
    if not upper:
        return
    query["_id"] = {"$lte": upper["_id"]}
    projection = {"image_id": 1, f"{field}_sha256": 1, "last_modified_at": 1}
    cursor = dataset._sample_collection.find(query, projection).sort("_id", 1).batch_size(500)
    if limit:
        cursor = cursor.limit(limit)
    try:
        with connection:
            for row in cursor:
                connection.execute(
                    "INSERT INTO inputs VALUES (?, ?, ?, ?, NULL, NULL, 0, NULL, NULL)",
                    (
                        str(row["_id"]),
                        row["image_id"],
                        row[f"{field}_sha256"],
                        row["last_modified_at"].isoformat(),
                    ),
                )
    finally:
        cursor.close()


def _inspect(cfg, dataset, row, policy):
    if not re.fullmatch(r"[0-9a-f]{64}", row["image_id"]):
        raise ValueError("Invalid image ID in the batch snapshot")
    sample = dataset[row["sample_id"]]
    sample.reload()
    raw = sample.to_mongo_dict(include_id=True)
    if raw["last_modified_at"].isoformat() != row["revision"]:
        return sample, None, None, "preserve", "Image changed after the batch snapshot"
    if sample["review_status"] not in (None, "unreviewed"):
        return sample, None, None, "preserve", "Existing review decision"
    source = sample["ground_truth_source"] or {}
    truth = sample["ground_truth"]
    if sample["review_initialized"] and truth is None:
        return sample, None, None, "preserve", "Previously initialized ground truth was cleared"
    if truth is not None and (
        source.get("kind") != "autolabel" or source.get("job_id") != policy["source_job_id"]
    ):
        return sample, None, None, "preserve", "Existing manual or different-source ground truth"
    if any(event.get("action") != "initialize" for event in (sample["review_history"] or [])):
        return sample, None, None, "preserve", "Existing review history"
    field = f"pred_{policy['source_job_id']}"
    path = (
        cfg.storage_dir
        / "runs"
        / policy["source_job_id"]
        / "predictions"
        / f"{row['image_id']}.json"
    )
    if (
        sample[f"{field}_sha256"] != row["prediction_sha256"]
        or sha256_file(path) != row["prediction_sha256"]
    ):
        raise ValueError("Prediction changed after the batch snapshot")
    prediction = json.loads(path.read_text())
    if (
        sample["image_id"] != row["image_id"]
        or prediction["image_id"] != row["image_id"]
        or prediction["width"] != sample.metadata.width
        or prediction["height"] != sample.metadata.height
    ):
        raise ValueError("Prediction and registered image coordinates differ")
    predicted_truth = to_detections(prediction)

    class Annotation(dict):
        metadata = sample.metadata

    content = annotation_content(
        Annotation(
            image_id=sample["image_id"],
            managed_sha256=sample["managed_sha256"],
            ground_truth=predicted_truth,
        ),
        cfg.classes,
    )
    if truth is not None and content_hash(annotation_content(sample, cfg.classes)) != content_hash(
        content
    ):
        return sample, None, None, "preserve", "Ground truth differs from the original prediction"
    decision = policy_decision(
        sampling_key(cfg, sample),
        [item.get("confidence") for item in prediction["instances"]],
        minimum=policy["minimum"],
        sample_rate=policy["sample_rate"],
        seed=policy["seed"],
    )
    return sample, predicted_truth, content, decision, None


def _apply_one(cfg, dataset, records, row, policy, batch_id, connection):
    # DB update may have succeeded just before a crash prevented the progress commit.
    sample = dataset[row["sample_id"]]
    sample.reload()
    if sample["review_batch_id"] == batch_id:
        return "recovered", None
    sample, truth, content, decision, detail = _inspect(cfg, dataset, row, policy)
    if decision == "preserve":
        return "preserved", detail
    if decision != row["decision"]:
        raise ValueError("Prediction no longer matches the preview decision")
    now = datetime.now(timezone.utc)
    source = {
        "kind": "autolabel",
        "job_id": policy["source_job_id"],
        "field": f"pred_{policy['source_job_id']}",
        "sha256": row["prediction_sha256"],
    }
    changes = {"review_batch_id": batch_id, "review_queue_job": policy["source_job_id"]}
    event = {"action": "queue", "batch_id": batch_id, "at": now.isoformat(), "reason": decision}
    if decision == "accept":
        if sha256_file(Path(sample.filepath)) != sample["managed_sha256"]:
            raise ValueError("Registered image content changed")
        record = {
            "schema_version": 1,
            "record_id": uuid4().hex,
            "image_id": sample["image_id"],
            "action": "auto_accept",
            "reviewer": policy["actor"],
            "at": now.isoformat(),
            "annotation": content,
            "label_hash": content_hash(content),
            "source": source,
            "batch_id": batch_id,
            "policy": policy,
            "review_kind": "automatic",
        }
        checksum = save_record(records, record)
        with connection:
            connection.execute(
                "UPDATE inputs SET record_id = ? WHERE sample_id = ?",
                (record["record_id"], row["sample_id"]),
            )
        changes.update(
            ground_truth=truth.to_mongo().to_dict(),
            ground_truth_source=source,
            review_initialized=True,
            review_status="auto_accepted",
            review_kind="automatic",
            review_approved_hash=record["label_hash"],
            review_approval_id=record["record_id"],
            review_approval_sha256=checksum,
            reviewed_by=policy["actor"],
            reviewed_at=now,
            review_reason="Automatic prediction adopted by batch policy",
            review_queue_reason=None,
        )
        event.update(action="auto_accept", record_id=record["record_id"], reviewer=policy["actor"])
    else:
        # Queue metadata is small; masks are initialized only when a review page is prepared.
        changes.update(review_queue_reason=decision)
    _save_change(dataset, sample, changes, event)
    return "accepted" if decision == "accept" else "queued", None


def _counts(connection):
    counts = dict.fromkeys(DECISIONS, 0)
    pending = 0
    for decision, count in connection.execute(
        "SELECT decision, COUNT(*) FROM inputs GROUP BY decision"
    ):
        if decision is None:
            pending = count
        else:
            counts[decision] = count
    outcomes = dict(
        connection.execute(
            "SELECT outcome, COUNT(*) FROM inputs WHERE applied = 1 GROUP BY outcome"
        )
    )
    return {
        "decisions": counts,
        "pending_preview": pending,
        "outcomes": outcomes,
        "total": pending + sum(counts.values()),
        "applied": sum(outcomes.values()),
    }


def _process(cfg, dataset, connection, records, directory, report, policy, batch_size, apply):
    report.update(_counts(connection))
    while True:
        query = (
            "SELECT * FROM inputs WHERE "
            + ("applied = 0" if apply else "decision IS NULL")
            + " ORDER BY sample_id LIMIT ?"
        )
        rows = connection.execute(query, (batch_size,)).fetchall()
        if not rows:
            return
        # Release the project lock between batches so manual operators can make progress.
        with project_lock(cfg):
            for row in rows:
                if apply:
                    try:
                        if row["decision"] in ("preserve", "error"):
                            outcome, detail = "skipped", row["detail"]
                        else:
                            outcome, detail = _apply_one(
                                cfg, dataset, records, row, policy, report["job_id"], connection
                            )
                    except (ValueError, OSError, KeyError, RuntimeError) as exc:
                        outcome, detail = "failed", str(exc)
                    with connection:
                        connection.execute(
                            "UPDATE inputs SET applied = 1, outcome = ?, detail = ? "
                            "WHERE sample_id = ?",
                            (outcome, detail, row["sample_id"]),
                        )
                    report["applied"] += 1
                    report["outcomes"][outcome] = report["outcomes"].get(outcome, 0) + 1
                else:
                    try:
                        _, _, _, decision, detail = _inspect(cfg, dataset, row, policy)
                    except (ValueError, OSError, KeyError, RuntimeError) as exc:
                        decision, detail = "error", str(exc)
                    connection.execute(
                        "UPDATE inputs SET decision = ?, detail = ? WHERE sample_id = ?",
                        (decision, detail, row["sample_id"]),
                    )
                    report["decisions"][decision] += 1
                    report["pending_preview"] -= 1
            connection.commit()
        write_json(directory / "report.json", report)
        print(
            f"Batch {report['job_id']}: "
            f"preview {report['total'] - report['pending_preview']}/{report['total']}, "
            f"applied {report['applied']}/{report['total']}",
            flush=True,
        )


def review_batch(
    cfg,
    *,
    job_id=None,
    minimum=None,
    sample_rate=None,
    actor=None,
    resume=None,
    apply=False,
    batch_size=100,
    limit=None,
):
    if type(batch_size) is not int or not 1 <= batch_size <= 1000:
        raise ValueError("batch-size must be between 1 and 1000")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    if resume:
        if not re.fullmatch(r"review_batch_\d{8}T\d{6}_[0-9a-f]{8}", resume):
            raise ValueError("Use a review_batch job ID")
        if any(value is not None for value in (job_id, minimum, sample_rate, actor, limit)):
            raise ValueError(
                "Resume uses the frozen source, policy and actor; do not override them"
            )
        directory = cfg.storage_dir / "runs" / resume
    else:
        if apply:
            raise ValueError("Create and inspect a preview before --resume JOB_ID --apply")
        if not job_id or not re.fullmatch(r"autolabel_\d{8}T\d{6}_[0-9a-f]{8}", job_id):
            raise ValueError("An explicit autolabel --job-id is required")
        if minimum is None or not math.isfinite(minimum) or not 0 <= minimum <= 1:
            raise ValueError("An explicit min-confidence in [0, 1] is required")
        sample_rate = 0.01 if sample_rate is None else sample_rate
        if not math.isfinite(sample_rate) or not 0 <= sample_rate <= 1:
            raise ValueError("sample-rate must be in [0, 1]")
        if not actor or not actor.strip():
            raise ValueError("actor is required for automatic adoption provenance")
        directory, report = start_run(cfg, "review_batch")
    # Separate lock avoids two processes resuming the same job, without holding the project lock.
    with project_lock(SimpleNamespace(storage_dir=directory)):
        if resume:
            report = json.loads((directory / "report.json").read_text())
        report.update(status="running", mode="apply" if apply else "preview")
        report.pop("error", None)
        try:
            with closing(_connect(directory / "samples.sqlite3")) as connection:
                if not resume:
                    dataset = load_review_dataset(cfg)
                    manifest = dataset.info.get("vloop_autolabel_jobs", {}).get(job_id)
                    if manifest is None:
                        raise ValueError("Unknown source auto-labeling job")
                    with project_lock(cfg):
                        ensure_review_schema(dataset, cfg)
                    policy = {
                        "source_job_id": job_id,
                        "minimum": minimum,
                        "sample_rate": sample_rate,
                        "sampling_method": "sha256_all_candidates_v2",
                        "sample_group_field": cfg.release_group_field,
                        "seed": cfg.seed,
                        "actor": actor.strip(),
                        "empty": "manual_review",
                        "source_manifest_hash": content_hash(manifest),
                    }
                    _snapshot(dataset, connection, f"pred_{job_id}", limit)
                    write_json(
                        directory / "manifest.json",
                        {
                            "schema_version": 1,
                            "implementation": _implementation(),
                            "policy": policy,
                            "inputs_sha256": _digest(connection),
                            "config_sha256": sha256_file(directory / "config.json"),
                        },
                    )
                    report["manifest_sha256"] = sha256_file(directory / "manifest.json")
                    write_json(directory / "report.json", report)
                if sha256_file(directory / "manifest.json") != report["manifest_sha256"]:
                    raise ValueError("Frozen batch policy was modified")
                manifest = json.loads((directory / "manifest.json").read_text())
                if manifest.get("implementation") != _implementation():
                    raise ValueError("Batch implementation changed; create a new preview")
                if manifest["config_sha256"] != sha256_file(directory / "config.json") or manifest[
                    "inputs_sha256"
                ] != _digest(connection):
                    raise ValueError("Frozen batch configuration or inputs were modified")
                snapshot = json.loads((directory / "config.json").read_text())
                frozen = config_from_dict(snapshot, Path(snapshot.pop("config_path")))
                if frozen.storage_dir != cfg.storage_dir or frozen.dataset_name != cfg.dataset_name:
                    raise ValueError("Batch belongs to a different project")
                dataset = load_review_dataset(frozen)
                policy = manifest["policy"]
                if (
                    content_hash(dataset.info["vloop_autolabel_jobs"][policy["source_job_id"]])
                    != policy["source_manifest_hash"]
                ):
                    raise ValueError("Source auto-labeling provenance changed")
                report["policy"] = policy
                if (
                    apply
                    and connection.execute(
                        "SELECT 1 FROM inputs WHERE decision IS NULL LIMIT 1"
                    ).fetchone()
                ):
                    raise ValueError("Finish the preview with --resume JOB_ID before applying it")
                with closing(connect_records(frozen)) as records:
                    _process(
                        frozen,
                        dataset,
                        connection,
                        records,
                        directory,
                        report,
                        policy,
                        batch_size,
                        apply,
                    )
                report["status"] = "completed" if apply else "ready"
                if apply and report["outcomes"].get("failed", 0):
                    report["status"] = "completed_with_errors"
        except KeyboardInterrupt:
            report.update(status="interrupted", error="Interrupted; resume with the same mode")
        except Exception as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            if not (directory / "manifest.json").is_file():
                report["error"] += "; snapshot initialization is incomplete, create a new preview"
        report["retry"] = f"vloop review-batch --resume {report['job_id']}" + (
            " --apply" if apply else ""
        )
        return finish_run(directory, report)
