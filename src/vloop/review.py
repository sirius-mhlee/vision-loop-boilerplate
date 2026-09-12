import json
import re
import time
from datetime import datetime, timedelta, timezone
from importlib.resources import files
from pathlib import Path
from uuid import uuid4

from .approval import annotation_content, approved_annotation, content_hash, record_path
from .config import config_from_dict
from .fiftyone import configure_fiftyone
from .labels import to_detections
from .runtime import finish_run, project_lock, sha256_file, start_run, write_json

STATUSES = {
    "unreviewed": "미검수",
    "in_progress": "수정 중",
    "completed": "완료",
    "auto_accepted": "자동 예측 채택",
    "excluded": "제외",
}


def load_review_dataset(cfg):
    fo = configure_fiftyone(cfg)
    if not fo.dataset_exists(cfg.dataset_name):
        raise ValueError("Dataset missing; run vloop ingest first")
    dataset = fo.load_dataset(cfg.dataset_name)
    dataset.reload()
    if dataset.info.get("vloop_storage_dir") != str(cfg.storage_dir):
        raise ValueError("Dataset does not belong to this project")
    mapping = [{"id": item.id, "name": item.name} for item in cfg.classes]
    if not mapping or dataset.info.get("vloop_classes") != mapping:
        raise ValueError("Project classes differ from the registered dataset")
    return dataset


def config_for_dataset(dataset):
    snapshot = dict(dataset.info.get("vloop_review_config") or {})
    path = snapshot.pop("config_path", None)
    if path is None:
        raise ValueError("Open this project with vloop review first")
    return config_from_dict(snapshot, Path(path))


def install_review_plugin(cfg):
    target = cfg.storage_dir / "fiftyone/plugins/vloop-review"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("__init__.py", "fiftyone.yml"):
        (target / name).write_text(files("vloop.review_plugin").joinpath(name).read_text())


def ensure_review_schema(dataset, cfg):
    import fiftyone as fo

    fields = {
        "ground_truth": (fo.EmbeddedDocumentField, {"embedded_doc_type": fo.Detections}),
        "review_status": (fo.StringField, {}),
        "review_initialized": (fo.BooleanField, {}),
        "ground_truth_source": (fo.DictField, {}),
        "review_approved_hash": (fo.StringField, {}),
        "review_approval_id": (fo.StringField, {}),
        "review_approval_sha256": (fo.StringField, {}),
        "reviewed_by": (fo.StringField, {}),
        "reviewed_at": (fo.DateTimeField, {}),
        "review_reason": (fo.StringField, {}),
        "review_history": (fo.ListField, {"subfield": fo.DictField}),
        "review_kind": (fo.StringField, {}),
        "review_batch_id": (fo.StringField, {}),
        "review_queue_reason": (fo.StringField, {}),
        "review_queue_job": (fo.StringField, {}),
        "review_checked_at": (fo.DateTimeField, {}),
    }
    for name, (field_type, kwargs) in fields.items():
        if not dataset.has_sample_field(name):
            dataset.add_sample_field(name, field_type, **kwargs)
    label_schema = {
        "type": "detections",
        "component": "dropdown",
        "classes": [c.name for c in cfg.classes],
        "attributes": [],
    }
    dataset.update_label_schema("ground_truth", label_schema)
    dataset.update_label_schema(
        "review_status",
        {"type": "str", "component": "dropdown", "values": list(STATUSES), "read_only": True},
    )
    dataset.activate_label_schemas(["ground_truth", "review_status"])
    for job in dataset.info.get("vloop_autolabel_jobs", {}):
        field = f"pred_{job}"
        if dataset.has_sample_field(field):
            dataset.update_label_schema(field, dict(label_schema, read_only=True))
    dataset.classes["ground_truth"] = [c.name for c in cfg.classes]
    dataset.info["vloop_review_config"] = cfg.to_dict()
    if dataset.info.get("vloop_review_schema_version", 1) < 2:
        # Adopt legacy manual labels without transferring any masks to Python.
        dataset._sample_collection.update_many(
            {"review_initialized": {"$ne": True}, "ground_truth": {"$ne": None}},
            {"$set": {"review_initialized": True}},
        )
        dataset.info["vloop_review_schema_version"] = 2
    collection = dataset._sample_collection
    collection.create_index(
        [("review_status", 1), ("last_modified_at", 1), ("_id", 1), ("review_checked_at", 1)]
    )
    collection.create_index([("review_status", 1), ("_id", 1)])
    collection.create_index([("review_initialized", 1), ("_id", 1)])
    collection.create_index([("review_queue_reason", 1), ("_id", 1)])
    dataset.save()
    for status, title in STATUSES.items():
        name = f"vloop-{status}"
        if not dataset.has_saved_view(name):
            dataset.save_view(
                name, dataset.match(fo.ViewField("review_status") == status), description=title
            )
    for reason in ("sample", "low_confidence", "empty"):
        name = f"vloop-queue-{reason}"
        if not dataset.has_saved_view(name):
            dataset.save_view(name, dataset.match(fo.ViewField("review_queue_reason") == reason))


def _save_change(dataset, sample, changes: dict, event: dict):
    """Atomically preserve concurrent annotation edits, including edits from the App."""
    raw = sample.to_mongo_dict(include_id=True)
    expected = {
        "_id": raw["_id"],
        "ground_truth": raw.get("ground_truth"),
        "review_status": raw.get("review_status"),
        "review_approval_id": raw.get("review_approval_id"),
        "last_modified_at": raw.get("last_modified_at"),
    }
    now = datetime.now(timezone.utc)
    changes = dict(changes, last_modified_at=now)
    if changes.get("review_status") in ("completed", "auto_accepted"):
        changes["review_checked_at"] = now
    update = {"$set": changes}
    if raw.get("review_history") is None:
        changes["review_history"] = [event]
    else:
        update["$push"] = {"review_history": event}
    # A single MongoDB update prevents a stale approval from overwriting a concurrent edit.
    result = dataset._sample_collection.update_one(expected, update)
    if result.matched_count != 1:
        raise RuntimeError("Image changed during review; reload it and try again")
    sample.reload()


def _clear_approval():
    return dict(
        review_approved_hash=None,
        review_approval_id=None,
        review_approval_sha256=None,
        reviewed_by=None,
        reviewed_at=None,
        review_kind=None,
        review_checked_at=None,
    )


def _transition(cfg, dataset, sample, action, reviewer, note="", confirm_empty=False):
    import fiftyone as fo

    sample.reload()
    now = datetime.now(timezone.utc)
    record_id = uuid4().hex
    record = {
        "schema_version": 1,
        "record_id": record_id,
        "image_id": sample["image_id"],
        "action": action,
        "reviewer": reviewer,
        "at": now.isoformat(),
        "note": note,
        "previous_status": sample["review_status"],
        "previous_approval_id": sample["review_approval_id"],
        "source": sample["ground_truth_source"],
    }
    changes = _clear_approval()
    changes.update(review_reason=note)
    if action == "complete":
        content = annotation_content(sample, cfg.classes)
        if not content["instances"] and not confirm_empty:
            raise ValueError("Confirm that the empty ground truth has been checked")
        if sha256_file(Path(sample.filepath)) != sample["managed_sha256"]:
            raise ValueError("Registered image content changed")
        digest = content_hash(content)
        record.update(
            annotation=content, label_hash=digest, confirmed_empty=not content["instances"]
        )
        changes.update(
            review_status="completed",
            review_approved_hash=digest,
            review_approval_id=record_id,
            reviewed_by=reviewer,
            reviewed_at=now,
            review_kind="manual",
            review_queue_reason=None,
        )
    elif action in ("start", "exclude", "invalidate"):
        changes["review_status"] = "excluded" if action == "exclude" else "in_progress"
        if action == "exclude":
            changes["review_queue_reason"] = None
        if action == "start" and sample["ground_truth"] is None:
            source_job = sample["review_queue_job"] or dataset.info.get("vloop_review_source_job")
            if not sample["review_initialized"] and source_job:
                initialize_sample(cfg, dataset, sample, source_job)
                sample.reload()
        if action == "start" and sample["ground_truth"] is None:
            changes.update(
                ground_truth=fo.Detections(detections=[]).to_mongo().to_dict(),
                review_initialized=True,
                ground_truth_source={"kind": "manual"},
            )
    else:
        raise ValueError(f"Unknown review action: {action}")
    path = record_path(cfg, sample, record_id)
    record["source"] = changes.get("ground_truth_source", sample["ground_truth_source"])
    write_json(path, record)
    if action == "complete":
        changes["review_approval_sha256"] = sha256_file(path)
    event = {key: record[key] for key in ("record_id", "action", "reviewer", "at", "note")}
    _save_change(dataset, sample, changes, event)


def change_reviews(cfg, sample_ids, action, reviewer, *, note="", confirm_empty=False):
    if action not in ("start", "complete", "exclude"):
        raise ValueError("Choose start, complete, or exclude")
    if not reviewer or not reviewer.strip():
        raise ValueError("Enter a reviewer name")
    selected = set()
    for sample_id in sample_ids:
        selected.add(sample_id)
        if len(selected) > 100:
            raise ValueError(
                "Manual review supports up to 100 images; use review-batch for adoption"
            )
    if not selected:
        raise ValueError("Open or select images to review")
    with project_lock(cfg):
        dataset = load_review_dataset(cfg)
        result = {"changed": 0, "failed": 0, "errors": []}
        for sample_id in sorted(selected):
            try:
                _transition(
                    cfg, dataset, dataset[sample_id], action, reviewer.strip(), note, confirm_empty
                )
                result["changed"] += 1
            except (ValueError, RuntimeError, OSError, KeyError) as exc:
                result["failed"] += 1
                result["errors"].append({"sample_id": sample_id, "error": str(exc)})
        return result


def audit_changed_reviews(cfg, dataset):
    """Follow indexed modification timestamps; never rehash all approved images on a timer."""
    from bson import ObjectId

    path = cfg.storage_dir / "reviews" / "audit-cursor.json"
    cursor = json.loads(path.read_text()) if path.exists() else None
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=2)
    query = {
        "review_status": {"$in": ["completed", "auto_accepted"]},
        "last_modified_at": {"$lte": cutoff},
    }
    if cursor:
        stamp = datetime.fromisoformat(cursor["at"])
        query["$or"] = [
            {"last_modified_at": {"$gt": stamp}},
            {"last_modified_at": stamp, "_id": {"$gt": ObjectId(cursor["id"])}},
        ]
    rows = (
        dataset._sample_collection.find(query, {"last_modified_at": 1, "review_checked_at": 1})
        .sort([("last_modified_at", 1), ("_id", 1)])
        .limit(cfg.review_audit_scan_size)
        .batch_size(500)
    )
    invalidated = 0
    checked = 0
    last = None
    for row in rows:
        last = row
        if row.get("review_checked_at") != row["last_modified_at"]:
            checked += 1
            sample = dataset[str(row["_id"])]
            sample.reload()
            if sample["review_status"] in ("completed", "auto_accepted"):
                try:
                    approved_annotation(cfg, sample, allow_auto=True)
                except (ValueError, OSError, KeyError) as exc:
                    _transition(cfg, dataset, sample, "invalidate", "vloop", str(exc))
                    invalidated += 1
                else:
                    # Updating the check marker must not change the annotation timestamp.
                    dataset._sample_collection.update_one(
                        {"_id": row["_id"], "last_modified_at": row["last_modified_at"]},
                        {"$set": {"review_checked_at": row["last_modified_at"]}},
                    )
        if checked >= cfg.review_audit_batch_size:
            break
    rows.close()
    if last:
        write_json(path, {"at": last["last_modified_at"].isoformat(), "id": str(last["_id"])})
    return invalidated


def initialize_sample(cfg, dataset, sample, job_id):
    if (
        sample["ground_truth"] is not None
        or sample["review_initialized"]
        or any(event.get("action") != "queue" for event in (sample["review_history"] or []))
    ):
        return False
    field = f"pred_{job_id}"
    if not dataset.has_sample_field(field) or sample[field] is None:
        return False
    path = cfg.storage_dir / "runs" / job_id / "predictions" / f"{sample['image_id']}.json"
    checksum = sha256_file(path)
    if checksum != sample[f"{field}_sha256"]:
        raise ValueError("Saved prediction was modified")
    prediction = json.loads(path.read_text())
    if prediction["image_id"] != sample["image_id"]:
        raise ValueError("Prediction image ID differs from registered image")
    truth = to_detections(prediction)
    source = {"kind": "autolabel", "job_id": job_id, "field": field, "sha256": checksum}
    _save_change(
        dataset,
        sample,
        {
            "ground_truth": truth.to_mongo().to_dict(),
            "ground_truth_source": source,
            "review_initialized": True,
            "review_status": "unreviewed",
        },
        {"action": "initialize", "at": datetime.now(timezone.utc).isoformat(), "source": source},
    )
    return True


def prepare_review(cfg, job_id=None, *, limit=None, queue=None):
    limit = cfg.review_prepare_limit if limit is None else limit
    if type(limit) is not int or limit < 1:
        raise ValueError("Review preparation limit must be positive")
    with project_lock(cfg):
        directory, report = start_run(cfg, "review")
        report.update(initialized=0, preserved=0, unavailable=0)
        try:
            install_review_plugin(cfg)
            dataset = load_review_dataset(cfg)
            ensure_review_schema(dataset, cfg)
            jobs = dataset.info.get("vloop_autolabel_jobs", {})
            if job_id is None:
                job_id = max(jobs, default=None)
            if job_id is not None and (
                not re.fullmatch(r"autolabel_\d{8}T\d{6}_[0-9a-f]{8}", job_id) or job_id not in jobs
            ):
                raise ValueError("Choose a registered auto-labeling job ID")
            report["source_job_id"] = job_id
            dataset.info["vloop_review_source_job"] = job_id
            dataset.save()
            report["invalidated"] = audit_changed_reviews(cfg, dataset)
            field = f"pred_{job_id}" if job_id else None
            query = {"review_initialized": {"$ne": True}}
            if field and not queue:
                query[f"{field}_sha256"] = {"$ne": None}
            if queue:
                query["review_queue_reason"] = queue
            report.update(limit=limit, queue=queue)
            for sample in dataset.match(query).sort_by("id").limit(limit):
                if (
                    sample["ground_truth"] is not None
                    or sample["review_initialized"]
                    or any(
                        event.get("action") != "queue" for event in (sample["review_history"] or [])
                    )
                    or sample["review_status"]
                    in ("in_progress", "completed", "auto_accepted", "excluded")
                ):
                    if not sample["review_initialized"]:
                        changes = {"review_initialized": True}
                        if sample["ground_truth_source"] is None:
                            changes["ground_truth_source"] = {"kind": "existing_ground_truth"}
                        _save_change(
                            dataset,
                            sample,
                            changes,
                            {
                                "action": "preserve_existing",
                                "at": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                    report["preserved"] += 1
                    continue
                source_job = sample["review_queue_job"] or job_id
                source_field = f"pred_{source_job}" if source_job else None
                if (
                    source_field is None
                    or not dataset.has_sample_field(source_field)
                    or sample[source_field] is None
                ):
                    report["unavailable"] += 1
                    continue
                report["initialized"] += int(initialize_sample(cfg, dataset, sample, source_job))
            report["status"] = "completed"
        except KeyboardInterrupt:
            report.update(status="interrupted", error="Review preparation interrupted")
        except Exception as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        return finish_run(directory, report)


def serve_review(cfg, *, no_browser=False, queue=None):
    fo = configure_fiftyone(cfg)
    dataset = load_review_dataset(cfg)
    view = dataset.match(fo.ViewField("review_queue_reason") == queue) if queue else dataset
    session = fo.launch_app(view, address="127.0.0.1", port=cfg.fiftyone_port, remote=no_browser)
    print(f"Review: http://127.0.0.1:{cfg.fiftyone_port} (Ctrl+C to stop)", flush=True)
    try:
        while True:
            time.sleep(cfg.review_poll_seconds)
            try:
                with project_lock(cfg):
                    changed = audit_changed_reviews(cfg, dataset)
                if changed:
                    session.refresh()
                    print(f"Review required again: {changed} images", flush=True)
            except RuntimeError as exc:
                if "Another vloop operation" not in str(exc):
                    print(f"Review audit: {exc}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
