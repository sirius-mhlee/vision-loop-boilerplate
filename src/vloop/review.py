import json
import re
import time
from datetime import datetime, timezone
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
    dataset.save()
    for status, title in STATUSES.items():
        name = f"vloop-{status}"
        if not dataset.has_saved_view(name):
            dataset.save_view(
                name, dataset.match(fo.ViewField("review_status") == status), description=title
            )


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
    changes = dict(changes, last_modified_at=datetime.now(timezone.utc))
    # A single MongoDB update prevents a stale approval from overwriting a concurrent edit.
    result = dataset._sample_collection.update_one(
        expected, {"$set": changes, "$push": {"review_history": event}}
    )
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
        )
    elif action in ("start", "exclude", "invalidate"):
        changes["review_status"] = "excluded" if action == "exclude" else "in_progress"
        if action == "start" and sample["ground_truth"] is None:
            changes.update(
                ground_truth=fo.Detections(detections=[]).to_mongo().to_dict(),
                review_initialized=True,
                ground_truth_source={"kind": "manual"},
            )
    else:
        raise ValueError(f"Unknown review action: {action}")
    path = record_path(cfg, sample, record_id)
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
    sample_ids = list(dict.fromkeys(sample_ids))
    if not sample_ids:
        raise ValueError("Open or select images to review")
    with project_lock(cfg):
        dataset = load_review_dataset(cfg)
        result = {"changed": 0, "failed": 0, "errors": []}
        for sample_id in sample_ids:
            try:
                _transition(
                    cfg, dataset, dataset[sample_id], action, reviewer.strip(), note, confirm_empty
                )
                result["changed"] += 1
            except (ValueError, RuntimeError, OSError, KeyError) as exc:
                result["failed"] += 1
                result["errors"].append({"sample_id": sample_id, "error": str(exc)})
        return result


def audit_reviews(cfg, dataset=None):
    import fiftyone as fo

    dataset = dataset or load_review_dataset(cfg)
    invalidated = 0
    for sample in dataset.match(fo.ViewField("review_status") == "completed"):
        sample.reload()
        try:
            approved_annotation(cfg, sample)
        except (ValueError, OSError, KeyError) as exc:
            _transition(cfg, dataset, sample, "invalidate", "vloop", str(exc))
            invalidated += 1
    return invalidated


def prepare_review(cfg, job_id=None):
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
            report["invalidated"] = audit_reviews(cfg, dataset)
            field = f"pred_{job_id}" if job_id else None
            for sample in dataset.iter_samples():
                if (
                    sample["ground_truth"] is not None
                    or sample["review_initialized"]
                    or sample["review_history"]
                    or sample["review_status"] in ("in_progress", "completed", "excluded")
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
                if field is None or sample[field] is None:
                    report["unavailable"] += 1
                    continue
                path = (
                    cfg.storage_dir / "runs" / job_id / "predictions" / f"{sample['image_id']}.json"
                )
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
                    {
                        "action": "initialize",
                        "at": datetime.now(timezone.utc).isoformat(),
                        "source": source,
                    },
                )
                report["initialized"] += 1
            report["status"] = "completed"
        except KeyboardInterrupt:
            report.update(status="interrupted", error="Review preparation interrupted")
        except Exception as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        return finish_run(directory, report)


def serve_review(cfg, *, no_browser=False):
    fo = configure_fiftyone(cfg)
    dataset = load_review_dataset(cfg)
    session = fo.launch_app(dataset, address="127.0.0.1", port=cfg.fiftyone_port, remote=no_browser)
    print(f"Review: http://127.0.0.1:{cfg.fiftyone_port} (Ctrl+C to stop)", flush=True)
    try:
        while True:
            time.sleep(2)
            try:
                with project_lock(cfg):
                    changed = audit_reviews(cfg, dataset)
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
