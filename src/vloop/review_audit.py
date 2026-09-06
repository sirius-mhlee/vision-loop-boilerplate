"""Explicit, resumable full integrity audit, outside the interactive review timer."""

import json
import re
from types import SimpleNamespace

from .approval import approved_annotation
from .review import _transition, load_review_dataset
from .runtime import finish_run, project_lock, start_run, write_json


def review_audit(cfg, *, resume=None):
    from bson import ObjectId

    dataset = load_review_dataset(cfg)
    if resume:
        if not re.fullmatch(r"review_audit_\d{8}T\d{6}_[0-9a-f]{8}", resume):
            raise ValueError("Use a review_audit job ID")
        directory = cfg.storage_dir / "runs" / resume
        report = json.loads((directory / "report.json").read_text())
    else:
        directory, report = start_run(cfg, "review_audit")
        last = dataset._sample_collection.find_one({}, {"_id": 1}, sort=[("_id", -1)])
        report.update(
            after=None, upper=str(last["_id"]) if last else None, checked=0, invalidated=0
        )
    with project_lock(SimpleNamespace(storage_dir=directory)):
        snapshot = json.loads((directory / "config.json").read_text())
        if snapshot["dataset_name"] != cfg.dataset_name or snapshot["storage_dir"] != str(
            cfg.storage_dir
        ):
            raise ValueError("Audit belongs to a different project")
        report.update(status="running")
        report.pop("error", None)
        try:
            while report["upper"]:
                query = {
                    "review_status": {"$in": ["completed", "auto_accepted"]},
                    "_id": {"$lte": ObjectId(report["upper"])},
                }
                if report["after"]:
                    query["_id"]["$gt"] = ObjectId(report["after"])
                ids = list(
                    dataset._sample_collection.find(query, {"_id": 1})
                    .sort("_id", 1)
                    .limit(cfg.review_audit_batch_size)
                )
                if not ids:
                    break
                with project_lock(cfg):
                    for row in ids:
                        sample = dataset[str(row["_id"])]
                        sample.reload()
                        if sample["review_status"] in ("completed", "auto_accepted"):
                            try:
                                approved_annotation(cfg, sample, allow_auto=True)
                            except (ValueError, OSError, KeyError) as exc:
                                _transition(cfg, dataset, sample, "invalidate", "vloop", str(exc))
                                report["invalidated"] += 1
                            report["checked"] += 1
                        report["after"] = str(row["_id"])
                    write_json(directory / "report.json", report)
                print(
                    f"Audit: {report['checked']} checked, {report['invalidated']} invalidated",
                    flush=True,
                )
            report["status"] = "completed"
        except KeyboardInterrupt:
            report.update(status="interrupted", error="Interrupted; resume this audit")
        except Exception as exc:
            report.update(status="failed", error=str(exc))
        report["retry"] = f"vloop review-audit --resume {report['job_id']}"
        return finish_run(directory, report)
