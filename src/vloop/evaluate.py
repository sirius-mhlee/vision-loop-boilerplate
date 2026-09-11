"""Evaluate a saved training job against a fixed release, with independent MLflow records."""

import gc
import json
import resource
import time
import traceback
from contextlib import closing
from types import SimpleNamespace

import numpy as np
from PIL import Image

from .evaluation_data import comparison_id, freeze_inputs, serialize_prediction, settings
from .evaluation_view import build_dataset, fields, serve
from .release_data import connect
from .release_dvc import descriptor, git, project_repo, restore_data, version_name
from .runtime import finish_run, project_lock, sha256_file, start_run, write_json
from .tracking import artifact_directory, client_for, create_run, manifest_hash, run_id_for_job


def predict(cfg, job_id, metadata, database, root, parameters, report):
    from .trained_model import load_model

    model, loaded_metadata = load_model(cfg, job_id)
    if loaded_metadata != metadata:
        raise ValueError("Source model metadata changed during evaluation")
    last_print = time.monotonic()
    with closing(connect(database)) as db:
        for row in db.execute("SELECT * FROM samples ORDER BY image_id"):
            try:
                annotation = json.loads(row["annotation"])
                path = root / row["relative_path"]
                if sha256_file(path) != annotation["managed_sha256"]:
                    raise ValueError("Evaluation image checksum changed")
                with Image.open(path) as image:
                    if image.size != (annotation["width"], annotation["height"]):
                        raise ValueError("Evaluation image dimensions changed")
                    pixels = np.asarray(image.convert("RGB"))
                output = model.predict(
                    pixels, threshold=parameters["confidence"], include_source_image=False
                )
                prediction = serialize_prediction(
                    output,
                    width=annotation["width"],
                    height=annotation["height"],
                    classes=metadata["classes"],
                    parameters=parameters,
                )
                db.execute(
                    "UPDATE samples SET prediction=? WHERE image_id=?",
                    (
                        json.dumps(prediction, allow_nan=False),
                        row["image_id"],
                    ),
                )
                report["predicted"] += 1
                if report["predicted"] % 20 == 0:
                    db.commit()
                if time.monotonic() - last_print > 30:
                    print(f"Predicted: {report['predicted']}/{report['images']}", flush=True)
                    last_print = time.monotonic()
            except Exception as exc:
                db.execute(
                    "UPDATE samples SET error=? WHERE image_id=?", (str(exc), row["image_id"])
                )
                db.commit()
                raise
        db.commit()
    del model
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate(
    cfg,
    *,
    job_id,
    dataset_version=None,
    split=None,
    limit=None,
    notes=None,
    confidence=None,
    display_confidence=None,
    max_detections=None,
):
    if split is None and cfg.eval_split != "val":
        raise ValueError(
            "Final test evaluation requires explicit --split test; use eval_split: val"
        )
    split = split or "val"
    if split not in ("val", "test") or (limit is not None and limit < 1):
        raise ValueError("Choose val/test and a positive limit")
    with project_lock(cfg):
        directory, report = start_run(cfg, "evaluate")
        report.update(source_job_id=job_id, predicted=0, split=split, limit=limit)
        print(f"Job ID: {report['job_id']}", flush=True)
        client = run_id = artifacts = None
        started = time.monotonic()
        try:
            client, run = create_run(cfg, report["job_id"], kind="evaluate", notes=notes)
            run_id, artifacts = run.info.run_id, artifact_directory(run)
            report["run_id"] = run_id
            write_json(directory / "report.json", report)
            source_run_id = run_id_for_job(client, cfg, job_id)
            from .trained_model import read_model_metadata

            metadata, _ = read_model_metadata(cfg, job_id)
            training = json.loads(
                (artifact_directory(client.get_run(source_run_id)) / "training.json").read_text()
            )
            if training.get("binding") != manifest_hash(training):
                raise ValueError("Frozen source training configuration changed")
            if (
                metadata["classes"] != training["dataset"]["descriptor"]["summary"]["classes"]
                or metadata["dataset_version"] != training["dataset"]["version"]
            ):
                raise ValueError("Model metadata does not match its source training data")
            parameters = settings(
                metadata,
                confidence=confidence,
                display_confidence=display_confidence,
                max_detections=max_detections,
            )
            dataset_version = version_name(dataset_version or metadata["dataset_version"])
            report["dataset_version"] = dataset_version
            repo = project_repo(cfg)
            info = descriptor(repo, dataset_version)
            commit = git(repo, "rev-parse", f"refs/tags/dataset/{dataset_version}^{{commit}}")
            if dataset_version == metadata["dataset_version"] and (
                info != training["dataset"]["descriptor"]
                or commit != training["dataset"]["git_commit"]
            ):
                raise ValueError("Dataset tag or metadata changed since training")
            classes = info["summary"]["classes"]
            if classes != metadata["classes"]:
                raise ValueError("Evaluation release classes do not match the saved model mapping")
            with project_lock(SimpleNamespace(storage_dir=cfg.storage_dir / "releases")):
                root, restored_info = restore_data(cfg, dataset_version)
            if restored_info != info:
                raise ValueError("Dataset changed during restoration")
            database = artifacts / "samples.sqlite3"
            selection = freeze_inputs(
                root / "metadata/snapshot.sqlite3",
                database,
                split=split,
                classes=classes,
                limit=limit,
            )
            manifest = {
                "schema_version": 1,
                "job_id": report["job_id"],
                "source_job_id": job_id,
                "model": metadata,
                "dataset": {"version": dataset_version, "git_commit": commit, "descriptor": info},
                "selection": selection,
                "parameters": parameters,
                "comparison_id": comparison_id(selection, parameters),
                "code": report["code"],
            }
            report.update(
                images=selection["images"],
                comparison_id=manifest["comparison_id"],
                subset=selection["images"] < info["summary"]["splits"][split]["images"],
            )
            write_json(artifacts / "evaluation.json", manifest)
            write_json(directory / "report.json", report)
            for key, value in {
                "source_job_id": job_id,
                "dataset_version": dataset_version,
                "comparison_id": manifest["comparison_id"],
            }.items():
                client.set_tag(run_id, "vloop." + key, value)
            for key, value in {
                "split": split,
                "images": selection["images"],
                "confidence": parameters["confidence"],
                "display_confidence": parameters["display_confidence"],
                "max_detections": parameters["max_detections"],
            }.items():
                client.log_param(run_id, key, value)
            predict(cfg, job_id, metadata, database, root, parameters, report)
            manifest["predictions_sha256"] = sha256_file(database)
            manifest["binding"] = manifest_hash(manifest)
            write_json(artifacts / "evaluation.json", manifest)
            dataset, metrics = build_dataset(cfg, manifest, database, root)
            write_json(artifacts / "metrics.json", metrics)
            for kind, values in metrics.items():
                for key in ("mAP", "AP50", "tp", "fp", "fn"):
                    if values[key] is not None:
                        client.log_metric(run_id, f"{kind}/{key}", values[key])
                for cls in classes:
                    for key, value in values["per_class"][cls["name"]].items():
                        if value is not None:
                            client.log_metric(run_id, f"{kind}/class_{cls['id']}/{key}", value)
            report.update(
                status="completed",
                metrics=metrics,
                evaluation_dataset=dataset.name,
                fields=fields(report["job_id"]),
                evaluation_sha256=sha256_file(artifacts / "evaluation.json"),
                metrics_sha256=sha256_file(artifacts / "metrics.json"),
            )
        except KeyboardInterrupt:
            report.update(
                status="interrupted", error="Evaluation interrupted; start a new evaluation job"
            )
        except Exception as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            (directory / "error.txt").write_text(traceback.format_exc())
        finally:
            report.update(
                elapsed_seconds=time.monotonic() - started,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            )
            if client and run_id:
                try:
                    for name in ("config.json", "dependencies.json", "error.txt"):
                        if (directory / name).exists():
                            client.log_artifact(run_id, str(directory / name))
                    client.set_tag(run_id, "vloop.status", report["status"])
                    client.set_terminated(
                        run_id,
                        {"completed": "FINISHED", "failed": "FAILED", "interrupted": "KILLED"}[
                            report["status"]
                        ],
                    )
                except Exception as exc:
                    report.update(status="failed", tracking_error=str(exc))
            finish_run(directory, report)
            if artifacts:
                write_json(artifacts / "report.json", report)
        return report


def load_evaluation(cfg, job_id):
    """Restore analysis from completed artifacts; never run the model or import review data."""
    with project_lock(cfg):
        client = client_for(cfg)
        run_id = run_id_for_job(client, cfg, job_id, command="evaluate")
        run = client.get_run(run_id)
        artifacts = artifact_directory(run)
        report = json.loads((cfg.storage_dir / "runs" / job_id / "report.json").read_text())
        if report["status"] != "completed" or run.info.status != "FINISHED":
            raise ValueError("Only completed evaluations can be opened")
        for name in ("evaluation", "metrics"):
            if sha256_file(artifacts / f"{name}.json") != report[f"{name}_sha256"]:
                raise ValueError(f"{name} artifact checksum mismatch")
        manifest = json.loads((artifacts / "evaluation.json").read_text())
        if manifest.get("job_id") != job_id or manifest.get("binding") != manifest_hash(manifest):
            raise ValueError("Frozen evaluation manifest changed")
        database = artifacts / "samples.sqlite3"
        if sha256_file(database) != manifest["predictions_sha256"]:
            raise ValueError("Prediction artifact checksum mismatch")
        version_info = manifest["dataset"]
        repo = project_repo(cfg)
        if (
            git(repo, "rev-parse", f"refs/tags/dataset/{version_info['version']}^{{commit}}")
            != version_info["git_commit"]
        ):
            raise ValueError("Evaluation dataset tag changed")
        with project_lock(SimpleNamespace(storage_dir=cfg.storage_dir / "releases")):
            root, info = restore_data(cfg, version_info["version"])
        if info != version_info["descriptor"]:
            raise ValueError("Evaluation release metadata changed")
        dataset, metrics = build_dataset(cfg, manifest, database, root)
        if metrics != json.loads((artifacts / "metrics.json").read_text()):
            raise ValueError("Rebuilt evaluation metrics differ from saved results")
        return dataset


def view_evaluation(cfg, job_id, *, no_browser=False):
    return serve(cfg, load_evaluation(cfg, job_id), no_browser=no_browser)
