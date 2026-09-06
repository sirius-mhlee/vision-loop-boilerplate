import hashlib
import json
import re
import sqlite3
import traceback
from contextlib import ExitStack, closing
from pathlib import Path

from .config import Config, config_from_dict
from .runtime import finish_run, project_lock, sha256_file, start_run, write_json
from .sam3 import Sam3Labeler, model_identity


def _open_job(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _input_digest(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    for row in connection.execute(
        "SELECT image_id, filepath, width, height, managed_sha256 FROM images ORDER BY image_id"
    ):
        digest.update((json.dumps(list(row), ensure_ascii=False) + "\n").encode())
    return digest.hexdigest()


def _snapshot(cfg: Config, directory: Path, limit: int | None) -> None:
    catalog = cfg.storage_dir / "catalog.sqlite3"
    if not catalog.is_file():
        raise ValueError("No registered images; run vloop ingest first")
    with closing(_open_job(directory / "samples.sqlite3")) as job:
        job.executescript("""
            CREATE TABLE images (
                image_id TEXT PRIMARY KEY, filepath TEXT NOT NULL, width INTEGER NOT NULL,
                height INTEGER NOT NULL, managed_sha256 TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                error TEXT, result_sha256 TEXT, object_count INTEGER
            );
            CREATE INDEX images_status ON images(status);
        """)
        with closing(sqlite3.connect(f"{catalog.as_uri()}?mode=ro", uri=True)) as source, job:
            for row in source.execute(
                "SELECT image_id, filepath, width, height, managed_sha256 "
                "FROM images ORDER BY image_id LIMIT ?",
                (limit or -1,),
            ):
                job.execute(
                    "INSERT INTO images (image_id, filepath, width, height, managed_sha256) "
                    "VALUES (?, ?, ?, ?, ?)",
                    row,
                )
        if not job.execute("SELECT COUNT(*) FROM images").fetchone()[0]:
            raise ValueError("No registered images; run vloop ingest first")
        write_json(
            directory / "manifest.json",
            {
                "schema_version": 1,
                "model": model_identity(cfg),
                "config_sha256": sha256_file(directory / "config.json"),
                "inputs_sha256": _input_digest(job),
            },
        )


def _counts(connection: sqlite3.Connection) -> dict:
    result = {"completed": 0, "failed": 0, "pending": 0, "processing": 0, "predicted": 0}
    result.update(dict(connection.execute("SELECT status, COUNT(*) FROM images GROUP BY status")))
    result["total"] = sum(result.values())
    result["empty"] = connection.execute(
        "SELECT COUNT(*) FROM images WHERE status = 'completed' AND object_count = 0"
    ).fetchone()[0]
    return result


class PredictionStore:
    """Write only this job's prediction field; review labels and states are untouched."""

    def __init__(self, cfg: Config, field: str, job_id: str, manifest: dict):
        from .fiftyone import configure_fiftyone

        self.fo = configure_fiftyone(cfg)
        if not self.fo.dataset_exists(cfg.dataset_name):
            raise ValueError("FiftyOne dataset missing; run vloop ingest without --local-only")
        self.dataset = self.fo.load_dataset(cfg.dataset_name)
        if self.dataset.info.get("vloop_storage_dir") != str(cfg.storage_dir):
            raise ValueError("FiftyOne dataset does not belong to this project")
        mapping = [{"id": item.id, "name": item.name} for item in cfg.classes]
        if self.dataset.info.get("vloop_classes") != mapping:
            raise ValueError("FiftyOne class mapping differs from the frozen job")
        self.field = field
        self.hash_field = f"{field}_sha256"
        if not self.dataset.has_sample_field(field):
            self.dataset.add_sample_field(
                field, self.fo.EmbeddedDocumentField, embedded_doc_type=self.fo.Detections
            )
        if not self.dataset.has_sample_field(self.hash_field):
            self.dataset.add_sample_field(self.hash_field, self.fo.StringField)
        jobs = self.dataset.info.setdefault("vloop_autolabel_jobs", {})
        if job_id in jobs and jobs[job_id] != manifest:
            raise ValueError("FiftyOne job provenance differs from the saved manifest")
        jobs[job_id] = manifest
        self.dataset.save()

    def put(self, prediction: dict, checksum: str) -> None:
        from .labels import to_detections

        view = self.dataset.match(self.fo.ViewField("image_id") == prediction["image_id"])
        if len(view) != 1:
            raise ValueError("Registered image is missing or duplicated in FiftyOne")
        sample = view.first()
        if Path(sample.filepath).resolve() != Path(prediction["image_path"]).resolve():
            raise ValueError("FiftyOne filepath differs from the frozen input")
        if sample[self.field] is not None:
            if sample[self.hash_field] != checksum:
                raise ValueError("Prediction field already contains a different result")
            return
        sample[self.field] = to_detections(prediction)
        sample[self.hash_field] = checksum
        sample.save()


def autolabel(cfg: Config, *, resume: str | None = None, limit: int | None = None) -> dict:
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    if resume is not None and (
        limit is not None or not re.fullmatch(r"autolabel_\d{8}T\d{6}_[0-9a-f]{8}", resume)
    ):
        raise ValueError("Use an autolabel job ID without --limit when resuming")
    with project_lock(cfg):
        if resume is None:
            if not cfg.classes:
                raise ValueError("Set classes before starting auto-labeling")
            directory, report = start_run(cfg, "autolabel")
            report.update(attempt=1, prediction_field=f"pred_{report['job_id']}")
            try:
                _snapshot(cfg, directory, limit)
            except KeyboardInterrupt:
                report.update(
                    status="interrupted",
                    error="Interrupted before input snapshot was completed; start a new job",
                    initialization_failed=True,
                    retry=f"vloop autolabel --config {cfg.config_path}",
                )
                return finish_run(directory, report)
            except Exception as exc:
                report.update(
                    status="failed",
                    error=str(exc),
                    initialization_failed=True,
                    retry=f"vloop autolabel --config {cfg.config_path}",
                )
                return finish_run(directory, report)
        else:
            directory = cfg.storage_dir / "runs" / resume
            report = json.loads((directory / "report.json").read_text())
            if (
                report.get("initialization_failed")
                or "prediction_field" not in report
                or not (directory / "manifest.json").is_file()
            ):
                raise ValueError("Job initialization failed; start a new autolabel job")
            write_json(directory / "attempts" / f"{report['attempt']:04d}.json", report)
            report.update(attempt=report["attempt"] + 1, status="running")
            report.pop("error", None)
            report.pop("finished_at", None)
            report.pop("runtime", None)
        report["retry"] = f"vloop autolabel --config {cfg.config_path} --resume {report['job_id']}"
        report.update(processed_this_attempt=0, reused_results=0)
        labeler = None
        try:
            manifest = json.loads((directory / "manifest.json").read_text())
            if manifest.get("schema_version") != 1:
                raise ValueError("Unsupported auto-labeling manifest schema")
            if sha256_file(directory / "config.json") != manifest["config_sha256"]:
                raise ValueError("Frozen job configuration was modified")
            snapshot = json.loads((directory / "config.json").read_text())
            frozen = config_from_dict(snapshot, Path(snapshot.pop("config_path")))
            if frozen.storage_dir != cfg.storage_dir:
                raise ValueError("Job storage differs from the current project")
            report["config_source"] = str(directory / "config.json")
            with (
                closing(_open_job(directory / "samples.sqlite3")) as connection,
                ExitStack() as stack,
            ):
                if _input_digest(connection) != manifest["inputs_sha256"]:
                    raise ValueError("Frozen input manifest was modified")
                for row in connection.execute("SELECT * FROM images WHERE status = 'completed'"):
                    path = directory / "predictions" / f"{row['image_id']}.json"
                    if sha256_file(path) != row["result_sha256"]:
                        raise ValueError(f"Completed result was modified: {path}")
                report.update(_counts(connection))
                if report["completed"] != report["total"]:
                    if model_identity(frozen) != manifest["model"]:
                        raise ValueError(
                            "Model, weights, adapter, or dependencies changed; start a new job"
                        )
                    store = PredictionStore(
                        frozen, report["prediction_field"], report["job_id"], manifest
                    )
                    for row in connection.execute(
                        "SELECT * FROM images WHERE status != 'completed' ORDER BY image_id"
                    ):
                        image = dict(row)
                        image_id = image["image_id"]
                        path = directory / "predictions" / f"{image_id}.json"
                        loading_model = False
                        try:
                            with connection:
                                connection.execute(
                                    "UPDATE images SET status = 'processing', "
                                    "attempts = attempts + 1, error = NULL WHERE image_id = ?",
                                    (image_id,),
                                )
                            if sha256_file(Path(image["filepath"])) != image["managed_sha256"]:
                                raise ValueError("Registered image content changed")
                            if image["result_sha256"]:
                                if sha256_file(path) != image["result_sha256"]:
                                    raise ValueError("Saved prediction was modified")
                                prediction = json.loads(path.read_text())
                                report["reused_results"] += 1
                            else:
                                if labeler is None:
                                    loading_model = True
                                    labeler = stack.enter_context(Sam3Labeler(frozen))
                                    loading_model = False
                                prediction = labeler.predict(image)
                                prediction.update(
                                    job_id=report["job_id"], image_path=image["filepath"]
                                )
                                if prediction["image_id"] != image_id:
                                    raise ValueError("Prediction image ID does not match input")
                                write_json(path, prediction)
                                with connection:
                                    connection.execute(
                                        "UPDATE images SET status = 'predicted', "
                                        "result_sha256 = ?, "
                                        "object_count = ? WHERE image_id = ?",
                                        (sha256_file(path), len(prediction["instances"]), image_id),
                                    )
                            store.put(prediction, sha256_file(path))
                            with connection:
                                connection.execute(
                                    "UPDATE images SET status = 'completed' WHERE image_id = ?",
                                    (image_id,),
                                )
                            report["processed_this_attempt"] += 1
                        except Exception as exc:
                            with connection:
                                connection.execute(
                                    "UPDATE images SET status = 'failed', error = ? "
                                    "WHERE image_id = ?",
                                    (f"{type(exc).__name__}: {exc}", image_id),
                                )
                            error_path = (
                                directory / "errors" / f"{image_id}-{report['attempt']}.txt"
                            )
                            error_path.parent.mkdir(exist_ok=True)
                            error_path.write_text(traceback.format_exc())
                            if type(exc).__name__ == "OutOfMemoryError" or loading_model:
                                raise
                        finally:
                            report.update(_counts(connection))
                            write_json(directory / "report.json", report)
                report["status"] = (
                    "completed" if report["completed"] == report["total"] else "failed"
                )
        except KeyboardInterrupt:
            report.update(status="interrupted", error="Interrupted; resume this job to continue")
        except Exception as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            if labeler is not None:
                report["runtime"] = labeler.metrics
        return finish_run(directory, report)
