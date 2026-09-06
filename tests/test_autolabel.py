import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from vloop import autolabel as module
from vloop.config import ClassConfig
from vloop.ingest import ingest
from vloop.runtime import sha256_file


@pytest.fixture
def backend(project, monkeypatch):
    for index, color in enumerate(("red", "blue", "green")):
        Image.new("RGB", (13, 7), color).save(project.image_dir / f"{index}.png")
    ingest(project, local_only=True)
    state = {
        "calls": [],
        "saved": {},
        "fail": set(),
        "interrupt_at": None,
        "store_error": False,
        "model_id": "original",
        "loads": 0,
    }

    class Labeler:
        def __init__(self, cfg):
            self.cfg = cfg
            self.metrics = {}

        def __enter__(self):
            state["loads"] += 1
            return self

        def __exit__(self, *args):
            pass

        def predict(self, image):
            state["calls"].append((image["image_id"], self.cfg.classes))
            if len(state["calls"]) == state["interrupt_at"]:
                raise KeyboardInterrupt
            if image["image_id"] in state["fail"]:
                raise RuntimeError("Inference failed")
            return {
                "image_id": image["image_id"],
                "width": image["width"],
                "height": image["height"],
                "instances": [],
            }

    class Store:
        def __init__(self, cfg, field, job_id, manifest):
            self.field = field

        def put(self, prediction, checksum):
            if state["store_error"]:
                raise RuntimeError("Database save failed")
            state["saved"][(self.field, prediction["image_id"])] = checksum

    monkeypatch.setattr(module, "Sam3Labeler", Labeler)
    monkeypatch.setattr(module, "PredictionStore", Store)
    monkeypatch.setattr(module, "model_identity", lambda cfg: {"checkpoint": state["model_id"]})
    return state


def test_completed_and_empty_predictions_are_skipped_on_resume(project, backend):
    report = module.autolabel(project)
    assert report["status"] == "completed", report
    assert report["completed"] == report["empty"] == 3
    hashes = dict(backend["saved"])
    resumed = module.autolabel(project, resume=report["job_id"])
    assert resumed["processed_this_attempt"] == 0
    assert len(backend["calls"]) == 3
    assert backend["loads"] == 1
    assert backend["saved"] == hashes


def test_interruption_freezes_inputs_and_prompts(project, backend):
    backend["interrupt_at"] = 2
    report = module.autolabel(project)
    assert report["status"] == "interrupted", report
    assert report["completed"] == 1
    first = backend["calls"][0][0]
    path = Path(report["result_dir"]) / "predictions" / f"{first}.json"
    before = sha256_file(path)
    Image.new("RGB", (13, 7), "yellow").save(project.image_dir / "new.png")
    ingest(project, local_only=True)
    changed = replace(project, classes=(ClassConfig(99, "different", ("different",)),))
    backend["interrupt_at"] = None
    resumed = module.autolabel(changed, resume=report["job_id"])
    assert resumed["status"] == "completed", resumed
    assert resumed["total"] == 3
    assert resumed["processed_this_attempt"] == 2
    assert sha256_file(path) == before
    assert all(classes == project.classes for _, classes in backend["calls"])
    assert (Path(report["result_dir"]) / "attempts/0001.json").is_file()


def test_failed_inference_can_resume_without_reprocessing_success(project, backend):
    bad_id = sha256_file(project.image_dir / "1.png")
    backend["fail"].add(bad_id)
    report = module.autolabel(project)
    assert report["status"] == "failed"
    assert (report["completed"], report["failed"]) == (2, 1)
    backend["fail"].clear()
    resumed = module.autolabel(project, resume=report["job_id"])
    assert resumed["status"] == "completed"
    assert resumed["processed_this_attempt"] == 1
    assert len(backend["calls"]) == 4


def test_database_retry_reuses_saved_inference(project, backend):
    backend["store_error"] = True
    report = module.autolabel(project)
    assert report["failed"] == 3
    assert report["empty"] == 0
    backend["store_error"] = False
    resumed = module.autolabel(project, resume=report["job_id"])
    assert resumed["status"] == "completed", resumed
    assert resumed["reused_results"] == 3
    assert len(backend["calls"]) == 3


def test_changed_model_rejects_incomplete_resume(project, backend):
    backend["interrupt_at"] = 1
    report = module.autolabel(project)
    backend["model_id"] = "different"
    resumed = module.autolabel(project, resume=report["job_id"])
    assert resumed["status"] == "failed"
    assert "changed" in resumed["error"]
    assert len(backend["calls"]) == 1


@pytest.mark.parametrize("target", ["configuration", "input", "result"])
def test_modified_artifacts_are_rejected(project, backend, target):
    report = module.autolabel(project, limit=1)
    directory = Path(report["result_dir"])
    if target == "configuration":
        (directory / "config.json").write_text("{}")
    elif target == "input":
        with sqlite3.connect(directory / "samples.sqlite3") as connection:
            connection.execute("UPDATE images SET width = 99")
    else:
        next((directory / "predictions").glob("*.json")).write_text("{}")
    resumed = module.autolabel(project, resume=report["job_id"])
    assert resumed["status"] == "failed"
    assert "modified" in resumed["error"]


def test_image_integrity_failure_does_not_stop_remaining_images(project, backend):
    with sqlite3.connect(project.storage_dir / "catalog.sqlite3") as connection:
        first = connection.execute("SELECT filepath FROM images ORDER BY image_id").fetchone()[0]
    Path(first).write_bytes(b"modified")
    report = module.autolabel(project)
    assert (report["completed"], report["failed"]) == (2, 1)
    assert len(backend["calls"]) == 2


def test_job_fields_are_distinct_and_limit_is_frozen(project, backend):
    first = module.autolabel(project, limit=1)
    second = module.autolabel(project, limit=2)
    assert first["prediction_field"] != second["prediction_field"]
    assert (first["total"], second["total"]) == (1, 2)
    manifest = json.loads((Path(first["result_dir"]) / "manifest.json").read_text())
    assert manifest["model"]["checkpoint"] == "original"


@pytest.mark.parametrize("resume,limit", [("../bad", None), (None, 0), (None, -1)])
def test_invalid_selection_rejected(project, resume, limit):
    with pytest.raises(ValueError):
        module.autolabel(project, resume=resume, limit=limit)


def test_interruption_during_snapshot_has_actionable_recovery(project, backend, monkeypatch):
    def interrupt(cfg, directory, limit):
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "_snapshot", interrupt)
    report = module.autolabel(project)
    assert report["status"] == "interrupted"
    assert report["initialization_failed"]
    assert "--resume" not in report["retry"]
    assert not backend["calls"]
    with pytest.raises(ValueError, match="start a new"):
        module.autolabel(project, resume=report["job_id"])


def test_hard_exit_before_snapshot_cannot_resume_incomplete_manifest(project, backend):
    directory, report = module.start_run(project, "autolabel")
    assert not (directory / "manifest.json").exists()
    with pytest.raises(ValueError, match="start a new"):
        module.autolabel(project, resume=report["job_id"])
