"""Real FiftyOne + MLflow evaluation with controlled geometry and artifact recovery."""

import json
import os
import sys
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from test_release import make_sample

from vloop.config import ClassConfig, config_from_dict
from vloop.evaluation_view import fields
from vloop.labels import decode_mask
from vloop.release_data import capture, connect, export_coco, image_relative
from vloop.runtime import finish_run, sha256_file, start_run, write_json
from vloop.tracking import artifact_directory, client_for, create_run, manifest_hash


def run(root):
    import vloop.evaluate as module
    import vloop.trained_model as model_module

    cfg = config_from_dict(
        {
            "image_dir": str(root / "input"),
            "storage_dir": str(root / "state"),
            "classes": [{"id": 7, "name": "test-object", "prompts": ["object"]}],
        },
        config_path=root / "project.yaml",
    )
    cfg = replace(cfg, classes=(*cfg.classes, ClassConfig(42, "absent", ("absent",))))
    cfg.image_dir.mkdir()
    samples = [make_sample(cfg, 1), make_sample(cfg, 2, empty=True)]
    dataset_root = root / "restored"
    snapshot = dataset_root / "metadata/snapshot.sqlite3"
    capture(cfg, iter(samples), snapshot)
    with closing(connect(snapshot)) as db:
        db.execute("UPDATE records SET split='val'")
        db.commit()
        records = list(db.execute("SELECT * FROM records ORDER BY image_id"))
    for row in records:
        target = dataset_root / image_relative(row)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(row["source_path"], target)
    info = {"summary": export_coco(cfg, snapshot, dataset_root)}
    truth = {int(Path(row["source_path"]).stem): json.loads(row["annotation"]) for row in records}
    original_hash = sha256_file(snapshot)
    training = {"dataset": {"version": "v001", "git_commit": "fixed", "descriptor": info}}
    training["binding"] = manifest_hash(training)
    metadata = {
        "schema_version": 1,
        "dataset_version": "v001",
        "classes": info["summary"]["classes"],
        "sha256": "fake-model",
        "postprocessing": {
            "eval_confidence": 0.001,
            "display_confidence": 0.5,
            "eval_max_detections": 100,
            "num_select": 100,
        },
    }
    directory, train_report = start_run(cfg, "train")
    client, train_run = create_run(cfg, train_report["job_id"])
    train_report.update(run_id=train_run.info.run_id, status="completed")
    finish_run(directory, train_report)
    write_json(artifact_directory(train_run) / "training.json", training)
    client.set_terminated(train_run.info.run_id)

    class Model:
        mode = "perfect"

        def predict(self, pixels, **kwargs):
            assert kwargs == {"threshold": 0.001, "include_source_image": False}
            content = truth[int(pixels[0, 0, 0])]
            objects = content["instances"]
            if self.mode == "empty":
                objects = []
            if self.mode == "false_positive" and not objects:
                objects = truth[1]["instances"]
            boxes, masks, scores = [], [], []
            for instance in objects:
                x, y, w, h = instance["bbox_xywh"]
                boxes.append(
                    [x, y, x + w, y + h] if self.mode != "wrong_box" else [100, 100, 110, 110]
                )
                mask = decode_mask(instance["segmentation"], 23, 13)
                masks.append(np.zeros_like(mask) if self.mode == "wrong_mask" else mask)
                scores.append(0.99 if int(pixels[0, 0, 0]) == 2 else 0.9)
            return SimpleNamespace(
                xyxy=np.array(boxes).reshape(-1, 4),
                confidence=np.array(scores),
                class_id=np.zeros(len(boxes), dtype=int),
                mask=np.array(masks).reshape(-1, 13, 23),
            )

    model = Model()
    with (
        patch.object(module, "project_repo", return_value=root),
        patch.object(module, "descriptor", return_value=info),
        patch.object(module, "git", return_value="fixed"),
        patch.object(module, "restore_data", return_value=(dataset_root, info)),
        patch.object(model_module, "read_model_metadata", return_value=(metadata, root)),
        patch.object(model_module, "load_model", return_value=(model, metadata)),
    ):
        reports = {}
        for mode in ("perfect", "wrong_mask", "wrong_box", "empty", "false_positive"):
            model.mode = mode
            report = module.evaluate(cfg, job_id=train_report["job_id"])
            assert report["status"] == "completed", report
            reports[mode] = report
        perfect = reports["perfect"]
        for kind in ("boxes", "masks"):
            assert perfect["metrics"][kind]["mAP"] == 1, perfect
            assert perfect["metrics"][kind]["per_class"]["absent"]["AP"] is None
            assert reports["empty"]["metrics"][kind]["mAP"] == 0
            assert reports["empty"]["metrics"][kind]["fn"] == 1
            assert reports["false_positive"]["metrics"][kind]["fp"] == 1
            assert 0 < reports["false_positive"]["metrics"][kind]["mAP"] < 1
        assert reports["wrong_mask"]["metrics"]["boxes"]["mAP"] == 1
        assert reports["wrong_mask"]["metrics"]["masks"]["mAP"] == 0
        assert reports["wrong_mask"]["metrics"]["masks"]["fp"] == 1
        assert reports["wrong_box"]["metrics"]["boxes"]["mAP"] == 0
        assert reports["wrong_box"]["metrics"]["masks"]["mAP"] == 1
        assert len({r["comparison_id"] for r in reports.values()}) == 1
        assert sha256_file(snapshot) == original_hash
        from vloop.fiftyone import configure_fiftyone

        fo = configure_fiftyone(cfg)
        assert not fo.dataset_exists(cfg.dataset_name)
        dataset = module.load_evaluation(cfg, perfect["job_id"])
        assert set(dataset.list_evaluations()) == {
            p["key"] for p in fields(perfect["job_id"]).values()
        }
        assert set(dataset.list_saved_views()) == {
            "boxes_fp",
            "boxes_fn",
            "masks_fp",
            "masks_fn",
            "all",
            "display_confidence",
            "empty_predictions",
        }
        fo.delete_dataset(dataset.name)
        with patch.object(model_module, "load_model", side_effect=AssertionError("must not infer")):
            restored = module.load_evaluation(cfg, perfect["job_id"])
            assert restored.info["metrics"] == perfect["metrics"]
        run = client_for(cfg).get_run(perfect["run_id"])
        assert run.data.tags["vloop.source_job_id"] == train_report["job_id"]
        assert run.data.metrics["masks/mAP"] == 1
        database = artifact_directory(run) / "samples.sqlite3"
        with database.open("ab") as handle:
            handle.write(b"corrupt")
        try:
            module.load_evaluation(cfg, perfect["job_id"])
        except ValueError as exc:
            assert "checksum" in str(exc)
        else:
            raise AssertionError("Corrupted prediction artifact was accepted")
        for name in fo.list_datasets():
            fo.delete_dataset(name)
        write_json(root / "result.json", reports)
        print(json.dumps({k: v["metrics"] for k, v in reports.items()}, indent=2))


if __name__ == "__main__":
    run(Path(sys.argv[1]))
