"""Two releases and GPU models with fixed validation data and preserved history.

This fixture uses generated images and known masks, not SAM 3 or human review.
The real ingest/review approval, FiftyOne, DVC, training and evaluation paths run
in separate processes. It checks the iteration contract, not model quality.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from vloop.config import config_from_dict, load_config
from vloop.release_data import connect, initial_split
from vloop.release_dvc import descriptor, git
from vloop.runtime import sha256_file, write_json


def known_mask():
    mask = np.zeros((13, 23), dtype=bool)
    mask[1:9, 1:10] = True
    mask[3:7, 3:8] = False
    mask[10:12, 18:21] = True
    return mask


def generate_image(path, number):
    pixels = np.full((13, 23, 3), number, dtype=np.uint8)
    if number != 2:
        pixels[known_mask()] = [230, 180, 40]
    Image.fromarray(pixels).save(path)


def prepare_release(cfg, version):
    from vloop.ingest import ingest
    from vloop.release import release
    from vloop.review import change_reviews, load_review_dataset, prepare_review

    cfg.image_dir.mkdir(exist_ok=True)
    for number in range(1, 5) if version == "v001" else [5]:
        generate_image(cfg.image_dir / f"{number}.png", number)
    registered = ingest(cfg)
    assert registered["status"] == "completed", registered
    assert registered["registered"] == (4 if version == "v001" else 1), registered
    prepared = prepare_review(cfg, limit=10)
    assert prepared["status"] == "completed", prepared
    dataset = load_review_dataset(cfg)
    import fiftyone as fo

    groups = {
        split: next(str(i) for i in range(1000) if initial_split("group:" + str(i), cfg) == split)
        for split in ("train", "val", "test")
    }
    if not dataset.has_sample_field("scene"):
        dataset.add_sample_field("scene", fo.StringField)
    for sample in dataset.iter_samples():
        number = int(Path(sample["source_paths"][0]).stem)
        if version == "v002" and number not in (1, 5):
            continue
        assert change_reviews(cfg, [sample.id], "start", "iteration-test")["changed"] == 1
        sample.reload()
        split = {3: "val", 4: "test"}.get(number, "train")
        sample["scene"] = groups[split]
        mask = known_mask()
        if number == 1 and version == "v001":
            mask[1, 1] = False  # Deliberate training-label correction in v002.
        sample["ground_truth"] = fo.Detections(
            detections=[]
            if number == 2
            else [
                fo.Detection(label="test-object", bounding_box=[0, 0, 1, 1], mask=mask),
            ]
        )
        sample.save()
        approved = change_reviews(
            cfg, [sample.id], "complete", "iteration-test", confirm_empty=True
        )
        assert approved["changed"] == 1, approved
    report = release(cfg, version=version)
    assert report["status"] == "completed", report
    return report


def verify_live_review(cfg):
    from vloop.review import load_review_dataset

    dataset = load_review_dataset(cfg)
    assert len(dataset) == 5
    for sample in dataset.iter_samples():
        number = int(Path(sample["source_paths"][0]).stem)
        assert sample["review_status"] == "completed"
        if number == 1:
            assert sample["ground_truth"].detections[0].mask[1, 1]


def snapshot_records(path):
    with closing(connect(path)) as db:
        return {row["image_id"]: dict(row) for row in db.execute("SELECT * FROM records")}


def latest_report(cfg, command):
    paths = sorted((cfg.storage_dir / "runs").glob(command + "_*/report.json"))
    report = json.loads(paths[-1].read_text())
    assert report["status"] == "completed", report
    return report


def run(destination):
    import torch

    from vloop.tracking import artifact_directory, client_for

    assert torch.cuda.is_available(), "GPU unavailable in this environment"
    source = Path(__file__).resolve().parents[1]
    weights = source / ".vloop/models/rfdetr/rf-detr-seg-nano.pt"
    assert weights.is_file(), "Official RF-DETR Seg Nano weights are required"
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="vloop-iteration-") as temporary:
        root = Path(temporary) / "repo"
        root.mkdir()
        shutil.copytree(source / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
        (root / ".gitignore").write_text(".vloop/\ninput/\nproject.yaml\n__pycache__/\n*.pyc\n")
        git(root, "init", "-b", "main")
        git(root, "config", "user.name", "Iteration Test")
        git(root, "config", "user.email", "test@example.invalid")
        git(root, "add", "src", ".gitignore")
        git(root, "commit", "-qm", "Snapshot implementation for isolated iteration test")
        head = git(root, "rev-parse", "HEAD")
        cfg = config_from_dict(
            {
                "image_dir": "input",
                "storage_dir": ".vloop",
                "dvc_remote": "../remote",
                "epochs": 1,
                "device": "cuda:0",
                "train_checkpoint": str(weights),
                "release_group_field": "scene",
                "classes": [
                    {"id": 7, "name": "test-object", "prompts": ["object"]},
                    {"id": 42, "name": "absent", "prompts": ["absent"]},
                ],
            },
            root / "project.yaml",
        )
        data = cfg.to_dict()
        data.pop("config_path")
        cfg.config_path.write_text(yaml.safe_dump(data))
        cfg.storage_dir.mkdir()
        environment = {
            **os.environ,
            "PYTHONPATH": str(root / "src") + os.pathsep + str(source / "tests"),
            "NO_ALBUMENTATIONS_UPDATE": "1",
            "MPLCONFIGDIR": str(Path(temporary) / "mpl"),
            "OMP_NUM_THREADS": "2",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }

        def execute(name, arguments):
            print(f"Iteration check: {name}", flush=True)
            log = cfg.storage_dir / f"{name}.log"
            with log.open("w") as output:
                result = subprocess.run(
                    [sys.executable, *arguments],
                    cwd=root,
                    env=environment,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                )
            assert result.returncode == 0, log.read_text()[-12000:]

        script = str(Path(__file__).resolve())
        client = client_for(cfg)
        versions = {}
        retained = {}
        for version in ("v001", "v002"):
            execute(
                "release-" + version,
                [
                    script,
                    "--mode",
                    "release",
                    "--config",
                    str(cfg.config_path),
                    "--dataset-version",
                    version,
                ],
            )
            release = latest_report(cfg, "release")
            execute(
                "train-" + version,
                [
                    "-m",
                    "vloop",
                    "train",
                    "--config",
                    str(cfg.config_path),
                    "--dataset-version",
                    version,
                ],
            )
            trained = latest_report(cfg, "train")
            assert trained["dataset_version"] == version and trained["resume_from_job_id"] is None
            execute(
                "evaluate-" + version,
                [
                    "-m",
                    "vloop",
                    "evaluate",
                    "--config",
                    str(cfg.config_path),
                    "--job-id",
                    trained["job_id"],
                    "--dataset-version",
                    "v001",
                    "--split",
                    "val",
                ],
            )
            evaluated = latest_report(cfg, "evaluate")
            assert evaluated["dataset_version"] == "v001" and evaluated["images"] == 1
            model_dir = artifact_directory(client.get_run(trained["run_id"]))
            eval_dir = artifact_directory(client.get_run(evaluated["run_id"]))
            model = json.loads((model_dir / "model/model.json").read_text())
            assert model["dataset_version"] == version
            assert sha256_file(model_dir / "model/best.pt") == model["sha256"]
            versions[version] = {"release": release, "train": trained, "evaluation": evaluated}
            if version == "v001":
                for directory in (
                    model_dir,
                    eval_dir,
                    Path(trained["result_dir"]),
                    Path(evaluated["result_dir"]),
                ):
                    for path in directory.rglob("*"):
                        if path.is_file():
                            retained[path] = sha256_file(path)
                retained_tag = git(root, "rev-parse", "refs/tags/dataset/v001")

        first, second = versions["v001"], versions["v002"]
        for split in ("val", "test"):
            assert (
                first["release"]["summary"]["splits"][split]
                == second["release"]["summary"]["splits"][split]
            )
        assert first["release"]["summary"]["splits"]["train"]["images"] == 2
        assert second["release"]["summary"]["splits"]["train"]["images"] == 3
        assert (
            first["release"]["summary"]["splits"]["train"]["data_id"]
            != second["release"]["summary"]["splits"]["train"]["data_id"]
        )
        assert descriptor(root, "v002")["parent"] == "v001"
        assert first["evaluation"]["comparison_id"] == second["evaluation"]["comparison_id"]

        # Even when version names differ, unchanged evaluation content has the same identity.
        execute(
            "evaluate-v002-default",
            [
                "-m",
                "vloop",
                "evaluate",
                "--config",
                str(cfg.config_path),
                "--job-id",
                second["train"]["job_id"],
                "--split",
                "val",
            ],
        )
        default_evaluation = latest_report(cfg, "evaluate")
        assert default_evaluation["dataset_version"] == "v002"
        assert default_evaluation["comparison_id"] == first["evaluation"]["comparison_id"]
        assert default_evaluation["metrics"] == second["evaluation"]["metrics"]

        execute(
            "restore-v001",
            ["-m", "vloop", "restore", "--config", str(cfg.config_path), "--version", "v001"],
        )
        execute(
            "verify-live-review",
            [script, "--mode", "verify-review", "--config", str(cfg.config_path)],
        )
        restored = cfg.storage_dir / "releases/restored"
        old = snapshot_records(restored / "v001/dataset/metadata/snapshot.sqlite3")
        new = snapshot_records(restored / "v002/dataset/metadata/snapshot.sqlite3")
        assert len(old) == 4 and len(new) == 5 and old.keys() <= new.keys()
        changed = []
        for image_id, row in old.items():
            assert row["split"] == new[image_id]["split"]
            assert row["coco_id"] == new[image_id]["coco_id"]
            if row["annotation"] != new[image_id]["annotation"]:
                changed.append(image_id)
                assert row["split"] == "train"
        assert len(changed) == 1
        for path, digest in retained.items():
            assert sha256_file(path) == digest, f"Historical artifact changed: {path}"
        assert git(root, "rev-parse", "refs/tags/dataset/v001") == retained_tag
        assert git(root, "rev-parse", "HEAD") == head and not git(root, "status", "--porcelain")
        for entry in versions.values():
            assert client.get_run(entry["train"]["run_id"]).info.status == "FINISHED"
            evaluated_run = client.get_run(entry["evaluation"]["run_id"])
            assert evaluated_run.info.status == "FINISHED"
            assert (
                evaluated_run.data.tags["vloop.comparison_id"]
                == first["evaluation"]["comparison_id"]
            )
        comparison = {}
        for kind in ("boxes", "masks"):
            comparison[kind] = {}
            for metric in ("mAP", "AP50"):
                before = first["evaluation"]["metrics"][kind][metric]
                after = second["evaluation"]["metrics"][kind][metric]
                comparison[kind][metric] = {"v001": before, "v002": after, "delta": after - before}
        result = {
            "status": "passed",
            "scope": "generated images and known masks; no SAM3 or human review",
            "elapsed_seconds": time.monotonic() - started,
            "versions": versions,
            "default_v002_evaluation": default_evaluation,
            "comparison": comparison,
            "preserved_artifacts": len(retained),
            "changed_train_labels": len(changed),
        }
        write_json(destination, result)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "status",
                        "scope",
                        "elapsed_seconds",
                        "comparison",
                        "preserved_artifacts",
                    )
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["release", "verify-review"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset-version", choices=["v001", "v002"])
    parser.add_argument("--output", type=Path, default=Path(".vloop/iteration-integration.json"))
    args = parser.parse_args()
    if args.mode == "release":
        prepare_release(load_config(args.config), args.dataset_version)
    elif args.mode == "verify-review":
        verify_live_review(load_config(args.config))
    else:
        run(args.output.resolve())
