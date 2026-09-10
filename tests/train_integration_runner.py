"""Isolated real DVC -> GPU training -> new-process resume -> artifact reload."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import yaml

from vloop.config import ClassConfig, config_from_dict, load_config
from vloop.release_data import initial_split
from vloop.release_dvc import git
from vloop.runtime import sha256_file, write_json


def child(mode, cfg_path, job_id=None):
    if mode == "release":
        from test_release import make_sample

        import vloop.release as release_module

        cfg = load_config(cfg_path)
        cfg.image_dir.mkdir()
        groups = {
            split: next(
                str(i) for i in range(1000) if initial_split("group:" + str(i), cfg) == split
            )
            for split in ("train", "val", "test")
        }
        samples = [
            make_sample(cfg, number + 1, empty=number == 1, group=groups[split])
            for number, split in enumerate(("train", "train", "val", "test"))
        ]
        release_module._samples = lambda *_args, **_kwargs: iter(samples)
        report = release_module.release(cfg, version="v001")
        assert report["status"] == "completed", report
    elif mode == "interrupt":
        from vloop.cli import main
        from vloop.training_engine import TrainCheckpoint

        original = TrainCheckpoint.on_train_epoch_end

        def stop_after_epoch(self, trainer, module):
            original(self, trainer, module)
            raise KeyboardInterrupt

        TrainCheckpoint.on_train_epoch_end = stop_after_epoch
        assert main(["train", "--config", str(cfg_path), "--dataset-version", "v001"]) == 130
    elif mode == "load":
        import numpy as np

        from vloop.tracking import artifact_directory, client_for, run_id_for_job
        from vloop.trained_model import load_model

        cfg = load_config(cfg_path)
        client = client_for(cfg)
        run_id = run_id_for_job(client, cfg, job_id)
        folder = artifact_directory(client.get_run(run_id)) / "model"
        metadata = json.loads((folder / "model.json").read_text())
        weights = folder / metadata["weights"]
        assert sha256_file(weights) == metadata["sha256"]
        model, metadata = load_model(cfg, job_id, device="cuda:0")
        assert model.model_config.num_classes == 2
        output = model.predict(np.zeros((13, 23, 3), dtype=np.uint8), threshold=0.0)
        assert output.mask.shape[1:] == (13, 23)
        assert set(output.class_id.tolist()) <= {0, 1}
        write_json(
            cfg.storage_dir / "model-reload.json",
            {
                "detections": len(output),
                "mask_shape": list(output.mask.shape),
                "classes": metadata["classes"],
            },
        )


def run(destination):
    import torch

    assert torch.cuda.is_available(), "GPU unavailable in this environment"
    source = Path(__file__).resolve().parents[1]
    weights = source / ".vloop/models/rfdetr/rf-detr-seg-nano.pt"
    assert weights.is_file(), "Download official RF-DETR weights before this integration test"
    with tempfile.TemporaryDirectory(prefix="vloop-train-") as temporary:
        root = Path(temporary) / "repo"
        root.mkdir()
        shutil.copytree(source / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
        (root / ".gitignore").write_text(".vloop/\ninput/\nproject.yaml\n__pycache__/\n*.pyc\n")
        git(root, "init", "-b", "main")
        git(root, "add", "src", ".gitignore")
        git(
            root,
            "-c",
            "user.name=Training Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "Snapshot training implementation for isolated integration",
        )
        git(root, "config", "user.name", "Training Test")
        git(root, "config", "user.email", "test@example.invalid")
        cfg = config_from_dict(
            {
                "image_dir": "input",
                "storage_dir": ".vloop",
                "dvc_remote": "../remote",
                "epochs": 2,
                "device": "cuda:0",
                "train_checkpoint": str(weights),
                "release_group_field": "scene",
            },
            root / "project.yaml",
        )
        cfg = replace(
            cfg,
            classes=(
                ClassConfig(7, "test-object", ("object",)),
                ClassConfig(42, "absent", ("absent",)),
            ),
        )
        data = cfg.to_dict()
        data.pop("config_path")
        cfg.config_path.write_text(yaml.safe_dump(data))
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

        def execute(name, args):
            log = cfg.storage_dir / f"{name}.log"
            cfg.storage_dir.mkdir(exist_ok=True)
            with log.open("w") as output:
                result = subprocess.run(
                    [sys.executable, *args],
                    cwd=root,
                    env=environment,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                )
            if result.returncode:
                errors = list((cfg.storage_dir / "runs").glob("train_*/error.txt"))
                detail = errors[-1].read_text() if errors else ""
                raise AssertionError(log.read_text()[-8000:] + "\n" + detail)

        script = str(Path(__file__).resolve())
        execute("release", [script, "--mode", "release", "--config", str(cfg.config_path)])
        before = git(root, "rev-parse", "HEAD")
        started = time.monotonic()
        execute("first", [script, "--mode", "interrupt", "--config", str(cfg.config_path)])
        reports = sorted((cfg.storage_dir / "runs").glob("train_*/report.json"))
        first = json.loads(reports[-1].read_text())
        assert first["status"] == "interrupted" and first["epochs_completed"] == 1, first
        from vloop.tracking import artifact_directory, client_for

        client = client_for(cfg)
        first_artifacts = artifact_directory(client.get_run(first["run_id"]))
        first_latest = json.loads((first_artifacts / "resume/latest.json").read_text())
        first_checkpoint = first_artifacts / "resume" / first_latest["filename"]
        state = torch.load(first_checkpoint, map_location="cpu", weights_only=False)
        assert state["optimizer_states"][0]["state"] and state["lr_schedulers"]
        assert state["global_step"] > 0
        del state
        # Current YAML deliberately differs: resume must still run only the original two epochs.
        data["epochs"] = 99
        data["learning_rate"] = 0.5
        cfg.config_path.write_text(yaml.safe_dump(data))
        execute(
            "resume",
            [
                "-m",
                "vloop",
                "train",
                "--config",
                str(cfg.config_path),
                "--resume",
                first["job_id"],
            ],
        )
        reports = sorted((cfg.storage_dir / "runs").glob("train_*/report.json"))
        second = json.loads(reports[-1].read_text())
        assert second["status"] == "completed" and second["epochs_completed"] == 2, second
        assert second["global_step"] > first["global_step"]
        assert second["resume_from_job_id"] == first["job_id"]
        assert second["job_id"] != first["job_id"]
        assert second["best_mask_map"] >= first["best_mask_map"]
        if second["best_mask_map"] == first["best_mask_map"]:
            assert second["best_epoch"] == first["best_epoch"]
        assert second["resumed_global_step"] == first["global_step"]
        assert client.get_run(first["run_id"]).info.status == "KILLED"
        assert client.get_run(second["run_id"]).info.status == "FINISHED"
        assert client.get_run(second["run_id"]).data.params["epochs"] == "2"
        assert client.get_metric_history(second["run_id"], "train/loss")
        assert client.get_metric_history(second["run_id"], "val/segm_mAP_50_95")
        assert sha256_file(first_checkpoint) == first_latest["sha256"]
        execute(
            "reload",
            [
                script,
                "--mode",
                "load",
                "--config",
                str(cfg.config_path),
                "--job-id",
                second["job_id"],
            ],
        )
        assert git(root, "rev-parse", "HEAD") == before
        assert not git(root, "status", "--porcelain")
        result = {
            "first": first,
            "resumed": second,
            "elapsed_seconds": time.monotonic() - started,
            "reload": json.loads((cfg.storage_dir / "model-reload.json").read_text()),
            "fixture": "4 generated images, sparse classes 7/42, real DVC + GPU + MLflow",
        }
        write_json(destination, result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["release", "interrupt", "load"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--output", type=Path, default=Path(".vloop/train-integration.json"))
    args = parser.parse_args()
    if args.mode:
        child(args.mode, args.config, args.job_id)
    else:
        run(args.output.resolve())
