"""Train immutable dataset releases and resume by vloop job ID."""

import traceback
from contextlib import ExitStack
from importlib.metadata import version as package_version
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from .config import config_from_dict
from .release_dvc import descriptor, git, project_repo, restore_data, version_name
from .runtime import finish_run, git_output, project_lock, sha256_file, start_run, write_json
from .tracking import artifact_directory, create_run, manifest_hash, read_resume, run_id_for_job

TRAIN_DEPENDENCIES = (
    "torch",
    "torchvision",
    "rfdetr",
    "pytorch-lightning",
    "mlflow",
    "numpy",
    "transformers",
    "pycocotools",
    "torchmetrics",
    "scipy",
    "Pillow",
)


def code_state():
    directory = Path(__file__).resolve().parent
    return {
        "commit": git_output(directory, "rev-parse", "HEAD"),
        "dirty": bool(git_output(directory, "status", "--porcelain")),
    }


def pretrained_weights(cfg):
    from rfdetr.assets.model_weights import download_pretrain_weights, validate_pretrain_weights

    path = cfg.train_checkpoint or cfg.storage_dir / "models/rfdetr/rf-detr-seg-nano.pt"
    if cfg.train_checkpoint is not None:
        if not path.is_file():
            raise ValueError(f"Training checkpoint not found: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        download_pretrain_weights(str(path))
        validate_pretrain_weights(str(path), strict=True)
    if path.stat().st_size == 0:
        raise ValueError(f"Empty training checkpoint: {path}")
    return path


def check_dataset(info):
    for split in ("train", "val"):
        counts = info["summary"]["splits"][split]
        if not counts["images"] or not any(c["instances"] for c in counts["classes"].values()):
            raise ValueError(f"Training requires {split} images with at least one labeled object")


def build_manifest(cfg, dataset_version, root, dataset, info, weights, code, dependencies, output):
    from rfdetr.config import RFDETRSegNanoConfig, SegmentationTrainConfig

    classes = info["summary"]["classes"]
    mc = RFDETRSegNanoConfig(
        num_classes=len(classes),
        pretrain_weights=str(weights),
        device="cuda:0" if cfg.device == "cuda" else cfg.device,
        gradient_checkpointing=cfg.train_gradient_checkpointing,
        model_name="RFDETRSegNano",
    )
    tc = SegmentationTrainConfig(
        dataset_dir=str(dataset),
        output_dir=str(output),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        grad_accum_steps=cfg.grad_accum_steps,
        lr=cfg.learning_rate,
        seed=cfg.seed,
        num_workers=cfg.train_num_workers,
        persistent_workers=False,
        tensorboard=False,
        mlflow=False,
        wandb=False,
        class_names=[c["name"] for c in classes],
        run_test=False,
        accelerator="cpu" if cfg.device == "cpu" else "gpu",
    )
    manifest = {
        "schema_version": 1,
        "config": cfg.to_dict(),
        "code": code,
        "dependencies": dependencies,
        "dataset": {
            "version": dataset_version,
            "git_commit": git(root, "rev-parse", f"refs/tags/dataset/{dataset_version}"),
            "descriptor": info,
        },
        "pretrained": {"path": str(weights), "sha256": sha256_file(weights)},
        "model_config": mc.model_dump(mode="json"),
        "train_config": tc.model_dump(mode="json"),
        "preprocessing": {
            "color": "RGB",
            "input_range": [0, 1],
            "resize": "square_bilinear",
            "resolution": mc.resolution,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "postprocessing": {
            "num_select": mc.num_select,
            "foreground_slots": list(range(len(classes))),
            "excluded_output_slot": len(classes),
            "score_activation": "sigmoid",
            "mask_logit_threshold": 0.0,
            "coordinate_space": "original_image",
            "eval_confidence": cfg.eval_confidence,
            "display_confidence": cfg.display_confidence,
            "eval_max_detections": cfg.eval_max_detections,
        },
        "adapter": "rfdetr-1.8.2-lightning-foreground-accumulation-v1",
    }
    manifest["binding"] = manifest_hash(manifest)
    return manifest


def train(cfg, *, dataset_version=None, resume=None, notes=None):
    if bool(dataset_version) == bool(resume):
        raise ValueError("Choose --dataset-version or --resume")
    # Same lock as ingest/autolabel: this project's SAM 3 and training run sequentially.
    with project_lock(cfg), ExitStack() as stack:
        directory, report = start_run(cfg, "train")
        report.update(dataset_version=dataset_version, resume_from_job_id=resume)
        write_json(directory / "report.json", report)
        print(f"Job ID: {report['job_id']}", flush=True)
        client = None
        run_id = None
        artifacts = None
        try:
            client, run = create_run(cfg, report["job_id"], notes=notes)
            run_id = run.info.run_id
            report["run_id"] = run_id  # Internal MLflow ID; the CLI uses job_id.
            write_json(directory / "report.json", report)
            artifacts = artifact_directory(run)
            resume_run_id = None
            if resume:
                resume_run_id = run_id_for_job(client, cfg, resume)
                report["resume_from_run_id"] = resume_run_id
                client.set_tag(run_id, "vloop.resume_from_job_id", resume)
                client.set_tag(run_id, "vloop.resume_from_run_id", resume_run_id)
                write_json(directory / "report.json", report)
            dependencies = {name: package_version(name) for name in TRAIN_DEPENDENCIES}
            for name, expected in (
                ("rfdetr", "1.8.2"),
                ("pytorch-lightning", "2.6.5"),
                ("mlflow", "3.16.0"),
            ):
                if dependencies[name] != expected:
                    raise ValueError(f"Training adapter requires {name}=={expected}")
            code = code_state()
            report["code"] = code
            if code["dirty"]:
                raise ValueError("Commit code changes before training so the run is reproducible")
            checkpoint = None
            manifest = None
            if resume_run_id:
                source = Path(
                    stack.enter_context(TemporaryDirectory(dir=directory, prefix="resume-"))
                )
                manifest, checkpoint, latest = read_resume(client, resume_run_id, source)
                if code != manifest["code"] or dependencies != manifest["dependencies"]:
                    raise ValueError(
                        "Resume requires the original committed code and training dependencies"
                    )
                frozen = dict(manifest["config"])
                frozen.pop("config_path", None)
                # Current local storage/remote paths can differ; learning settings stay frozen.
                frozen.update(storage_dir=str(cfg.storage_dir), dvc_remote=str(cfg.dvc_remote))
                cfg = config_from_dict(frozen, cfg.config_path)
                dataset_version = manifest["dataset"]["version"]
                report.update(
                    resumed_epoch=latest["epoch"], resumed_global_step=latest["global_step"]
                )
            write_json(directory / "config.json", cfg.to_dict())
            version_name(dataset_version)
            report["dataset_version"] = dataset_version
            root = project_repo(cfg)
            info = descriptor(root, dataset_version)
            if manifest and (
                git(root, "rev-parse", f"refs/tags/dataset/{dataset_version}")
                != manifest["dataset"]["git_commit"]
                or info != manifest["dataset"]["descriptor"]
            ):
                raise ValueError("Dataset tag or release metadata changed since the original run")
            check_dataset(info)
            with project_lock(SimpleNamespace(storage_dir=cfg.storage_dir / "releases")):
                dataset, info = restore_data(cfg, dataset_version)
            if manifest is None:
                weights = pretrained_weights(cfg)
                manifest = build_manifest(
                    cfg,
                    dataset_version,
                    root,
                    dataset,
                    info,
                    weights,
                    code,
                    dependencies,
                    directory / "engine",
                )
            report["dataset_dir"] = str(dataset)
            report["annotation_bytes"] = {
                split: (dataset / folder / "_annotations.coco.json").stat().st_size
                for split, folder in (("train", "train"), ("val", "valid"))
            }
            write_json(artifacts / "training.json", manifest)
            client.log_artifact(run_id, str(directory / "dependencies.json"))
            from mlflow.entities import Param, RunTag

            client.log_batch(
                run_id,
                params=[
                    Param(key, str(value))
                    for key, value in {
                        "dataset_version": dataset_version,
                        "train_model": cfg.train_model,
                        "epochs": cfg.epochs,
                        "batch_size": cfg.batch_size,
                        "grad_accum_steps": cfg.grad_accum_steps,
                        "learning_rate": cfg.learning_rate,
                        "seed": cfg.seed,
                        "num_workers": cfg.train_num_workers,
                        "monitor": "val/segm_mAP_50_95",
                    }.items()
                ],
                tags=[
                    RunTag("mlflow.source.git.commit", code["commit"]),
                    RunTag("vloop.dataset_commit", manifest["dataset"]["git_commit"]),
                    RunTag("vloop.dataset_version", dataset_version),
                ],
            )
            from .training_engine import fit

            fit(manifest, checkpoint, client, run_id, artifacts, directory, report)
            report["status"] = "completed"
        except KeyboardInterrupt:
            report.update(
                status="interrupted",
                error="Training interrupted; resume from the last completed epoch",
            )
        except Exception as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            (directory / "error.txt").write_text(traceback.format_exc())
        finally:
            has_checkpoint = artifacts and (artifacts / "resume/latest.json").is_file()
            if has_checkpoint and report["status"] != "completed":
                report["retry"] = (
                    f"vloop train --config {cfg.config_path} --resume {report['job_id']}"
                )
            elif not has_checkpoint and report["status"] == "interrupted":
                report["error"] = (
                    "Interrupted before the first epoch checkpoint; start a new training run"
                )
            if client and run_id:
                try:
                    for name in ("error.txt", "config.json"):
                        if (directory / name).exists():
                            client.log_artifact(run_id, str(directory / name))
                    client.set_tag(run_id, "vloop.status", report["status"])
                    client.set_terminated(
                        run_id,
                        {
                            "completed": "FINISHED",
                            "interrupted": "KILLED",
                            "failed": "FAILED",
                        }[report["status"]],
                    )
                except Exception as exc:
                    report.update(status="failed", tracking_error=str(exc))
            finish_run(directory, report)
            if artifacts:
                write_json(artifacts / "report.json", report)
        return report
