"""RF-DETR 1.8.2's Lightning primitives with local, atomic MLflow checkpoints."""

import math
import random
import resource
import time
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Checkpoint
from rfdetr.config import RFDETRSegNanoConfig, SegmentationTrainConfig
from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer
from rfdetr.training.callbacks import COCOEvalCallback, RFDETREMACallback

from .runtime import sha256_file, write_json
from .trained_model import ForegroundPostProcess

MONITOR = "val/segm_mAP_50_95"
EMA_MONITOR = "val/ema_segm_mAP_50_95"


def atomic_torch_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class TrainingModule(RFDETRModelModule):
    def training_step(self, batch, batch_idx):
        # RF-DETR 1.8.2 divides the returned loss by accumulation steps; Lightning
        # 2.6 divides it again. Undo RF-DETR's division so gradients are averaged once.
        output = super().training_step(batch, batch_idx)
        loss = output["loss"] if isinstance(output, dict) else output
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Training loss is not finite")
        factor = self.trainer.accumulate_grad_batches
        if isinstance(output, dict):
            return {**output, "loss": output["loss"] * factor}
        return output * factor


class TrainCheckpoint(Checkpoint):
    def __init__(self, client, run_id, artifacts, manifest, report, directory):
        self.client, self.run_id, self.artifacts = client, run_id, artifacts
        self.manifest, self.report, self.directory = manifest, report, directory
        self.best = None
        self.rng = None

    def state_dict(self):
        # Include best weights in resume state so a new run retains the old winner
        # even if none of its subsequent validation scores improve.
        return {"best": self.best}

    def load_state_dict(self, state_dict):
        self.best = state_dict["best"]

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["vloop"] = {
            "binding": self.manifest["binding"],
            "run_id": self.run_id,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        self.rng = checkpoint["vloop"]
        if self.rng["binding"] != self.manifest["binding"]:
            raise ValueError("Checkpoint belongs to different data, code, or training settings")
        if not checkpoint.get("optimizer_states") or not checkpoint.get("lr_schedulers"):
            raise ValueError("Checkpoint lacks optimizer/scheduler state; cannot resume training")

    def on_train_start(self, trainer, pl_module):
        if self.rng:
            random.setstate(self.rng["python_rng"])
            np.random.set_state(self.rng["numpy_rng"])
            torch.set_rng_state(self.rng["torch_rng"].cpu())
            if self.rng["cuda_rng"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in self.rng["cuda_rng"]])
        if self.best:
            self.export_best()

    def export_best(self):
        model_dir = self.artifacts / "model"
        path = model_dir / "best.pt"
        atomic_torch_save(self.best["checkpoint"], path)
        write_json(
            model_dir / "model.json",
            {
                "schema_version": 1,
                "weights": "best.pt",
                "sha256": sha256_file(path),
                "model_config": self.manifest["model_config"],
                "classes": self.manifest["dataset"]["descriptor"]["summary"]["classes"],
                "preprocessing": self.manifest["preprocessing"],
                "postprocessing": self.manifest["postprocessing"],
                "dataset_version": self.manifest["dataset"]["version"],
                "monitor": MONITOR,
                "score": self.best["score"],
                "epoch": self.best["epoch"],
                "weights_kind": self.best["kind"],
            },
        )

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = {
            key: float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
            for key, value in trainer.callback_metrics.items()
            if not isinstance(value, torch.Tensor) or value.numel() == 1
        }
        score = metrics.get(MONITOR)
        if score is None or not math.isfinite(score) or score < 0:
            raise ValueError("Validation mask mAP is unavailable; check validation ground truth")
        epoch = trainer.current_epoch
        kind = "regular"
        if metrics.get(EMA_MONITOR, -1) > score:
            score, kind = metrics[EMA_MONITOR], "ema"
        if self.best is None or score > self.best["score"]:
            if kind == "ema":
                ema = next(c for c in trainer.callbacks if isinstance(c, RFDETREMACallback))
                state = ema.get_ema_model_state_dict()
            else:
                state = pl_module.model.state_dict()
            weights = {key: value.detach().cpu().clone() for key, value in state.items()}
            self.best = {
                "score": score,
                "epoch": epoch,
                "kind": kind,
                "checkpoint": {
                    "model": weights,
                    "model_name": "RFDETRSegNano",
                    "model_config": self.manifest["model_config"],
                    "args": {**self.manifest["train_config"], **self.manifest["model_config"]},
                    "epoch": epoch,
                    "rfdetr_version": "1.8.2",
                },
            }
            self.export_best()

        folder = self.artifacts / "resume"
        folder.mkdir(exist_ok=True)
        path = folder / f"epoch-{epoch:06d}.ckpt"
        temporary = path.with_suffix(".tmp")
        trainer.save_checkpoint(temporary)
        temporary.replace(path)
        # Publish the manifest after the checkpoint is complete. Keep the previous
        # file until then, so an interrupted save leaves the previous epoch usable.
        write_json(
            folder / "latest.json",
            {
                "filename": path.name,
                "sha256": sha256_file(path),
                "epoch": epoch,
                "global_step": trainer.global_step,
                "binding": self.manifest["binding"],
            },
        )
        for previous in folder.glob("epoch-*.ckpt"):
            if previous != path:
                previous.unlink()

        from mlflow.entities import Metric

        metrics["epoch"] = epoch
        metrics["global_step"] = trainer.global_step
        metrics["learning_rate"] = trainer.optimizers[0].param_groups[0]["lr"]
        self.client.log_batch(
            self.run_id,
            metrics=[
                Metric(key, value, int(time.time() * 1000), epoch)
                for key, value in metrics.items()
                if math.isfinite(value)
            ],
        )
        self.report.update(
            epochs_completed=epoch + 1,
            global_step=trainer.global_step,
            best_mask_map=self.best["score"],
            best_epoch=self.best["epoch"],
            checkpoint=str(path),
            model_dir=str(self.artifacts / "model"),
        )
        write_json(self.directory / "report.json", self.report)
        print(f"Epoch {epoch + 1}: mask mAP={score:.6f}, checkpoint saved", flush=True)


def fit(manifest, checkpoint, client, run_id, artifacts, directory, report):
    tc = SegmentationTrainConfig(**manifest["train_config"])
    tc.dataset_dir = report["dataset_dir"]
    tc.output_dir = str(directory / "engine")
    mc = RFDETRSegNanoConfig(**manifest["model_config"])
    if checkpoint:
        # None would download DINO backbone weights. Initialize from the local
        # checkpoint, then let Lightning restore optimizer/loop/callback states.
        mc.pretrain_weights = str(checkpoint)
    elif sha256_file(Path(mc.pretrain_weights)) != manifest["pretrained"]["sha256"]:
        raise ValueError("Pretrained weights changed before model loading")
    seed_everything(tc.seed, workers=True)
    if mc.device.startswith("cuda"):
        torch.cuda.set_device(mc.device)
        torch.cuda.reset_peak_memory_stats()
    module = TrainingModule(mc, tc)
    module.strict_loading = True
    module.postprocess = ForegroundPostProcess(mc.num_classes, mc.num_select)
    data = RFDETRDataModule(mc, tc)
    data.setup("fit")
    report["loader_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    write_json(directory / "report.json", report)
    callback = TrainCheckpoint(client, run_id, artifacts, manifest, report, directory)
    callbacks = [
        RFDETREMACallback(
            decay=tc.ema_decay, tau=tc.ema_tau, update_interval_steps=tc.ema_update_interval
        ),
        COCOEvalCallback(
            max_dets=tc.eval_max_dets,
            segmentation=True,
            eval_interval=1,
            log_per_class_metrics=True,
        ),
        callback,
    ]
    trainer = build_trainer(
        tc,
        mc,
        accelerator="cpu" if mc.device == "cpu" else "gpu",
        devices=1 if mc.device == "cpu" else mc.device.split(":")[-1] + ",",
        callbacks=callbacks,
        logger=False,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    manifest["runtime"] = {"precision": str(trainer.precision)}
    write_json(artifacts / "training.json", manifest)
    try:
        try:
            trainer.fit(
                module, data, ckpt_path=str(checkpoint) if checkpoint else None, weights_only=False
            )
        except SystemExit as exc:
            # Lightning converts Ctrl+C to sys.exit(1). Preserve vloop's interrupted
            # report and KILLED MLflow run rather than escaping the orchestrator.
            if trainer.interrupted:
                raise KeyboardInterrupt from exc
            raise RuntimeError(f"Training process exited: {exc.code}") from exc
        if trainer.interrupted:
            raise KeyboardInterrupt
        if report.get("epochs_completed", 0) != tc.epochs:
            raise RuntimeError("Training ended before the configured epoch count")
    finally:
        report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        if mc.device.startswith("cuda"):
            report["cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            report["cuda_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
