import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from vloop.cli import parse_args
from vloop.config import config_from_dict
from vloop.runtime import sha256_file, write_json


def test_training_cli_and_config(project):
    assert parse_args(["train", "--dataset-version", "v001"]).dataset_version == "v001"
    assert parse_args(["train", "--resume", "abc"]).resume == "abc"
    assert parse_args(["experiments", "--no-browser"]).no_browser
    with pytest.raises(SystemExit):
        parse_args(["train", "--dataset-version", "v001", "--resume", "abc"])
    with pytest.raises(SystemExit):
        parse_args(["train", "--resume-run-id", "abc"])
    for values in ({"train_num_workers": -1}, {"train_gradient_checkpointing": "false"}):
        with pytest.raises(ValueError):
            config_from_dict(values, project.config_path)
    cfg = config_from_dict({"train_checkpoint": "weights.pt"}, project.config_path)
    assert cfg.train_checkpoint == project.config_path.parent / "weights.pt"


@pytest.fixture
def training_project(project, monkeypatch):
    pytest.importorskip("mlflow")
    pytest.importorskip("rfdetr.training")
    import vloop.train as training

    cfg = replace(project, epochs=2, device="cpu")
    root = cfg.config_path.parent
    dataset = root / "dataset"
    for folder in ("train", "valid"):
        (dataset / folder).mkdir(parents=True)
        (dataset / folder / "_annotations.coco.json").write_text("{}")
    weights = root / "weights.pt"
    weights.write_bytes(b"fixture weights")
    info = {
        "summary": {
            "classes": [{"id": 7, "name": "test-object", "model_index": 0}],
            "splits": {
                split: {"images": 2, "classes": {"7": {"instances": 1}}}
                for split in ("train", "val")
            },
        },
        "snapshot_sha256": "fixture",
        "dvc_targets": ["images.dvc"],
    }
    monkeypatch.setattr(training, "code_state", lambda: {"commit": "a" * 40, "dirty": False})
    monkeypatch.setattr(training, "project_repo", lambda cfg: root)
    monkeypatch.setattr(training, "descriptor", lambda root, version: info)
    monkeypatch.setattr(training, "git", lambda *args: "b" * 40)
    monkeypatch.setattr(training, "restore_data", lambda *args: (dataset, info))
    monkeypatch.setattr(training, "pretrained_weights", lambda cfg: weights)
    return cfg, info


def test_run_lifecycle_frozen_resume_and_failed_preflight(training_project, monkeypatch):
    import vloop.train as training
    import vloop.training_engine as engine
    from vloop.tracking import artifact_directory, client_for, manifest_hash

    cfg, info = training_project
    manifests = []

    def interrupted(manifest, checkpoint, client, run_id, artifacts, directory, report):
        assert checkpoint is None
        manifests.append(manifest)
        path = artifacts / "resume/epoch-000000.ckpt"
        path.parent.mkdir()
        path.write_bytes(b"completed epoch zero")
        write_json(
            path.parent / "latest.json",
            {
                "filename": path.name,
                "sha256": sha256_file(path),
                "epoch": 0,
                "global_step": 10,
                "binding": manifest["binding"],
            },
        )
        report.update(epochs_completed=1)
        raise KeyboardInterrupt

    monkeypatch.setattr(engine, "fit", interrupted)
    first = training.train(cfg, dataset_version="v001", notes="baseline")
    client = client_for(cfg)
    source = client.get_run(first["run_id"])
    assert first["status"] == "interrupted" and source.info.status == "KILLED"
    assert source.data.tags["mlflow.note.content"] == "baseline"
    assert first["retry"].endswith("--resume " + first["job_id"])

    def resumed(manifest, checkpoint, client, run_id, artifacts, directory, report):
        assert manifest["train_config"]["epochs"] == 2
        assert manifest["train_config"]["lr"] == cfg.learning_rate
        assert manifest["binding"] == manifest_hash(manifest)
        assert checkpoint.read_bytes() == b"completed epoch zero"
        report.update(epochs_completed=2)

    monkeypatch.setattr(engine, "fit", resumed)
    second = training.train(replace(cfg, epochs=99, learning_rate=0.5), resume=first["job_id"])
    resumed_run = client.get_run(second["run_id"])
    assert second["status"] == "completed" and resumed_run.info.status == "FINISHED"
    assert resumed_run.data.tags["vloop.resume_from_run_id"] == first["run_id"]
    assert resumed_run.data.tags["vloop.resume_from_job_id"] == first["job_id"]
    assert second["resume_from_job_id"] == first["job_id"]
    assert second["job_id"] != first["job_id"]
    assert second["run_id"] != first["run_id"] and second["resumed_global_step"] == 10
    assert client.get_run(first["run_id"]).info.status == "KILLED"
    used = json.loads((artifact_directory(resumed_run) / "config.json").read_text())
    assert used["epochs"] == 2
    assert not list((cfg.storage_dir / "runs" / second["job_id"]).glob("resume-*"))

    info["snapshot_sha256"] = "changed"
    changed = training.train(cfg, resume=first["job_id"])
    assert changed["status"] == "failed" and "metadata changed" in changed["error"]
    assert client.get_run(changed["run_id"]).info.status == "FAILED"

    source_dir = artifact_directory(source)
    saved = json.loads((source_dir / "training.json").read_text())
    saved["train_config"]["epochs"] = 99
    write_json(source_dir / "training.json", saved)
    tampered = training.train(cfg, resume=first["job_id"])
    assert "Frozen training configuration changed" in tampered["error"]

    monkeypatch.setattr(training, "code_state", lambda: {"commit": "a" * 40, "dirty": True})
    dirty = training.train(cfg, dataset_version="v001")
    assert dirty["status"] == "failed" and "Commit code" in dirty["error"]
    assert client.get_run(dirty["run_id"]).info.status == "FAILED"


def test_mlflow_initialization_failure_keeps_local_job(project, monkeypatch, capsys):
    import vloop.cli as cli
    import vloop.train as training
    from vloop.tracking import run_id_for_job

    def unavailable(*args, **kwargs):
        raise RuntimeError("MLflow initialization failed")

    monkeypatch.setattr(cli, "load_config", lambda *args: project)
    monkeypatch.setattr(training, "create_run", unavailable)
    assert cli.main(["train", "--dataset-version", "v001"]) == 1
    folder = next((project.storage_dir / "runs").glob("train_*"))
    report = json.loads((folder / "report.json").read_text())
    assert report["status"] == "failed" and report["dataset_version"] == "v001"
    assert "MLflow initialization failed" in (folder / "error.txt").read_text()
    assert (folder / "config.json").is_file() and report.get("run_id") is None
    output = capsys.readouterr()
    assert f"Job ID: {report['job_id']}" in output.out
    assert "MLflow run ID:" not in output.out
    with pytest.raises(ValueError, match="no MLflow run or checkpoint"):
        run_id_for_job(None, project, report["job_id"])


def test_job_mapping_rejects_a_different_mlflow_run(training_project):
    from vloop.runtime import start_run
    from vloop.tracking import create_run, run_id_for_job

    cfg, _ = training_project
    folder, report = start_run(cfg, "train")
    client, run = create_run(cfg, report["job_id"])
    report["run_id"] = run.info.run_id
    write_json(folder / "report.json", report)
    assert run_id_for_job(client, cfg, report["job_id"]) == run.info.run_id

    _, other_report = start_run(cfg, "train")
    _, other = create_run(cfg, other_report["job_id"])
    report["run_id"] = other.info.run_id
    write_json(folder / "report.json", report)
    with pytest.raises(ValueError, match="different vloop job"):
        run_id_for_job(client, cfg, report["job_id"])
    with pytest.raises(ValueError, match="vloop training job ID"):
        run_id_for_job(client, cfg, "../other")
    with pytest.raises(ValueError, match="report not found"):
        run_id_for_job(client, cfg, "train_20000101T000000_00000000")


def test_checkpoint_corruption_and_completed_run_are_rejected(training_project):
    from vloop.tracking import artifact_directory, create_run, manifest_hash, read_resume

    cfg, _ = training_project
    client, run = create_run(cfg, "test")
    folder = artifact_directory(run)
    manifest = {"train_config": {"epochs": 2}}
    manifest["binding"] = manifest_hash(manifest)
    write_json(folder / "training.json", manifest)
    path = folder / "resume/epoch-000001.ckpt"
    path.parent.mkdir()
    path.write_bytes(b"valid")
    pointer = {
        "filename": path.name,
        "sha256": sha256_file(path),
        "epoch": 1,
        "binding": manifest["binding"],
    }
    write_json(path.parent / "latest.json", pointer)
    with pytest.raises(ValueError, match="already reached"):
        read_resume(client, run.info.run_id, cfg.storage_dir / "download")
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_resume(client, run.info.run_id, cfg.storage_dir / "download")


def test_gradient_accumulation_matches_a_full_batch():
    torch = pytest.importorskip("torch")
    pl = pytest.importorskip("pytorch_lightning")
    pytest.importorskip("rfdetr.training")
    from torch.utils.data import DataLoader, TensorDataset

    from vloop.training_engine import TrainingModule

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, samples, targets):
            return self.weight * samples

    class Criterion(torch.nn.Module):
        weight_dict = {"mse": 1}

        def forward(self, output, targets):
            return {"mse": ((output - targets) ** 2).mean()}

    class Tiny(TrainingModule):
        def __init__(self):
            pl.LightningModule.__init__(self)
            self.model = Model()
            self.criterion = Criterion()
            self._use_manual_optimization = False
            self.model_config = SimpleNamespace(fused_optimizer=False)
            self.train_config = SimpleNamespace(
                train_log_sync_dist=False,
                train_log_on_step=False,
                compute_train_metrics=False,
                seed=None,
            )

        def _log_train_progress_metrics(self, *args, **kwargs):
            pass

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

        def on_train_epoch_start(self):
            pass

        def on_train_batch_start(self, batch, batch_idx):
            pass

    values = TensorDataset(torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([2.0, 1.0, 3.0, 2.0]))
    weights = []
    for batch, accumulation in ((4, 1), (1, 4)):
        model = Tiny()
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            accumulate_grad_batches=accumulation,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(model, DataLoader(values, batch_size=batch))
        weights.append(float(model.model.weight.detach()))
    assert weights[0] == pytest.approx(0.8)
    assert weights[1] == pytest.approx(weights[0], abs=1e-6)


def test_postprocess_excludes_unmapped_slot_before_topk():
    torch = pytest.importorskip("torch")
    pytest.importorskip("rfdetr")
    from vloop.trained_model import ForegroundPostProcess

    process = ForegroundPostProcess(num_classes=2, num_select=1)
    result = process(
        {
            "pred_logits": torch.tensor([[[1.0, -1.0, 100.0]]]),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 1.0, 1.0]]]),
            "pred_masks": torch.ones((1, 1, 2, 2)),
        },
        torch.tensor([[4, 6]]),
    )[0]
    assert result["labels"].tolist() == [0]
    assert result["scores"].item() == pytest.approx(torch.sigmoid(torch.tensor(1.0)).item())
    assert result["boxes"].tolist() == [[0.0, 0.0, 6.0, 4.0]]
    assert result["masks"].shape == (1, 1, 4, 6)


def test_best_mask_model_survives_resume_and_interrupted_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("mlflow")
    pytest.importorskip("rfdetr.training")
    from vloop.training_engine import EMA_MONITOR, MONITOR, RFDETREMACallback, TrainCheckpoint

    manifest = {
        "binding": "fixture",
        "model_config": {},
        "train_config": {},
        "preprocessing": {},
        "postprocessing": {},
        "dataset": {"version": "v001", "descriptor": {"summary": {"classes": []}}},
    }
    client = SimpleNamespace(log_batch=lambda *args, **kwargs: None)

    def callback(name):
        folder = tmp_path / name
        folder.mkdir()
        return TrainCheckpoint(client, name, folder, manifest, {}, folder)

    first = callback("first")
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1)
    module = SimpleNamespace(model=model)
    ema = RFDETREMACallback()
    ema.get_ema_model_state_dict = lambda: {"weight": torch.tensor([[2.0]])}
    trainer = SimpleNamespace(
        current_epoch=0,
        global_step=10,
        callback_metrics={MONITOR: 0.6, EMA_MONITOR: 0.7, "val/mAP_50_95": 0.9},
        callbacks=[ema, first],
        optimizers=[SimpleNamespace(param_groups=[{"lr": 1e-4}])],
        save_checkpoint=lambda path: torch.save(first.state_dict(), path),
    )
    first.on_train_epoch_end(trainer, module)
    source_path = first.artifacts / "resume/epoch-000000.ckpt"
    source_hash = sha256_file(source_path)
    second = callback("second")
    second.load_state_dict(torch.load(source_path, weights_only=True))
    second.on_train_start(trainer, module)
    trainer.callbacks = [ema, second]
    trainer.current_epoch = 1
    trainer.global_step = 20
    trainer.callback_metrics = {MONITOR: 0.5, EMA_MONITOR: 0.4, "val/mAP_50_95": 1.0}
    trainer.save_checkpoint = lambda path: torch.save(second.state_dict(), path)
    model.weight.data.fill_(3)
    second.on_train_epoch_end(trainer, module)

    saved = torch.load(second.artifacts / "model/best.pt", weights_only=True)
    metadata = json.loads((second.artifacts / "model/model.json").read_text())
    assert saved["model"]["weight"].item() == 2.0
    assert metadata["score"] == 0.7 and metadata["epoch"] == 0
    assert metadata["weights_kind"] == "ema"
    assert sha256_file(source_path) == source_hash
    pointer = second.artifacts / "resume/latest.json"
    previous = json.loads(pointer.read_text())

    def interrupted_save(path):
        path.write_bytes(b"partial checkpoint")
        raise KeyboardInterrupt

    trainer.current_epoch = 2
    trainer.save_checkpoint = interrupted_save
    with pytest.raises(KeyboardInterrupt):
        second.on_train_epoch_end(trainer, module)
    assert json.loads(pointer.read_text()) == previous
    assert sha256_file(pointer.parent / previous["filename"]) == previous["sha256"]
