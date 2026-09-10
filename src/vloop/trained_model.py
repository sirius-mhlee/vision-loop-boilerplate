"""Restore the inference contract stored with a training run."""

import json
from pathlib import Path

from rfdetr.models.postprocess import PostProcess

from .runtime import sha256_file
from .tracking import client_for, run_id_for_job


class ForegroundPostProcess(PostProcess):
    def __init__(self, num_classes, num_select):
        super().__init__(num_select=num_select)
        self.num_classes = num_classes

    def _select_topk(self, out_logits):
        # RF-DETR adds one output slot beyond the N remapped project classes.
        # Exclude it before top-k so it cannot consume real detections' slots.
        if out_logits.shape[-1] != self.num_classes + 1:
            raise ValueError("Model output slots do not match the recorded class mapping")
        return super()._select_topk(out_logits[..., : self.num_classes])


def load_model(cfg, job_id, *, device=None):
    from rfdetr import RFDETR

    client = client_for(cfg)
    run_id = run_id_for_job(client, cfg, job_id)
    destination = cfg.storage_dir / "models/mlflow" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = client.download_artifacts(run_id, "model/model.json", str(destination))
    metadata = json.loads(Path(metadata_path).read_text())
    if metadata["schema_version"] != 1 or metadata["weights"] != "best.pt":
        raise ValueError("Unsupported model artifact format")
    weights = Path(metadata_path).parent / "best.pt"
    if not weights.is_file() or sha256_file(weights) != metadata["sha256"]:
        weights = Path(client.download_artifacts(run_id, "model/best.pt", str(destination)))
    if sha256_file(weights) != metadata["sha256"]:
        raise ValueError("Model weights checksum mismatch")
    model = RFDETR.from_checkpoint(str(weights), device=device or cfg.device)
    if model.model_config.num_classes != len(metadata["classes"]):
        raise ValueError("Model classes do not match the saved mapping")
    model.means = metadata["preprocessing"]["mean"]
    model.stds = metadata["preprocessing"]["std"]
    model.model.postprocess = ForegroundPostProcess(
        len(metadata["classes"]),
        metadata["postprocessing"]["num_select"],
    )
    return model, metadata
