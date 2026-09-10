"""Local MLflow runs, artifacts, and the localhost experiment UI."""

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from urllib.parse import unquote, urlparse

from .runtime import sha256_file


def client_for(cfg):
    try:
        from mlflow import MlflowClient
    except ImportError as exc:
        raise RuntimeError(
            "Install training dependencies: python -m pip install -e '.[pipeline]'"
        ) from exc

    (cfg.storage_dir / "mlflow").mkdir(parents=True, exist_ok=True)
    return MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)


def create_run(cfg, job_id, *, parent=None, notes=None):
    client = client_for(cfg)
    experiment = client.get_experiment_by_name(cfg.dataset_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            cfg.dataset_name,
            artifact_location=(cfg.storage_dir / "mlflow/artifacts").as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id
    tags = {"vloop.kind": "train", "vloop.job_id": job_id, "mlflow.runName": job_id}
    if parent:
        tags["vloop.resume_from_run_id"] = parent
    if notes:
        tags["mlflow.note.content"] = notes
    return client, client.create_run(experiment_id, tags=tags)


def artifact_directory(run):
    """Our SQLite tracking store uses local artifacts, enabling atomic checkpoint writes."""
    uri = urlparse(run.info.artifact_uri)
    if uri.scheme != "file" or uri.netloc not in ("", "localhost"):
        raise ValueError("This version requires a local MLflow artifact directory")
    path = Path(unquote(uri.path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_id_for_job(client, cfg, job_id):
    """Resolve the recorded ID mapping; never guess a run from its name or timestamp."""
    if not isinstance(job_id, str) or not re.fullmatch(
        r"train_[0-9]{8}T[0-9]{6}_[0-9a-f]{8}", job_id
    ):
        raise ValueError("Use a vloop training job ID: train_<timestamp>_<suffix>")
    path = cfg.storage_dir / "runs" / job_id / "report.json"
    if not path.is_file():
        raise ValueError(f"Training job report not found: {path}")
    report = json.loads(path.read_text())
    if report.get("job_id") != job_id or report.get("command") != "train":
        raise ValueError("Training job report does not match the requested job")
    run_id = report.get("run_id")
    if run_id is None:
        raise ValueError(
            f"Job {job_id} has no MLflow run or checkpoint; inspect its error report, "
            "fix initialization, and start a new training job"
        )
    if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("Invalid MLflow ID in the training job report")
    run = client.get_run(run_id)
    if run.data.tags.get("vloop.kind") != "train" or run.data.tags.get("vloop.job_id") != job_id:
        raise ValueError("The recorded MLflow run belongs to a different vloop job")
    return run_id


def read_resume(client, run_id, destination):
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("Invalid MLflow run ID")
    run = client.get_run(run_id)
    if run.data.tags.get("vloop.kind") != "train":
        raise ValueError("The source run is not a vloop training run")
    destination.mkdir(parents=True, exist_ok=True)

    def read(name):
        return Path(client.download_artifacts(run_id, name, str(destination)))

    manifest = json.loads(read("training.json").read_text())
    if manifest.get("binding") != manifest_hash(manifest):
        raise ValueError("Frozen training configuration changed")
    latest = json.loads(read("resume/latest.json").read_text())
    if latest.get("binding") != manifest["binding"]:
        raise ValueError("Resume metadata belongs to different training settings")
    if not re.fullmatch(r"epoch-[0-9]+\.ckpt", latest["filename"]):
        raise ValueError("Invalid resume checkpoint filename")
    checkpoint = read("resume/" + latest["filename"])
    if sha256_file(checkpoint) != latest["sha256"]:
        raise ValueError("Resume checkpoint checksum mismatch")
    if latest["epoch"] + 1 >= manifest["train_config"]["epochs"]:
        raise ValueError("This run already reached its configured epoch count; start a new run")
    return manifest, checkpoint, latest


def manifest_hash(manifest):
    frozen = {k: v for k, v in manifest.items() if k not in ("binding", "runtime")}
    return hashlib.sha256(
        json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def experiments(cfg, *, no_browser=False):
    # Initialize/migrate the same SQLite store used by training before serving it.
    client_for(cfg)
    url = f"http://127.0.0.1:{cfg.mlflow_port}"
    print(f"MLflow: {url}", flush=True)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--backend-store-uri",
            cfg.mlflow_tracking_uri,
            "--default-artifact-root",
            (cfg.storage_dir / "mlflow/artifacts").as_uri(),
            "--no-serve-artifacts",
            "--host",
            "127.0.0.1",
            "--port",
            str(cfg.mlflow_port),
            "--workers",
            "1",
        ],
        start_new_session=True,
    )
    try:
        if not no_browser:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    return process.returncode
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1):
                        break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.2)
            else:
                raise RuntimeError(f"MLflow did not become ready at {url}")
            webbrowser.open(url)
        return process.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
