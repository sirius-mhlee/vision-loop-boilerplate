import json
import re
import subprocess
import sys
import tempfile
import traceback
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

from .config import Config
from .runtime import finish_run, git_output, project_lock, sha256_file, start_run

EXPECTED_VERSIONS = {"fiftyone": "1.21.0", "rfdetr": "1.8.2", "mlflow": "3.16.0"}
DEPENDENCIES = (
    "PyYAML",
    "Pillow",
    "torch",
    "torchvision",
    "fiftyone",
    "sam3",
    "rfdetr",
    "mlflow",
    "dvc",
)


def check_cuda(device: str) -> str:
    code = """
import json
import sys
import torch
device = sys.argv[1]
if not torch.cuda.is_available():
    raise RuntimeError('torch.cuda.is_available() is false')
if not device.startswith('cuda'):
    device = 'cuda:0'
x = torch.ones((32, 32), device=device)
y = x @ x
torch.cuda.synchronize()
if not torch.equal(y.cpu(), torch.full((32, 32), 32.0)):
    raise RuntimeError('CUDA matrix multiplication returned incorrect values')
print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'device': torch.cuda.get_device_name(device),
                  'capability': torch.cuda.get_device_capability(device),
                  'operation': '32x32 matrix multiplication passed'}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, device], capture_output=True, text=True, timeout=90
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout.strip()


def check_checkpoint(cfg: Config) -> str:
    if cfg.sam3_checkpoint is None:
        raise ValueError(
            "Set sam3_checkpoint after obtaining access at https://huggingface.co/facebook/sam3"
        )
    if not cfg.sam3_checkpoint.is_file() or cfg.sam3_checkpoint.stat().st_size == 0:
        raise ValueError(f"Checkpoint missing or empty: {cfg.sam3_checkpoint}")
    return (
        f"Readable; sha256={sha256_file(cfg.sam3_checkpoint)}; "
        "model loading checked only by --sam3-image"
    )


def check_sam3_source(cfg: Config) -> str:
    if cfg.sam3_source_dir is None or cfg.sam3_commit is None:
        raise ValueError(
            "Set sam3_source_dir and sam3_commit after installing a pinned SAM 3 checkout"
        )
    commit = git_output(cfg.sam3_source_dir, "rev-parse", "HEAD")
    if commit != cfg.sam3_commit:
        raise ValueError(f"SAM 3 checkout mismatch: {commit}")
    if git_output(cfg.sam3_source_dir, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("SAM 3 checkout has uncommitted modifications")
    metadata = distribution("sam3").read_text("direct_url.json")
    if not metadata:
        raise ValueError("Install SAM 3 from the pinned checkout with pip install -e")
    direct = json.loads(metadata)
    if direct.get("url") != cfg.sam3_source_dir.as_uri() or not direct.get("dir_info", {}).get(
        "editable"
    ):
        raise ValueError("Installed SAM 3 does not point to sam3_source_dir as an editable install")
    return commit


def check_writable(path: Path) -> str:
    parent = path
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir():
        raise ValueError(f"Not a directory: {parent}")
    with tempfile.NamedTemporaryFile(prefix=".vloop-doctor-", dir=parent):
        pass
    return f"Writable: {path} (checked existing parent {parent})"


def check_remote(cfg: Config) -> str:
    root = Path(git_output(cfg.config_path.parent, "rev-parse", "--show-toplevel"))
    if cfg.dvc_remote.is_relative_to(root) or root.is_relative_to(cfg.dvc_remote):
        raise ValueError("dvc_remote must be separate from the Git repository")
    if cfg.dvc_remote.is_relative_to(cfg.storage_dir) or cfg.storage_dir.is_relative_to(
        cfg.dvc_remote
    ):
        raise ValueError("dvc_remote must be separate from storage_dir")
    return check_writable(cfg.dvc_remote)


def doctor(cfg: Config, *, sam3_image: Path | None = None) -> dict:
    with project_lock(cfg):
        directory, report = start_run(cfg, "doctor")
        checks = []

        def check(name, function):
            try:
                detail = function()
                checks.append({"name": name, "status": "passed", "detail": str(detail)})
            except Exception as exc:
                error_name = re.sub(r"[^a-zA-Z0-9_-]", "-", name)
                (directory / f"{error_name}-error.txt").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
                checks.append(
                    {"name": name, "status": "failed", "detail": f"{type(exc).__name__}: {exc}"}
                )

        def python_version():
            if sys.version_info[:2] != (3, 12):
                raise ValueError(f"Expected Python 3.12, got {sys.version.split()[0]}")
            return sys.version.split()[0]

        def inputs():
            errors = cfg.input_errors()
            if errors:
                raise ValueError("; ".join(errors))
            return f"{cfg.image_dir}; {len(cfg.classes)} classes"

        def dependency(name):
            try:
                installed = version(name)
            except PackageNotFoundError as exc:
                raise ValueError(f"{name} is not installed; see README.md") from exc
            expected = EXPECTED_VERSIONS.get(name)
            if expected is not None and installed != expected:
                raise ValueError(f"Compatibility target {expected}; installed {installed}")
            return installed

        try:
            check("python", python_version)
            check("inputs", inputs)
            for name in DEPENDENCIES:
                check(f"dependency:{name}", lambda name=name: dependency(name))
            for name in ("images", "runs", "fiftyone/db", "mlflow", "models", "releases"):
                check(f"storage:{name}", lambda name=name: check_writable(cfg.storage_dir / name))
            check("dvc_remote", lambda: check_remote(cfg))
            check(
                "nvidia-smi",
                lambda: subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,driver_version,memory.total",
                        "--format=csv,noheader",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=20,
                    check=True,
                ).stdout.strip(),
            )
            check("cuda", lambda: check_cuda(cfg.device))
            check("sam3_checkpoint", lambda: check_checkpoint(cfg))
            check("sam3_source", lambda: check_sam3_source(cfg))
            report["sam3_inference"] = "not_requested"
            if sam3_image is not None:
                from .sam3 import smoke_predict

                check("sam3_inference", lambda: smoke_predict(cfg, sam3_image, directory))
                report["sam3_inference"] = checks[-1]["status"]
            report["status"] = (
                "failed" if any(c["status"] == "failed" for c in checks) else "completed"
            )
        except KeyboardInterrupt:
            report.update(status="interrupted", error="Environment check interrupted")
        report.update(
            checks=checks,
            passed=sum(c["status"] == "passed" for c in checks),
            failed=sum(c["status"] == "failed" for c in checks),
        )
        return finish_run(directory, report)
