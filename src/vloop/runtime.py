import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
from uuid import uuid4

from .config import Config


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def git_output(directory: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(directory), *args], capture_output=True, text=True, timeout=15, check=True
    )
    return result.stdout.strip()


def start_run(cfg: Config, command: str) -> tuple[Path, dict]:
    job_id = f"{command}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
    directory = cfg.storage_dir / "runs" / job_id
    directory.mkdir(parents=True)
    write_json(directory / "config.json", cfg.to_dict())
    write_json(
        directory / "dependencies.json",
        {dist.metadata["Name"]: dist.version for dist in distributions() if dist.metadata["Name"]},
    )
    try:
        code_dir = Path(__file__).resolve().parent
        code = {
            "commit": git_output(code_dir, "rev-parse", "HEAD"),
            "dirty": bool(git_output(code_dir, "status", "--porcelain")),
        }
    except (OSError, subprocess.SubprocessError):
        code = {"commit": None, "dirty": None}
    report = {
        "job_id": job_id,
        "command": command,
        "status": "running",
        "dataset_version": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "result_dir": str(directory),
        "code": code,
    }
    write_json(directory / "report.json", report)
    return directory, report


def finish_run(directory: Path, report: dict) -> dict:
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(directory / "report.json", report)
    return report


@contextmanager
def project_lock(cfg: Config):
    cfg.storage_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.storage_dir / "project.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another vloop operation is running for this project") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
