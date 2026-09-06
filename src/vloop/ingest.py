import json
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from PIL import Image, ImageOps

from .config import Config
from .runtime import finish_run, project_lock, sha256_file, start_run, write_json

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def open_catalog(cfg: Config) -> sqlite3.Connection:
    connection = sqlite3.connect(cfg.storage_dir / "catalog.sqlite3")
    connection.row_factory = sqlite3.Row
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS images (
            image_id TEXT PRIMARY KEY,
            filepath TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            managed_sha256 TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sources (
            image_id TEXT NOT NULL REFERENCES images(image_id),
            source_path TEXT NOT NULL,
            PRIMARY KEY (image_id, source_path)
        );
    """)
    return connection


def iter_files(directory: Path):
    def on_error(error):
        raise error

    for root, directories, filenames in os.walk(directory, followlinks=False, onerror=on_error):
        directories.sort()
        for name in sorted(filenames):
            yield Path(root) / name


def normalize_image(source: Path, destination: Path) -> tuple[int, int]:
    """Freeze the coordinate system in a lossless RGB PNG, one image at a time."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Multi-frame images are not supported")
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        normalized.info.clear()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, suffix=".png", delete=False
            ) as handle:
                temporary = Path(handle.name)
                normalized.save(handle, format="PNG")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return normalized.size


def _register(connection: sqlite3.Connection, cfg: Config, source: Path) -> tuple[str, str]:
    image_id = sha256_file(source)
    existing = connection.execute("SELECT * FROM images WHERE image_id = ?", (image_id,)).fetchone()
    destination = cfg.storage_dir / "images" / image_id[:2] / f"{image_id}.png"
    state = "duplicate"
    if existing is None or not Path(existing["filepath"]).is_file():
        width, height = normalize_image(source, destination)
        if sha256_file(source) != image_id:
            destination.unlink(missing_ok=True)
            raise ValueError("Source changed during ingest; retry when the file is stable")
        managed_hash = sha256_file(destination)
        if existing is not None and managed_hash != existing["managed_sha256"]:
            destination.unlink(missing_ok=True)
            raise ValueError("Restored pixels differ from the registered image")
        with connection:
            connection.execute(
                "INSERT OR IGNORE INTO images VALUES (?, ?, ?, ?, ?)",
                (image_id, str(destination), width, height, managed_hash),
            )
        state = "registered" if existing is None else "repaired"
    elif sha256_file(Path(existing["filepath"])) != existing["managed_sha256"]:
        raise ValueError(f"Managed image was modified: {existing['filepath']}")
    with connection:
        connection.execute(
            "INSERT OR IGNORE INTO sources VALUES (?, ?)", (image_id, str(source.resolve()))
        )
    return image_id, state


def ingest(cfg: Config, *, local_only: bool = False) -> dict:
    errors = cfg.input_errors()
    if errors:
        raise ValueError("; ".join(errors))
    with project_lock(cfg):
        directory, report = start_run(cfg, "ingest")
        report.update(
            registered=0,
            duplicate=0,
            repaired=0,
            failed=0,
            sync_status="not_requested" if local_only else "pending",
        )
        try:
            with closing(open_catalog(cfg)) as connection:
                with (directory / "files.jsonl").open("w", encoding="utf-8") as log:
                    for source in iter_files(cfg.image_dir):
                        try:
                            if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
                                raise ValueError(
                                    f"Unsupported file extension: {source.suffix or '(none)'}"
                                )
                            image_id, state = _register(connection, cfg, source)
                            report[state] += 1
                            item = {"source": str(source), "image_id": image_id, "status": state}
                        except (OSError, ValueError, Image.DecompressionBombError) as exc:
                            report["failed"] += 1
                            item = {"source": str(source), "status": "failed", "error": str(exc)}
                        log.write(json.dumps(item, ensure_ascii=False) + "\n")
                        log.flush()
                        write_json(directory / "report.json", report)
                if (
                    sum(report[key] for key in ("registered", "duplicate", "repaired", "failed"))
                    == 0
                ):
                    raise ValueError("Input directory contains no files")
                if not local_only:
                    from .fiftyone import sync_catalog

                    report["synced"] = sync_catalog(cfg, connection)
                    report["sync_status"] = "completed"
            report["status"] = "failed" if report["failed"] else "completed"
        except KeyboardInterrupt:
            report.update(status="interrupted", error="Interrupted; rerun ingest to continue")
        except Exception as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            if report["sync_status"] == "pending":
                report["sync_status"] = "incomplete"
        report["retry"] = f"vloop ingest --config {cfg.config_path}"
        if local_only:
            report["retry"] += " --local-only"
        return finish_run(directory, report)
