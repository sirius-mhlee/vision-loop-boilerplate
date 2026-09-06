import json
import shutil
from contextlib import closing
from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageOps

from vloop.cli import main
from vloop.ingest import ingest, open_catalog
from vloop.runtime import project_lock, sha256_file


def test_deduplication_preserves_sources_and_original(project):
    source = project.image_dir / "first.png"
    Image.new("RGB", (13, 7), "red").save(source)
    original_hash = sha256_file(source)
    nested = project.image_dir / "nested"
    nested.mkdir()
    shutil.copy2(source, nested / "same-content.png")
    first = ingest(project, local_only=True)
    second = ingest(project, local_only=True)
    assert first["status"] == second["status"] == "completed"
    assert (first["registered"], first["duplicate"]) == (1, 1)
    assert (second["registered"], second["duplicate"]) == (0, 2)
    with closing(open_catalog(project)) as db:
        assert db.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 2
    assert sha256_file(source) == original_hash
    assert json.loads((Path(first["result_dir"]) / "config.json").read_text())["seed"] == 42


def test_orientation_and_pixels_are_frozen(project):
    source = project.image_dir / "rotated.jpg"
    image = Image.new("RGB", (13, 7), "red")
    image.paste("blue", (0, 0, 6, 7))
    exif = Image.Exif()
    exif[274] = 6
    image.save(source, exif=exif)
    with Image.open(source) as original:
        expected = ImageOps.exif_transpose(original).convert("RGB")
    ingest(project, local_only=True)
    with closing(open_catalog(project)) as db:
        row = db.execute("SELECT * FROM images").fetchone()
    assert (row["width"], row["height"]) == (7, 13)
    with Image.open(row["filepath"]) as managed:
        assert ImageChops.difference(expected, managed).getbbox() is None
        assert managed.getexif().get(274) is None


def test_bad_files_do_not_prevent_good_images_and_exit_is_failure(project):
    Image.new("RGB", (10, 5)).save(project.image_dir / "valid.png")
    (project.image_dir / "broken.jpg").write_bytes(b"not an image")
    (project.image_dir / "notes.txt").write_text("unsupported")
    report = ingest(project, local_only=True)
    assert report["registered"] == 1
    assert report["failed"] == 2
    assert report["status"] == "failed"
    items = [
        json.loads(line)
        for line in (Path(report["result_dir"]) / "files.jsonl").read_text().splitlines()
    ]
    assert sum("error" in item for item in items) == 2
    assert main(["ingest", "--config", str(project.config_path), "--local-only"]) == 1


def test_missing_managed_image_is_repaired_and_modification_is_rejected(project):
    Image.new("RGB", (4, 2), "red").save(project.image_dir / "source.png")
    ingest(project, local_only=True)
    with closing(open_catalog(project)) as db:
        row = db.execute("SELECT * FROM images").fetchone()
    destination = Path(row["filepath"])
    destination.unlink()
    assert ingest(project, local_only=True)["repaired"] == 1
    destination.write_bytes(b"changed")
    report = ingest(project, local_only=True)
    assert report["status"] == "failed"
    assert report["failed"] == 1


def test_sync_failure_preserves_local_registration_for_retry(project, monkeypatch):
    Image.new("RGB", (4, 2)).save(project.image_dir / "source.png")

    def fail(*args):
        raise RuntimeError("Database unavailable")

    monkeypatch.setattr("vloop.fiftyone.sync_catalog", fail)
    report = ingest(project)
    assert report["status"] == "failed"
    assert report["registered"] == 1
    assert report["sync_status"] == "incomplete"
    monkeypatch.setattr("vloop.fiftyone.sync_catalog", lambda *args: 1)
    retry = ingest(project)
    assert retry["status"] == "completed"
    assert retry["registered"] == 0
    assert retry["duplicate"] == 1


def test_interruption_is_recorded_and_retry_is_idempotent(project, monkeypatch):
    Image.new("RGB", (4, 2)).save(project.image_dir / "source.png")
    from vloop import ingest as module

    register = module._register

    def interrupt(*args):
        register(*args)
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "_register", interrupt)
    report = ingest(project, local_only=True)
    assert report["status"] == "interrupted"
    assert (
        json.loads((Path(report["result_dir"]) / "report.json").read_text())["status"]
        == "interrupted"
    )
    monkeypatch.setattr(module, "_register", register)
    assert ingest(project, local_only=True)["duplicate"] == 1


def test_empty_input_is_not_a_success(project):
    assert ingest(project, local_only=True)["status"] == "failed"


def test_concurrent_project_operation_is_rejected(project):
    with project_lock(project), pytest.raises(RuntimeError, match="Another vloop"):
        ingest(project, local_only=True)
