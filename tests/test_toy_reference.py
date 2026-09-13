import io
import runpy
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

TOY = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/pennfudan_toy.py"))
reference_instances = TOY["reference_instances"]


def test_reference_instance_ids_preserve_holes_and_disconnected_pixels(tmp_path):
    labels = np.zeros((13, 23), dtype=np.uint8)
    labels[1:9, 2:10] = 1
    labels[3:6, 4:7] = 0
    labels[11, 21] = 1
    labels[0:2, 19:23] = 7
    path = tmp_path / "mask.png"
    Image.fromarray(labels).save(path)

    shape, instances = reference_instances(path)
    assert shape == (13, 23)
    assert len(instances) == 2
    restored = np.zeros_like(labels)
    for value, (x, y, w, h, mask) in zip((1, 7), instances, strict=True):
        assert mask.dtype == bool
        restored[y : y + h, x : x + w][mask] = value
    np.testing.assert_array_equal(restored, labels)


def test_reference_rejects_rgb_mask_instead_of_treating_channels_as_instances(tmp_path):
    path = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="single-channel"):
        reference_instances(path)


@pytest.fixture
def toy_archive(tmp_path):
    workspace = tmp_path / "toy"
    archive = workspace / "source/PennFudanPed.zip"
    archive.parent.mkdir(parents=True)
    with ZipFile(archive, "w") as contents:
        for index in range(200):
            pixels, mask = io.BytesIO(), io.BytesIO()
            Image.new("RGB", (8, 6), (index, 17, 29)).save(pixels, format="PNG")
            Image.new("L", (8, 6), 1).save(mask, format="PNG")
            contents.writestr(f"PennFudanPed/PNGImages/photo-{index}.png", pixels.getvalue())
            contents.writestr(f"PennFudanPed/PedMasks/photo-{index}_mask.png", mask.getvalue())
    return workspace


@pytest.mark.parametrize("failure", ["copy", "commit", "config"])
def test_toy_initialization_can_retry_without_replacing_completed_code(
    project, toy_archive, monkeypatch, failure
):
    setup = TOY["setup"]
    original_copy, original_git, original_write = (
        TOY["shutil"].copytree,
        TOY["git"],
        Path.write_text,
    )

    def copy(*args, **kwargs):
        raise OSError("interrupted copy")

    def git(root, *args):
        if args[0] == "commit":
            raise OSError("interrupted commit")
        return original_git(root, *args)

    def write(path, *args, **kwargs):
        if path.name == "project.yaml":
            raise OSError("interrupted config write")
        return original_write(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(TOY["shutil"], "copytree", copy if failure == "copy" else original_copy)
        patch.setitem(setup.__globals__, "git", git if failure == "commit" else original_git)
        patch.setattr(Path, "write_text", write if failure == "config" else original_write)
        with pytest.raises(OSError, match="interrupted"):
            setup(project, toy_archive)
    assert not (toy_archive / "repo").exists()
    assert not list(toy_archive.glob(".repo-*"))

    cfg, manifest = setup(project, toy_archive)
    root = cfg.config_path.parent
    assert cfg.image_dir.is_dir()
    assert len(manifest["images"]) == 8
    assert original_git(root, "status", "--porcelain") == ""
    commit = original_git(root, "rev-parse", "HEAD")
    assert setup(replace(project, seed=99), toy_archive) == (cfg, manifest)
    assert original_git(root, "rev-parse", "HEAD") == commit


def test_toy_refuses_to_overwrite_an_unrecognized_repo(project, toy_archive):
    root = toy_archive / "repo"
    root.mkdir()
    existing = root / "notes.txt"
    existing.write_text("existing work")
    with pytest.raises(ValueError, match="use a new --workspace"):
        TOY["setup"](project, toy_archive)
    assert existing.read_text() == "existing work"
