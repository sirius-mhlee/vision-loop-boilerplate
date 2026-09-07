import hashlib
import json
import os
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image

from vloop.approval import annotation_content, content_hash, record_path
from vloop.config import ClassConfig
from vloop.release_data import capture, connect, export_coco, initial_split, validate_coco
from vloop.release_dvc import git, image_relative, restore_data
from vloop.runtime import sha256_file, write_json


class Sample(dict):
    metadata = SimpleNamespace(width=23, height=13)


def approve(cfg, sample, *, automatic=False):
    content = annotation_content(sample, cfg.classes)
    record_id = uuid4().hex
    record = {
        "record_id": record_id,
        "action": "auto_accept" if automatic else "complete",
        "reviewer": "release-test",
        "label_hash": content_hash(content),
        "annotation": content,
        "source": {"kind": "manual"},
    }
    sample.update(
        review_status="auto_accepted" if automatic else "completed",
        review_approved_hash=record["label_hash"],
        review_approval_id=record_id,
        review_history=[{"action": record["action"], "record_id": record_id}],
        ground_truth_source=record["source"],
    )
    if automatic:
        from vloop.review_store import connect_records, save_record

        with closing(connect_records(cfg)) as db:
            sample["review_approval_sha256"] = save_record(db, record)
    else:
        path = record_path(cfg, sample, record_id)
        write_json(path, record)
        sample["review_approval_sha256"] = sha256_file(path)


def make_sample(cfg, number, *, empty=False, automatic=False, group=None):
    image_id = hashlib.sha256(str(number).encode()).hexdigest()
    path = cfg.image_dir / f"{number}.png"
    Image.new("RGB", (23, 13), (number % 256, (number // 256) % 256, 42)).save(path)
    mask = np.zeros((13, 23), dtype=bool)
    mask[1:9, 1:10] = True
    mask[3:7, 3:8] = False  # Hole
    mask[10:12, 18:21] = True  # Disconnected part of the same instance
    detection = SimpleNamespace(
        label="test-object", mask=mask, mask_path=None, bounding_box=[0, 0, 1, 1]
    )
    sample = Sample(
        image_id=image_id,
        managed_sha256=sha256_file(path),
        scene=group,
        ground_truth=SimpleNamespace(detections=[] if empty else [detection]),
    )
    sample.filepath = str(path)
    approve(cfg, sample, automatic=automatic)
    return sample


@pytest.fixture
def release_project(project):
    return replace(project, classes=(*project.classes, ClassConfig(42, "absent", ("absent",))))


def test_streamed_export_empty_rle_provenance_and_tampering(release_project, tmp_path):
    cfg = release_project
    samples = [make_sample(cfg, 1), make_sample(cfg, 2, empty=True)]
    snapshot = tmp_path / "export/metadata/snapshot.sqlite3"
    capture(cfg, iter(samples), snapshot)
    summary = export_coco(cfg, snapshot, snapshot.parent.parent)
    validate_coco(cfg, snapshot, snapshot.parent.parent)
    assert sum(s["images"] for s in summary["splits"].values()) == 2
    assert sum(s["empty"] for s in summary["splits"].values()) == 1
    assert all(s["classes"]["42"]["evaluation"] == "N/A" for s in summary["splits"].values())
    with closing(connect(snapshot)) as db:
        row = db.execute(
            "SELECT * FROM records WHERE image_id=?", (samples[0]["image_id"],)
        ).fetchone()
        assert json.loads(row["provenance"])["approval"]["reviewer"] == "release-test"
        split = summary["split_directories"][row["split"]]
    path = snapshot.parent.parent / split / "_annotations.coco.json"
    data = json.loads(path.read_text())
    data["annotations"][0]["area"] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="annotation differs"):
        validate_coco(cfg, snapshot, snapshot.parent.parent)


def test_changed_approval_fails_closed(release_project, tmp_path):
    cfg = release_project
    sample = make_sample(cfg, 1)
    sample["ground_truth"].detections[0].mask[0, 0] = True
    path = tmp_path / "snapshot.sqlite3"
    with pytest.raises(ValueError, match="changed after approval"):
        capture(cfg, [sample], path)
    assert not path.exists()


def test_split_stability_groups_auto_policy_and_eval_identity(release_project, tmp_path):
    cfg = replace(release_project, release_group_field="scene")
    names = {
        split: next(str(i) for i in range(1000) if initial_split("group:" + str(i), cfg) == split)
        for split in ("train", "val", "test")
    }
    samples = [
        make_sample(cfg, i + 1, group=names[split])
        for i, split in enumerate(("train", "val", "test"))
    ]
    samples.append(make_sample(cfg, 4, group=names["val"]))
    first = tmp_path / "first/metadata/snapshot.sqlite3"
    capture(cfg, samples, first)
    original = export_coco(cfg, first, first.parent.parent)
    second = tmp_path / "second/metadata/snapshot.sqlite3"
    new = make_sample(cfg, 5, group="new", automatic=True)
    held = make_sample(cfg, 6, group=names["test"])
    capture(cfg, [*reversed(samples), new, held], second, parent=first, include_auto=True)
    updated = export_coco(cfg, second, second.parent.parent)
    assert updated["held"] == 1
    for split in ("val", "test"):
        assert updated["splits"][split]["data_id"] == original["splits"][split]["data_id"]
    with closing(connect(second)) as db:
        assert (
            db.execute("SELECT split FROM records WHERE image_id=?", (new["image_id"],)).fetchone()[
                0
            ]
            == "train"
        )
    # Changing a reviewed validation label changes only its evaluation fingerprint.
    samples[1]["ground_truth"].detections[0].mask[0, 0] = True
    approve(cfg, samples[1])
    third = tmp_path / "third/metadata/snapshot.sqlite3"
    capture(cfg, samples, third, parent=second)
    changed = export_coco(cfg, third, third.parent.parent)
    assert changed["splits"]["val"]["data_id"] != original["splits"]["val"]["data_id"]
    assert changed["splits"]["test"]["data_id"] == original["splits"]["test"]["data_id"]
    # Omitting an image from one version does not erase its held-out assignment.
    assert changed["splits"]["val"]["images"] == 2


def test_automatic_is_opt_in_and_first_release_group_stays_together(release_project, tmp_path):
    cfg = replace(release_project, release_group_field="scene")
    manual = make_sample(cfg, 1, group="together")
    auto = make_sample(cfg, 2, automatic=True, group="together")
    first = tmp_path / "manual.sqlite3"
    capture(cfg, [manual, auto], first)
    with closing(connect(first)) as db:
        assert db.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 1
    second = tmp_path / "auto.sqlite3"
    capture(cfg, [manual, auto], second, include_auto=True)
    with closing(connect(second)) as db:
        assert {r[0] for r in db.execute("SELECT split FROM records")} == {"train"}


@pytest.mark.skipif(os.environ.get("VLOOP_TEST_RELEASE") != "1", reason="Requires DVC and RF-DETR")
def test_real_dvc_git_restore_resume_and_rfdetr(release_project, tmp_path, monkeypatch):
    from rfdetr.datasets.coco import CocoDetection

    import vloop.release as module

    root = tmp_path / "repo"
    root.mkdir()
    cfg = replace(
        release_project,
        config_path=root / "project.yaml",
        storage_dir=root / ".vloop",
        dvc_remote=tmp_path / "remote",
    )
    git(root, "init", "-q")
    git(root, "config", "user.name", "Release Test")
    git(root, "config", "user.email", "release@example.invalid")
    (root / ".gitignore").write_text(".vloop/\nproject.yaml\n")
    (root / "code.py").write_text("original\n")
    git(root, "add", ".gitignore", "code.py")
    git(root, "commit", "-qm", "Initial code")
    head = git(root, "rev-parse", "HEAD")
    (root / "code.py").write_text("staged edit\n")
    git(root, "add", "code.py")
    (root / "code.py").write_text("unstaged edit\n")
    before = git(root, "diff", "--cached"), git(root, "diff")
    samples = [make_sample(cfg, i + 1, empty=i == 1) for i in range(5)]
    monkeypatch.setattr(module, "_samples", lambda cfg, **kwargs: iter(samples))
    prepared = module.release(cfg, version="v001", prepare_only=True)
    assert prepared["status"] == "ready", prepared
    assert prepared["storage_plan"]["new_images"] == 5
    assert not cfg.dvc_remote.exists()
    assert not (cfg.storage_dir / "releases/pool").exists()
    assert git(root, "tag", "--list", "dataset/v001") == ""
    first = module.release(cfg, resume=prepared["job_id"])
    assert first["status"] == "completed", first
    assert git(root, "rev-parse", "HEAD") == head
    assert before == (git(root, "diff", "--cached"), git(root, "diff"))
    assert git(root, "show", "dataset/v001:code.py") == "original"
    assert "/dataset/images/*/" in git(root, "show", "dataset/v001:vloop-dataset/.gitignore")
    assert not (Path(first["result_dir"]) / "dvc-work").exists()
    tag_files = git(root, "ls-tree", "-r", "--name-only", "dataset/v001").splitlines()
    assert not any(name.endswith((".png", ".sqlite3")) for name in tag_files)
    with pytest.raises(ValueError, match="cannot be overwritten"):
        module.release(cfg, version="v001")
    data_root, info = restore_data(cfg, "v001")
    expected = {s["image_id"]: s for s in samples}
    for folder in ("train", "valid", "test"):
        dataset = CocoDetection(
            data_root / folder,
            data_root / folder / "_annotations.coco.json",
            transforms=None,
            include_masks=True,
            remap_category_ids=True,
        )
        assert dataset.cat2label == {7: 0, 42: 1}
        for i, image_id in enumerate(dataset.ids):
            image, target = dataset[i]
            sample = expected[dataset.coco.imgs[image_id]["vloop_image_id"]]
            assert image.size == (23, 13)
            assert len(target["masks"]) == len(sample["ground_truth"].detections)
            if len(target["masks"]):
                assert np.array_equal(
                    target["masks"][0].numpy(), sample["ground_truth"].detections[0].mask
                )
                assert target["labels"].tolist() == [0]
    # Removing the restored tree and local cache proves restoration from remote alone.
    import shutil

    shutil.rmtree(cfg.storage_dir / "releases/restored")
    shutil.rmtree(cfg.storage_dir / "releases/cache")
    data_root, _ = restore_data(cfg, "v001")
    source_hash = sha256_file(Path(samples[0].filepath))
    image_path = data_root / image_relative({"managed_sha256": source_hash})
    inode = image_path.stat().st_ino
    assert restore_data(cfg, "v001")[0] == data_root
    assert image_path.stat().st_ino == inode
    # A label-only version reuses every image and a failed upload creates no tag.
    original_push = module.add_and_push

    def interrupt_after_upload(*args, **kwargs):
        original_push(*args, **kwargs)
        raise RuntimeError("upload response interrupted")

    monkeypatch.setattr(module, "add_and_push", interrupt_after_upload)
    failed = module.release(cfg, version="v002")
    assert failed["status"] == "failed", failed
    assert git(root, "tag", "--list", "dataset/v002") == ""
    monkeypatch.setattr(module, "add_and_push", original_push)
    import subprocess
    import sys

    import yaml

    cfg.config_path.write_text(
        yaml.safe_dump({k: v for k, v in cfg.to_dict().items() if k != "config_path"})
    )
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "vloop",
            "--config",
            str(cfg.config_path),
            "release",
            "--resume",
            failed["job_id"],
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    resumed = json.loads((Path(failed["result_dir"]) / "report.json").read_text())
    assert resumed["status"] == "completed", resumed
    assert resumed["storage"]["new_image_bytes"] == 0
    assert git(root, "rev-parse", "HEAD") == head
    assert before == (git(root, "diff", "--cached"), git(root, "diff"))
    # Mutable review files cannot change an old release's bytes.
    Path(samples[0].filepath).write_bytes(b"changed original")
    assert sha256_file(image_path) == source_hash
    failed = module.release(cfg, version="v003")
    assert failed["status"] == "failed" and "Registered image changed" in failed["error"]
    assert git(root, "tag", "--list", "dataset/v003") == ""


def test_missing_group_and_changed_existing_group_are_rejected(release_project, tmp_path):
    cfg = replace(release_project, release_group_field="scene")
    sample = make_sample(cfg, 1)
    first = tmp_path / "first.sqlite3"
    with pytest.raises(ValueError, match="group field"):
        capture(cfg, [sample], first)
    sample["scene"] = "original"
    capture(cfg, [sample], first)
    sample["scene"] = "changed"
    with pytest.raises(ValueError, match="Group changed"):
        capture(cfg, [sample], tmp_path / "second.sqlite3", parent=first)


def test_interrupted_snapshot_can_restart_without_partial_release(release_project, tmp_path):
    cfg = release_project
    sample = make_sample(cfg, 1)

    def interrupted():
        yield sample
        raise KeyboardInterrupt

    path = tmp_path / "snapshot.sqlite3"
    with pytest.raises(KeyboardInterrupt):
        capture(cfg, interrupted(), path)
    assert not path.exists()
    capture(cfg, [sample], path)
    assert json.loads(path.with_suffix(".seal.json").read_text())["sha256"] == sha256_file(path)


def test_remote_validation_reads_content_not_only_names(release_project, tmp_path):
    import yaml

    from vloop.release_dvc import check_remote

    cfg = replace(release_project, dvc_remote=tmp_path / "remote")
    cfg.storage_dir.mkdir(parents=True)
    digest = hashlib.md5(b"original").hexdigest()
    object_path = cfg.dvc_remote / "files/md5" / digest[:2] / digest[2:]
    object_path.parent.mkdir(parents=True)
    object_path.write_bytes(b"original")
    pointer = tmp_path / "data.dvc"
    pointer.write_text(yaml.safe_dump({"outs": [{"md5": digest, "path": "data"}]}))
    check_remote(cfg, tmp_path, ["data.dvc"])
    object_path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="remote object corrupted"):
        check_remote(cfg, tmp_path, ["data.dvc"])
