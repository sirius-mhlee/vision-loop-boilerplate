"""Run a small public-data iteration in a separate, persistent toy project.

Download PennFudanPed.zip from its official source first (see docs/TOY.md).
SAM 3 predicts real photographs. Provided reference masks replace manual editing
in this demo; approval records explicitly identify this as a scripted toy step.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import yaml
from PIL import Image

from vloop.config import ClassConfig, load_config
from vloop.release_data import initial_split
from vloop.release_dvc import descriptor, git
from vloop.runtime import sha256_file, write_json

SOURCE_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"


def reference_instances(mask_path):
    with Image.open(mask_path) as image:
        labels = np.asarray(image)
    if labels.ndim != 2 or labels.dtype.kind not in "iu":
        raise ValueError("Expected a single-channel instance-ID mask")
    result = []
    for value in np.unique(labels):
        if value == 0:
            continue
        mask = labels == value
        ys, xs = np.where(mask)
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max()) + 1 - x, int(ys.max()) + 1 - y
        result.append((x, y, w, h, mask[y : y + h, x : x + w]))
    return labels.shape, result


def setup(base, workspace):
    root = workspace / "repo"
    manifest_path = workspace / "source/selection.json"
    if (root / "project.yaml").exists():
        return load_config(root / "project.yaml"), json.loads(manifest_path.read_text())
    archive = workspace / "source/PennFudanPed.zip"
    if not archive.is_file():
        raise ValueError(f"Download {SOURCE_URL} to {archive} first")
    source = Path(__file__).resolve().parents[1]
    cfg = replace(
        base,
        config_path=root / "project.yaml",
        image_dir=workspace / "input",
        storage_dir=workspace / "state",
        dvc_remote=workspace / "dvc-remote",
        dataset_name="vloop-pennfudan-toy",
        classes=(ClassConfig(1, "person", ("person",)),),
        epochs=2,
        release_group_field=None,
        train_num_workers=0,
        train_checkpoint=base.train_checkpoint
        or base.storage_dir / "models/rfdetr/rf-detr-seg-nano.pt",
    )
    selected = {"train": [], "val": [], "test": []}
    counts = {"train": 6, "val": 1, "test": 1}
    with ZipFile(archive) as contents:
        names = sorted(
            n
            for n in contents.namelist()
            if n.startswith("PennFudanPed/PNGImages/") and n.endswith(".png")
        )
        for name in names:
            pixels = contents.read(name)
            digest = hashlib.sha256(pixels).hexdigest()
            split = initial_split("image:" + digest, cfg)
            if len(selected[split]) == counts[split]:
                continue
            filename = Path(name).name
            mask_name = "PennFudanPed/PedMasks/" + Path(filename).stem + "_mask.png"
            mask = contents.read(mask_name)
            for folder, data in (("images", pixels), ("masks", mask)):
                target = workspace / "source" / folder / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            selected[split].append(
                {
                    "filename": filename,
                    "split": split,
                    "sha256": digest,
                    "mask_sha256": hashlib.sha256(mask).hexdigest(),
                }
            )
            if all(len(selected[s]) == counts[s] for s in counts):
                break
    if any(len(selected[s]) != counts[s] for s in counts):
        raise ValueError("Archive did not contain enough images for the toy split")
    entries = []
    for split, images in selected.items():
        for index, item in enumerate(images):
            item["version"] = "v002" if split == "train" and index >= 4 else "v001"
            entries.append(item)
    manifest = {
        "source_url": SOURCE_URL,
        "archive_sha256": sha256_file(archive),
        "images": entries,
        "purpose": "toy workflow check; not a benchmark or human annotation study",
    }
    write_json(manifest_path, manifest)
    root.mkdir(parents=True)
    shutil.copytree(source / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(Path(__file__), root / "toy.py")
    (root / ".gitignore").write_text("project.yaml\n__pycache__/\n*.pyc\n")
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Vloop Toy Example")
    git(root, "config", "user.email", "toy@example.invalid")
    git(root, "add", "src", "toy.py", ".gitignore")
    git(root, "commit", "-qm", "Snapshot code for Penn-Fudan toy iteration")
    data = cfg.to_dict()
    data.pop("config_path")
    cfg.config_path.write_text(yaml.safe_dump(data))
    cfg.image_dir.mkdir()
    return cfg, manifest


def review_reference(cfg, version):
    from vloop.review import change_reviews, load_review_dataset

    workspace = cfg.config_path.parent.parent
    selection = json.loads((workspace / "source/selection.json").read_text())
    entries = {item["filename"]: item for item in selection["images"]}
    dataset = load_review_dataset(cfg)
    import fiftyone as fo

    for sample in dataset.iter_samples():
        filename = Path(sample["source_paths"][0]).name
        entry = entries[filename]
        # Existing approvals are preserved during the second round.
        if entry["version"] != version:
            continue
        mask_path = workspace / "source/masks" / filename
        assert sha256_file(mask_path) == entry["mask_sha256"]
        shape, instances = reference_instances(mask_path)
        height, width = shape
        assert (sample.metadata.width, sample.metadata.height) == (width, height)
        changed = change_reviews(cfg, [sample.id], "start", "toy-reference-import")
        assert changed["changed"] == 1, changed
        sample.reload()
        sample["ground_truth"] = fo.Detections(
            detections=[
                fo.Detection(
                    label="person",
                    bounding_box=[x / width, y / height, w / width, h / height],
                    mask=mask,
                )
                for x, y, w, h, mask in instances
            ]
        )
        sample.save()
        changed = change_reviews(
            cfg,
            [sample.id],
            "complete",
            "toy-reference-import",
            note=(
                f"Toy-only scripted verification against Penn-Fudan provided mask: {filename}; "
                "not human review"
            ),
            confirm_empty=True,
        )
        assert changed["changed"] == 1, changed


def run(base, workspace):
    cfg, selection = setup(base, workspace)
    root = cfg.config_path.parent
    environment = {
        **os.environ,
        "PYTHONPATH": str(root / "src"),
        "NO_ALBUMENTATIONS_UPDATE": "1",
        "OMP_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "MPLCONFIGDIR": str(workspace / "mpl"),
    }
    progress_path = workspace / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    logs = workspace / "logs"
    logs.mkdir(exist_ok=True)

    def execute(name, arguments, command=None):
        if name in progress:
            return progress[name]
        print(f"Toy step: {name}", flush=True)
        before = set((cfg.storage_dir / "runs").glob("*/report.json"))
        with (logs / f"{name}.log").open("w") as output:
            result = subprocess.run(
                [sys.executable, *arguments],
                cwd=root,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=1200,
            )
        if result.returncode:
            raise RuntimeError(
                f"{name} failed ({result.returncode}):\n"
                + (logs / f"{name}.log").read_text()[-10000:]
            )
        report = {"status": "completed"}
        if command:
            created = set((cfg.storage_dir / "runs").glob(command + "_*/report.json")) - before
            assert len(created) == 1, created
            report = json.loads(created.pop().read_text())
            assert report["status"] == "completed", report
        progress[name] = report
        write_json(progress_path, progress)
        return report

    def cli(name, command, *options):
        return execute(
            name, ["-m", "vloop", command, "--config", str(cfg.config_path), *options], command
        )

    started = time.monotonic()
    baseline = None
    for version in ("v001", "v002"):
        for entry in selection["images"]:
            if entry["version"] == version:
                source = workspace / "source/images" / entry["filename"]
                assert sha256_file(source) == entry["sha256"]
                target = cfg.image_dir / entry["filename"]
                if not target.exists():
                    shutil.copy2(source, target)
        cli("ingest-" + version, "ingest")
        autolabel = cli("autolabel-" + version, "autolabel")
        cli(
            "review-" + version,
            "review",
            "--job-id",
            autolabel["job_id"],
            "--limit",
            "8",
            "--prepare-only",
        )
        execute(
            "reference-" + version,
            [str(root / "toy.py"), "--review-reference", version, "--config", str(cfg.config_path)],
        )
        cli("release-" + version, "release", "--version", version)
        trained = cli(
            "train-" + version,
            "train",
            "--dataset-version",
            version,
            "--notes",
            f"Penn-Fudan toy {version}; provided reference masks, two epochs",
        )
        evaluated = cli(
            "evaluate-" + version,
            "evaluate",
            "--job-id",
            trained["job_id"],
            "--dataset-version",
            "v001",
            "--split",
            "val",
        )
        if baseline is None:
            baseline = evaluated
        else:
            assert evaluated["comparison_id"] == baseline["comparison_id"]
    first, second = descriptor(root, "v001"), descriptor(root, "v002")
    assert first["summary"]["splits"]["train"]["images"] == 4
    assert second["summary"]["splits"]["train"]["images"] == 6
    for split in ("val", "test"):
        assert first["summary"]["splits"][split] == second["summary"]["splits"][split]
    cli("restore-v001", "restore", "--version", "v001")
    comparison = {
        kind: {
            metric: {
                "v001": progress["evaluate-v001"]["metrics"][kind][metric],
                "v002": progress["evaluate-v002"]["metrics"][kind][metric],
            }
            for metric in ("mAP", "AP50")
        }
        for kind in ("boxes", "masks")
    }
    result = {
        "status": "completed",
        "config": str(cfg.config_path),
        "selection": selection,
        "elapsed_seconds_this_attempt": time.monotonic() - started,
        "comparison_id": baseline["comparison_id"],
        "comparison": comparison,
        "steps": progress,
        "scope": "8 public photos; scripted reference-mask approvals; two epochs per model",
    }
    write_json(workspace / "result.json", result)
    print(
        json.dumps(
            {"status": result["status"], "config": result["config"], "comparison": comparison},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("project.yaml"))
    parser.add_argument("--workspace", type=Path, default=Path(".vloop/toy-pennfudan"))
    parser.add_argument("--review-reference", choices=["v001", "v002"])
    args = parser.parse_args()
    if args.review_reference:
        review_reference(load_config(args.config), args.review_reference)
    else:
        run(load_config(args.config), args.workspace.resolve())
