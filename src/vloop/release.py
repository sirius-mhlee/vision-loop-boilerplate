"""Stage 4: approved COCO releases, DVC storage, and isolated restoration."""

import json
import re
import shutil
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

from .config import config_from_dict
from .release_data import SPLITS, capture, export_coco, validate_coco
from .release_dvc import (
    add_and_push,
    check_remote,
    descriptor,
    git,
    materialize,
    project_repo,
    publish,
    restore_data,
    storage_plan,
    version_name,
    versions,
)
from .runtime import cli_command, finish_run, project_lock, sha256_file, start_run, write_json


def _samples(cfg, *, include_auto=False):
    from .review import load_review_dataset

    dataset = load_review_dataset(cfg)
    from fiftyone import ViewField as F

    if not dataset.has_sample_field("review_status"):
        raise ValueError("Run vloop review and approve labels before release")
    fields = [
        "image_id",
        "managed_sha256",
        "metadata",
        "ground_truth",
        "ground_truth_source",
        "review_status",
        "review_approved_hash",
        "review_approval_id",
        "review_approval_sha256",
        "review_history",
    ]
    if cfg.release_group_field:
        if not dataset.has_sample_field(cfg.release_group_field):
            raise ValueError(f"Missing FiftyOne group field: {cfg.release_group_field}")
        fields.append(cfg.release_group_field)
    statuses = ["completed", "auto_accepted"] if include_auto else ["completed"]
    view = dataset.match(F("review_status").is_in(statuses))
    # Exclude prediction fields and avoid dataset.values(), which materializes every ID.
    yield from view.select_fields(fields).iter_samples(progress=False)


def release(
    cfg,
    *,
    version=None,
    resume=None,
    include_auto_train=False,
    manual_to_val_test=False,
    prepare_only=False,
):
    if find_spec("dvc") is None or find_spec("ijson") is None:
        raise RuntimeError(
            "Install release dependencies: python -m pip install 'dvc>=3,<4' 'ijson>=3.4,<4'"
        )
    if resume:
        if version is not None or include_auto_train or manual_to_val_test:
            raise ValueError("--resume uses the frozen version and label/split policies")
        if not re.fullmatch(r"release_\d{8}T\d{6}_[0-9a-f]{8}", resume):
            raise ValueError("Invalid release job ID")
        directory = cfg.storage_dir / "runs" / resume
        frozen = json.loads((directory / "config.json").read_text())
        cfg = config_from_dict(
            {k: v for k, v in frozen.items() if k != "config_path"}, Path(frozen["config_path"])
        )
        report = json.loads((directory / "report.json").read_text())
        info = json.loads((directory / "release-job.json").read_text())
        version = info["version"]
    else:
        version_name(version)
        if not cfg.classes:
            raise ValueError("Set project classes before release")
        root = project_repo(cfg)
        prior = versions(root)
        if prior and int(version[1:]) <= int(prior[-1][1:]):
            raise ValueError(f"Use a new version after {prior[-1]}; versions cannot be overwritten")
        git(root, "var", "GIT_AUTHOR_IDENT")
        code_commit = git(root, "rev-parse", "HEAD")
        directory, report = start_run(cfg, "release")
        info = {
            "version": version,
            "dataset_name": cfg.dataset_name,
            "parent": prior[-1] if prior else None,
            "include_auto_train": include_auto_train,
            "manual_to_val_test": manual_to_val_test,
            "code_commit": code_commit,
            "job_id": report["job_id"],
        }
        write_json(directory / "release-job.json", info)
    report.update(
        status="running",
        dataset_version=version,
        retry=cli_command(cfg, "release", "--resume", report["job_id"]),
    )
    report.pop("error", None)
    # Separate from the review lock; reviewers can keep working. The successful scan
    # freezes individually verified approvals, not a global MongoDB transaction.
    with project_lock(SimpleNamespace(storage_dir=cfg.storage_dir / "releases")):
        work = directory / "dvc-work"
        try:
            root = project_repo(cfg)
            prior = versions(root)
            if version in prior:
                published = descriptor(root, version)
                if published["job_id"] != report["job_id"]:
                    raise ValueError("Dataset version already belongs to another release")
                report.update(
                    status="completed",
                    git_commit=git(root, "rev-parse", f"refs/tags/dataset/{version}"),
                    summary=published["summary"],
                )
                if work.exists():
                    shutil.rmtree(work)
                return finish_run(directory, report)
            if (prior[-1] if prior else None) != info["parent"]:
                raise ValueError("Another dataset version was released; start a new release job")
            parent = None
            if info["parent"]:
                parent_root, parent_info = restore_data(cfg, info["parent"], metadata_only=True)
                if parent_info["dataset_name"] != cfg.dataset_name:
                    raise ValueError("The parent release belongs to a different FiftyOne dataset")
                mapping = [(c.id, c.name) for c in sorted(cfg.classes, key=lambda c: c.id)]
                previous = [(c["id"], c["name"]) for c in parent_info["summary"]["classes"]]
                if (
                    mapping != previous
                    or cfg.release_group_field != parent_info["summary"]["group_field"]
                ):
                    raise ValueError("Class mapping and group field must match the parent release")
                parent = parent_root / "metadata/snapshot.sqlite3"
            snapshot = work / "dataset/metadata/snapshot.sqlite3"
            report["phase"] = "snapshot"
            write_json(directory / "report.json", report)
            if not snapshot.exists():
                capture(
                    cfg,
                    _samples(cfg, include_auto=info["include_auto_train"]),
                    snapshot,
                    parent=parent,
                    include_auto=info["include_auto_train"],
                    manual_to_val_test=info.get("manual_to_val_test", False),
                )
            if "snapshot_sha256" not in report:
                report["snapshot_sha256"] = json.loads(
                    snapshot.with_suffix(".seal.json").read_text()
                )["sha256"]
                write_json(directory / "report.json", report)
            if sha256_file(snapshot) != report.get("snapshot_sha256"):
                raise ValueError("Frozen release snapshot was modified")
            report["phase"] = "export"
            report["summary"] = export_coco(
                cfg,
                snapshot,
                work / "dataset",
                manual_to_val_test=info.get("manual_to_val_test", False),
            )
            validate_coco(cfg, snapshot, work / "dataset")
            report["storage_plan"] = storage_plan(cfg, snapshot, verify_sources=prepare_only)
            write_json(directory / "report.json", report)
            if prepare_only:
                report.update(status="ready", phase="prepared")
                return finish_run(directory, report)
            if not any(s["images"] for s in report["summary"]["splits"].values()):
                raise ValueError("All images are held; inspect summary.held_reasons and re-review")
            report["storage"] = materialize(cfg, snapshot, work)
            report["phase"] = "upload"

            def progress(done, total):
                report.update(dvc_shards_done=done, dvc_shards_total=total)
                write_json(directory / "report.json", report)

            targets = add_and_push(cfg, work, progress)
            report["phase"] = "verify_remote"
            write_json(directory / "report.json", report)
            check_remote(cfg, work, targets)
            info.update(
                schema_version=1,
                summary=report["summary"],
                dvc_targets=targets,
                snapshot_sha256=report["snapshot_sha256"],
                label_checksums={
                    f"{folder}/_annotations.coco.json": sha256_file(
                        work / "dataset" / folder / "_annotations.coco.json"
                    )
                    for folder in SPLITS.values()
                },
            )
            info["label_checksums"]["metadata/summary.json"] = sha256_file(
                work / "dataset/metadata/summary.json"
            )
            write_json(work / "release.json", info)
            report["phase"] = "publish"
            write_json(directory / "report.json", report)
            report["git_commit"] = publish(root, work, info)
            report.update(status="completed", tag=f"dataset/{version}")
            # Only the shared content cache/pool survive. No permanent per-release image tree.
            shutil.rmtree(work)
        except KeyboardInterrupt:
            report["status"] = "interrupted"
        except Exception as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        return finish_run(directory, report)


def restore(cfg, *, version):
    version_name(version)
    directory, report = start_run(cfg, "restore")
    report["dataset_version"] = version
    with project_lock(SimpleNamespace(storage_dir=cfg.storage_dir / "releases")):
        try:
            path, info = restore_data(cfg, version)
            report.update(status="completed", dataset_dir=str(path), summary=info["summary"])
        except KeyboardInterrupt:
            report["status"] = "interrupted"
        except Exception as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        return finish_run(directory, report)
