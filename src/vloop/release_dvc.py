"""Isolated DVC workspaces and metadata-only Git tags; never change the user's index."""

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
from contextlib import closing, contextmanager
from pathlib import Path

from .release_data import connect, image_relative, verify_images
from .runtime import sha256_file, write_json

PREFIX = "vloop-dataset"
LABEL_TARGETS = ["dataset/metadata", "dataset/train", "dataset/valid", "dataset/test"]
METADATA_FILES = [".dvc/config", ".gitignore", "release.json"]


def git(root, *args, data=None, env=None):
    result = subprocess.run(
        ["git", "-C", str(root), *args], input=data, capture_output=True, env=env, timeout=60
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return result.stdout.decode().strip()


def version_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"v[0-9]{3,}", value):
        raise ValueError("Dataset version must look like v001")
    return value


def project_repo(cfg):
    root = Path(git(cfg.config_path.parent, "rev-parse", "--show-toplevel"))
    if cfg.dvc_remote.is_relative_to(root):
        raise ValueError("dvc_remote must be a local directory outside the Git repository")
    if cfg.dvc_remote.is_relative_to(cfg.storage_dir) or cfg.storage_dir.is_relative_to(
        cfg.dvc_remote
    ):
        raise ValueError("dvc_remote and storage_dir must not overlap")
    if cfg.storage_dir.is_relative_to(root):
        result = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "-q", str(cfg.storage_dir / "releases")]
        )
        if result.returncode:
            raise ValueError("Add storage_dir to .gitignore before creating releases")
    return root


def versions(root):
    return sorted(
        (
            v.removeprefix("dataset/")
            for v in git(root, "tag", "--list", "dataset/v*").splitlines()
            if re.fullmatch(r"dataset/v[0-9]{3,}", v)
        ),
        key=lambda v: int(v[1:]),
    )


def descriptor(root, version):
    return json.loads(
        git(root, "show", f"refs/tags/dataset/{version_name(version)}:{PREFIX}/release.json")
    )


@contextmanager
def dvc_repo(cfg, work):
    try:
        from dvc.repo import Repo
    except ImportError as exc:
        raise RuntimeError("Install DVC: python -m pip install 'dvc>=3,<4'") from exc
    folder = work / ".dvc"
    folder.mkdir(parents=True, exist_ok=True)
    (work / ".gitignore").write_text(
        "/.dvc/config.local\n/.dvc/tmp/\n/.dvc/cache/\n/restored.json\n"
        "/dataset/images/*/\n/dataset/metadata/\n/dataset/train/\n/dataset/valid/\n/dataset/test/\n"
    )
    if not (folder / "config").exists():
        (folder / "config").write_text(
            "[core]\n    no_scm = true\n    analytics = false\n"
            "[cache]\n    type = reflink,hardlink\n"
        )
    # Local-only paths do not enter the dataset tag. All restores share this cache.
    from configobj import ConfigObj

    local = ConfigObj()
    local.filename = str(folder / "config.local")
    local["core"] = {
        "remote": "release",
        "site_cache_dir": str(cfg.storage_dir / "releases/site-cache"),
    }
    local["cache"] = {"dir": str(cfg.storage_dir / "releases/cache")}
    local['remote "release"'] = {"url": str(cfg.dvc_remote), "verify": "true"}
    local.write()
    with Repo(str(work)) as repo:
        yield repo


def _safe_link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".link-tmp")
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    temporary.replace(destination)


def storage_plan(cfg, snapshot, *, verify_sources=False):
    """Conservative byte estimate, not a reservation or a filesystem quota guarantee."""
    remote_parent = cfg.dvc_remote
    while not remote_parent.exists():
        remote_parent = remote_parent.parent
    stats = {
        "new_images": 0,
        "reused_images": 0,
        "new_image_bytes": 0,
        "local_free_bytes": shutil.disk_usage(cfg.storage_dir).free,
        "remote_free_bytes": shutil.disk_usage(remote_parent).free,
        "same_filesystem": cfg.storage_dir.stat().st_dev == remote_parent.stat().st_dev,
    }
    with closing(connect(snapshot)) as db:
        if verify_sources:
            for row in db.execute(
                "SELECT source_path, managed_sha256 FROM records WHERE split IS NOT NULL"
            ):
                if sha256_file(Path(row["source_path"])) != row["managed_sha256"]:
                    raise ValueError(f"Registered image changed: {row['source_path']}")
        for row in db.execute(
            "SELECT managed_sha256, MIN(source_path) AS source_path FROM records "
            "WHERE split IS NOT NULL GROUP BY managed_sha256"
        ):
            cached = cfg.storage_dir / "releases/pool" / image_relative(row)
            if cached.exists():
                stats["reused_images"] += 1
            else:
                stats["new_images"] += 1
                stats["new_image_bytes"] += Path(row["source_path"]).stat().st_size
    # Staging + local cache + remote, before reflink/hardlink savings. Label files,
    # directory entries and temporary SQLite indexes are additional to this estimate.
    stats["image_bytes_upper_estimate_local"] = stats["new_image_bytes"] * (
        3 if stats["same_filesystem"] else 2
    )
    stats["image_bytes_upper_estimate_remote"] = stats["new_image_bytes"]
    return stats


def materialize(cfg, snapshot, work):
    """A mutable review image is copied once, then an immutable pool shares DVC's files."""
    pool = cfg.storage_dir / "releases/pool"
    root = work / "dataset"
    stats = {"reused_images": 0, "new_images": 0, "new_image_bytes": 0}
    with closing(connect(snapshot)) as db:
        # Verify every input source, even if several inputs normalize to the same PNG.
        for row in db.execute(
            "SELECT source_path, managed_sha256 FROM records WHERE split IS NOT NULL"
        ):
            if sha256_file(Path(row["source_path"])) != row["managed_sha256"]:
                raise ValueError(f"Registered image changed: {row['source_path']}")
        for row in db.execute(
            "SELECT managed_sha256, MIN(source_path) AS source_path FROM records "
            "WHERE split IS NOT NULL GROUP BY managed_sha256"
        ):
            relative = image_relative(row)
            destination, cached = root / relative, pool / relative
            if destination.exists():
                if sha256_file(destination) != row["managed_sha256"]:
                    raise ValueError("Staged release image was modified")
                stats["reused_images"] += 1
                continue
            if cached.exists():
                if sha256_file(cached) != row["managed_sha256"]:
                    raise ValueError("Immutable image pool was modified; restore it from remote")
                _safe_link(cached, destination)
                stats["reused_images"] += 1
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".copy-tmp")
                # Never hardlink an editable managed image into an immutable DVC cache.
                shutil.copyfile(row["source_path"], temporary)
                if sha256_file(temporary) != row["managed_sha256"]:
                    temporary.unlink()
                    raise ValueError("Image changed while being copied")
                temporary.replace(destination)
                stats["new_images"] += 1
                stats["new_image_bytes"] += destination.stat().st_size
    return stats


def add_and_push(cfg, work, progress=None):
    """At most one hash-prefix shard is loaded by DVC at a time."""
    targets = LABEL_TARGETS + [
        p.relative_to(work).as_posix()
        for p in sorted((work / "dataset/images").iterdir())
        if p.is_dir()
    ]
    with dvc_repo(cfg, work) as repo:
        for index, target in enumerate(targets):
            absolute = str(work / target)
            repo.add(absolute)
            repo.push(targets=[absolute + ".dvc"], jobs=4, remote="release")
            if target.startswith("dataset/images/"):
                for path in (work / target).iterdir():
                    path.chmod(0o444)
                    relative = path.relative_to(work / "dataset")
                    _safe_link(path, cfg.storage_dir / "releases/pool" / relative)
            if progress:
                progress(index + 1, len(targets))
    return [target + ".dvc" for target in targets]


def check_remote(cfg, work, targets):
    """Verify local remote bytes, not just their existence, before publishing a tag.

    Directory manifests are shard-sized. This layout is the DVC 3 md5 object format.
    """
    import yaml

    with tempfile.TemporaryDirectory(dir=cfg.storage_dir, prefix="remote-check-") as folder:
        with closing(sqlite3.connect(Path(folder) / "seen.sqlite3")) as seen:
            seen.execute("CREATE TABLE objects (hash TEXT PRIMARY KEY)")

            def verify(digest):
                if not re.fullmatch(r"[0-9a-f]{32}(?:\.dir)?", digest):
                    raise ValueError("Unsupported DVC object hash; expected DVC 3 md5")
                if not seen.execute("INSERT OR IGNORE INTO objects VALUES (?)", (digest,)).rowcount:
                    return None
                path = cfg.dvc_remote / "files/md5" / digest[:2] / digest[2:]
                with path.open("rb") as handle:
                    actual = hashlib.file_digest(handle, "md5").hexdigest()
                if actual != digest.removesuffix(".dir"):
                    raise ValueError(f"DVC remote object corrupted: {digest}")
                return path

            for target in targets:
                out = yaml.safe_load((work / target).read_text())["outs"][0]
                digest = out["md5"]
                path = verify(digest)
                if digest.endswith(".dir") and path:
                    for item in json.loads(path.read_text()):
                        verify(item["md5"])
                seen.commit()


def publish(root, work, info):
    """Create a dataset-only commit from frozen HEAD, then atomically reserve its tag."""
    base = info["code_commit"]
    with tempfile.TemporaryDirectory(dir=work.parent, prefix="git-index-") as directory:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(directory) / "index")}
        git(root, "read-tree", base, env=env)
        old = git(root, "ls-tree", "-r", "--name-only", base, "--", PREFIX).splitlines()
        for name in old:
            git(root, "update-index", "--force-remove", name, env=env)
        names = [*METADATA_FILES, *info["dvc_targets"]]
        for name in names:
            blob = git(root, "hash-object", "-w", "--stdin", data=(work / name).read_bytes())
            git(
                root,
                "update-index",
                "--add",
                "--cacheinfo",
                "100644",
                blob,
                f"{PREFIX}/{name}",
                env=env,
            )
        tree = git(root, "write-tree", env=env)
        commit = git(
            root,
            "commit-tree",
            tree,
            "-p",
            base,
            data=f"Release dataset/{info['version']}\n".encode(),
        )
        git(root, "update-ref", f"refs/tags/dataset/{info['version']}", commit, "0" * len(commit))
    return commit


def restore_data(cfg, version, *, metadata_only=False):
    root = project_repo(cfg)
    info = descriptor(root, version)
    commit = git(root, "rev-parse", f"refs/tags/dataset/{version}^{{commit}}")
    work = cfg.storage_dir / "releases/restored" / version
    marker = work / "restored.json"
    work.mkdir(parents=True, exist_ok=True)
    if marker.exists() and json.loads(marker.read_text())["commit"] != commit:
        raise ValueError("Dataset tag changed after restoration")
    for name in [*METADATA_FILES, *info["dvc_targets"]]:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Invalid dataset metadata path")
        content = git(root, "show", f"{commit}:{PREFIX}/{name}") + "\n"
        destination = work / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)
    targets = ["dataset/metadata.dvc"] if metadata_only else info["dvc_targets"]
    with dvc_repo(cfg, work) as repo:
        for target in targets:
            repo.pull(targets=[str(work / target)], jobs=4, remote="release")
    snapshot = work / "dataset/metadata/snapshot.sqlite3"
    if sha256_file(snapshot) != info["snapshot_sha256"]:
        raise ValueError("Restored snapshot checksum mismatch")
    if not metadata_only:
        verify_images(snapshot, work / "dataset")
        for name, checksum in info["label_checksums"].items():
            if sha256_file(work / "dataset" / name) != checksum:
                raise ValueError(f"Restored label checksum mismatch: {name}")
    write_json(marker, {"commit": commit, "metadata_only": metadata_only})
    return work / "dataset", info
