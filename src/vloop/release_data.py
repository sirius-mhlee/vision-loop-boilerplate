"""Disk-backed release snapshots and streaming COCO export."""

import hashlib
import json
import re
import sqlite3
from contextlib import ExitStack, closing
from itertools import zip_longest
from pathlib import Path

from .approval import approved_snapshot, content_hash
from .review_store import connect_records
from .runtime import sha256_file, write_json

SPLITS = {"train": "train", "val": "valid", "test": "test"}


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str)


def connect(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA cache_size=-8192")
    db.execute("PRAGMA temp_store=FILE")
    return db


def initial_split(key, cfg):
    value = int(hashlib.sha256(f"{cfg.seed}:{key}".encode()).hexdigest()[:16], 16) / 2**64
    if value < cfg.split_ratios[0]:
        return "train"
    return "val" if value < sum(cfg.split_ratios[:2]) else "test"


def capture(cfg, samples, path, *, parent=None, include_auto=False):
    """Publish a snapshot only after its entire approval scan succeeds.

    A retry of an interrupted scan takes a new snapshot; a finished snapshot is frozen.
    All large collections, including historical split assignments, stay in SQLite.
    """
    temporary = path.with_suffix(".partial.sqlite3")
    temporary.unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        db = stack.enter_context(closing(connect(temporary)))
        record_db = None
        if include_auto and (cfg.storage_dir / "reviews/records.sqlite3").exists():
            record_db = stack.enter_context(closing(connect_records(cfg, readonly=True)))
        db.executescript("""
            CREATE TABLE ledger (image_id TEXT PRIMARY KEY, group_key TEXT NOT NULL,
                                 split TEXT NOT NULL, coco_id INTEGER UNIQUE NOT NULL);
            CREATE TABLE groups (group_key TEXT PRIMARY KEY, split TEXT NOT NULL);
            CREATE TABLE records (image_id TEXT PRIMARY KEY, source_path TEXT NOT NULL,
                managed_sha256 TEXT NOT NULL, annotation TEXT NOT NULL, provenance TEXT NOT NULL,
                automatic INTEGER NOT NULL, group_key TEXT NOT NULL, split TEXT,
                coco_id INTEGER, held_reason TEXT);
            CREATE INDEX records_split ON records(split, image_id);
            CREATE INDEX records_group ON records(group_key, automatic);
        """)
        if parent:
            db.execute("ATTACH DATABASE ? AS previous", (str(parent),))
            db.execute("INSERT INTO ledger SELECT * FROM previous.ledger")
            db.execute("INSERT INTO groups SELECT * FROM previous.groups")
            db.commit()
            db.execute("DETACH DATABASE previous")
        count = 0
        for sample in samples:
            status = sample["review_status"]
            if status != "completed" and not (include_auto and status == "auto_accepted"):
                continue
            image_id = sample["image_id"]
            if not re.fullmatch(r"[0-9a-f]{64}", image_id or ""):
                raise ValueError("Invalid release image ID")
            try:
                annotation, record = approved_snapshot(
                    cfg,
                    sample,
                    verify_image=False,
                    allow_auto=include_auto,
                    record_connection=record_db,
                )
                if cfg.release_group_field:
                    group = sample[cfg.release_group_field]
                    if not isinstance(group, str) or not group.strip():
                        raise ValueError("Configured group field must be a nonempty string")
                    group_key = "group:" + group
                else:
                    group_key = "image:" + image_id
                provenance = {
                    "approval": record,
                    "approval_sha256": sample["review_approval_sha256"],
                    "history": sample["review_history"],
                    "source": sample["ground_truth_source"],
                    "review_status": status,
                }
                db.execute(
                    "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)",
                    (
                        image_id,
                        sample.filepath,
                        annotation["managed_sha256"],
                        dumps(annotation),
                        dumps(provenance),
                        int(status == "auto_accepted"),
                        group_key,
                    ),
                )
            except (ValueError, OSError, KeyError) as exc:
                raise ValueError(f"Image {image_id}: {exc}") from exc
            count += 1
            if count % 250 == 0:
                db.commit()
        if not count:
            raise ValueError("No approved images; complete review or use --include-auto-accepted")
        # Groups containing automatic labels are train-only on the first release.
        # A historical held-out group is never silently moved to train.
        if parent is None:
            db.execute(
                "INSERT INTO groups SELECT DISTINCT group_key, 'train' FROM records "
                "WHERE automatic=1"
            )
        next_id = db.execute("SELECT COALESCE(MAX(coco_id), 0)+1 FROM ledger").fetchone()[0]
        for row in db.execute(
            "SELECT image_id, group_key, automatic FROM records ORDER BY image_id"
        ):
            old = db.execute("SELECT * FROM ledger WHERE image_id=?", (row["image_id"],)).fetchone()
            group = db.execute(
                "SELECT split FROM groups WHERE group_key=?", (row["group_key"],)
            ).fetchone()
            if old:
                if old["group_key"] != row["group_key"]:
                    raise ValueError(f"Group changed for existing image {row['image_id']}")
                split, coco_id = old["split"], old["coco_id"]
                if row["automatic"] and split != "train":
                    raise ValueError("An existing val/test image needs manual approval")
            elif parent and group and group[0] != "train":
                db.execute(
                    "UPDATE records SET held_reason='new_image_in_heldout_group' WHERE image_id=?",
                    (row["image_id"],),
                )
                continue
            else:
                split = (
                    group[0]
                    if group
                    else ("train" if parent else initial_split(row["group_key"], cfg))
                )
                coco_id, next_id = next_id, next_id + 1
                db.execute(
                    "INSERT INTO ledger VALUES (?, ?, ?, ?)",
                    (row["image_id"], row["group_key"], split, coco_id),
                )
                db.execute("INSERT OR IGNORE INTO groups VALUES (?, ?)", (row["group_key"], split))
            db.execute(
                "UPDATE records SET split=?, coco_id=? WHERE image_id=?",
                (split, coco_id, row["image_id"]),
            )
        db.commit()
        if not db.execute("SELECT 1 FROM records WHERE split IS NOT NULL LIMIT 1").fetchone():
            raise ValueError("All new images belong to held-out groups; no releasable images")
    write_json(path.with_suffix(".seal.json"), {"sha256": sha256_file(temporary)})
    temporary.replace(path)


def image_relative(row):
    digest = row["managed_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", digest or ""):
        raise ValueError("Invalid managed image checksum")
    return Path("images") / digest[:2] / f"{digest}.png"


def _array(handle, values):
    handle.write("[")
    for index, value in enumerate(values):
        if index:
            handle.write(",")
        handle.write(dumps(value))
    handle.write("]")


def export_coco(cfg, snapshot, root):
    classes = [
        {"id": c.id, "name": c.name, "supercategory": "object"}
        for c in sorted(cfg.classes, key=lambda c: c.id)
    ]
    summary = {
        "splits": {},
        "split_directories": SPLITS,
        "classes": [{**c, "model_index": i} for i, c in enumerate(classes)],
        "group_field": cfg.release_group_field,
        "seed": cfg.seed,
        "split_ratios": list(cfg.split_ratios),
        "split_method": "sha256_group_threshold_v1",
    }
    with closing(connect(snapshot)) as db:
        summary["held"] = db.execute("SELECT COUNT(*) FROM records WHERE split IS NULL").fetchone()[
            0
        ]
        for split, folder in SPLITS.items():
            directory = root / folder
            directory.mkdir(parents=True, exist_ok=True)
            counts = {c.id: {"images": 0, "instances": 0} for c in cfg.classes}
            stats = {"images": 0, "empty": 0, "manual": 0, "automatic": 0}
            fingerprint = hashlib.sha256(dumps(classes).encode())

            def images():
                for row in db.execute(
                    "SELECT * FROM records WHERE split=? ORDER BY image_id", (split,)
                ):
                    content = json.loads(row["annotation"])
                    stats["images"] += 1
                    stats["automatic" if row["automatic"] else "manual"] += 1
                    stats["empty"] += int(not content["instances"])
                    present = set()
                    for instance in content["instances"]:
                        counts[instance["class_id"]]["instances"] += 1
                        present.add(instance["class_id"])
                    for class_id in present:
                        counts[class_id]["images"] += 1
                    fingerprint.update((content_hash(content) + "\n").encode())
                    yield {
                        "id": row["coco_id"],
                        "file_name": "../" + image_relative(row).as_posix(),
                        "width": content["width"],
                        "height": content["height"],
                        "vloop_image_id": row["image_id"],
                        "sha256": row["managed_sha256"],
                    }

            def annotations():
                annotation_id = 0
                for row in db.execute(
                    "SELECT coco_id, annotation FROM records WHERE split=? ORDER BY image_id",
                    (split,),
                ):
                    for instance in json.loads(row["annotation"])["instances"]:
                        annotation_id += 1
                        yield {
                            "id": annotation_id,
                            "image_id": row["coco_id"],
                            "category_id": instance["class_id"],
                            "bbox": instance["bbox_xywh"],
                            "area": instance["area"],
                            "segmentation": instance["segmentation"],
                            "iscrowd": 0,
                        }

            temporary = directory / "_annotations.coco.json.tmp"
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write('{"images":')
                _array(handle, images())
                handle.write(',"annotations":')
                _array(handle, annotations())
                handle.write(',"categories":')
                _array(handle, classes)
                handle.write("}\n")
            temporary.replace(directory / "_annotations.coco.json")
            stats["classes"] = {
                str(key): {**value, "evaluation": "N/A" if not value["instances"] else "available"}
                for key, value in counts.items()
            }
            stats["data_id"] = fingerprint.hexdigest()
            summary["splits"][split] = stats
    write_json(root / "metadata" / "summary.json", summary)
    return summary


def verify_images(snapshot, root):
    with closing(connect(snapshot)) as db:
        for row in db.execute(
            "SELECT DISTINCT managed_sha256 FROM records WHERE split IS NOT NULL"
        ):
            if sha256_file(root / image_relative(row)) != row["managed_sha256"]:
                raise ValueError(f"Release image checksum mismatch: {row['managed_sha256']}")


def validate_coco(cfg, snapshot, root):
    """Read exported JSON back incrementally and compare it to the frozen approvals."""
    import ijson

    from .labels import decode_mask

    with closing(connect(snapshot)) as db:
        for split, folder in SPLITS.items():
            path = root / folder / "_annotations.coco.json"
            with path.open("rb") as handle:
                actual = ijson.items(handle, "images.item")
                rows = db.execute("SELECT * FROM records WHERE split=? ORDER BY image_id", (split,))
                for image, row in zip_longest(actual, rows):
                    if image is None or row is None:
                        raise ValueError("COCO image count differs from snapshot")
                    content = json.loads(row["annotation"])
                    expected = {
                        "id": row["coco_id"],
                        "file_name": "../" + image_relative(row).as_posix(),
                        "width": content["width"],
                        "height": content["height"],
                        "vloop_image_id": row["image_id"],
                        "sha256": row["managed_sha256"],
                    }
                    if image != expected:
                        raise ValueError("COCO image metadata differs from snapshot")

            def expected_annotations():
                annotation_id = 0
                for row in db.execute(
                    "SELECT coco_id, annotation FROM records WHERE split=? ORDER BY image_id",
                    (split,),
                ):
                    content = json.loads(row["annotation"])
                    for instance in content["instances"]:
                        annotation_id += 1
                        yield (
                            {
                                "id": annotation_id,
                                "image_id": row["coco_id"],
                                "category_id": instance["class_id"],
                                "bbox": instance["bbox_xywh"],
                                "segmentation": instance["segmentation"],
                                "area": instance["area"],
                                "iscrowd": 0,
                            },
                            content["width"],
                            content["height"],
                        )

            with path.open("rb") as handle:
                for actual, expected in zip_longest(
                    ijson.items(handle, "annotations.item"), expected_annotations()
                ):
                    if actual is None or expected is None or actual != expected[0]:
                        raise ValueError("COCO annotation differs from snapshot")
                    mask = decode_mask(actual["segmentation"], expected[1], expected[2])
                    if int(mask.sum()) != actual["area"]:
                        raise ValueError("COCO mask area differs from annotation")
            with path.open("rb") as handle:
                actual = list(ijson.items(handle, "categories.item"))
                expected = [
                    {"id": c.id, "name": c.name, "supercategory": "object"}
                    for c in sorted(cfg.classes, key=lambda c: c.id)
                ]
                if actual != expected:
                    raise ValueError("COCO class mapping differs from project")
