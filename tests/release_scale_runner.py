"""Synthetic metadata-only export benchmark; does not benchmark images, MongoDB, or DVC."""

import argparse
import json
import resource
import sqlite3
import tempfile
import time
from pathlib import Path

from vloop.config import ClassConfig, Config
from vloop.release_data import export_coco, validate_coco
from vloop.runtime import write_json


def run(count, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="release-scale-") as folder:
        root = Path(folder)
        (root / "metadata").mkdir()
        snapshot = root / "metadata/snapshot.sqlite3"
        with sqlite3.connect(snapshot) as db:
            db.execute("PRAGMA cache_size=-8192")
            db.execute(
                "CREATE TABLE records (image_id TEXT PRIMARY KEY, managed_sha256 TEXT, "
                "annotation TEXT, split TEXT, automatic INTEGER, coco_id INTEGER, held_reason TEXT)"
            )
            db.execute("CREATE INDEX records_split ON records(split, image_id)")
            for i in range(count):
                digest = f"{i:064x}"
                content = {
                    "schema_version": 1,
                    "image_id": digest,
                    "managed_sha256": digest,
                    "width": 23,
                    "height": 13,
                    "classes": [["object", 7]],
                    "instances": [],
                }
                split = "train" if i % 10 < 8 else "val" if i % 10 == 8 else "test"
                db.execute(
                    "INSERT INTO records VALUES (?, ?, ?, ?, 0, ?, NULL)",
                    (digest, digest, json.dumps(content), split, i + 1),
                )
                if i % 5000 == 0:
                    db.commit()
        populated = time.perf_counter()
        cfg = Config(
            classes=(ClassConfig(7, "object", ("object",)),),
            release_group_field=None,
            seed=42,
            split_ratios=(0.8, 0.1, 0.1),
        )
        summary = export_coco(cfg, snapshot, root)
        exported = time.perf_counter()
        validate_coco(cfg, snapshot, root)
        report = {
            "scope": "synthetic metadata only; no actual images, MongoDB, or DVC transfers",
            "count": count,
            "create_seconds": populated - started,
            "export_seconds": exported - populated,
            "validation_seconds": time.perf_counter() - exported,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "sqlite_mib": snapshot.stat().st_size / 2**20,
            "coco_mib": sum(p.stat().st_size for p in root.glob("*/_annotations.coco.json"))
            / 2**20,
            "splits": {name: stats["images"] for name, stats in summary["splits"].items()},
        }
        write_json(output, report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, default=Path(".vloop/release-scale.json"))
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    run(args.count, args.output)
