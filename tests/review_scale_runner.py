"""Optional million-row metadata benchmark; does not create images or run inference."""

import argparse
import json
import resource
import sqlite3
import tempfile
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from vloop.review_batch import _snapshot


class MetadataCursor:
    def __init__(self, count):
        self.count = count

    def sort(self, *_):
        return self

    def batch_size(self, size):
        assert size == 500
        return self

    def limit(self, count):
        self.count = min(count, self.count)
        return self

    def close(self):
        pass

    def __iter__(self):
        stamp = datetime(2026, 1, 1)
        for i in range(self.count):
            yield {
                "_id": f"{i:024x}",
                "image_id": f"{i:064x}",
                "pred_job_sha256": "a" * 64,
                "last_modified_at": stamp,
            }


class MetadataCollection:
    def __init__(self, count):
        self.count = count

    def find_one(self, *_args, **_kwargs):
        return {"_id": f"{self.count - 1:024x}"}

    def find(self, query, projection):
        assert set(projection) == {"image_id", "pred_job_sha256", "last_modified_at"}
        assert "ground_truth" not in projection
        return MetadataCursor(self.count)


def run(count):
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="vloop-scale-") as root:
        path = Path(root) / "samples.sqlite3"
        with sqlite3.connect(path) as connection:
            _snapshot(
                SimpleNamespace(_sample_collection=MetadataCollection(count)),
                connection,
                "pred_job",
                None,
            )
            assert connection.execute("SELECT COUNT(*) FROM inputs").fetchone()[0] == count
            first_page = connection.execute(
                "SELECT sample_id FROM inputs WHERE decision IS NULL ORDER BY sample_id LIMIT 100"
            ).fetchall()
            assert len(first_page) == min(100, count)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        assert peak - baseline < 64, "Snapshot memory grew beyond the bounded-memory budget"
        return {
            "status": "passed",
            "metadata_rows": count,
            "seconds": time.monotonic() - started,
            "peak_rss_mib": peak,
            "rss_growth_mib": peak - baseline,
            "sqlite_mib": path.stat().st_size / 1024**2,
            "scope": "Synthetic metadata cursor to SQLite; not a million-image benchmark",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1_000_000)
    args = parser.parse_args()
    print(json.dumps(run(args.count), indent=2))
