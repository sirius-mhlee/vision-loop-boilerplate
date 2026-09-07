"""Exercise the real FiftyOne -> approval -> DVC -> restore boundary in a fresh process."""

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from vloop.config import load_config
from vloop.ingest import ingest
from vloop.release import release, restore
from vloop.release_dvc import git
from vloop.review import change_reviews, load_review_dataset, prepare_review


def run(path):
    root = path.parent
    data = yaml.safe_load(path.read_text())
    data["dvc_remote"] = str(root.parent / (root.name + "-remote"))
    path.write_text(yaml.safe_dump(data))
    cfg = load_config(path)
    git(root, "init", "-q")
    git(root, "config", "user.name", "Integration Test")
    git(root, "config", "user.email", "test@example.invalid")
    (root / ".gitignore").write_text("state/\ninput/\nproject.yaml\n")
    git(root, "add", ".gitignore")
    git(root, "commit", "-qm", "Test fixture")
    for index, color in enumerate(("red", "green", "blue")):
        Image.new("RGB", (13, 7), color).save(cfg.image_dir / f"{index}.png")
    assert ingest(cfg)["status"] == "completed"
    assert prepare_review(cfg)["status"] == "completed"
    dataset = load_review_dataset(cfg)
    import fiftyone as fo

    try:
        ids = dataset.values("id")
        for index, sample_id in enumerate(ids):
            assert change_reviews(cfg, [sample_id], "start", "test")["changed"] == 1
            if index == 0:
                sample = dataset[sample_id]
                sample.reload()
                mask = np.ones((7, 13), dtype=bool)
                mask[2:5, 3:8] = False
                sample["ground_truth"] = fo.Detections(
                    detections=[
                        fo.Detection(label="test-object", bounding_box=[0, 0, 1, 1], mask=mask)
                    ]
                )
                sample.save()
            assert (
                change_reviews(cfg, [sample_id], "complete", "test", confirm_empty=True)["changed"]
                == 1
            )
        report = release(cfg, version="v001")
        assert report["status"] == "completed", report
        restored = restore(cfg, version="v001")
        assert restored["status"] == "completed", restored
        assert sum(s["images"] for s in report["summary"]["splits"].values()) == 3
        assert sum(s["empty"] for s in report["summary"]["splits"].values()) == 2
        # A fresh SampleView must observe an edit after approval.
        sample = dataset[ids[0]]
        sample.reload()
        sample["ground_truth"].detections[0].label = "unknown"
        sample.save()
        failed = release(cfg, version="v002")
        assert failed["status"] == "failed" and "Unknown ground truth class" in failed["error"], (
            failed
        )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "images": 3,
                    "empty": 2,
                    "checks": [
                        "FiftyOne SampleView",
                        "manual approvals",
                        "DVC remote",
                        "restoration",
                        "edited approval rejection",
                    ],
                }
            )
        )
    finally:
        dataset.delete()


if __name__ == "__main__":
    run(Path(sys.argv[1]))
