"""Measure RF-DETR's COCO index RAM; no images are loaded and no model is trained."""

import argparse
import json
import resource
import tempfile
import time
from pathlib import Path

from rfdetr.datasets.coco import CocoDetection

from vloop.runtime import write_json


def measure(count, output):
    with tempfile.TemporaryDirectory(prefix="vloop-loader-scale-") as temporary:
        root = Path(temporary)
        path = root / "annotations.json"
        with path.open("w") as handle:
            handle.write('{"images":[')
            for index in range(count):
                if index:
                    handle.write(",")
                json.dump(
                    {"id": index, "file_name": f"{index:064x}.png", "width": 64, "height": 64},
                    handle,
                )
            handle.write('],"annotations":[')
            for index in range(count):
                if index:
                    handle.write(",")
                json.dump(
                    {
                        "id": index,
                        "image_id": index,
                        "category_id": 7,
                        "bbox": [0, 0, 64, 64],
                        "area": 4096,
                        "iscrowd": 0,
                        "segmentation": {"size": [64, 64], "counts": [0, 4096]},
                    },
                    handle,
                )
            handle.write('],"categories":[{"id":7,"name":"object"}]}')
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        started = time.monotonic()
        dataset = CocoDetection(
            root, path, transforms=None, include_masks=True, remap_category_ids=True
        )
        elapsed = time.monotonic() - started
        assert len(dataset) == count
        result = {
            "images": count,
            "annotations": count,
            "annotation_bytes": path.stat().st_size,
            "load_seconds": elapsed,
            "baseline_peak_rss_mib": before,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "scope": "Synthetic COCO index with one fixed-size RLE per image; no pixels/train/eval",
        }
        write_json(output, result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--output", type=Path, default=Path(".vloop/train-loader-scale.json"))
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    measure(args.count, args.output)
