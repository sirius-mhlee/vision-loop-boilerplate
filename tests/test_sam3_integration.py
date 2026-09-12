import json
import os
import shutil
import subprocess
import sys

import pytest
import yaml

from vloop.config import load_config
from vloop.runtime import sha256_file


@pytest.mark.skipif(
    not os.environ.get("VLOOP_TEST_SAM3_CONFIG"),
    reason="Set VLOOP_TEST_SAM3_CONFIG to run the real CUDA checkpoint test",
)
def test_real_gpu_interruption_resume_and_human_label_preservation(tmp_path):
    base = load_config(os.environ["VLOOP_TEST_SAM3_CONFIG"])
    images = tmp_path / "input"
    images.mkdir()
    for name in ("truck.jpg", "groceries.jpg"):
        shutil.copy2(base.sam3_source_dir / "assets/images" / name, images / name)
    data = base.to_dict()
    data.pop("config_path")
    data.update(
        image_dir=str(images),
        storage_dir=str(tmp_path / "state"),
        dataset_name="vloop-gpu-integration",
        device="cuda",
        classes=[
            {"id": 7, "name": "truck", "prompts": ["truck"]},
            {"id": 42, "name": "apple", "prompts": ["apple"]},
        ],
    )
    config_path = tmp_path / "project.yaml"
    config_path.write_text(yaml.safe_dump(data))
    setup = r"""
import json
import sys
import numpy as np
from vloop.config import load_config
from vloop.ingest import ingest
from vloop.fiftyone import configure_fiftyone
from vloop import autolabel as module
cfg = load_config(sys.argv[1])
assert ingest(cfg)["status"] == "completed"
fo = configure_fiftyone(cfg)
dataset = fo.load_dataset(cfg.dataset_name)
sample = dataset.first()
sample["ground_truth"] = fo.Detections(detections=[fo.Detection(
    label="truck", bounding_box=[0.1, 0.1, 0.2, 0.2], mask=np.ones((5, 5), dtype=bool))])
sample["review_status"] = "completed"
sample.save()
guard = {"image_id": sample["image_id"], "ground_truth": sample["ground_truth"].to_json()}
(cfg.storage_dir / "human-label.json").write_text(json.dumps(guard))
original = module.Sam3Labeler.predict
calls = 0
def interrupt_after_first(self, image):
    global calls
    calls += 1
    if calls == 2:
        raise KeyboardInterrupt
    return original(self, image)
module.Sam3Labeler.predict = interrupt_after_first
report = module.autolabel(cfg)
assert report["status"] == "interrupted", report
assert report["completed"] == 1, report
"""
    first = subprocess.run(
        [sys.executable, "-c", setup, str(config_path)], capture_output=True, text=True, timeout=240
    )
    assert first.returncode == 0, first.stdout + first.stderr
    directory = next((tmp_path / "state/runs").glob("autolabel_*"))
    completed = next((directory / "predictions").glob("*.json"))
    original_hash = sha256_file(completed)
    # A changed project prompt must not affect an in-progress job.
    changed = dict(data, classes=[{"id": 99, "name": "changed", "prompts": ["changed"]}])
    config_path.write_text(yaml.safe_dump(changed))
    second = subprocess.run(
        [
            sys.executable,
            "-m",
            "vloop",
            "autolabel",
            "--config",
            str(config_path),
            "--resume",
            directory.name,
        ],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert second.returncode == 0, second.stdout + second.stderr
    report = json.loads((directory / "report.json").read_text())
    assert report["completed"] == 2
    assert report["processed_this_attempt"] == 1
    assert sha256_file(completed) == original_hash
    predictions = [
        json.loads(path.read_text()) for path in (directory / "predictions").glob("*.json")
    ]
    items = [item for prediction in predictions for item in prediction["instances"]]
    assert any(item["class_id"] == 7 and item["class_name"] == "truck" for item in items)
    assert all(item["class_id"] in (7, 42) for item in items)
    verify = r"""
import json
import sys
from pathlib import Path
from vloop.config import load_config
from vloop.fiftyone import configure_fiftyone
from vloop.labels import decode_mask
cfg = load_config(sys.argv[1])
fo = configure_fiftyone(cfg)
dataset = fo.load_dataset(cfg.dataset_name)
guard = json.loads((cfg.storage_dir / "human-label.json").read_text())
sample = dataset.match(fo.ViewField("image_id") == guard["image_id"]).first()
assert sample["ground_truth"].to_json() == guard["ground_truth"]
assert sample["review_status"] == "completed"
directory = Path(sys.argv[2])
report = json.loads((directory / "report.json").read_text())
for path in (directory / "predictions").glob("*.json"):
    prediction = json.loads(path.read_text())
    sample = dataset.match(fo.ViewField("image_id") == prediction["image_id"]).first()
    detections = sample[report["prediction_field"]].detections
    assert len(detections) == len(prediction["instances"])
    for detection, item in zip(detections, prediction["instances"]):
        mask = decode_mask(item["segmentation"], prediction["width"], prediction["height"])
        assert int(detection.mask.sum()) == int(mask.sum()) == item["area"]
        assert detection["class_id"] == item["class_id"]
dataset.delete()
"""
    config_path.write_text(yaml.safe_dump(data))
    verified = subprocess.run(
        [sys.executable, "-c", verify, str(config_path), str(directory)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert verified.returncode == 0, verified.stdout + verified.stderr
    print(f"GPU validation: {directory}")
    print(f"Resume runtime: {report['runtime']}")
