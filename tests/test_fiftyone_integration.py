import os
import subprocess
import sys

import pytest
from PIL import Image

from vloop.fiftyone import configure_fiftyone
from vloop.ingest import ingest


@pytest.mark.skipif(
    os.environ.get("VLOOP_TEST_FIFTYONE") != "1",
    reason="Set VLOOP_TEST_FIFTYONE=1 to start the real local FiftyOne database",
)
def test_real_database_registration_preserves_labels_across_processes(project):
    Image.new("RGB", (13, 7), "red").save(project.image_dir / "input.png")
    report = ingest(project)
    assert report["status"] == "completed", report
    fo = configure_fiftyone(project)
    dataset = fo.load_dataset(project.dataset_name)
    try:
        assert len(dataset) == 1
        sample = dataset.first()
        sample["ground_truth"] = fo.Detections(
            detections=[fo.Detection(label="test-object", bounding_box=[0.1, 0.1, 0.5, 0.5])]
        )
        sample["review_status"] = "completed"
        sample.save()
        result = subprocess.run(
            [sys.executable, "-m", "vloop", "ingest", "--config", str(project.config_path)],
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        dataset.reload()
        assert len(dataset) == 1
        sample = dataset.first()
        assert sample["review_status"] == "completed"
        assert sample["ground_truth"].detections[0].label == "test-object"
        assert sample["metadata"].width == 13
        assert sample["metadata"].height == 7
    finally:
        dataset.delete()
