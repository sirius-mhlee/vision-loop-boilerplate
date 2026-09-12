"""Keep the real FiftyOne service inside a process that ends with this test."""

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from PIL import Image

from vloop.autolabel import PredictionStore
from vloop.config import load_config
from vloop.fiftyone import configure_fiftyone
from vloop.ingest import ingest


def run(config_path):
    project = load_config(config_path)
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
        # A second project must not silently use this process's already running MongoDB.
        other = project.to_dict()
        other.pop("config_path")
        other.update(
            storage_dir=str(project.storage_dir.parent / "other-state"),
            dataset_name="other-project",
        )
        other_config = project.config_path.parent / "other.yaml"
        other_config.write_text(yaml.safe_dump(other))
        for inherit_port in (True, False):
            environment = dict(os.environ)
            if not inherit_port:
                environment.pop("FIFTYONE_PRIVATE_DATABASE_PORT", None)
            rejected = subprocess.run(
                [sys.executable, "-m", "vloop", "ingest", "--config", str(other_config)],
                env=environment,
                capture_output=True,
                text=True,
                timeout=90,
            )
            assert rejected.returncode == 1, rejected.stdout + rejected.stderr
            assert "different database directory" in rejected.stderr, (
                rejected.stdout + rejected.stderr
            )
        assert not fo.dataset_exists("other-project")
        for report_path in (Path(other["storage_dir"]) / "runs").glob("ingest_*/report.json"):
            rejected_report = json.loads(report_path.read_text())
            assert rejected_report["sync_status"] == "incomplete"
        dataset.reload()
        assert len(dataset) == 1
        assert sample["metadata"].width == 13
        assert sample["metadata"].height == 7
        # Simulate interruption after the prediction field, before its checksum field.
        field = "pred_autolabel_partial_setup"
        dataset.add_sample_field(field, fo.EmbeddedDocumentField, embedded_doc_type=fo.Detections)
        store = PredictionStore(project, field, "partial_setup", {"schema_version": 1})
        store.put(
            {
                "image_id": sample["image_id"],
                "image_path": sample.filepath,
                "width": 13,
                "height": 7,
                "instances": [],
            },
            "saved-empty-result",
        )
        sample.reload()
        assert sample[field].detections == []
        assert sample[f"{field}_sha256"] == "saved-empty-result"
        assert sample["review_status"] == "completed"
        assert sample["ground_truth"].detections[0].label == "test-object"
    finally:
        dataset.delete()


if __name__ == "__main__":
    run(Path(sys.argv[1]))
