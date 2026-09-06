from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from vloop.approval import annotation_content, approved_annotation, content_hash, record_path
from vloop.config import ClassConfig
from vloop.labels import decode_mask
from vloop.runtime import sha256_file, write_json


class Sample(dict):
    metadata = SimpleNamespace(width=23, height=13)


@pytest.fixture
def annotation():
    mask = np.ones((7, 11), dtype=bool)
    mask[2:5, 3:7] = False
    detection = SimpleNamespace(
        label="object",
        class_id=999,
        mask=mask,
        mask_path=None,
        bounding_box=[4 / 23, 3 / 13, 11 / 23, 7 / 13],
    )
    sample = Sample(
        image_id="a" * 64,
        managed_sha256="b" * 64,
        ground_truth=SimpleNamespace(detections=[detection]),
    )
    return sample, (ClassConfig(7, "object", ("object",)), ClassConfig(42, "other", ("other",)))


def test_canonical_annotation_preserves_mask_and_uses_edited_class(annotation):
    sample, classes = annotation
    content = annotation_content(sample, classes)
    item = content["instances"][0]
    assert item["class_id"] == 7
    mask = decode_mask(item["segmentation"], 23, 13)
    assert np.array_equal(mask[3:10, 4:15], sample["ground_truth"].detections[0].mask)
    assert mask.sum() == item["area"] == 65
    sample["ground_truth"].detections[0].label = "other"
    changed = annotation_content(sample, classes)
    assert changed["instances"][0]["class_id"] == 42
    assert content_hash(changed) != content_hash(content)


@pytest.mark.parametrize("change", ["mask", "box", "class", "image", "size", "mapping"])
def test_approval_fingerprint_changes_with_training_content(annotation, change):
    sample, classes = annotation
    original = content_hash(annotation_content(sample, classes))
    detection = sample["ground_truth"].detections[0]
    if change == "mask":
        detection.mask[0, 0] = False
    elif change == "box":
        detection.bounding_box[0] += 1 / 23
    elif change == "class":
        detection.label = "other"
    elif change == "image":
        sample["managed_sha256"] = "c" * 64
    elif change == "size":
        sample.metadata = SimpleNamespace(width=46, height=26)
        detection.bounding_box = [v / 2 for v in detection.bounding_box]
    else:
        classes = (ClassConfig(8, "object", ("object",)), classes[1])
    assert content_hash(annotation_content(sample, classes)) != original


def test_hash_ignores_detection_order_and_prediction_metadata(annotation):
    sample, classes = annotation
    other = deepcopy(sample["ground_truth"].detections[0])
    other.label = "other"
    sample["ground_truth"].detections.append(other)
    original = content_hash(annotation_content(sample, classes))
    sample["ground_truth"].detections.reverse()
    other.confidence = 0.9
    other.id = "new-id"
    assert content_hash(annotation_content(sample, classes)) == original


def test_missing_ground_truth_is_not_an_empty_approved_annotation(annotation):
    sample, classes = annotation
    sample["ground_truth"] = None
    with pytest.raises(ValueError, match="missing"):
        annotation_content(sample, classes)
    sample["ground_truth"] = SimpleNamespace(detections=[])
    assert annotation_content(sample, classes)["instances"] == []


@pytest.mark.parametrize(
    "invalid",
    ["class", "missing_mask", "mask_path", "shape", "empty_mask", "bounds", "nan", "nonbinary"],
)
def test_invalid_instances_cannot_be_approved(annotation, invalid):
    sample, classes = annotation
    detection = sample["ground_truth"].detections[0]
    if invalid == "class":
        detection.label = "unknown"
    elif invalid == "missing_mask":
        detection.mask = None
    elif invalid == "mask_path":
        detection.mask_path = "/tmp/external.png"
    elif invalid == "shape":
        detection.mask = np.ones((2, 2, 2), dtype=bool)
    elif invalid == "empty_mask":
        detection.mask[:] = False
    elif invalid == "bounds":
        detection.bounding_box[0] = -0.1
    elif invalid == "nan":
        detection.bounding_box[0] = float("nan")
    else:
        detection.mask = np.full((7, 11), 2, dtype=np.uint8)
    with pytest.raises(ValueError):
        annotation_content(sample, classes)


def test_native_editor_mask_projection_preserves_source_and_disconnected_regions(annotation):
    sample, classes = annotation
    detection = sample["ground_truth"].detections[0]
    detection.bounding_box = [0.11, 0.15, 0.26, 0.31]
    detection.mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    item = annotation_content(sample, classes)["instances"][0]
    assert item["bbox_xywh"] == [3, 2, 6, 4]
    assert item["area"] == 12
    expected = np.zeros((13, 23), dtype=bool)
    expected[2:4, 3:6] = True
    expected[4:6, 6:9] = True
    assert np.array_equal(decode_mask(item["segmentation"], 23, 13), expected)
    assert np.array_equal(decode_mask(item["editor_geometry"]["mask"], 2, 2), detection.mask)


@pytest.mark.parametrize(
    "change", [None, "status_only", "record_missing", "record_changed", "record_mismatch", "image"]
)
def test_approval_requires_matching_record_and_image(annotation, tmp_path, change):
    sample, classes = annotation
    cfg = SimpleNamespace(storage_dir=tmp_path, classes=classes)
    image = tmp_path / "image.png"
    image.write_bytes(b"registered image")
    sample.filepath = str(image)
    sample["managed_sha256"] = sha256_file(image)
    content = annotation_content(sample, classes)
    digest = content_hash(content)
    sample.update(
        review_status="completed", review_approved_hash=digest, review_approval_id="c" * 32
    )
    path = record_path(cfg, sample, sample["review_approval_id"])
    record = dict(action="complete", label_hash=digest, annotation=content, reviewer="test")
    write_json(path, record)
    sample["review_approval_sha256"] = sha256_file(path)
    if change == "status_only":
        sample["review_approved_hash"] = None
    elif change == "record_missing":
        path.unlink()
    elif change == "record_changed":
        path.write_text("{}")
    elif change == "record_mismatch":
        record["annotation"]["instances"] = []
        write_json(path, record)
        sample["review_approval_sha256"] = sha256_file(path)
    elif change == "image":
        image.write_bytes(b"different image")
    if change is None:
        assert approved_annotation(cfg, sample) == content
    else:
        with pytest.raises((ValueError, OSError)):
            approved_annotation(cfg, sample)
