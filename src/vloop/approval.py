"""Canonical annotations and approval checks shared by review and future releases."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .labels import encode_mask, project_review_mask
from .runtime import sha256_file


def content_hash(value: dict) -> str:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(data.encode()).hexdigest()


def annotation_content(sample, classes) -> dict:
    truth = sample["ground_truth"]
    if truth is None:
        raise ValueError("Ground truth is missing; start review before approving an empty image")
    width, height = sample.metadata.width, sample.metadata.height
    if not width or not height or width < 1 or height < 1:
        raise ValueError("Registered image dimensions are missing")
    mapping = {item.name: item.id for item in classes}
    annotations = []
    for detection in truth.detections:
        if detection.label not in mapping:
            raise ValueError(f"Unknown ground truth class: {detection.label}")
        if detection.mask is None or detection.mask_path:
            raise ValueError("Each instance needs an embedded mask; draw a mask before approving")
        crop = np.asarray(detection.mask)
        box, mask = project_review_mask(detection.bounding_box, crop, width, height)
        # The edited class name is authoritative; copied prediction IDs may be stale.
        annotations.append(
            {
                "class_id": mapping[detection.label],
                "class_name": detection.label,
                "bbox_xywh": box,
                "segmentation": encode_mask(mask),
                "area": int(mask.sum()),
                "editor_geometry": {
                    "bounding_box": list(detection.bounding_box),
                    "mask": encode_mask(crop.astype(bool)),
                    "projection": "round_edges_nearest_v1",
                },
            }
        )
    annotations.sort(key=lambda item: json.dumps(item, sort_keys=True))
    return {
        "schema_version": 1,
        "image_id": sample["image_id"],
        "managed_sha256": sample["managed_sha256"],
        "width": width,
        "height": height,
        "classes": sorted(mapping.items()),
        "instances": annotations,
    }


def record_path(cfg, sample, record_id: str) -> Path:
    image_id = sample["image_id"]
    if not re.fullmatch(r"[0-9a-f]{64}", image_id or "") or not re.fullmatch(
        r"[0-9a-f]{32}", record_id or ""
    ):
        raise ValueError("Invalid review record identifier")
    return cfg.storage_dir / "reviews" / image_id / f"{record_id}.json"


def approved_annotation(cfg, sample, *, verify_image: bool = True) -> dict:
    """Fail closed: a status string alone never makes an annotation releasable."""
    if sample["review_status"] != "completed":
        raise ValueError("Image is not approved")
    content = annotation_content(sample, cfg.classes)
    digest = content_hash(content)
    if digest != sample["review_approved_hash"]:
        raise ValueError("Ground truth changed after approval")
    path = record_path(cfg, sample, sample["review_approval_id"])
    if sha256_file(path) != sample["review_approval_sha256"]:
        raise ValueError("Approval record changed after approval")
    record = json.loads(path.read_text())
    if (
        record.get("action") != "complete"
        or record.get("label_hash") != digest
        or content_hash(record.get("annotation", {})) != digest
        or not record.get("reviewer")
    ):
        raise ValueError("Approval record does not match the current annotation")
    if verify_image and sha256_file(Path(sample.filepath)) != content["managed_sha256"]:
        raise ValueError("Registered image content changed after approval")
    return content
