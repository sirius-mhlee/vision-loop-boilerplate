"""Stream immutable evaluation inputs and original-image predictions through SQLite."""

import hashlib
import json
import math
from contextlib import closing
from importlib.metadata import version

import numpy as np

from .approval import content_hash
from .labels import encode_mask
from .release_data import connect, image_relative


def settings(metadata, *, confidence=None, display_confidence=None, max_detections=None):
    saved = metadata["postprocessing"]
    confidence = saved["eval_confidence"] if confidence is None else confidence
    display = saved["display_confidence"] if display_confidence is None else display_confidence
    maximum = saved["eval_max_detections"] if max_detections is None else max_detections
    if not all(math.isfinite(n) and 0 <= n <= 1 for n in (confidence, display)):
        raise ValueError("Evaluation and display confidence must be between 0 and 1")
    if display < confidence:
        raise ValueError("Display confidence cannot be lower than inference confidence")
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or not 1 <= maximum <= saved["num_select"]
    ):
        raise ValueError(f"max-detections must be 1..{saved['num_select']} (saved model top-k)")
    return {
        "confidence": confidence,
        "display_confidence": display,
        "max_detections": maximum,
        "method": "fiftyone_coco",
        "fiftyone_version": version("fiftyone"),
        "iou_thresholds": [0.5 + 0.05 * i for i in range(10)],
        "analysis_iou": 0.5,
        "recall_points": 101,
        "classwise": True,
        "mask_tolerance": None,
        "prediction_format": "original_xywh_dense_mask_v1",
    }


def freeze_inputs(snapshot, destination, *, split, classes, limit=None):
    if limit is not None and (isinstance(limit, bool) or limit < 1):
        raise ValueError("limit must be positive")
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256()
    total = 0
    with closing(connect(snapshot)) as source, closing(connect(destination)) as db:
        db.execute("""CREATE TABLE samples (
            image_id TEXT PRIMARY KEY, relative_path TEXT NOT NULL,
            annotation TEXT NOT NULL, prediction TEXT, error TEXT)""")
        for row in source.execute(
            "SELECT * FROM records WHERE split=? ORDER BY image_id LIMIT ?", (split, limit or -1)
        ):
            annotation = json.loads(row["annotation"])
            fingerprint.update((content_hash(annotation) + "\n").encode())
            db.execute(
                "INSERT INTO samples(image_id, relative_path, annotation) VALUES (?,?,?)",
                (row["image_id"], image_relative(row).as_posix(), row["annotation"]),
            )
            total += 1
            if total % 1000 == 0:
                db.commit()
        db.commit()
    if not total:
        raise ValueError(f"The selected {split} split has no images")
    return {
        "split": split,
        "images": total,
        "content_sha256": fingerprint.hexdigest(),
        "classes": classes,
    }


def comparison_id(selection, parameters):
    # Display filtering, source model/run, dataset version and local paths do not affect scores.
    metric_settings = {k: v for k, v in parameters.items() if k != "display_confidence"}
    return content_hash({"selection": selection, "settings": metric_settings})


def serialize_prediction(output, *, width, height, classes, parameters):
    """Keep floating model boxes and full masks independent; never infer one from the other."""
    boxes = np.asarray(output.xyxy)
    scores = np.asarray(output.confidence)
    indices = np.asarray(output.class_id)
    count = len(boxes)
    if boxes.shape != (count, 4) or scores.shape != (count,) or indices.shape != (count,):
        raise ValueError("Invalid prediction array shapes")
    if not (np.isfinite(boxes).all() and np.isfinite(scores).all()):
        raise ValueError("Non-finite prediction coordinates or confidence")
    if np.any(boxes[:, 2:] < boxes[:, :2]) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Invalid prediction box or confidence")
    if count and (
        indices.dtype.kind not in "iu" or np.any((indices < 0) | (indices >= len(classes)))
    ):
        raise ValueError("Predicted class is outside the saved model mapping")
    masks = np.asarray(output.mask) if output.mask is not None else None
    if count and (
        masks is None or masks.shape != (count, height, width) or masks.dtype != np.bool_
    ):
        raise ValueError("Predictions must contain boolean masks in original-image coordinates")
    order = np.argsort(-scores, kind="stable")
    order = [i for i in order if scores[i] > parameters["confidence"]][
        : parameters["max_detections"]
    ]
    instances = []
    for i in order:
        x1, y1, x2, y2 = map(float, boxes[i])
        cls = classes[int(indices[i])]
        instances.append(
            {
                "class_id": cls["id"],
                "class_name": cls["name"],
                "confidence": float(scores[i]),
                "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                "segmentation": encode_mask(masks[i]),
            }
        )
    return {"width": width, "height": height, "instances": instances}
