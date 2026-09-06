"""Lossless instance masks and the pixel/normalized coordinate boundary."""

import math

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask


def encode_mask(mask: np.ndarray) -> dict:
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("An instance mask must be a 2D boolean array")
    rle = coco_mask.encode(np.asfortranarray(mask, dtype=np.uint8))
    return {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")}


def decode_mask(rle: dict, width: int, height: int) -> np.ndarray:
    if rle.get("size") != [height, width] or not isinstance(rle.get("counts"), str):
        raise ValueError("RLE dimensions must match the registered image")
    return coco_mask.decode(
        {"size": [height, width], "counts": rle["counts"].encode("ascii")}
    ).astype(bool)


def pixel_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    values = np.asarray(box, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Bounding box must contain four finite values")
    x, y, w, h = values * [width, height, width, height]
    edges = np.array([x, y, x + w, y + h])
    rounded = np.rint(edges).astype(int)
    if not np.allclose(edges, rounded, atol=1e-4, rtol=0):
        raise ValueError("SAM 3 mask bounds must align to registered image pixels")
    x1, y1, x2, y2 = rounded.tolist()
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Bounding box lies outside the registered image")
    return x1, y1, x2 - x1, y2 - y1


def project_review_mask(box, crop: np.ndarray, width: int, height: int):
    """Project the native editor's viewport-sized mask onto registered image pixels."""
    values = np.asarray(box, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Ground truth box must contain four finite numbers")
    x, y, w, h = values.tolist()
    if not (0 <= x < x + w <= 1 + 1e-9 and 0 <= y < y + h <= 1 + 1e-9):
        raise ValueError("Ground truth box lies outside the registered image")
    if crop.ndim != 2 or not np.isin(crop, [0, 1]).all() or not crop.any():
        raise ValueError("Instance mask must be non-empty and binary")
    edges = np.rint(np.array([x, y, x + w, y + h]) * [width, height, width, height]).astype(int)
    x1, y1, x2, y2 = np.clip(edges, 0, [width, height, width, height]).tolist()
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Instance occupies less than one registered image pixel")
    rendered = crop.astype(bool)
    if rendered.shape != (y2 - y1, x2 - x1):
        rendered = np.asarray(
            Image.fromarray(crop.astype(np.uint8)).resize(
                (x2 - x1, y2 - y1), Image.Resampling.NEAREST
            )
        ).astype(bool)
    if not rendered.any():
        raise ValueError("Instance disappears at registered image resolution")
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = rendered
    return [x1, y1, x2 - x1, y2 - y1], mask


def from_detection(detection, class_config, prompt: str, width: int, height: int) -> dict:
    x, y, w, h = pixel_box(detection.bounding_box, width, height)
    crop = np.asarray(detection.mask)
    if crop.shape != (h, w) or crop.dtype != np.bool_ or not crop.any():
        raise ValueError("Detection mask must be non-empty and match its pixel bounding box")
    confidence = detection.confidence
    if confidence is None or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Detection confidence must be in [0, 1]")
    mask = np.zeros((height, width), dtype=bool)
    mask[y : y + h, x : x + w] = crop
    return {
        "class_id": class_config.id,
        "class_name": class_config.name,
        "prompt": prompt,
        "confidence": float(confidence),
        "bbox_xywh": [x, y, w, h],
        "segmentation": encode_mask(mask),
        "area": int(mask.sum()),
    }


def to_detections(prediction: dict):
    import fiftyone as fo

    width, height = prediction["width"], prediction["height"]
    detections = []
    for item in prediction["instances"]:
        x, y, w, h = item["bbox_xywh"]
        if any(type(value) is not int for value in (x, y, w, h)):
            raise ValueError("Pixel bounding boxes must be integers")
        pixel_box([x / width, y / height, w / width, h / height], width, height)
        mask = decode_mask(item["segmentation"], width, height)
        crop = mask[y : y + h, x : x + w]
        if int(crop.sum()) != int(mask.sum()) or int(mask.sum()) != item["area"]:
            raise ValueError("Mask area is inconsistent or lies outside its bounding box")
        detections.append(
            fo.Detection(
                label=item["class_name"],
                class_id=item["class_id"],
                prompt=item["prompt"],
                confidence=item["confidence"],
                bounding_box=[x / width, y / height, w / width, h / height],
                mask=crop,
            )
        )
    return fo.Detections(detections=detections)
