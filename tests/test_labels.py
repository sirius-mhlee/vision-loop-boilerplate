from types import SimpleNamespace

import numpy as np
import pytest

from vloop.config import ClassConfig
from vloop.labels import decode_mask, encode_mask, from_detection, pixel_box


def test_non_square_mask_with_hole_and_disconnected_region_is_lossless():
    crop = np.zeros((7, 11), dtype=bool)
    crop[0:5, 0:5] = True
    crop[1:4, 1:4] = False
    crop[6, 10] = True
    detection = SimpleNamespace(
        bounding_box=[3 / 23, 2 / 13, 11 / 23, 7 / 13], mask=crop, confidence=0.8
    )
    item = from_detection(
        detection, ClassConfig(7, "object", ("test object",)), "test object", 23, 13
    )
    decoded = decode_mask(item["segmentation"], 23, 13)
    assert item["class_id"] == 7
    assert item["bbox_xywh"] == [3, 2, 11, 7]
    assert item["area"] == 17
    assert np.array_equal(decoded[2:9, 3:14], crop)
    assert decoded.sum() == crop.sum()
    assert not decoded[3:6, 4:7].any()


def test_rle_rejects_wrong_dimensions():
    rle = encode_mask(np.zeros((7, 13), dtype=bool))
    with pytest.raises(ValueError):
        decode_mask(rle, 7, 13)


@pytest.mark.parametrize(
    "box", [[-1, 0, 1, 1], [0, 0, 0, 1], [0, 0, 2, 1], [float("nan"), 0, 1, 1]]
)
def test_invalid_box_rejected(box):
    with pytest.raises(ValueError):
        pixel_box(box, 23, 13)


def test_mask_shape_mismatch_is_not_silently_resized():
    detection = SimpleNamespace(
        bounding_box=[0, 0, 1, 1], mask=np.ones((3, 3), dtype=bool), confidence=0.8
    )
    with pytest.raises(ValueError, match="match its pixel"):
        from_detection(detection, ClassConfig(0, "a", ("a",)), "a", 23, 13)
