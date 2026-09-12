import runpy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

reference_instances = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "examples/pennfudan_toy.py")
)["reference_instances"]


def test_reference_instance_ids_preserve_holes_and_disconnected_pixels(tmp_path):
    labels = np.zeros((13, 23), dtype=np.uint8)
    labels[1:9, 2:10] = 1
    labels[3:6, 4:7] = 0
    labels[11, 21] = 1
    labels[0:2, 19:23] = 7
    path = tmp_path / "mask.png"
    Image.fromarray(labels).save(path)

    shape, instances = reference_instances(path)
    assert shape == (13, 23)
    assert len(instances) == 2
    restored = np.zeros_like(labels)
    for value, (x, y, w, h, mask) in zip((1, 7), instances, strict=True):
        assert mask.dtype == bool
        restored[y : y + h, x : x + w][mask] = value
    np.testing.assert_array_equal(restored, labels)


def test_reference_rejects_rgb_mask_instead_of_treating_channels_as_instances(tmp_path):
    path = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="single-channel"):
        reference_instances(path)
