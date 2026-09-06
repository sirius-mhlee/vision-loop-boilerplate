import pytest
import yaml

from vloop.config import load_config


@pytest.fixture
def project(tmp_path):
    images = tmp_path / "input"
    images.mkdir()
    path = tmp_path / "project.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "image_dir": "input",
                "storage_dir": "state",
                "classes": [{"id": 7, "name": "test-object", "prompts": ["test object"]}],
            }
        ),
        encoding="utf-8",
    )
    return load_config(path)
