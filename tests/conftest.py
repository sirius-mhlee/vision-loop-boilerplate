import io

import pytest
import yaml

from vloop.config import load_config


@pytest.fixture
def progress_bars(monkeypatch):
    """Record real terminal bars, including their final counts after close."""
    import vloop.progress as module

    class TerminalBuffer(io.StringIO):
        def isatty(self):
            return True

    output = TerminalBuffer()
    bars = []
    original = module.tqdm

    def create(*args, **kwargs):
        kwargs["file"] = output
        bar = original(*args, **kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(module, "tqdm", create)
    return bars


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
