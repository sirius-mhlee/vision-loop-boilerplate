import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("name", ["review", "release", "train_loader"])
def test_metadata_benchmarks_execute_with_current_schemas(tmp_path, name):
    if name == "train_loader":
        pytest.importorskip("rfdetr")
    script = Path(__file__).with_name(f"{name}_scale_runner.py")
    output = tmp_path / "result.json"
    arguments = [sys.executable, str(script), "--count", "100"]
    if name != "review":
        arguments.extend(["--output", str(output)])
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout if name == "review" else output.read_text())
    if name == "release":
        assert report["splits"] == {"train": 80, "val": 10, "test": 10}
    else:
        assert report["metadata_rows" if name == "review" else "images"] == 100
