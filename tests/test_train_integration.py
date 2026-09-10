import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("VLOOP_TEST_TRAIN") != "1", reason="Requires GPU and RF-DETR weights"
)
def test_real_gpu_training_and_new_process_resume(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("train_integration_runner.py")),
            "--output",
            str(tmp_path / "training.json"),
        ],
        capture_output=True,
        text=True,
        timeout=1500,
    )
    assert result.returncode == 0, result.stdout + result.stderr
