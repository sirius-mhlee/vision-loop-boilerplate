import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("VLOOP_TEST_FIFTYONE") != "1", reason="Requires local FiftyOne server"
)
def test_box_mask_metrics_and_artifact_rebuild(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("evaluate_integration_runner.py")),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
