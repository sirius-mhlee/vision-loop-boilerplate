import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("VLOOP_TEST_ITERATION") != "1",
    reason="Requires GPU, RF-DETR weights and local FiftyOne/DVC services",
)
def test_two_versions_fixed_validation_and_preserved_history(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("iteration_integration_runner.py")),
            "--output",
            str(tmp_path / "iteration.json"),
        ],
        capture_output=True,
        text=True,
        timeout=1500,
    )
    assert result.returncode == 0, result.stdout + result.stderr
