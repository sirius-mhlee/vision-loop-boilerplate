import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("VLOOP_TEST_FIFTYONE") != "1",
    reason="Set VLOOP_TEST_FIFTYONE=1 to start the real local FiftyOne database",
)
def test_real_database_registration_preserves_labels_across_processes(project):
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("fiftyone_integration_runner.py")),
            str(project.config_path),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
