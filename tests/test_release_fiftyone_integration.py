import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("VLOOP_TEST_FIFTYONE") != "1" or os.environ.get("VLOOP_TEST_RELEASE") != "1",
    reason="Requires FiftyOne DB and DVC",
)
def test_approved_fiftyone_release(project):
    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("release_integration_runner.py")),
            str(project.config_path),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    print(process.stdout)
