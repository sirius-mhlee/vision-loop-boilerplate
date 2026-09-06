import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("VLOOP_TEST_FIFTYONE") != "1", reason="Requires the real local FiftyOne DB"
)
def test_review_lifecycle_in_real_database(project):
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("review_integration_runner.py")),
            str(project.config_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)
