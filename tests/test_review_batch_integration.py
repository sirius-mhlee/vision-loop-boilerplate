import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("VLOOP_TEST_FIFTYONE") != "1", reason="Requires the local FiftyOne DB"
)
def test_resumable_batch_and_incremental_audit(project):
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("review_batch_runner.py")),
            str(project.config_path),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)
