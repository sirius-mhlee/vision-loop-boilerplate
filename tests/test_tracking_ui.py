import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import replace

import pytest
import yaml


@pytest.mark.skipif(
    os.environ.get("VLOOP_TEST_MLFLOW") != "1", reason="Starts a localhost MLflow server"
)
def test_experiments_serves_training_runs_and_artifacts(project):
    from vloop.tracking import client_for, create_run

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    cfg = replace(project, mlflow_port=port)
    data = cfg.to_dict()
    data.pop("config_path")
    cfg.config_path.write_text(yaml.safe_dump(data))
    client, run = create_run(cfg, "ui-test", notes="UI integration")
    artifact = cfg.storage_dir / "sample.txt"
    artifact.write_text("training artifact")
    client.log_artifact(run.info.run_id, str(artifact))
    client.set_terminated(run.info.run_id)
    log = cfg.storage_dir / "ui.log"
    with log.open("w") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "vloop",
                "experiments",
                "--config",
                str(cfg.config_path),
                "--no-browser",
            ],
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 45
            while True:
                assert process.poll() is None, log.read_text()
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1) as response:
                        assert response.status == 200
                    break
                except (OSError, urllib.error.URLError):
                    assert time.monotonic() < deadline, log.read_text()
                    time.sleep(0.2)
            with urllib.request.urlopen(url, timeout=5) as response:
                assert b"<html" in response.read().lower()
            with urllib.request.urlopen(
                url + "/api/2.0/mlflow/runs/get?run_id=" + run.info.run_id,
                timeout=5,
            ) as response:
                content = json.load(response)
                assert content["run"]["info"]["status"] == "FINISHED"
            with urllib.request.urlopen(
                url + "/api/2.0/mlflow/artifacts/list?run_id=" + run.info.run_id,
                timeout=5,
            ) as response:
                assert json.load(response)["files"][0]["path"] == "sample.txt"
        finally:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=20)
    assert process.returncode == 130
    assert client_for(cfg).get_run(run.info.run_id).info.status == "FINISHED"
