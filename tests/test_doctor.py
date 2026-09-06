import json
from pathlib import Path

from vloop.cli import main
from vloop.doctor import doctor


def test_doctor_records_missing_requirements_and_cuda_failure(project, monkeypatch):
    import vloop.doctor as module

    def fail_cuda(device):
        raise RuntimeError("CUDA device unavailable")

    monkeypatch.setattr(module, "check_cuda", fail_cuda)
    monkeypatch.setattr(module, "version", lambda name: "1.0")
    report = doctor(project)
    checks = {item["name"]: item for item in report["checks"]}
    assert report["status"] == "failed"
    assert checks["cuda"]["status"] == "failed"
    assert checks["sam3_checkpoint"]["status"] == "failed"
    assert checks["dependency:fiftyone"]["status"] == "failed"
    assert report["sam3_inference"] == "not_requested"
    assert json.loads((Path(report["result_dir"]) / "report.json").read_text()) == report
    assert main(["doctor", "--config", str(project.config_path)]) == 1
