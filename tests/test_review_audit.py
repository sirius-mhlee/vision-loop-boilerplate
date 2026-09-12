import json
from unittest.mock import Mock

import pytest

from vloop import review_audit as module


@pytest.fixture
def collection(project, monkeypatch):
    pytest.importorskip("bson")
    dataset = Mock()
    monkeypatch.setattr(module, "load_review_dataset", lambda cfg: dataset)
    return dataset._sample_collection


def test_uninitialized_audit_requires_a_new_job(project, collection):
    _, report = module.start_run(project, "review_audit")
    with pytest.raises(ValueError, match="start a new review-audit"):
        module.review_audit(project, resume=report["job_id"])


def test_audit_saves_bounds_before_first_batch_and_resumes(project, collection):
    from bson import ObjectId

    upper = ObjectId()
    collection.find_one.return_value = {"_id": upper}

    def interrupt(query, projection):
        path = next((project.storage_dir / "runs").glob("review_audit_*/report.json"))
        saved = json.loads(path.read_text())
        assert saved["upper"] == str(upper)
        assert saved["after"] is None and saved["checked"] == 0
        assert "finished_at" not in saved
        raise KeyboardInterrupt

    collection.find.side_effect = interrupt
    report = module.review_audit(project)
    assert report["status"] == "interrupted", report
    assert str(project.config_path) in report["retry"]
    collection.find.side_effect = None
    collection.find.return_value.sort.return_value.limit.return_value = []
    resumed = module.review_audit(project, resume=report["job_id"])
    assert resumed["status"] == "completed", resumed
    assert resumed["upper"] == str(upper)
    assert collection.find_one.call_count == 1


def test_audit_initialization_failure_records_a_non_resume_retry(project, collection):
    collection.find_one.side_effect = RuntimeError("Database unavailable")
    report = module.review_audit(project)
    assert report["status"] == "failed", report
    assert report["error"] == "Database unavailable"
    assert "--resume" not in report["retry"]
