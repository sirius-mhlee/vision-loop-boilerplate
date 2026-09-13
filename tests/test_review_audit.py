import json
from dataclasses import replace
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


def test_audit_progress_resumes_checked_count(project, collection, monkeypatch, progress_bars):
    from bson import ObjectId

    first, second = ObjectId(), ObjectId()

    class Sample(dict):
        def reload(self):
            pass

    class Dataset(dict):
        _sample_collection = collection

    dataset = Dataset({str(key): Sample(review_status="completed") for key in (first, second)})
    monkeypatch.setattr(module, "load_review_dataset", lambda cfg: dataset)
    monkeypatch.setattr(module, "approved_annotation", lambda *args, **kwargs: None)
    collection.find_one.return_value = {"_id": second}
    page = Mock()
    page.sort.return_value.limit.return_value = [{"_id": first}]
    collection.find.side_effect = [page, KeyboardInterrupt()]
    cfg = replace(project, review_audit_batch_size=1)
    report = module.review_audit(cfg)
    assert report["status"] == "interrupted"
    assert (progress_bars[-1].n, progress_bars[-1].total) == (1, None)
    next_page, end = Mock(), Mock()
    next_page.sort.return_value.limit.return_value = [{"_id": second}]
    end.sort.return_value.limit.return_value = []
    collection.find.side_effect = [next_page, end]
    resumed = module.review_audit(cfg, resume=report["job_id"])
    assert resumed["status"] == "completed"
    assert (progress_bars[-1].initial, progress_bars[-1].n) == (1, 2)
