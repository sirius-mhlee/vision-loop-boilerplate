import math
import sqlite3
from contextlib import closing
from dataclasses import replace

import pytest

from vloop.cli import parse_args
from vloop.progress import Progress
from vloop.review import change_reviews
from vloop.review_batch import policy_decision, review_batch, sampling_key
from vloop.review_store import connect_records, read_record, save_record


@pytest.mark.parametrize(
    "scores, expected",
    [
        ([], "empty"),
        ([0.95], "accept"),
        ([0.95, 0.7], "low_confidence"),
        ([None], "low_confidence"),
        ([math.nan], "low_confidence"),
        ([1.1], "low_confidence"),
    ],
)
def test_policy_reserves_empty_and_uncertain_predictions(scores, expected):
    assert policy_decision("a" * 64, scores, minimum=0.9, sample_rate=0, seed=42) == expected


def test_sampling_is_repeatable_and_does_not_depend_on_batch_order():
    def choose(image_id, seed=42):
        return policy_decision(image_id, [0.99], minimum=0.9, sample_rate=0.01, seed=seed)

    forward = {str(i): choose(str(i)) for i in range(10000)}
    backward = {str(i): choose(str(i)) for i in reversed(range(10000))}
    assert forward == backward
    assert 50 < list(forward.values()).count("sample") < 150
    assert any(choose(str(i), seed=43) != forward[str(i)] for i in range(10000))
    assert policy_decision("x", [1.0], minimum=0.9, sample_rate=1, seed=42) == "sample"


def test_sampling_precedes_quality_checks_for_the_entire_candidate_set():
    selected = []
    for i in range(1000):
        decisions = [
            policy_decision(str(i), scores, minimum=0.9, sample_rate=0.2, seed=42)
            for scores in ([0.99], [0.2], [])
        ]
        assert decisions in (["sample"] * 3, ["accept", "low_confidence", "empty"])
        if decisions[0] == "sample":
            selected.append(i)
    assert 150 < len(selected) < 250
    for scores in ([], [0.1], [None]):
        assert policy_decision("x", scores, minimum=0.9, sample_rate=1, seed=42) == "sample"


def test_sampling_uses_the_configured_scene_group(project):
    first = {"image_id": "a", "scene": "shared"}
    second = {"image_id": "b", "scene": "shared"}
    assert sampling_key(project, first) != sampling_key(project, second)
    grouped = replace(project, release_group_field="scene")
    assert sampling_key(grouped, first) == sampling_key(grouped, second) == "group:shared"
    with pytest.raises(ValueError, match="group field"):
        sampling_key(grouped, {"image_id": "a", "scene": ""})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"apply": True},
        {"minimum": 0.9},
        {"batch_size": 0},
        {"resume": "../../escape"},
        {"resume": "review_batch_20260906T000000_12345678", "minimum": 0.8},
    ],
)
def test_batch_requires_explicit_frozen_policy(project, kwargs):
    with pytest.raises(ValueError):
        review_batch(project, **kwargs)


def test_batch_cli_and_queue_options():
    args = parse_args(["review-batch", "--resume", "job", "--apply", "--batch-size", "10"])
    assert args.resume == "job" and args.apply and args.batch_size == 10
    args = parse_args(["review", "--queue", "sample", "--limit", "20"])
    assert args.queue == "sample" and args.limit == 20


def test_compact_records_are_immutable_and_missing_stores_fail_closed(project):
    with pytest.raises(ValueError, match="cannot be read"):
        read_record(project, "missing")
    connection = connect_records(project)
    try:
        record = {"record_id": "x", "action": "auto_accept", "value": [1, 2]}
        digest = save_record(connection, record)
        assert len(digest) == 64
        assert read_record(project, "x") == record
        with pytest.raises(sqlite3.IntegrityError):
            save_record(connection, record)
        with pytest.raises(ValueError, match="missing"):
            read_record(project, "not-found")
    finally:
        connection.close()


def test_manual_review_cannot_materialize_a_million_selected_ids(project):
    consumed = 0

    def ids():
        nonlocal consumed
        for value in range(1_000_000):
            consumed += 1
            yield str(value)

    with pytest.raises(ValueError, match="100 images"):
        change_reviews(project, ids(), "complete", "test")
    assert consumed == 101


def test_batch_progress_resumes_preview_and_counts_failed_or_skipped_apply(
    project, tmp_path, monkeypatch, progress_bars
):
    import vloop.review_batch as module

    with closing(sqlite3.connect(":memory:")) as db:
        db.row_factory = sqlite3.Row
        db.executescript("""
            CREATE TABLE inputs (sample_id TEXT PRIMARY KEY, decision TEXT, detail TEXT,
                                 applied INTEGER DEFAULT 0, outcome TEXT);
            INSERT INTO inputs (sample_id) VALUES ('first'), ('second');
        """)
        report = {"job_id": "progress-test"}

        def inspect(cfg, dataset, row, policy):
            if row["sample_id"] == "second":
                raise KeyboardInterrupt
            return None, None, None, "accept", ""

        monkeypatch.setattr(module, "_inspect", inspect)

        def process(apply):
            with Progress("batch") as progress:
                module._process(project, None, db, None, tmp_path, report, {}, 1, apply, progress)

        with pytest.raises(KeyboardInterrupt):
            process(False)
        assert (progress_bars[-1].n, progress_bars[-1].total) == (1, 2)
        monkeypatch.setattr(module, "_inspect", lambda *args: (None, None, None, "error", "bad"))
        process(False)
        bar = progress_bars[-1]
        assert (bar.initial, bar.n, bar.total) == (1, 2, 2)
        assert "errors=1" in bar.postfix

        def fail(*args):
            raise RuntimeError("Database update failed")

        monkeypatch.setattr(module, "_apply_one", fail)
        process(True)
        assert (progress_bars[-1].n, progress_bars[-1].total) == (2, 2)
        assert "failed=1" in progress_bars[-1].postfix
        assert "skipped=1" in progress_bars[-1].postfix
        process(True)  # Already attempted entries are not applied twice.
        assert progress_bars[-1].initial == progress_bars[-1].n == 2
        assert "failed=1" in progress_bars[-1].postfix
