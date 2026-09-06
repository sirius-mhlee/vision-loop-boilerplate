import math
import sqlite3

import pytest

from vloop.cli import parse_args
from vloop.review import change_reviews
from vloop.review_batch import policy_decision, review_batch
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
