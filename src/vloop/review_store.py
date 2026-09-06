"""Compact approval storage: avoid creating one file per automatically accepted image."""

import json
import sqlite3
from contextlib import closing

from .approval import content_hash


def connect_records(cfg, *, readonly=False):
    path = cfg.storage_dir / "reviews" / "records.sqlite3"
    if readonly:
        return sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS records "
        "(id TEXT PRIMARY KEY, content TEXT NOT NULL, sha256 TEXT NOT NULL)"
    )
    return connection


def save_record(connection, record):
    digest = content_hash(record)
    with connection:
        connection.execute(
            "INSERT INTO records VALUES (?, ?, ?)",
            (record["record_id"], json.dumps(record, ensure_ascii=False), digest),
        )
    return digest


def read_record(cfg, record_id):
    try:
        with closing(connect_records(cfg, readonly=True)) as connection:
            row = connection.execute(
                "SELECT content FROM records WHERE id = ?", (record_id,)
            ).fetchone()
    except sqlite3.Error as exc:
        raise ValueError("Approval record store cannot be read") from exc
    if row is None:
        raise ValueError("Approval record is missing")
    return json.loads(row[0])
