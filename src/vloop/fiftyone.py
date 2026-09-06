import os
import sqlite3
import sys
from pathlib import Path

from .config import Config
from .runtime import sha256_file, write_json


def configure_fiftyone(cfg: Config):
    database_dir = cfg.storage_dir / "fiftyone" / "db"
    if os.environ.get("FIFTYONE_DATABASE_URI"):
        raise ValueError("Unset FIFTYONE_DATABASE_URI to use the project-local database")
    existing = sys.modules.get("fiftyone")
    if existing is not None and (
        existing.config.database_uri or Path(existing.config.database_dir).resolve() != database_dir
    ):
        raise RuntimeError("FiftyOne already uses another database; start a new vloop process")
    database_dir.mkdir(parents=True, exist_ok=True)
    config_path = database_dir.parent / "config.json"
    write_json(config_path, {"database_uri": None, "database_dir": str(database_dir)})
    os.environ["FIFTYONE_CONFIG_PATH"] = str(config_path)
    os.environ["FIFTYONE_DATABASE_DIR"] = str(database_dir)
    os.environ["FIFTYONE_DEFAULT_APP_ADDRESS"] = "127.0.0.1"
    os.environ["FIFTYONE_DEFAULT_APP_PORT"] = str(cfg.fiftyone_port)
    os.environ["FIFTYONE_MODEL_ZOO_DIR"] = str(cfg.storage_dir / "models")
    import fiftyone as fo

    fo.config.database_dir = str(database_dir)
    fo.config.default_app_address = "127.0.0.1"
    fo.config.default_app_port = cfg.fiftyone_port
    fo.config.model_zoo_dir = str(cfg.storage_dir / "models")
    return fo


def sync_catalog(cfg: Config, connection: sqlite3.Connection) -> int:
    fo = configure_fiftyone(cfg)
    dataset = (
        fo.load_dataset(cfg.dataset_name)
        if fo.dataset_exists(cfg.dataset_name)
        else fo.Dataset(cfg.dataset_name, persistent=True)
    )
    mapping = [{"id": item.id, "name": item.name} for item in cfg.classes]
    owner = str(cfg.storage_dir)
    if dataset.info.get("vloop_storage_dir") not in (None, owner):
        raise ValueError("FiftyOne dataset belongs to another vloop project")
    if len(dataset) and dataset.info.get("vloop_storage_dir") is None:
        raise ValueError("Refusing to reuse an existing dataset without vloop provenance")
    if dataset.info.get("vloop_classes", mapping) != mapping:
        raise ValueError("Class mapping changed; existing dataset requires an explicit migration")
    dataset.info.update(vloop_storage_dir=owner, vloop_classes=mapping)
    if not dataset.has_sample_field("image_id"):
        dataset.add_sample_field("image_id", fo.StringField)
    dataset.create_index("image_id", unique=True)
    dataset.save()
    synced = 0
    for row in connection.execute("SELECT * FROM images ORDER BY image_id"):
        filepath = Path(row["filepath"])
        if sha256_file(filepath) != row["managed_sha256"]:
            raise ValueError(f"Managed image integrity check failed: {filepath}")
        sources = [
            item[0]
            for item in connection.execute(
                "SELECT source_path FROM sources WHERE image_id = ? ORDER BY source_path",
                (row["image_id"],),
            )
        ]
        view = dataset.match(fo.ViewField("image_id") == row["image_id"])
        if len(view):
            sample = view.first()
            sample["source_paths"] = sources
            sample.save()
        else:
            sample = fo.Sample(
                filepath=str(filepath),
                image_id=row["image_id"],
                source_paths=sources,
                review_status="unreviewed",
                managed_sha256=row["managed_sha256"],
                metadata=fo.ImageMetadata(width=row["width"], height=row["height"], num_channels=3),
            )
            dataset.add_sample(sample)
        synced += 1
    return synced
