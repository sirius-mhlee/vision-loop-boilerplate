import math
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ClassConfig:
    id: int
    name: str
    prompts: tuple[str, ...]


@dataclass(frozen=True)
class Config:
    image_dir: Path | None = None
    classes: tuple[ClassConfig, ...] = ()
    dataset_name: str = "vision-loop"
    storage_dir: Path = Path(".vloop")
    device: str = "cuda:0"
    seed: int = 42

    sam3_model: str = "segment-anything-3-image-torch"
    sam3_checkpoint: Path | None = None
    sam3_source_dir: Path | None = None
    sam3_commit: str | None = None
    autolabel_batch_size: int = 1
    autolabel_confidence: float = 0.5
    sam3_precision: str = "bfloat16"

    fiftyone_port: int = 5151
    mlflow_port: int = 5000
    dvc_remote: Path = Path("../vision-loop-dvc-remote")

    train_model: str = "RFDETRSegNano"
    epochs: int = 30
    batch_size: int = 1
    grad_accum_steps: int = 8
    learning_rate: float = 1e-4

    split_ratios: tuple[float, ...] = (0.8, 0.1, 0.1)
    eval_split: str = "val"
    eval_confidence: float = 0.001
    display_confidence: float = 0.5
    eval_max_detections: int = 100
    config_path: Path = field(default=Path("project.yaml"), repr=False)

    @property
    def class_to_index(self) -> dict[int, int]:
        return {item.id: index for index, item in enumerate(self.classes)}

    @property
    def prompt_to_class(self) -> dict[str, ClassConfig]:
        return {prompt: item for item in self.classes for prompt in item.prompts}

    @property
    def mlflow_tracking_uri(self) -> str:
        return "sqlite:///" + str(self.storage_dir / "mlflow" / "mlflow.db")

    def to_dict(self) -> dict[str, Any]:
        def convert(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))

    def input_errors(self) -> list[str]:
        errors = []
        if self.image_dir is None:
            errors.append("Set image_dir in project.yaml")
        elif not self.image_dir.is_dir():
            errors.append(f"Image directory not found: {self.image_dir}")
        if not self.classes:
            errors.append("Set classes with id, name, and prompts in project.yaml")
        return errors


def find_config(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    current = Path.cwd()
    for directory in (current, *current.parents):
        candidate = directory / "project.yaml"
        if candidate.is_file():
            return candidate
        if (directory / ".git").exists():
            break
    raise FileNotFoundError("project.yaml not found; copy project.example.yaml to project.yaml")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _classes(value: Any) -> tuple[ClassConfig, ...]:
    if not isinstance(value, list):
        raise ValueError("classes must be a list")
    result, ids, names, prompts_seen = [], set(), set(), set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"id", "name", "prompts"}:
            raise ValueError("Each class requires exactly id, name, and prompts")
        class_id = item["id"]
        if type(class_id) is not int or class_id < 0 or class_id in ids:
            raise ValueError("Class IDs must be unique non-negative integers")
        name = _text(item["name"], "class name")
        if name in names:
            raise ValueError(f"Duplicate class name: {name}")
        if not isinstance(item["prompts"], list) or not item["prompts"]:
            raise ValueError(f"Class {name} requires a non-empty prompts list")
        prompts = tuple(_text(prompt, "prompt") for prompt in item["prompts"])
        for prompt in prompts:
            if prompt in prompts_seen:
                raise ValueError(f"Ambiguous or duplicate prompt: {prompt}")
            prompts_seen.add(prompt)
        ids.add(class_id)
        names.add(name)
        result.append(ClassConfig(class_id, name, prompts))
    return tuple(result)


def load_config(path: str | Path | None = None) -> Config:
    config_path = find_config(path)
    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return config_from_dict(data, config_path)


def config_from_dict(data: dict, config_path: Path) -> Config:
    """Validate both user YAML and frozen job configurations through the same path."""
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping")
    data = dict(data)
    allowed = {item.name for item in fields(Config)} - {"config_path"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(sorted(map(str, unknown)))}")
    data["classes"] = _classes(data.get("classes", []))
    defaults = Config()
    for key in ("image_dir", "storage_dir", "dvc_remote", "sam3_checkpoint", "sam3_source_dir"):
        value = data.get(key, getattr(defaults, key))
        if value is None and key in ("image_dir", "sam3_checkpoint", "sam3_source_dir"):
            data[key] = None
        else:
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"{key} must be a non-empty path")
            data[key] = (config_path.parent / Path(value).expanduser()).resolve()
    for key in ("dataset_name", "device", "sam3_model", "train_model", "eval_split"):
        data[key] = _text(data.get(key, getattr(defaults, key)), key)
    for key in (
        "seed",
        "epochs",
        "batch_size",
        "grad_accum_steps",
        "autolabel_batch_size",
        "fiftyone_port",
        "mlflow_port",
        "eval_max_detections",
    ):
        value = data.get(key, getattr(defaults, key))
        if type(value) is not int or value < (0 if key == "seed" else 1):
            raise ValueError(
                f"{key} must be a {'non-negative' if key == 'seed' else 'positive'} integer"
            )
    for key in ("learning_rate", "autolabel_confidence", "eval_confidence", "display_confidence"):
        value = data.get(key, getattr(defaults, key))
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        if (key == "learning_rate" and value <= 0) or (
            key != "learning_rate" and not 0 <= value <= 1
        ):
            raise ValueError(f"Invalid {key}: {value}")
    ratios = data.get("split_ratios", defaults.split_ratios)
    if (
        not isinstance(ratios, (list, tuple))
        or len(ratios) != 3
        or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in ratios)
        or not math.isclose(sum(ratios), 1.0)
    ):
        raise ValueError("split_ratios must contain three positive numbers summing to 1")
    data["split_ratios"] = tuple(ratios)
    cfg = Config(**data, config_path=config_path)
    if cfg.sam3_precision not in ("bfloat16", "float32"):
        raise ValueError("sam3_precision must be bfloat16 or float32")
    if cfg.eval_split not in ("val", "test"):
        raise ValueError("eval_split must be val or test")
    if cfg.sam3_model != defaults.sam3_model or cfg.train_model != defaults.train_model:
        raise ValueError("The first version supports SAM 3 image and RFDETRSegNano only")
    if cfg.autolabel_batch_size != 1:
        raise ValueError("autolabel_batch_size must be 1 for the initial validation")
    if cfg.device != "cpu" and not re.fullmatch(r"cuda(?::\d+)?", cfg.device):
        raise ValueError("device must be cpu, cuda, or cuda:<index>")
    if any(port > 65535 for port in (cfg.fiftyone_port, cfg.mlflow_port)):
        raise ValueError("Ports must be in the range 1..65535")
    if cfg.fiftyone_port == cfg.mlflow_port:
        raise ValueError("FiftyOne and MLflow must use different ports")
    if cfg.sam3_commit is not None and (
        not isinstance(cfg.sam3_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", cfg.sam3_commit)
    ):
        raise ValueError("sam3_commit must be a full 40-character Git commit")
    if cfg.image_dir is not None and (
        cfg.storage_dir.is_relative_to(cfg.image_dir)
        or cfg.image_dir.is_relative_to(cfg.storage_dir)
    ):
        raise ValueError("image_dir and storage_dir must not overlap")
    return cfg
