import json

import pytest
import yaml

from vloop.cli import main, parse_args
from vloop.config import load_config


def test_relative_paths_and_explicit_class_mapping(project, monkeypatch, tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    cfg = load_config(project.config_path)
    assert cfg.image_dir == project.config_path.parent / "input"
    assert cfg.class_to_index == {7: 0}
    assert cfg.prompt_to_class["test object"].id == 7
    assert json.loads(json.dumps(cfg.to_dict()))["classes"][0]["id"] == 7


def test_find_config_from_child(project, monkeypatch):
    monkeypatch.chdir(project.image_dir)
    assert load_config().config_path == project.config_path


@pytest.mark.parametrize(
    "patch",
    [
        {"batch_szie": 2},
        {"seed": True},
        {"batch_size": "1"},
        {"epochs": 0},
        {"learning_rate": float("nan")},
        {"learning_rate": 0},
        {"autolabel_confidence": 1.1},
        {"autolabel_batch_size": 2},
        {"split_ratios": [0.9, 0.1, 0.1]},
        {"split_ratios": [float("inf"), 0.1, 0.1]},
        {"classes": [{"id": True, "name": "test", "prompts": ["object"]}]},
        {"classes": [{"id": 1, "name": "test", "prompts": "object"}]},
        {"classes": [{"id": 1, "name": "test", "prompts": ["object", "object"]}]},
        {
            "classes": [
                {"id": 1, "name": "a", "prompts": ["object"]},
                {"id": 2, "name": "b", "prompts": ["object"]},
            ]
        },
        {"storage_dir": "input/state"},
        {"storage_dir": "."},
        {"storage_dir": None},
        {"device": "gpu"},
        {"sam3_commit": "main"},
        {"fiftyone_port": 70000},
        {"fiftyone_port": 5000},
        {"eval_split": "train"},
    ],
)
def test_invalid_config_is_rejected(project, patch):
    data = yaml.safe_load(project.config_path.read_text())
    data.update(patch)
    project.config_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_config(project.config_path)


def test_examples_never_supply_real_classes(tmp_path):
    path = tmp_path / "project.yaml"
    path.write_text("image_dir: null\nclasses: []\n")
    cfg = load_config(path)
    assert len(cfg.input_errors()) == 2
    assert main(["ingest", "--config", str(path), "--local-only"]) == 2
    assert not cfg.storage_dir.exists()


def test_common_config_option_works_in_both_positions():
    for args in (["--config", "chosen.yaml", "doctor"], ["doctor", "--config", "chosen.yaml"]):
        assert parse_args(args).config == "chosen.yaml"


def test_malformed_yaml_is_a_cli_error(tmp_path, capsys):
    path = tmp_path / "project.yaml"
    path.write_text("image_dir: [unclosed")
    assert main(["ingest", "--config", str(path)]) == 2
    assert "vloop:" in capsys.readouterr().err
