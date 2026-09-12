import runpy
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace

import pytest

CHECK = runpy.run_path(str(Path(__file__).resolve().parents[1] / "requirements/check.py"))


def test_missing_nested_extra_dependency_is_reported():
    packages = {
        "model": SimpleNamespace(version="1.0", requires=['metrics[detection]; extra == "train"']),
        "metrics": SimpleNamespace(version="2.0", requires=['masks>=3; extra == "detection"']),
    }

    def distribution(name):
        if name not in packages:
            raise PackageNotFoundError(name)
        return packages[name]

    assert CHECK["check_requirements"](["model"], {"model": "1.0"}, distribution) == []
    errors = CHECK["check_requirements"](
        ["model[train]"], {"model": "1.0", "metrics": "2.0"}, distribution
    )
    assert errors == ["Missing lock entry: masks", "Not installed: masks"]


def test_changed_dependency_requirement_is_reported():
    package = SimpleNamespace(version="1.0", requires=[])
    errors = CHECK["check_requirements"](["model>=2"], {"model": "1.0"}, lambda name: package)
    assert errors == ["model>=2 is incompatible with installed 1.0"]


def test_lock_rejects_paths_ranges_and_conflicting_build_pins(tmp_path):
    runtime, build = tmp_path / "runtime.lock", tmp_path / "build.lock"
    runtime.write_text("setuptools==78.1.0\n")
    build.write_text("setuptools==80.0.0\n")
    with pytest.raises(ValueError, match="Conflicting pins"):
        CHECK["read_pins"]([runtime, build])
    for invalid in ("-e /home/user/project", "numpy>=1.26"):
        runtime.write_text(invalid)
        with pytest.raises(ValueError, match="exact version pin"):
            CHECK["read_pins"]([runtime])
