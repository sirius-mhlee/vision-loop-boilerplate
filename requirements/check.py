"""Check the installed full environment, including dependencies activated by extras."""

import platform
import re
import sys
import tomllib
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
LOCAL = {"sam3", "vision-loop-boilerplate"}


def read_pins(paths):
    pins = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line or line.startswith(("#", "--index-url ", "--extra-index-url ")):
                continue
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
            if not match:
                raise ValueError(f"Expected an exact version pin: {path}: {line}")
            name, version = canonicalize_name(match[1]), match[2]
            if name in pins and pins[name] != version:
                raise ValueError(f"Conflicting pins for {name}")
            pins[name] = version
    return pins


def check_requirements(requirements, pins, distribution=metadata.distribution):
    """Follow active extras as well as ordinary dependencies; pip check omits extras."""
    pending = [Requirement(item) for item in requirements]
    seen, errors = set(), set()
    while pending:
        req = pending.pop()
        if req.marker and not req.marker.evaluate({"extra": ""}):
            continue
        name = canonicalize_name(req.name)
        if name not in LOCAL and name not in pins:
            errors.add(f"Missing lock entry: {name}")
        try:
            installed = distribution(name)
        except metadata.PackageNotFoundError:
            errors.add(f"Not installed: {name}")
            continue
        if not req.specifier.contains(installed.version, prereleases=True):
            errors.add(f"{req} is incompatible with installed {installed.version}")
        for extra in req.extras | {""}:
            key = name, extra
            if key in seen:
                continue
            seen.add(key)
            for item in installed.requires or ():
                dependency = Requirement(item)
                if not dependency.marker or dependency.marker.evaluate({"extra": extra}):
                    dependency.marker = None
                    pending.append(dependency)
    return sorted(errors)


def main():
    if (
        sys.version_info[:2] != (3, 12)
        or platform.python_implementation() != "CPython"
        or (platform.system(), platform.machine()) != ("Linux", "x86_64")
    ):
        raise SystemExit("This lock targets Linux x86_64 and CPython 3.12")
    pins = read_pins(
        [ROOT / "requirements/bootstrap.lock", ROOT / "requirements/linux-py312-cu128.lock"]
    )
    errors = []
    for name, expected in sorted(pins.items()):
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            errors.append(f"Not installed: {name}=={expected}")
            continue
        if actual != expected:
            errors.append(f"Version differs: {name} expected {expected}, installed {actual}")
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = config["project"]
    requirements = [*project["dependencies"], *config["build-system"]["requires"]]
    for extra in ("dev", "autolabel", "pipeline"):
        requirements.extend(project["optional-dependencies"][extra])
    requirements.extend(
        ["sam3==0.1.0", f"{project['name']}=={project['version']}", "torch", "torchvision"]
    )
    errors.extend(check_requirements(requirements, pins))
    if errors:
        raise SystemExit("\n".join(sorted(set(errors))))
    print(f"Dependency lock verified: {len(pins)} external packages and 2 local packages")
    print("SAM 3 source commit and GPU/checkpoint checks: vloop doctor")


if __name__ == "__main__":
    main()
