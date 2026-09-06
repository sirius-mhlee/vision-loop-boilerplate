import argparse
import sys
from pathlib import Path

import yaml

from . import __version__
from .config import load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local computer vision iteration pipeline")
    parser.add_argument("--version", action="version", version=f"vloop {__version__}")
    parser.add_argument("--config", help="Project YAML (defaults to nearest project.yaml)")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser(
        "doctor", help="Check configuration, storage, dependencies, and CUDA"
    )
    doctor.add_argument("--sam3-image", type=Path, help="Also run SAM 3 on one image")
    ingest = commands.add_parser("ingest", help="Register local images and sync to FiftyOne")
    ingest.add_argument(
        "--local-only", action="store_true", help="Register locally without FiftyOne"
    )
    for command in (doctor, ingest):
        command.add_argument("--config", default=argparse.SUPPRESS, help="Path to project YAML")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)
        if args.command == "doctor":
            from .doctor import doctor

            report = doctor(cfg, sam3_image=args.sam3_image)
        else:
            from .ingest import ingest

            report = ingest(cfg, local_only=args.local_only)
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as exc:
        print(f"vloop: {exc}", file=sys.stderr)
        return 2
    print(f"Job ID: {report['job_id']}")
    print(f"Status: {report['status']}")
    print(f"Dataset version: {report['dataset_version'] or 'unreleased'}")
    if args.command == "doctor":
        for check in report["checks"]:
            print(f"[{check['status']}] {check['name']}: {check['detail']}")
        print(f"Checks: {report['passed']} passed, {report['failed']} failed")
        print(f"SAM 3 inference: {report.get('sam3_inference', 'not_run')}")
    else:
        print(
            f"Images: {report['registered']} registered, {report['duplicate']} duplicates, "
            f"{report['repaired']} repaired, {report['failed']} failed"
        )
        print(f"FiftyOne sync: {report['sync_status']}")
    if report.get("error"):
        print(f"Error: {report['error']}", file=sys.stderr)
    if report["status"] != "completed" and "retry" in report:
        print(f"Retry: {report['retry']}")
    print(f"Results: {report['result_dir']}")
    return 0 if report["status"] == "completed" else 130 if report["status"] == "interrupted" else 1
