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
    autolabel = commands.add_parser(
        "autolabel", help="Run or resume sequential SAM 3 auto-labeling"
    )
    selection = autolabel.add_mutually_exclusive_group()
    selection.add_argument(
        "--resume", metavar="JOB_ID", help="Resume with the job's frozen configuration"
    )
    selection.add_argument(
        "--limit", type=int, help="Freeze only the first N registered images in a new job"
    )
    review = commands.add_parser(
        "review", help="Prepare ground truth and open the FiftyOne review App"
    )
    review.add_argument(
        "--job-id", help="Prediction job to initialize missing ground truth (default: latest)"
    )
    review.add_argument(
        "--prepare-only", action="store_true", help="Prepare the dataset without starting the App"
    )
    review.add_argument(
        "--no-browser", action="store_true", help="Serve the App without opening a browser"
    )
    review.add_argument(
        "--limit", type=int, help="Maximum labels to prepare (default: config, 100)"
    )
    review.add_argument(
        "--queue", choices=["sample", "low_confidence", "empty"], help="Open a focused review queue"
    )
    batch = commands.add_parser(
        "review-batch", help="Preview or resume automatic prediction adoption"
    )
    batch.add_argument("--job-id", help="Source autolabel job ID for a new preview")
    batch.add_argument(
        "--min-confidence", type=float, help="Minimum confidence of every predicted object"
    )
    batch.add_argument(
        "--sample-rate",
        type=float,
        help="Sample all eligible images/groups before confidence checks (default: 0.01)",
    )
    batch.add_argument("--actor", help="Person choosing the automatic adoption policy")
    batch.add_argument("--limit", type=int, help="Bound the new preview's input count")
    batch.add_argument(
        "--batch-size", type=int, default=100, help="Number of IDs held in memory (1..1000)"
    )
    batch.add_argument("--resume", help="Continue a frozen review_batch job")
    batch.add_argument(
        "--apply", action="store_true", help="Apply a completed preview with --resume"
    )
    audit = commands.add_parser(
        "review-audit", help="Run a resumable full approval integrity audit"
    )
    audit.add_argument("--resume", help="Continue a review_audit job")
    release = commands.add_parser("release", help="Freeze approved COCO data and publish a DVC tag")
    selection = release.add_mutually_exclusive_group(required=True)
    selection.add_argument("--version", help="New dataset version, e.g. v001")
    selection.add_argument("--resume", help="Resume a release job")
    release.add_argument(
        "--include-auto-accepted",
        action="store_true",
        help="Explicitly include automatically accepted labels in train only",
    )
    release.add_argument(
        "--manual-to-val-test",
        action="store_true",
        default=None,
        help="Assign new manual groups to val/test; set on the first release, inherited later",
    )
    release.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate labels and estimate image storage without copying/uploading images",
    )
    restore = commands.add_parser("restore", help="Restore a tagged dataset into a separate cache")
    restore.add_argument("--version", required=True, help="Dataset version, e.g. v001")
    train = commands.add_parser("train", help="Train a dataset release and record an MLflow run")
    selection = train.add_mutually_exclusive_group(required=True)
    selection.add_argument("--dataset-version", help="Dataset version, e.g. v001")
    selection.add_argument(
        "--resume", metavar="JOB_ID", help="Resume a recorded vloop training job"
    )
    train.add_argument("--notes", help="Experiment purpose or observations recorded in MLflow")
    evaluate = commands.add_parser("evaluate", help="Evaluate a training job on fixed release data")
    selection = evaluate.add_mutually_exclusive_group(required=True)
    selection.add_argument("--job-id", help="Source vloop training job ID")
    selection.add_argument("--view", metavar="JOB_ID", help="Open a completed evaluation job")
    evaluate.add_argument(
        "--dataset-version", help="Evaluation release (default: training release)"
    )
    evaluate.add_argument(
        "--split", choices=["val", "test"], help="Default: val; test requires this option"
    )
    evaluate.add_argument("--limit", type=int, help="Evaluate only the first N images by image ID")
    evaluate.add_argument("--confidence", type=float, help="Override the saved inference threshold")
    evaluate.add_argument(
        "--display-confidence", type=float, help="Override the saved display threshold"
    )
    evaluate.add_argument("--max-detections", type=int, help="Maximum predictions per image")
    evaluate.add_argument("--notes", help="Evaluation purpose or observations recorded in MLflow")
    evaluate.add_argument(
        "--no-browser", action="store_true", help="Serve --view without opening a browser"
    )
    experiments = commands.add_parser("experiments", help="Serve the local MLflow experiment UI")
    experiments.add_argument("--no-browser", action="store_true", help="Do not open a browser")
    for command in (
        doctor,
        ingest,
        autolabel,
        review,
        batch,
        audit,
        release,
        restore,
        train,
        evaluate,
        experiments,
    ):
        command.add_argument("--config", default=argparse.SUPPRESS, help="Path to project YAML")
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        if args.view and any(
            getattr(args, name) is not None
            for name in (
                "dataset_version",
                "split",
                "limit",
                "confidence",
                "display_confidence",
                "max_detections",
                "notes",
            )
        ):
            parser.error("--view uses saved evaluation settings; omit inference options")
        if args.no_browser and not args.view:
            parser.error("--no-browser requires --view")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)
        if args.command == "experiments":
            from .tracking import experiments

            return experiments(cfg, no_browser=args.no_browser)
        if args.command == "evaluate" and args.view:
            from .evaluate import view_evaluation

            return view_evaluation(cfg, args.view, no_browser=args.no_browser)
        if args.command == "doctor":
            from .doctor import doctor

            report = doctor(cfg, sam3_image=args.sam3_image)
        elif args.command == "ingest":
            from .ingest import ingest

            report = ingest(cfg, local_only=args.local_only)
        elif args.command == "autolabel":
            from .autolabel import autolabel

            report = autolabel(cfg, resume=args.resume, limit=args.limit)
        elif args.command == "review":
            from .review import prepare_review

            report = prepare_review(cfg, job_id=args.job_id, limit=args.limit, queue=args.queue)
        elif args.command == "review-batch":
            from .review_batch import review_batch

            report = review_batch(
                cfg,
                job_id=args.job_id,
                minimum=args.min_confidence,
                sample_rate=args.sample_rate,
                actor=args.actor,
                limit=args.limit,
                resume=args.resume,
                apply=args.apply,
                batch_size=args.batch_size,
            )
        elif args.command == "review-audit":
            from .review_audit import review_audit

            report = review_audit(cfg, resume=args.resume)
        elif args.command == "release":
            from .release import release

            report = release(
                cfg,
                version=args.version,
                resume=args.resume,
                include_auto_train=args.include_auto_accepted,
                manual_to_val_test=args.manual_to_val_test,
                prepare_only=args.prepare_only,
            )
        elif args.command == "restore":
            from .release import restore

            report = restore(cfg, version=args.version)
        elif args.command == "train":
            from .train import train

            report = train(
                cfg,
                dataset_version=args.dataset_version,
                resume=args.resume,
                notes=args.notes,
            )
        else:
            from .evaluate import evaluate

            report = evaluate(
                cfg,
                job_id=args.job_id,
                dataset_version=args.dataset_version,
                split=args.split,
                limit=args.limit,
                notes=args.notes,
                confidence=args.confidence,
                display_confidence=args.display_confidence,
                max_detections=args.max_detections,
            )
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
    elif args.command == "ingest":
        print(
            f"Images: {report['registered']} registered, {report['duplicate']} duplicates, "
            f"{report['repaired']} repaired, {report['failed']} failed"
        )
        print(f"FiftyOne sync: {report['sync_status']}")
    elif args.command == "autolabel":
        unfinished = sum(report.get(key, 0) for key in ("pending", "processing", "predicted"))
        print(
            f"Images: {report.get('completed', 0)} completed "
            f"({report.get('empty', 0)} empty), {report.get('failed', 0)} failed, "
            f"{unfinished} unfinished"
        )
        print(f"Prediction field: {report['prediction_field']}")
        print(f"Processed this attempt: {report.get('processed_this_attempt', 0)}")
        if report.get("config_source"):
            print(f"Frozen config: {report['config_source']}")
    elif args.command == "review":
        print(f"Source job: {report.get('source_job_id') or 'manual review'}")
        print(
            f"Ground truth: {report['initialized']} initialized, {report['preserved']} preserved, "
            f"{report['unavailable']} without a prediction"
        )
        print(f"Approvals invalidated: {report.get('invalidated', 0)}")
    elif args.command == "review-batch":
        print(f"Decisions: {report.get('decisions', {})}")
        print(f"Applied: {report.get('applied', 0)}, outcomes: {report.get('outcomes', {})}")
        if report["status"] == "ready":
            print(f"Preview only. Apply: vloop review-batch --resume {report['job_id']} --apply")
    elif args.command == "review-audit":
        print(f"Audited: {report.get('checked', 0)}, invalidated: {report.get('invalidated', 0)}")
    elif args.command == "train":
        print(f"Epochs completed: {report.get('epochs_completed', 0)}")
        if report.get("best_mask_map") is not None:
            print(f"Best validation mask mAP: {report['best_mask_map']:.6f}")
        if report.get("model_dir"):
            print(f"Model: {report['model_dir']}")
    elif args.command == "evaluate":
        print(f"Source training job: {report['source_job_id']}")
        print(
            f"Split: {report['split']}, images: {report.get('images', 0)}, "
            f"predicted: {report['predicted']}"
        )
        for kind, values in report.get("metrics", {}).items():
            print(
                f"{kind}: mAP={values['mAP']}, AP50={values['AP50']}, "
                f"FP={values['fp']}, FN={values['fn']}"
            )
        if report["status"] == "completed":
            print(f"Comparison ID: {report['comparison_id']}")
            print(f"View: vloop evaluate --config {cfg.config_path} --view {report['job_id']}")
    else:
        for split, counts in report.get("summary", {}).get("splits", {}).items():
            print(
                f"{split}: {counts['images']} images ({counts['empty']} empty), "
                f"{counts['automatic']} automatic"
            )
        print(f"Held for group review: {report.get('summary', {}).get('held', 0)}")
        if report.get("git_commit"):
            print(f"Tag: dataset/{report['dataset_version']} ({report['git_commit']})")
        if report.get("dataset_dir"):
            print(f"Dataset directory: {report['dataset_dir']}")
        if report.get("storage_plan"):
            plan = report["storage_plan"]
            print(
                f"Images to store: {plan['new_images']} new, {plan['reused_images']} reusable, "
                f"{plan['new_image_bytes'] / 2**30:.3f} GiB new image bytes"
            )
            print(
                f"Local image-space estimate before link savings: "
                f"{plan['image_bytes_upper_estimate_local'] / 2**30:.3f} GiB; "
                f"free: {plan['local_free_bytes'] / 2**30:.3f} GiB (labels/metadata extra)"
            )
    if report.get("error"):
        print(f"Error: {report['error']}", file=sys.stderr)
    if report["status"] != "completed" and "retry" in report:
        print(f"Retry: {report['retry']}")
    print(f"Results: {report['result_dir']}")
    if args.command == "review" and report["status"] == "completed" and not args.prepare_only:
        from .review import serve_review

        try:
            serve_review(cfg, no_browser=args.no_browser, queue=args.queue)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"vloop review: {exc}", file=sys.stderr)
            return 1
    return (
        0
        if report["status"] in ("completed", "ready")
        else 130
        if report["status"] == "interrupted"
        else 1
    )
