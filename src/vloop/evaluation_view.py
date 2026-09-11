"""FiftyOne COCO box/mask evaluation and a reproducible analysis dataset."""

import json
import threading
from contextlib import closing
from importlib.metadata import version

import numpy as np

from .fiftyone import configure_fiftyone
from .labels import decode_mask
from .release_data import connect


def fields(job_id):
    return {
        kind: {
            "gt": f"ground_truth_{kind}",
            "pred": f"pred_{kind}_{job_id}",
            "key": f"{kind}_{job_id}",
        }
        for kind in ("boxes", "masks")
    }


def detections(fo, annotation, *, masks):
    width, height = annotation["width"], annotation["height"]
    labels = []
    for instance in annotation["instances"]:
        x, y, w, h = instance["bbox_xywh"]
        mask = None
        if masks:
            full = decode_mask(instance["segmentation"], width, height)
            ys, xs = np.where(full)
            if len(xs):
                x, y = int(xs.min()), int(ys.min())
                w, h = int(xs.max()) + 1 - x, int(ys.max()) + 1 - y
                mask = full[y : y + h, x : x + w]
            else:
                # A zero-area mask is still a prediction and must count as a false positive.
                x, y, w, h = 0, 0, 1, 1
                mask = np.zeros((1, 1), dtype=bool)
        labels.append(
            fo.Detection(
                label=instance["class_name"],
                bounding_box=[x / width, y / height, w / width, h / height],
                mask=mask,
                confidence=instance.get("confidence"),
                iscrowd=False,
                class_id=instance["class_id"],
            )
        )
    return fo.Detections(detections=labels)


def build_dataset(cfg, manifest, database, root):
    fo = configure_fiftyone(cfg)
    if version("fiftyone") != manifest["parameters"]["fiftyone_version"]:
        raise ValueError("Rebuilding evaluation requires its recorded FiftyOne version")
    name = "vloop-eval-" + manifest["job_id"]
    owner = {"vloop_storage_dir": str(cfg.storage_dir), "binding": manifest["binding"]}
    if fo.dataset_exists(name):
        dataset = fo.load_dataset(name)
        if any(dataset.info.get(k) != v for k, v in owner.items()):
            raise ValueError("Evaluation dataset belongs to different artifacts or storage")
        if dataset.info.get("status") == "completed":
            return dataset, dataset.info["metrics"]
        fo.delete_dataset(name)  # Rebuild only our own interrupted derived dataset.
    dataset = fo.Dataset(name, persistent=True)
    dataset.info = {**owner, "status": "building", "comparison_id": manifest["comparison_id"]}
    dataset.save()
    names = [c["name"] for c in manifest["selection"]["classes"]]
    mapping = fields(manifest["job_id"])
    for pair in mapping.values():
        for field in (pair["gt"], pair["pred"]):
            dataset.add_sample_field(
                field, fo.EmbeddedDocumentField, embedded_doc_type=fo.Detections
            )
            dataset.classes[field] = names
    batch = []
    with closing(connect(database)) as db:
        for row in db.execute("SELECT * FROM samples ORDER BY image_id"):
            if row["prediction"] is None or row["error"]:
                raise ValueError("Evaluation contains incomplete predictions")
            gt, pred = json.loads(row["annotation"]), json.loads(row["prediction"])
            sample = fo.Sample(
                filepath=str(root / row["relative_path"]),
                image_id=row["image_id"],
                metadata=fo.ImageMetadata(width=gt["width"], height=gt["height"]),
                empty_prediction=not pred["instances"],
            )
            for kind, pair in mapping.items():
                sample[pair["gt"]] = detections(fo, gt, masks=kind == "masks")
                sample[pair["pred"]] = detections(fo, pred, masks=kind == "masks")
            batch.append(sample)
            if len(batch) == 20:
                dataset.add_samples(batch)
                batch.clear()
        if batch:
            dataset.add_samples(batch)
    metrics = {}
    parameters = manifest["parameters"]
    for kind, pair in mapping.items():
        print(f"Evaluating {kind}: {len(dataset)} images", flush=True)
        result = dataset.evaluate_detections(
            pair["pred"],
            gt_field=pair["gt"],
            eval_key=pair["key"],
            method="coco",
            classes=names,
            iou=parameters["analysis_iou"],
            classwise=True,
            compute_mAP=True,
            iou_threshs=parameters["iou_thresholds"],
            max_preds=parameters["max_detections"],
            use_masks=kind == "masks",
            tolerance=None,
            error_level=0,
        )
        per_class = {}
        for index, label in enumerate(result.classes):
            values = result.precision[:, index, :]
            valid = values[values >= 0]
            first = values[0][values[0] >= 0]
            per_class[str(label)] = {
                "AP": float(valid.mean()) if valid.size else None,
                "AP50": float(first.mean()) if first.size else None,
            }
        aps = [c["AP"] for c in per_class.values() if c["AP"] is not None]
        ap50s = [c["AP50"] for c in per_class.values() if c["AP50"] is not None]
        metrics[kind] = {
            "mAP": float(np.mean(aps)) if aps else None,
            "AP50": float(np.mean(ap50s)) if ap50s else None,
            "per_class": per_class,
            **{key: int(dataset.sum(pair["key"] + "_" + key)) for key in ("tp", "fp", "fn")},
        }
        for field in (pair["gt"], pair["pred"]):
            dataset.update_label_schema(
                field,
                {
                    "type": "detections",
                    "component": "dropdown",
                    "classes": names,
                    "attributes": [],
                    "read_only": True,
                },
            )
        for outcome in ("fp", "fn"):
            dataset.save_view(
                f"{kind}_{outcome}", dataset.match(fo.ViewField(pair["key"] + "_" + outcome) > 0)
            )
    label_fields = [p[key] for p in mapping.values() for key in ("gt", "pred")]
    dataset.activate_label_schemas(label_fields)
    display = dataset.view()
    for pair in mapping.values():
        display = display.filter_labels(
            pair["pred"],
            fo.ViewField("confidence") > parameters["display_confidence"],
            only_matches=False,
        )
    dataset.save_view("display_confidence", display)
    dataset.save_view("empty_predictions", dataset.match(fo.ViewField("empty_prediction")))
    dataset.save_view("all", dataset.view())
    dataset.info.update(status="completed", metrics=metrics)
    dataset.save()
    return dataset, metrics


def serve(cfg, dataset, *, no_browser=False):
    fo = configure_fiftyone(cfg)
    session = fo.launch_app(
        dataset.load_saved_view("display_confidence"),
        address="127.0.0.1",
        port=cfg.fiftyone_port,
        remote=no_browser,
    )
    print(f"FiftyOne: http://127.0.0.1:{cfg.fiftyone_port}", flush=True)
    try:
        session.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        close_session(session)
    return 0


def close_session(session):
    """Bound FiftyOne 1.21's otherwise unbounded web-server shutdown wait.

    Hypercorn may retain an active event-stream worker after SIGTERM. Only the
    server service owned by this Python session is eligible for forced shutdown;
    an existing shared server and the project MongoDB service are left alone.
    """
    import psutil
    from fiftyone.core.session import session as session_module

    service = session_module._server_services.get(session.server_port)
    child = getattr(service, "child", None)
    if child is None:
        session.close()
        return
    try:
        processes = [child, *child.children(recursive=True)]
    except psutil.NoSuchProcess:
        processes = []

    def force_close():
        for process in reversed(processes):
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass

    timer = threading.Timer(5, force_close)
    timer.daemon = True
    timer.start()
    try:
        session.close()
    finally:
        timer.cancel()
