from pathlib import Path

from .config import Config
from .fiftyone import configure_fiftyone
from .ingest import normalize_image
from .runtime import write_json


def smoke_predict(cfg: Config, source: Path, result_dir: Path) -> str:
    """Check the official concept-mode adapter without modifying a review dataset."""
    from .doctor import check_checkpoint, check_sam3_source

    errors = cfg.input_errors()
    if errors:
        raise ValueError("; ".join(errors))
    check_checkpoint(cfg)
    check_sam3_source(cfg)
    fo = configure_fiftyone(cfg)
    import fiftyone.zoo as foz
    import numpy as np
    from PIL import Image

    source = source.expanduser().resolve()
    normalized = result_dir / "sam3-input.png"
    width, height = normalize_image(source, normalized)
    dataset = fo.Dataset()
    try:
        dataset.add_sample(fo.Sample(filepath=str(normalized)))
        model = foz.load_zoo_model(
            cfg.sam3_model,
            operation_mode="concept",
            classes=list(cfg.prompt_to_class),
            confidence_thresh=cfg.autolabel_confidence,
            device=cfg.device,
            entrypoint_args={"checkpoint_path": str(cfg.sam3_checkpoint)},
        )
        dataset.apply_model(model, label_field="smoke_predictions", batch_size=1)
        detections = dataset.first()["smoke_predictions"]
        if detections is None:
            raise RuntimeError(
                "SAM 3 returned no prediction field; an empty Detections is required"
            )
        predictions = []
        for index, detection in enumerate(detections.detections):
            if detection.label not in cfg.prompt_to_class:
                raise ValueError(f"Unknown model output class: {detection.label}")
            item = cfg.prompt_to_class[detection.label]
            box = np.asarray(detection.bounding_box, dtype=float)
            if (
                box.shape != (4,)
                or not np.isfinite(box).all()
                or (box < 0).any()
                or (box[2:] <= 0).any()
                or (box[:2] + box[2:] > 1.00001).any()
            ):
                raise ValueError("SAM 3 returned an invalid normalized bounding box")
            mask = np.asarray(detection.mask)
            if mask.ndim != 2 or not mask.size:
                raise ValueError("SAM 3 returned a detection without a 2D instance mask")
            mask_path = result_dir / f"mask-{index:04d}.png"
            Image.fromarray((mask.astype(bool) * 255).astype("uint8")).save(mask_path)
            x, y, w, h = box.tolist()
            predictions.append(
                {
                    "class_id": item.id,
                    "class_name": item.name,
                    "prompt": detection.label,
                    "confidence": detection.confidence,
                    "bbox_xywh": [x * width, y * height, w * width, h * height],
                    "mask_path": str(mask_path),
                    "mask_space": "bounding_box",
                }
            )
        write_json(
            result_dir / "sam3-predictions.json",
            {
                "source": str(source),
                "image_path": str(normalized),
                "width": width,
                "height": height,
                "predictions": predictions,
                "review_status": "unreviewed",
            },
        )
        return (
            f"Inference completed: {len(predictions)} objects; "
            f"{result_dir / 'sam3-predictions.json'}"
        )
    finally:
        dataset.delete()
