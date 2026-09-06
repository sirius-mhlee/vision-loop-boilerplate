import gc
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from PIL import Image

from .config import Config
from .fiftyone import configure_fiftyone
from .ingest import normalize_image
from .runtime import sha256_file, write_json


def model_identity(cfg: Config) -> dict:
    from .doctor import check_checkpoint, check_sam3_source

    check_sam3_source(cfg)
    check_checkpoint(cfg)
    vocabulary = cfg.sam3_source_dir / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    return {
        "model": cfg.sam3_model,
        "sam3_commit": cfg.sam3_commit,
        "checkpoint_sha256": sha256_file(cfg.sam3_checkpoint),
        "vocabulary_sha256": sha256_file(vocabulary),
        "precision": cfg.sam3_precision,
        "prompt_batch_size": 1,
        "image_batch_size": 1,
        "dependencies": {
            name: version(name)
            for name in ("torch", "torchvision", "fiftyone", "sam3", "numpy", "pycocotools")
        },
        "adapter_sha256": sha256_file(Path(__file__)),
        "labels_sha256": sha256_file(Path(__file__).with_name("labels.py")),
    }


class Sam3Labeler:
    """The pinned FiftyOne model, with one image and one concept per forward pass."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None
        self.metrics = {}

    def __enter__(self):
        import torch

        self.torch = torch
        self.device = torch.device(self.cfg.device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.cuda.init()
            if self.cfg.sam3_precision == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise ValueError("This CUDA device does not support bfloat16")
            torch.cuda.reset_peak_memory_stats(self.device)
        elif self.cfg.sam3_precision != "float32":
            raise ValueError("CPU inference requires sam3_precision: float32")
        configure_fiftyone(self.cfg)
        import fiftyone.zoo as foz

        self.started = perf_counter()
        try:
            self.model = foz.load_zoo_model(
                self.cfg.sam3_model,
                operation_mode="concept",
                classes=[],
                cache=False,
                confidence_thresh=self.cfg.autolabel_confidence,
                device=str(self.device),
                entrypoint_args={
                    "checkpoint_path": str(self.cfg.sam3_checkpoint),
                    "bpe_path": str(
                        self.cfg.sam3_source_dir / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
                    ),
                    "load_from_HF": False,
                    "compile": False,
                },
            )
            # FiftyOne 1.21 otherwise filters scores at 0.5 before the configured threshold.
            self.model._concept_output_processor.mask_thresh = self.cfg.autolabel_confidence
            self.model.__enter__()
        except BaseException:
            self.model = None
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            raise
        return self

    def predict(self, image: dict) -> dict:
        from .labels import from_detection

        torch = self.torch
        predictions = []
        started = perf_counter()
        for prompt, item in self.cfg.prompt_to_class.items():
            self.model.config.classes = [prompt]
            get_item = self.model.build_get_item()
            inputs = get_item({"id": image["image_id"], "filepath": image["filepath"]})
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if self.device.type == "cuda" and self.cfg.sam3_precision == "bfloat16"
                else nullcontext()
            )
            with torch.inference_mode(), autocast:
                output = self.model.predict(inputs)
            if output is None:
                raise RuntimeError("SAM 3 returned no prediction; empty Detections is required")
            for detection in output.detections:
                if detection.label != prompt:
                    raise ValueError(f"Unexpected SAM 3 label: {detection.label}")
                predictions.append(
                    from_detection(detection, item, prompt, image["width"], image["height"])
                )
            del output, inputs
        self._update_metrics()
        return {
            "image_id": image["image_id"],
            "width": image["width"],
            "height": image["height"],
            "instances": predictions,
            "elapsed_seconds": perf_counter() - started,
        }

    def _update_metrics(self):
        self.metrics["elapsed_seconds"] = perf_counter() - self.started
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
            self.metrics.update(
                peak_allocated_gib=self.torch.cuda.max_memory_allocated(self.device) / 1024**3,
                peak_reserved_gib=self.torch.cuda.max_memory_reserved(self.device) / 1024**3,
            )

    def __exit__(self, *args):
        try:
            self._update_metrics()
        finally:
            if self.model is not None:
                self.model.__exit__(*args)
            self.model = None
            gc.collect()
            if self.device.type == "cuda":
                self.torch.cuda.empty_cache()


def smoke_predict(cfg: Config, source: Path, result_dir: Path) -> str:
    from .labels import decode_mask

    if not cfg.classes:
        raise ValueError("Set classes and prompts before running SAM 3")
    identity = model_identity(cfg)
    write_json(result_dir / "sam3-model.json", identity)
    source = source.expanduser().resolve()
    normalized = result_dir / "sam3-input.png"
    width, height = normalize_image(source, normalized)
    image = {
        "image_id": sha256_file(source),
        "filepath": str(normalized),
        "width": width,
        "height": height,
    }
    labeler = Sam3Labeler(cfg)
    try:
        with labeler:
            prediction = labeler.predict(image)
        prediction.update(
            source=str(source), image_path=str(normalized), review_status="unreviewed"
        )
        write_json(result_dir / "sam3-predictions.json", prediction)
        for index, item in enumerate(prediction["instances"]):
            mask = decode_mask(item["segmentation"], width, height)
            Image.fromarray(mask).save(result_dir / f"mask-{index:04d}.png")
        return f"Inference completed: {len(prediction['instances'])} objects; {result_dir}"
    finally:
        write_json(result_dir / "sam3-memory.json", labeler.metrics)
