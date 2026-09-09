# Trains the exclusion model on downscaled images.
#
# A second copy of the inner/outer part model for datasets captured at lower
# optical resolution, such as data/widiv. The scale factor was measured from
# the imagery rather than assumed.
#
# Companion to train_exclusive.py, which is left untouched. Weights land beside
# the existing exclusion model:
#
#     models/runs/segment/runs/exclusion_model/inner_part_detector/         <- existing
#     models/runs/segment/runs/exclusion_model/inner_part_detector_lowres/  <- this one
#
# WHY THIS IS NEEDED
# The exclusion model marks which part of a quadrant is valid cortex; cells
# outside it are discarded before measurement. On widiv it fails in the most
# damaging way possible -- it does not fail loudly, it silently over-excludes.
# On a representative widiv quadrant the "inner-part" mask covered 62.6% of the
# image, swallowing a large region of ordinary cortex that is visually identical
# to the region it kept, with no stele boundary anywhere inside it. Every cell in
# that region is dropped, so the measurement comes out quietly undercounted
# rather than obviously broken.
#
# That is the same domain gap the cell segmenter had: both models were trained on
# imagery roughly 1.5x higher resolution than widiv. Measured on the actual data,
# exclusion training images have a median width of 1074 px against 735 px for
# widiv quadrants -- a ratio of 1.46, i.e. a scale factor of 0.68. DEFAULT_SCALE
# is 0.70, matching the value derived independently for the cell model in
# train_robust_lowres.py.
#
# The dataset builder is imported from train_robust_lowres rather than copied, so
# both low-resolution models are produced by exactly the same transformation.
#
# Note this dataset carries TWO classes (inner-part, outer-part), unlike the cell
# dataset's one; the builder reads the class list from the source data.yaml, so
# that is handled automatically.

from pathlib import Path

import torch
from code.yolo.training_output import split_run_output, report
from ultralytics import YOLO

from code.yolo.train_robust_lowres import build_downscaled_dataset

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

SRC_DATASET = PROJECT_ROOT / "models" / "exclusion_dataset"
DST_DATASET = PROJECT_ROOT / "models" / "exclusion_dataset_lowres"

# Same factor as the cell model: 1074 px -> 735 px is 0.68, rounded to match.
DEFAULT_SCALE = 0.70


def train_exclusion_lowres(
    scale=DEFAULT_SCALE,
    model_size=None,                     # default: models/yolov8l-seg.pt
    epochs=50,                           # matches train_exclusive.py
    batch_size=4,
    image_size=640,
    project=None,                        # default: beside the existing weights
    run_name="inner_part_detector_lowres",
    device=None,
    rebuild_dataset=False,
):
    if model_size is None:
        model_size = str(PROJECT_ROOT / "models" / "yolov8l-seg.pt")
    if project is None:
        project = str(PROJECT_ROOT / "models" / "runs" / "segment" / "runs" / "exclusion_model")
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    print(f"Building downscaled exclusion dataset (scale={scale}) ...")
    data_yaml = build_downscaled_dataset(
        scale=scale, src=SRC_DATASET, dst=DST_DATASET, force=rebuild_dataset)

    print(f"\nstarting weights : {model_size}")
    print(f"device           : {device}")
    print(f"output           : {project}/{run_name}")

    model = YOLO(model_size)

    # Hyperparameters mirror train_exclusive.py so the only variable is image
    # scale. save_period is the one addition -- it writes periodic checkpoints
    # so an interrupted run is not lost.
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        project=project,
        name=run_name,
        exist_ok=True,
        task='segment',
        verbose=True,
        save=True,
        save_period=10,
    )

    # Figures and previews move to results/training/; only weights stay here.

    run_dir = Path(project) / run_name

    report(split_run_output(run_dir, "inner_part_detector_lowres"), "inner_part_detector_lowres")

    weights_path = run_dir / "weights" / "inner_part_detector_lowres_best.pt"

    print(f"Weights saved to: {weights_path}")
    return weights_path


def main():
    # python -m code.yolo.train_exclusion_lowres
    class cfg:
        # image downscale factor
        scale = DEFAULT_SCALE
        epochs = 50
        # reduce to 2 if the GPU runs out of memory
        batch = 4
        imgsz = 640
        # '0' for GPU, 'cpu' to force CPU, None to let ultralytics choose
        device = None
        # regenerate the scaled dataset even if it already exists
        rebuild_dataset = False
        # build the scaled dataset and stop, without training
        dataset_only = False

    if cfg.dataset_only:
        build_downscaled_dataset(scale=cfg.scale, src=SRC_DATASET, dst=DST_DATASET, force=cfg.rebuild_dataset)
    else:
        train_exclusion_lowres(
            scale=cfg.scale, epochs=cfg.epochs, batch_size=cfg.batch,
            image_size=cfg.imgsz, device=cfg.device,
            rebuild_dataset=cfg.rebuild_dataset,
        )


if __name__ == '__main__':
    main()
