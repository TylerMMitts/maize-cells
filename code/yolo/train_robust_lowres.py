# Trains the cell segmenter on downscaled images.
#
# A second copy of the cell model for datasets captured at lower optical
# resolution, such as data/widiv, where the full-resolution model finds far
# fewer cells than it should.
#
# Companion to train_robust.py, which is left untouched. Weights land beside the
# existing model:
#
#     models/runs/segment/runs/robust_segmentation/robust_cell_detector/        <- existing
#     models/runs/segment/runs/robust_segmentation/robust_cell_detector_lowres/ <- this one
#
# WHAT THE MEASUREMENTS SAY (worth reading before spending a training run on it)
# Cell size measured as a fraction of image width -- the thing that actually
# determines how big a cell looks to the network, since YOLO rescales every
# image to `imgsz` regardless of its file dimensions:
#
#     training set (7,913 annotated cells)   2.52 % of width   ~33 px per cell
#     widiv quadrants (after root crop)      2.40-2.68 %       ~23 px per cell
#     existing pipeline quadrants            3.29 %            ~46 px per cell
#
# So widiv cells arrive at the network at essentially the SAME size the model was
# trained on (15-17 px at imgsz=640 vs 16 px for the training set). The gap is
# not object scale -- it is optical detail: a widiv cell is carried by ~23 raw
# pixels where a training cell has ~33, roughly half the pixel area.
#
# Two consequences:
#
#   1. Downscaling the image files does NOT make cells appear smaller to the
#      network. A 1304 px image with 33 px cells and its 652 px copy with 16 px
#      cells both become ~16 px cells once YOLO resizes to 640. What downscaling
#      does change is how much real detail survives -- which is exactly the gap
#      that separates widiv from the training set, so the idea is sound; it just
#      works through sharpness rather than scale.
#
#   2. The factor that MATCHES widiv is ~0.70 (33 px -> 23 px), not 0.50.
#      At 0.50 the training images end up softer than widiv actually is, and the
#      model would be tuned for a blur the real data does not have.
#
# DEFAULT_SCALE is therefore 0.70. Pass scale=0.5 to train the originally
# requested variant; both are one-argument changes.
#
# Note also that the observed failure on widiv (masks fragmenting into
# overlapping rectangles) is consistent with an appearance gap as much as a
# sharpness one -- widiv is markedly more saturated and lower-contrast than the
# training imagery, and the training set is only ~23 images. If this run does not
# close the gap, hand-annotating a handful of widiv quadrants and adding them to
# the training set addresses both problems directly.

import shutil
from pathlib import Path

import torch
import yaml
from PIL import Image
from code.yolo.training_output import split_run_output, report
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

SRC_DATASET = PROJECT_ROOT / "models" / "cell_dataset"
DST_DATASET = PROJECT_ROOT / "models" / "cell_dataset_lowres"

# 33 px -> 23 px, matching measured widiv cell detail. See module docstring.
DEFAULT_SCALE = 0.70


def build_downscaled_dataset(scale=DEFAULT_SCALE, src=SRC_DATASET, dst=DST_DATASET,
                             splits=("train", "valid", "test"), force=False):
    # Write a scaled copy of the dataset and return the path to its data.yaml.
    #
    # Label files are copied byte-for-byte on purpose: YOLO segmentation labels
    # are polygon coordinates normalised to [0, 1], so they describe the same
    # region of the image no matter what pixel dimensions it has. Rescaling them
    # would actually corrupt the dataset.
    if dst.exists() and force:
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    total_imgs = 0
    present = []
    for split in splits:
        src_img, src_lbl = src / split / "images", src / split / "labels"
        if not src_img.is_dir():
            continue
        dst_img, dst_lbl = dst / split / "images", dst / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        n = 0
        for img_path in sorted(src_img.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                continue
            out_path = dst_img / img_path.name
            if not out_path.exists() or force:
                im = Image.open(img_path).convert("RGB")
                w, h = im.size
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
                im.save(out_path, quality=95)
            label = src_lbl / f"{img_path.stem}.txt"
            if label.exists():
                shutil.copyfile(label, dst_lbl / label.name)
            n += 1

        if n:
            present.append(split)
            total_imgs += n
            print(f"  {split}: {n} images scaled by {scale}")

    if not total_imgs:
        raise SystemExit(f"No images found under {src}")

    # Absolute paths: the source data.yaml uses Roboflow-style relative paths
    # that only resolve from one working directory.
    train_split = "train" if "train" in present else present[0]
    val_split = "valid" if "valid" in present else train_split
    if val_split == train_split:
        print("  NOTE: no separate validation split exists, so val == train "
              "(inherited from the original dataset; val metrics will be optimistic)")

    names = ["maize-cells"]
    src_yaml = src / "data.yaml"
    if src_yaml.exists():
        try:
            names = yaml.safe_load(src_yaml.read_text(encoding="utf-8")).get("names", names)
        except Exception:
            pass

    dst_yaml = dst / "data.yaml"
    dst_yaml.write_text(yaml.safe_dump({
        "path": str(dst),
        "train": f"{train_split}/images",
        "val": f"{val_split}/images",
        "nc": len(names),
        "names": names,
    }, sort_keys=False), encoding="utf-8")
    print(f"  wrote {dst_yaml}")
    return dst_yaml


def train_robust_lowres_model(
    scale=DEFAULT_SCALE,
    model_size=None,                     # default: models/yolov8l-seg.pt
    epochs=100,
    max_det=3000,
    batch_size=4,
    image_size=640,
    project=None,                        # default: beside the existing weights
    run_name="robust_cell_detector_lowres",
    device=None,                         # default: cuda when available
    rebuild_dataset=False,
):
    if model_size is None:
        model_size = str(PROJECT_ROOT / "models" / "yolov8l-seg.pt")
    if project is None:
        project = str(PROJECT_ROOT / "models" / "runs" / "segment" / "runs" / "robust_segmentation")
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    print(f"Building downscaled dataset (scale={scale}) ...")
    data_yaml = build_downscaled_dataset(scale=scale, force=rebuild_dataset)

    print(f"\nstarting weights : {model_size}")
    print(f"device           : {device}")
    print(f"output           : {project}/{run_name}")
    if device == "cpu":
        print("WARNING: yolov8l-seg on CPU will take days. Use a smaller backbone or a GPU.")

    model = YOLO(model_size)

    # Hyperparameters deliberately identical to train_robust.py so the only
    # variable is the image scale -- otherwise a difference in results could
    # not be attributed to the downscaling.
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        workers=0,
        max_det=max_det,
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=5,
        translate=0.1,
        scale=0.2,
        fliplr=0.5,
        flipud=0.1,
        mosaic=0.5,
        mixup=0.1,
        copy_paste=0.1,
        project=project,
        name=run_name,
        exist_ok=True,
        task='segment',
        amp=True,
        patience=20,
        dropout=0.1,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        save=True,
        save_period=10,
        verbose=True
    )

    # Figures and previews move to results/training/; only weights stay here.

    run_dir = Path(project) / run_name

    report(split_run_output(run_dir, "robust_cell_detector_lowres"), "robust_cell_detector_lowres")

    weights_path = run_dir / "weights" / "robust_cell_detector_lowres_best.pt"

    print(f"Weights saved to: {weights_path}")
    return weights_path


def main():
    # python -m code.yolo.train_robust_lowres
    class cfg:
        # image downscale factor; pass 0.5 for a half-size run
        scale = DEFAULT_SCALE
        epochs = 100
        # reduce to 2 if the GPU runs out of memory (yolov8l-seg is large)
        batch = 4
        imgsz = 640
        # '0' for GPU, 'cpu' to force CPU, None to let ultralytics choose
        device = None
        # regenerate the scaled dataset even if it already exists
        rebuild_dataset = False
        # build the scaled dataset and stop, without training
        dataset_only = False

    if cfg.dataset_only:
        build_downscaled_dataset(scale=cfg.scale, force=cfg.rebuild_dataset)
    else:
        train_robust_lowres_model(
            scale=cfg.scale, epochs=cfg.epochs, batch_size=cfg.batch,
            image_size=cfg.imgsz, device=cfg.device,
            rebuild_dataset=cfg.rebuild_dataset,
        )


if __name__ == '__main__':
    main()
