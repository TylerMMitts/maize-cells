# Trains the cell segmentation model.
#
# Uses the nano backbone deliberately: cells are small, numerous and
# repetitive, and nano at high confidence gave cleaner masks and much faster
# inference over thousands of quadrants than a larger model did.

from pathlib import Path

from ultralytics import YOLO
import torch
from code.yolo.training_output import split_run_output, report
from code.config import MODELS_FOLDER

def train_robust_model(
    dataset_yaml=MODELS_FOLDER / 'cell_dataset' / 'data.yaml',
    model_size="yolov8n-seg.pt",
    epochs=100,
    max_det=3000,
    batch_size=4,
    image_size=640,
    project=MODELS_FOLDER / 'runs' / 'segment' / 'runs',
    run_name="robust_segmentation",
    device='cpu'
):
    # Use the device parameter passed in (respects config setting)
    
    model = YOLO(model_size)
    
    # Added data augmentation parameters for more robust training
    results = model.train(
        data=dataset_yaml,
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
    
    report(split_run_output(run_dir, "robust_cell_detector"), "robust_cell_detector")
    
    weights_path = run_dir / "weights" / "robust_cell_detector_best.pt"
    
    print(f"Weights saved to: {weights_path}")
    
    return weights_path

if __name__ == "__main__":
    train_robust_model()
