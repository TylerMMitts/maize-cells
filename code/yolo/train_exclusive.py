# Trains the exclusion model that marks valid cortex.
#
# yolov8l-seg, because this mask is one large region per image rather than
# many small ones and benefits from the larger backbone.

from pathlib import Path

from ultralytics import YOLO
import torch
from code.yolo.training_output import split_run_output, report
from code.config import MODELS_FOLDER

def train_exclusion(
    dataset_yaml=MODELS_FOLDER / 'exclusion_dataset' / 'data.yaml',
    model_size="yolov8l-seg.pt",
    epochs=50,
    batch_size=4,
    image_size=640,
    project=MODELS_FOLDER / 'runs' / 'segment' / 'runs',
    run_name="exclusion_model",
    device='cpu'
): 
    # Use the device parameter passed in (respects config setting)
    
    model = YOLO(model_size)
    
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        project=project,
        name=run_name,
        exist_ok=True,
        task='segment',
        verbose=True
    )
    
    # Figures and previews move to results/training/; only weights stay here.
    
    run_dir = Path(project) / run_name
    
    report(split_run_output(run_dir, "inner_part_detector"), "inner_part_detector")
    
    weights_path = run_dir / "weights" / "inner_part_detector_best.pt"
    
    print(f"Exclusion model saved to: {weights_path}")
    return weights_path

if __name__ == "__main__":
    train_exclusion()