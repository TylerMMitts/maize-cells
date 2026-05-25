# train_exclusion.py
from ultralytics import YOLO
import torch

def train_exclusion(
    dataset_yaml="exclusion_dataset/data.yaml",
    model_size="yolov8l-seg.pt",
    epochs=50,
    batch_size=4,
    image_size=640,
    project="runs/exclusion_model",
    run_name="inner_part_detector"
):
    device = 0 if torch.cuda.is_available() else 'cpu'
    
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
    
    weights_path = f"{project}/{run_name}/weights/best.pt"
    print(f"\nExclusion model saved to: {weights_path}")
    return weights_path

if __name__ == "__main__":
    train_exclusion()