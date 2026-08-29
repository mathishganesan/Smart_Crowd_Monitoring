import os
import torch
from ultralytics import YOLO

def start_training():
    print("[INFO] Checking GPU status for training...")
    if not torch.cuda.is_available():
        print("[WARNING] CUDA is not available! Training on CPU will be slow.")
        device = 'cpu'
    else:
        print(f"[INFO] CUDA Active! Training on GPU: {torch.cuda.get_device_name(0)}")
        device = 0

    # Explicit path to data.yaml
    data_yaml_path = r"B:\Mini_Project\Smart_Crowd_Monitoring\dataset\data.yaml"

    print(f"[INFO] Using dataset configuration from: {data_yaml_path}")

    # Load base YOLOv10 model weights (use 'yolov10n.pt' or 'yolov10s.pt')
    model = YOLO("yolov10n.pt")

    print("[INFO] Starting YOLOv10 fine-tuning on custom dataset...")
    
    # Run fine-tuning
    results = model.train(
        data=data_yaml_path,
        epochs=30,
        imgsz=640,
        batch=16,
        device=device,
        workers=2,
        amp=True,
        name="yolov10_head_monitoring",
        project="runs/detect",
        save=True
    )
    
    print("[SUCCESS] Training completed! Best model saved at: runs/detect/yolov10_head_monitoring/weights/best.pt")

if __name__ == "__main__":
    start_training()