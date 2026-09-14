from ultralytics import YOLO
import torch

def start_training():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Executing training on: {device} ({torch.cuda.get_device_name(0) if device == 0 else 'CPU'})")

    # Warm-start from your Part A trained weights
    model = YOLO(r"runs/detect/runs/detect/yolov10_head_monitoring-6/weights/best.pt")

    # Fine-tune on the merged SCUT-HEAD-AB dataset
    model.train(
        data="dataset/data.yaml",
        imgsz=640,
        batch=8,           # Optimal for 4GB VRAM
        epochs=25,         # Transfer learning domain adaptation
        lr0=0.005,         # Moderate learning rate to preserve learned features
        workers=2,         # Safe for Windows file indexing
        device=device,
        amp=True,          # Automatic Mixed Precision
        name="yolov10_scut_ab_combined"
    )

if __name__ == "__main__":
    start_training()