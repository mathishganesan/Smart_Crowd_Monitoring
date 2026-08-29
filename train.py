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

    # Load base model (use 'yolov10n.pt' for Nano or 'yolov10s.pt' for Small)
    model = YOLO("yolov10n.pt")

    print("[INFO] Starting YOLOv10 fine-tuning on custom dataset...")
    
    # Run fine-tuning process
    results = model.train(
        data="data.yaml",
        epochs=30,           # 30-50 epochs recommended
        imgsz=640,           # Input image resolution
        batch=16,            # Optimized for 4GB VRAM (drop to 8 if OOM occurs)
        device=device,
        workers=2,
        amp=True,            # Mixed precision FP16 (faster & lower VRAM)
        name="yolov10_head_monitoring",
        project="runs/detect",
        save=True
    )
    
    print("[SUCCESS] Training completed! Best model saved at: runs/detect/yolov10_head_monitoring/weights/best.pt")

if __name__ == "__main__":
    start_training()