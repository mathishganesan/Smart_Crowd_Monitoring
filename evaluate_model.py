import os
import torch
from ultralytics import YOLO

def main():
    # 1. Select available model weights
    model_path = r"runs/detect/yolov10_scut_ab_combined/weights/best.pt"
    if not os.path.exists(model_path):
        model_path = r"runs/detect/runs/detect/yolov10_head_monitoring-6/weights/best.pt"

    print(f"[INFO] Loading checkpoint: {model_path}")
    model = YOLO(model_path)

    # 2. Select compute device (GPU if available)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running validation on device: {device}")

    # 3. Path to your dataset configuration YAML
    data_yaml = "dataset/data.yaml" if os.path.exists("dataset/data.yaml") else "data.yaml"

    # 4. Run model evaluation on the test split
    print("[INFO] Evaluating test split...")
    results = model.val(
        data=data_yaml,
        split="test",          # Maps to 'test: test/images' in your YAML
        imgsz=960,             # Target surveillance image size
        batch=8,
        conf=0.20,
        iou=0.45,
        device=device,
        workers=0,             # Set to 0 to prevent Windows multiprocessing bootstrap errors
        save_json=False
    )

    # 5. Extract and display the calculated accuracy metrics
    precision = results.box.mp * 100
    recall = results.box.mr * 100
    map50 = results.box.map50 * 100
    map50_95 = results.box.map * 100

    print("\n" + "=" * 50)
    print("          MODEL ACCURACY EVALUATION RESULTS       ")
    print("=" * 50)
    print(f" Class Evaluated:          {results.names[0]}")
    print(f" Precision (P):            {precision:.2f}%")
    print(f" Recall (R):               {recall:.2f}%")
    print(f" mAP @ IoU 0.50 (mAP@50):  {map50:.2f}%")
    print(f" mAP @ IoU 0.50:0.95:      {map50_95:.2f}%")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    main()