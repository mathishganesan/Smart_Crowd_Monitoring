import cv2
import time
import torch
from ultralytics import YOLO

class CrowdDetector:
    def __init__(self, 
                 model_path=r"runs/detect/runs/detect/yolov10_head_monitoring-6/weights/best.pt", 
                 conf_threshold=0.15,      # Lowered confidence threshold for small/occluded heads
                 warning_limit=5, 
                 danger_limit=12):
        
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Initializing CrowdDetector on device: {self.device}")
        
        # Load fine-tuned weights
        self.model = YOLO(model_path)
        
        self.conf_threshold = conf_threshold
        self.warning_limit = warning_limit
        self.danger_limit = danger_limit
        
        # EMA metrics smoothing
        self.prev_time = time.time()
        self.fps_smoothed = 0.0
        self.alpha = 0.1

    def classify_density(self, count):
        if count < self.warning_limit:
            return "NORMAL", (0, 255, 0)       # Green
        elif self.warning_limit <= count < self.danger_limit:
            return "WARNING", (0, 215, 255)    # Amber
        else:
            return "DANGER", (0, 0, 255)       # Red

    def process_frame(self, frame):
        curr_time = time.time()
        fps = 1.0 / max(curr_time - self.prev_time, 1e-5)
        self.prev_time = curr_time
        self.fps_smoothed = (self.alpha * fps) + ((1.0 - self.alpha) * self.fps_smoothed)

        t_start = time.time()
        # High-resolution inference (imgsz=960) resolves small heads without dropping GPU FPS below 30
        results = self.model.predict(
    source=frame,
    imgsz=960,
    conf=0.12,          # Catches faint head contours on zebra crossings
    iou=0.40,           # Eliminates duplicate overlapping boxes
    augment=True,       # Test-Time Augmentation (TTA) for multi-scale resilience
    device=self.device,
    verbose=False
)[0]
        inference_time_ms = (time.time() - t_start) * 1000.0

        boxes = results.boxes
        head_count = len(boxes) if boxes is not None else 0
        status, status_color = self.classify_density(head_count)

        # Draw detections
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
                
                # Draw head label
                cv2.putText(
                    frame,
                    f"H {conf:.2f}",
                    (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    status_color,
                    1,
                    cv2.LINE_AA
                )

        # Draw Video Overlay Banner
        cv2.rectangle(frame, (15, 15), (290, 95), (20, 20, 20), -1)
        cv2.rectangle(frame, (15, 15), (290, 95), status_color, 2)
        cv2.putText(frame, f"Head Count: {head_count}", (25, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (25, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 2)
        cv2.putText(frame, f"FPS: {self.fps_smoothed:.1f} (GPU)", (25, 87),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

        metrics = {
            "count": head_count,
            "status": status,
            "fps": round(self.fps_smoothed, 1),
            "inference_time_ms": round(inference_time_ms, 1)
        }

        return frame, metrics