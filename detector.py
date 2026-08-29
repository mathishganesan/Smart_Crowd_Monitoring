import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO

class CrowdDetector:
    def __init__(self, 
                 model_path=r"runs/detect/runs/detect/yolov10_head_monitoring-6/weights/best.pt", 
                 conf_threshold=0.35, 
                 warning_limit=3, 
                 danger_limit=6):
        
        # Select device
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Initializing CrowdDetector on device: {self.device}")
        
        # Load model weights
        self.model = YOLO(model_path)
        
        # Configurable thresholds
        self.conf_threshold = conf_threshold
        self.warning_limit = warning_limit
        self.danger_limit = danger_limit
        
        # EMA metrics tracking
        self.prev_time = time.time()
        self.fps_smoothed = 0.0
        self.alpha = 0.1  # Smoothing factor

    def classify_density(self, count):
        """Dynamic 3-tier safety classification logic"""
        if count < self.warning_limit:
            return "NORMAL", (0, 255, 0)       # Green
        elif self.warning_limit <= count < self.danger_limit:
            return "WARNING", (0, 215, 255)    # Yellow / Amber
        else:
            return "DANGER", (0, 0, 255)       # Red

    def process_frame(self, frame):
        # 1. Temporal latency calculation
        curr_time = time.time()
        fps = 1.0 / max(curr_time - self.prev_time, 1e-5)
        self.prev_time = curr_time
        self.fps_smoothed = (self.alpha * fps) + ((1.0 - self.alpha) * self.fps_smoothed)

        # 2. Downsample for faster streaming throughput
        h, w = frame.shape[:2]
        target_w = 640
        target_h = int((target_w / w) * h)
        resized_frame = cv2.resize(frame, (target_w, target_h))

        # 3. GPU-accelerated head inference
        t_start = time.time()
        results = self.model.predict(
            source=resized_frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )[0]
        inference_time_ms = (time.time() - t_start) * 1000.0

        head_count = 0
        boxes = results.boxes

        if boxes is not None:
            head_count = len(boxes)

        status, status_color = self.classify_density(head_count)

        # 4. Draw detections on the frame
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                
                # Draw bounding box
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), status_color, 2)
                cv2.putText(
                    resized_frame,
                    f"Head {conf:.2f}",
                    (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    status_color,
                    1,
                    cv2.LINE_AA
                )

        # 5. Overlay Status Banner
        cv2.rectangle(resized_frame, (10, 10), (280, 80), (20, 20, 20), -1)
        cv2.rectangle(resized_frame, (10, 10), (280, 80), status_color, 2)
        cv2.putText(resized_frame, f"Head Count: {head_count}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(resized_frame, f"Status: {status}", (20, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 2)
        cv2.putText(resized_frame, f"FPS: {self.fps_smoothed:.1f} (GPU)", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

        metrics = {
            "count": head_count,
            "status": status,
            "fps": round(self.fps_smoothed, 1),
            "inference_time_ms": round(inference_time_ms, 1)
        }

        return resized_frame, metrics