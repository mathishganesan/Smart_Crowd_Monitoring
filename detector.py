import cv2
import time
import torch
from ultralytics import YOLO

class CrowdDetector:
    def __init__(self, 
                 model_path=r"runs/detect/runs/detect/yolov10_head_monitoring-6/weights/best.pt", 
                 warning_limit=5, 
                 danger_limit=12):
        
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Initializing CrowdDetector on device: {self.device}")
        
        self.model = YOLO(model_path)
        self.warning_limit = warning_limit
        self.danger_limit = danger_limit
        
        # EMA metrics smoothing
        self.prev_time = time.time()
        self.fps_smoothed = 0.0
        self.alpha = 0.1

    def classify_density(self, count):
        if count < self.warning_limit:
            return "NORMAL", (0, 255, 0)
        elif self.warning_limit <= count < self.danger_limit:
            return "WARNING", (0, 215, 255)
        else:
            return "DANGER", (0, 0, 255)

    def process_frame(self, frame, is_webcam=False):
        curr_time = time.time()
        fps = 1.0 / max(curr_time - self.prev_time, 1e-5)
        self.prev_time = curr_time
        self.fps_smoothed = (self.alpha * fps) + ((1.0 - self.alpha) * self.fps_smoothed)

        # Source-adaptive hyperparameters
        # Webcam needs higher confidence to reject indoor artifacts (grommets, curtains, lights)
        conf_thresh = 0.40 if is_webcam else 0.20
        inference_size = 640 if is_webcam else 960

        t_start = time.time()
        results = self.model.predict(
            source=frame,
            imgsz=inference_size,
            conf=conf_thresh,
            iou=0.45,
            device=self.device,
            verbose=False
        )[0]
        inference_time_ms = (time.time() - t_start) * 1000.0

        boxes = results.boxes
        valid_boxes = []

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                w, h = x2 - x1, y2 - y1

                # Aspect ratio & minimum size check for live webcam
         # Robust heuristic filter for live desktop webcam feeds
                if is_webcam:
                    # 1. Clear high-confidence curtain rings/eyelets
                    if conf < 0.72:
                        continue

                    # 2. Minimum dimension check: a human head near a laptop webcam is typically >= 45px
                    if w < 45 or h < 45:
                        continue

                    # 3. Suppress extreme ceiling/curtain-rod fixtures (top 15% margin)
                    frame_h = frame.shape[0]
                    if y1 < (frame_h * 0.15):
                        continue

                    # 4. Standard biometric aspect-ratio check
                    aspect_ratio = w / float(h)
                    if aspect_ratio < 0.65 or aspect_ratio > 1.45:
                        continue

                valid_boxes.append((x1, y1, x2, y2, conf))

        head_count = len(valid_boxes)
        status, status_color = self.classify_density(head_count)

        # Render valid bounding boxes
        for x1, y1, x2, y2, conf in valid_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
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

        # Video HUD banner
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