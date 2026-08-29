import time
import cv2
from ultralytics import YOLO

class CrowdDetector:
    def __init__(self, model_path="yolov10n.pt", conf_threshold=0.35, target_classes=[0], warning_limit=3, danger_limit=6):
        print(f"[INFO] Initializing YOLOv10 Engine from '{model_path}'...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes
        
        # Density Threshold Limits
        self.warning_limit = warning_limit
        self.danger_limit = danger_limit
        
        # Performance optimization variables
        self.frame_counter = 0
        self.skip_frames = 3
        self.cached_boxes = []
        self.cached_count = 0

        # Smooth FPS tracking
        self.prev_time = time.time()
        self.fps = 0.0

    def get_density_status(self, count):
        if count >= self.danger_limit:
            return "DANGER", (0, 0, 255)       # Red (BGR)
        elif count >= self.warning_limit:
            return "WARNING", (0, 255, 255)    # Yellow (BGR)
        else:
            return "NORMAL", (0, 255, 0)       # Green (BGR)

    def process_frame(self, frame):
        current_time = time.time()
        time_diff = current_time - self.prev_time
        self.prev_time = current_time

        if time_diff > 0:
            instant_fps = 1.0 / time_diff
            self.fps = 0.85 * self.fps + 0.15 * instant_fps if self.fps > 0 else instant_fps

        self.frame_counter += 1

        height, width = frame.shape[:2]
        if width > 640:
            target_width = 640
            target_height = int(height * (640 / width))
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        if self.frame_counter % self.skip_frames == 0 or not self.cached_boxes:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                classes=self.target_classes,
                device=0,  # Forces inference onto NVIDIA RTX 2050 GPU
                verbose=False,
                imgsz=416
            )[0]

            self.cached_boxes = []
            boxes = results.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    self.cached_boxes.append((x1, y1, x2, y2, conf))

            self.cached_count = len(self.cached_boxes)

        # Get density status & status color
        status, color = self.get_density_status(self.cached_count)

        # Draw cached detections on the frame
        for (x1, y1, x2, y2, conf) in self.cached_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"Person {conf:.2f}"
            cv2.putText(
                frame, label, (x1, max(y1 - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        # Overlay Top Status Banner
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        status_text = f"Count: {self.cached_count} | Status: {status} | FPS: {self.fps:.1f}"
        cv2.putText(
            frame, status_text, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

        return frame, self.cached_count, self.fps, status