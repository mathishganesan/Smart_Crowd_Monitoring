import os
import cv2
import threading
from flask import Flask, render_template, Response, jsonify, request
from detector import CrowdDetector

app = Flask(__name__)

# Initialize custom detector with trained weights
detector = CrowdDetector(
    model_path=r"runs/detect/runs/detect/yolov10_head_monitoring-6/weights/best.pt",
    conf_threshold=0.15,
    warning_limit=5,
    danger_limit=12
)

# Folder containing downloaded clips
VIDEO_DIR = "Sample_videos"

# Global camera and video state tracking
camera = None
current_source = None
video_lock = threading.Lock()
latest_metrics = {"count": 0, "status": "NORMAL", "fps": 0.0, "inference_time_ms": 0.0}

def get_available_videos():
    """Lists video files inside Sample_video folder"""
    if not os.path.exists(VIDEO_DIR):
        return []
    valid_exts = ('.mp4', '.avi', '.mov', '.mkv')
    return [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(valid_exts)]

def set_video_source(source_name):
    """Safely updates the active video capture stream"""
    global camera, current_source
    with video_lock:
        if camera is not None:
            camera.release()

        if source_name == "webcam":
            print("[INFO] Switching feed to: Live Webcam (Index 0)")
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            current_source = "webcam"
        else:
            video_path = os.path.join(VIDEO_DIR, source_name)
            if os.path.exists(video_path):
                print(f"[INFO] Ingesting video stream from: {video_path}")
                camera = cv2.VideoCapture(video_path)
                current_source = source_name
            else:
                print(f"[WARNING] Video {video_path} not found. Defaulting to webcam.")
                camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                current_source = "webcam"

# Auto-select the first available video on startup
initial_videos = get_available_videos()
if initial_videos:
    set_video_source(initial_videos[0])
else:
    set_video_source("webcam")

def generate_frames():
    global latest_metrics, camera
    while True:
        with video_lock:
            if camera is None or not camera.isOpened():
                continue
            success, frame = camera.read()

            # Auto-loop video file upon reaching the end
            if not success and current_source != "webcam":
                camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = camera.read()
                if not success:
                    continue

        if not success or frame is None:
            continue

        # Run custom head detection
        processed_frame, metrics = detector.process_frame(frame)
        latest_metrics = metrics

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    videos = get_available_videos()
    return render_template('index.html', videos=videos, current_video=current_source)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/metrics')
def get_metrics():
    return jsonify(latest_metrics)

@app.route('/api/set_source', methods=['POST'])
def update_source():
    data = request.get_json()
    source = data.get('source', 'webcam')
    set_video_source(source)
    return jsonify({"status": "success", "active_source": source})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)