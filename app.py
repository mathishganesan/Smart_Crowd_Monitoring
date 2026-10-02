import os
import cv2
import time
import threading
import io
import csv
from flask import Flask, render_template, Response, jsonify, request
from detector import CrowdDetector
from database.database import (
    init_db,
    register_camera_source,
    log_crowd_record,
    log_alert_event,
    fetch_recent_logs
)

app = Flask(__name__)

# Initialize SQLite database schema
init_db()

# Resolve correct model weights path with safety fallback
primary_weights = r"runs/detect/yolov10_scut_ab_combined/weights/best.pt"
fallback_weights = r"runs/detect/runs/detect/yolov10_head_monitoring-6/weights/best.pt"

selected_model_path = primary_weights if os.path.exists(primary_weights) else fallback_weights
print(f"[INFO] Initializing CrowdDetector with model weights: {selected_model_path}")

# Initialize custom detector with fine-tuned weights
detector = CrowdDetector(model_path=selected_model_path)
detector.conf = 0.15
detector.warning_limit = 5
detector.danger_limit = 12

# Folder containing downloaded clips
VIDEO_DIR = "Sample_videos"

# Global camera, video state, and telemetry tracking
camera = None
current_source = None
video_lock = threading.Lock()
latest_metrics = {"count": 0, "status": "NORMAL", "fps": 0.0, "inference_time_ms": 0.0}

# Telemetry logging state
last_db_log_time = 0.0
last_logged_status = "NORMAL"

def get_available_videos():
    """Lists video files inside Sample_videos folder."""
    if not os.path.exists(VIDEO_DIR):
        return []
    valid_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v')
    return [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(valid_exts)]

def set_video_source(source_name):
    """Safely updates the active video capture stream and registers source in DB."""
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

        # Index the active camera/source in the database
        register_camera_source(current_source)

# Auto-select the first available video on startup
initial_videos = get_available_videos()
if initial_videos:
    set_video_source(initial_videos[0])
else:
    set_video_source("webcam")

def generate_frames():
    global latest_metrics, camera, current_source, last_db_log_time, last_logged_status
    while True:
        frame = None
        is_webcam = False
        active_source = "unknown"

        # 1. Fetch frame inside lock and release immediately
        with video_lock:
            if camera is None or not camera.isOpened():
                time.sleep(0.01)
                continue
            success, frame = camera.read()

            if not success and current_source != "webcam":
                camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = camera.read()
                if not success:
                    time.sleep(0.01)
                    continue

            if not success or frame is None:
                time.sleep(0.01)
                continue

            is_webcam = (current_source == "webcam")
            active_source = current_source if current_source else "unknown"

        # 2. Run inference OUTSIDE video_lock (prevents pipeline starvation)
        processed_frame, metrics = detector.process_frame(frame, is_webcam=is_webcam)
        latest_metrics = metrics

        # 3. Periodic SQLite logging directly every 2 seconds
        curr_t = time.time()
        if curr_t - last_db_log_time >= 2.0:
            last_db_log_time = curr_t
            cnt = metrics.get("count", 0)
            st = metrics.get("status", "NORMAL")
            fps = metrics.get("fps", 0.0)
            lat = metrics.get("inference_time_ms", 0.0)

            # Insert time-series crowd telemetry
            log_crowd_record(active_source, cnt, st, fps, lat)

            # Log incident alert on risk status escalation
            if st in ("WARNING", "DANGER") and st != last_logged_status:
                log_alert_event(active_source, st, cnt, f"Crowd density threshold reached: {cnt} people detected.")
                last_logged_status = st
            elif st == "NORMAL":
                last_logged_status = "NORMAL"

        # 4. Stream frame as JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

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
    data = request.get_json() or {}
    source = data.get('source', 'webcam')
    set_video_source(source)
    return jsonify({"status": "success", "active_source": source})

@app.route('/api/export_logs')
def export_logs():
    try:
        rows = fetch_recent_logs(limit=500)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Timestamp', 'Source Name', 'Head Count', 'Density Status', 'FPS', 'Latency (ms)'])

        if rows:
            writer.writerows(rows)

        csv_data = output.getvalue()

        return Response(
            csv_data,
            mimetype="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=crowd_telemetry_logs.csv",
                "Content-Type": "text/csv; charset=utf-8"
            }
        )
    except Exception as e:
        print(f"[ERROR] Export logs route failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)