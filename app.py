from flask import Flask, render_template, Response, jsonify
import cv2
from detector import CrowdDetector

app = Flask(__name__)

# Initialize custom detector
detector = CrowdDetector(
    model_path=r"runs/detect/runs/detect/yolov10_head_monitoring-6/weights/best.pt",
    conf_threshold=0.35,
    warning_limit=3,
    danger_limit=6
)

# Open camera with DirectShow backend for Windows stability
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

latest_metrics = {"count": 0, "status": "NORMAL", "fps": 0.0, "inference_time_ms": 0.0}

def generate_frames():
    global latest_metrics
    while True:
        success, frame = camera.read()
        if not success:
            # Fallback if camera index 0 fails: try reopening or yield dummy
            continue

        # Process with custom YOLOv10 detector
        processed_frame, metrics = detector.process_frame(frame)
        latest_metrics = metrics

        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()

        # Yield multipart stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/metrics')
def get_metrics():
    return jsonify(latest_metrics)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)