import cv2
from flask import Flask, render_template, Response, jsonify
from detector import CrowdDetector

app = Flask(__name__)

# Initialize detector with density thresholds (Warning at 3, Danger at 6)
detector = CrowdDetector(model_path="yolov10n.pt", conf_threshold=0.35, warning_limit=3, danger_limit=6)

current_metrics = {
    "count": 0,
    "fps": 0.0,
    "status": "NORMAL"
}


def generate_frames():
    global current_metrics
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Failed to open video source for Flask streaming.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Process frame
        annotated_frame, count, fps, status = detector.process_frame(frame)

        # Update global metric state
        current_metrics["count"] = count
        current_metrics["fps"] = round(fps, 1)
        current_metrics["status"] = status

        # Compress to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/metrics')
def get_metrics():
    return jsonify(current_metrics)


if __name__ == '__main__':
    print("[INFO] Starting Smart Crowd Monitoring Server on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)