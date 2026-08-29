import sys
import cv2
import flask
from ultralytics import YOLO

print("--- ENVIRONMENT VERIFICATION LOGS ---")
print(f"1. Current Python Version: {sys.version.split()[0]}")
print(f"2. Local OpenCV Version: {cv2.__version__}")
print(f"3. Local Flask Version: {flask.__version__}")

try:
    print("\n4. Initializing Pretrained YOLOv10 Engine...")
    model = YOLO("yolov10n.pt")
    
    print("\nSUCCESS: All components initialized without errors!")
    print(f"Loaded model architecture successfully: {type(model)}")
except Exception as e:
    print(f"\nCRITICAL INGESTION ERROR: {str(e)}")