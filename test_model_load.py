"""
Quick test to verify YOLO model loads correctly with PyTorch 2.8
"""

import torch
print(f"PyTorch version: {torch.__version__}")

# Fix PyTorch 2.8 weights_only issue
from ultralytics.nn.tasks import DetectionModel
torch.serialization.add_safe_globals([DetectionModel])

# Patch torch.load
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

# Try loading model
print("\nAttempting to load YOLOv8 model...")
from ultralytics import YOLO

model_path = "runs/detect/acne_yolov8_production/weights/best.pt"
model = YOLO(model_path)

print("✅ Model loaded successfully!")
print(f"Model classes: {model.names}")
print(f"Model task: {model.task}")
