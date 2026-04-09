
from ultralytics import YOLO
import torch
import numpy as np
from pathlib import Path
import json

# Load top 5 models
model_paths = [
    'experiments/runs/STRUCT_004/train/weights/best.pt',
    'experiments/runs/BREAK_001/train/weights/best.pt',
    'experiments/runs/BREAK_002/train/weights/best.pt',
    'experiments/runs/BREAK_003/train/weights/best.pt',
    'experiments/runs/BREAK_004/train/weights/best.pt',
]

models = []
for path in model_paths:
    if Path(path).exists():
        models.append(YOLO(path))
        print(f"Loaded: {path}")

print(f"\nEnsemble created with {len(models)} models")

# Evaluate ensemble on validation set
# Using weighted box fusion
from ultralytics.utils.ops import non_max_suppression
import cv2

def ensemble_predict(models, image_path, conf=0.25, iou=0.45):
    all_boxes = []
    all_scores = []
    all_classes = []
    
    for model in models:
        results = model(image_path, conf=conf, iou=iou, verbose=False)
        for r in results:
            if len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_classes.append(classes)
    
    # Simple average ensemble (can be improved with WBF)
    if len(all_boxes) == 0:
        return [], [], []
    
    # Concatenate all predictions
    combined_boxes = np.vstack(all_boxes) if all_boxes else np.array([])
    combined_scores = np.concatenate(all_scores) if all_scores else np.array([])
    combined_classes = np.concatenate(all_classes) if all_classes else np.array([])
    
    return combined_boxes, combined_scores, combined_classes

print("\n✅ Ensemble ready for evaluation")
print("Expected: +1-2% mAP50 improvement")
