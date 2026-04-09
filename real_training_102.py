#!/usr/bin/env python3
"""
BREAK_102: High Resolution Training (1280px)
Real training with increased resolution for better B4 detection
"""

import os
os.chdir("/workspace/Hermes-YOLO")

from ultralytics import YOLO
import json

print("🚀 BREAK_102: High Resolution Training (1280px)")
print("="*60)

model = YOLO('yolov8n.pt')

print("\nTraining with 1280px resolution...")
results = model.train(
    data='/workspace/Hermes-YOLO/dataset_absolute.yaml',
    epochs=50,  # Fewer epochs due to larger images
    imgsz=1280,
    batch=8,  # Smaller batch for memory
    workers=4,  # Optimized: less CPU overhead
    device=0,
    seed=42,
    patience=15,
    project='experiments/runs/BREAK_102',
    name='train',
    exist_ok=True,
    verbose=True
)

metrics = results.results_dict if hasattr(results, 'results_dict') else {}
result_data = {
    'experiment_id': 'BREAK_102',
    'name': 'HighRes_1280px',
    'map50': float(metrics.get('metrics/mAP50(B)', 0)),
    'map75': float(metrics.get('metrics/mAP50-95(B)', 0)),
    'precision': float(metrics.get('metrics/precision(B)', 0)),
    'recall': float(metrics.get('metrics/recall(B)', 0)),
    'resolution': 1280
}

with open('experiments/runs/BREAK_102/results.json', 'w') as f:
    json.dump(result_data, f, indent=2)

print(f"\n✅ BREAK_102 COMPLETE!")
print(f"   mAP50: {result_data['map50']:.4f}")
print(f"   Recall: {result_data['recall']:.4f}")
