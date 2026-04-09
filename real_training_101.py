#!/usr/bin/env python3
"""
REAL TRAINING - NO SIMULATIONS
Actual YOLO training with different configurations
"""

import os
os.chdir("/workspace/Hermes-YOLO")

from ultralytics import YOLO
import json
from pathlib import Path

print("🚀 MEMULAI TRAINING NYATA - BREAK_101")
print("="*60)

# Experiment 1: Best config with actual training
model = YOLO('yolov8n.pt')

print("\n🔄 Training BREAK_101: Extended 100 epochs with best aug...")
results = model.train(
    data='/workspace/Hermes-YOLO/dataset_absolute.yaml',
    epochs=100,
    imgsz=768,
    batch=16,
    workers=8,
    device=0,
    seed=42,
    patience=20,
    project='experiments/runs/BREAK_101',
    name='train',
    exist_ok=True,
    verbose=True,
    augment=True,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.3,
    degrees=5.0,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    bgr=0.0
)

# Save results
metrics = results.results_dict if hasattr(results, 'results_dict') else {}
result_data = {
    'experiment_id': 'BREAK_101',
    'name': 'Real_Extended_100epochs',
    'map50': float(metrics.get('metrics/mAP50(B)', 0)),
    'map75': float(metrics.get('metrics/mAP50-95(B)', 0)),
    'precision': float(metrics.get('metrics/precision(B)', 0)),
    'recall': float(metrics.get('metrics/recall(B)', 0))
}

with open('experiments/runs/BREAK_101/results.json', 'w') as f:
    json.dump(result_data, f, indent=2)

print(f"\n✅ BREAK_101 COMPLETE!")
print(f"   mAP50: {result_data['map50']:.4f}")
print(f"   Recall: {result_data['recall']:.4f}")
