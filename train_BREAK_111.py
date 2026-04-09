#!/usr/bin/env python3
import os
os.chdir("/workspace/Hermes-YOLO")
from ultralytics import YOLO
import json

print(f"🚀 Starting BREAK_111: High_Mosaic")

model = YOLO('yolov8n.pt')
results = model.train(
    data='/workspace/Hermes-YOLO/dataset_absolute.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    workers=8,
    device=0,
    seed=42,
    patience=15,
    project='experiments/runs/BREAK_111',
    name='train',
    exist_ok=True,
    verbose=True,
    optimizer='auto',
    lr0=0.01,
    lrf=0.01,
    mosaic=1.0,
    mixup=0.3,
    copy_paste=0.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    augment=true,
    rect=false
)

metrics = results.results_dict if hasattr(results, 'results_dict') else {}
result_data = {
    'experiment_id': 'BREAK_111',
    'name': 'High_Mosaic',
    'map50': float(metrics.get('metrics/mAP50(B)', 0)),
    'map75': float(metrics.get('metrics/mAP50-95(B)', 0)),
    'precision': float(metrics.get('metrics/precision(B)', 0)),
    'recall': float(metrics.get('metrics/recall(B)', 0))
}

with open(f'experiments/runs/BREAK_111/results.json', 'w') as f:
    json.dump(result_data, f, indent=2)

print(f"✅ BREAK_111 COMPLETE: mAP50={result_data['map50']:.4f}")
