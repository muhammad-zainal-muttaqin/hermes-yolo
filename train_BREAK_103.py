#!/usr/bin/env python3
import os
os.chdir("/workspace/Hermes-YOLO")
from ultralytics import YOLO
import json

print(f"🚀 Starting BREAK_103: Heavy_Augmentation")

model = YOLO('yolov8n.pt')
results = model.train(
    data='/workspace/Hermes-YOLO/dataset_absolute.yaml',
    epochs=50,
    imgsz=768,
    batch=16,
    workers=4,  # Optimized: less CPU overhead
    device=0,
    seed=42,
    patience=15,
    project='experiments/runs/BREAK_103',
    name='train',
    exist_ok=True,
    verbose=True,
    mosaic=1.0,
    mixup=0.2,
    copy_paste=0.4
)

metrics = results.results_dict if hasattr(results, 'results_dict') else {}
result_data = {
    'experiment_id': 'BREAK_103',
    'name': 'Heavy_Augmentation',
    'map50': float(metrics.get('metrics/mAP50(B)', 0)),
    'map75': float(metrics.get('metrics/mAP50-95(B)', 0)),
    'precision': float(metrics.get('metrics/precision(B)', 0)),
    'recall': float(metrics.get('metrics/recall(B)', 0))
}

with open(f'experiments/runs/BREAK_103/results.json', 'w') as f:
    json.dump(result_data, f, indent=2)

print(f"✅ BREAK_103 COMPLETE: mAP50={result_data['map50']:.4f}")
