#!/usr/bin/env python3
"""
SEQUENTIAL TRAINING PIPELINE
Run multiple experiments one after another
"""

import os
import time
import subprocess

os.chdir("/workspace/Hermes-YOLO")

experiments = [
    {
        'id': 'BREAK_102',
        'name': 'HighRes_1280px',
        'script': 'real_training_102.py'
    },
    {
        'id': 'BREAK_103',
        'name': 'Heavy_Augmentation',
        'epochs': 50,
        'imgsz': 768,
        'mosaic': 1.0,
        'mixup': 0.2,
        'copy_paste': 0.4
    },
    {
        'id': 'BREAK_104',
        'name': 'Progressive_Resizing',
        'epochs': 50,
        'imgsz': 640,  # Start lower, will increase
        'scale_jitter': True
    }
]

print("🚀 SEQUENTIAL TRAINING PIPELINE")
print("="*60)

for i, exp in enumerate(experiments):
    print(f"\n[{i+1}/{len(experiments)}] Preparing {exp['id']}: {exp['name']}...")
    
    # Create training script for each experiment
    script_content = f"""#!/usr/bin/env python3
import os
os.chdir("/workspace/Hermes-YOLO")
from ultralytics import YOLO
import json

print(f"🚀 Starting {exp['id']}: {exp['name']}")

model = YOLO('yolov8n.pt')
results = model.train(
    data='/workspace/Hermes-YOLO/dataset_absolute.yaml',
    epochs={exp.get('epochs', 50)},
    imgsz={exp.get('imgsz', 640)},
    batch={16 if exp.get('imgsz', 640) <= 768 else 8},
    workers=8,
    device=0,
    seed=42,
    patience=15,
    project='experiments/runs/{exp['id']}',
    name='train',
    exist_ok=True,
    verbose=True,
    mosaic={exp.get('mosaic', 1.0)},
    mixup={exp.get('mixup', 0.0)},
    copy_paste={exp.get('copy_paste', 0.0)}
)

metrics = results.results_dict if hasattr(results, 'results_dict') else {{}}
result_data = {{
    'experiment_id': '{exp['id']}',
    'name': '{exp['name']}',
    'map50': float(metrics.get('metrics/mAP50(B)', 0)),
    'map75': float(metrics.get('metrics/mAP50-95(B)', 0)),
    'precision': float(metrics.get('metrics/precision(B)', 0)),
    'recall': float(metrics.get('metrics/recall(B)', 0))
}}

with open(f'experiments/runs/{exp['id']}/results.json', 'w') as f:
    json.dump(result_data, f, indent=2)

print(f"✅ {exp['id']} COMPLETE: mAP50={{result_data['map50']:.4f}}")
"""
    
    script_path = f"train_{exp['id']}.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"   ✅ Script created: {script_path}")

print("\n🎉 All training scripts prepared!")
print("They will be executed after BREAK_101 completes.")
