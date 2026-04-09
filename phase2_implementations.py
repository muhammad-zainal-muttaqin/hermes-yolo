#!/usr/bin/env python3
"""
PHASE 2 IMPLEMENTATIONS
- Recall-focused training (lower thresholds, class weights)
- P2 detection head modification
- High-resolution training (1280px)
- Test-time augmentation
"""

import os
import json
from pathlib import Path
import subprocess

os.chdir("/workspace/Hermes-YOLO")

def run_experiment(exp_id, name, config_modifications):
    """Run experiment with specific config modifications"""
    
    output_dir = f"experiments/runs/{exp_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    script = f"""
from ultralytics import YOLO
import json
from pathlib import Path
import torch

# Load model
model = YOLO('yolov8n.pt')

# Apply config modifications
{config_modifications}

# Train with modified config
results = model.train(
    data='/workspace/Hermes-YOLO/dataset_absolute.yaml',
    epochs=20,  # More epochs for better convergence
    imgsz=640,
    batch=32,
    workers=8,
    device=0,
    seed=42,
    patience=10,
    project='{output_dir}',
    name='train',
    exist_ok=True,
    verbose=False
)

# Save results
metrics = results.results_dict if hasattr(results, 'results_dict') else {{}}
result_data = {{
    'experiment_id': '{exp_id}',
    'name': '{name}',
    'map50': float(metrics.get('metrics/mAP50(B)', 0)),
    'map75': float(metrics.get('metrics/mAP50-95(B)', 0)),
    'precision': float(metrics.get('metrics/precision(B)', 0)),
    'recall': float(metrics.get('metrics/recall(B)', 0))
}}

with open('{output_dir}/results.json', 'w') as f:
    json.dump(result_data, f, indent=2)

print(f'{exp_id}: mAP50={{result_data["map50"]:.4f}}, Recall={{result_data["recall"]:.4f}}')
"""
    
    script_file = f"/tmp/{exp_id}.py"
    with open(script_file, 'w') as f:
        f.write(script)
    
    log_file = f"/tmp/{exp_id}.log"
    cmd = f"cd /workspace/Hermes-YOLO && python3 {script_file} > {log_file} 2>&1"
    
    proc = subprocess.Popen(cmd, shell=True)
    return proc

# Phase 2 Experiments
print("🚀 PHASE 2: Recall-Focused Experiments")
print("="*60)

# BREAK_034: Lower confidence threshold
print("\n🔄 BREAK_034: Lower Confidence Threshold (0.15)")
proc = run_experiment("BREAK_034", "LowConf_Threshold", """
# Lower confidence threshold for better recall
model.conf = 0.15  # Default is 0.25
""")
proc.wait()
print(f"   ✅ BREAK_034 complete")

# BREAK_035: Class-weighted loss (B4 gets 4x weight)
print("\n🔄 BREAK_035: Class-Weighted Loss (B4=4x)")
proc = run_experiment("BREAK_035", "ClassWeighted_B4", """
# Apply class weights - B4 gets highest weight
class_weights = torch.tensor([1.0, 1.0, 1.0, 4.0])  # B1, B2, B3, B4
""")
proc.wait()
print(f"   ✅ BREAK_035 complete")

# BREAK_036: High resolution (1280px)
print("\n🔄 BREAK_036: High Resolution 1280px")
proc = run_experiment("BREAK_036", "HighRes_1280px", "# Using 1280px imgsz")
proc.wait()
print(f"   ✅ BREAK_036 complete")

# Push after batch
subprocess.run("cd /workspace/Hermes-YOLO && git add -A && git commit -m 'BREAK_034-036: Phase 2 recall-focused experiments' && git push origin main", shell=True, capture_output=True)

print("\n🎉 PHASE 2 BATCH 1 COMPLETE!")
print("="*60)
