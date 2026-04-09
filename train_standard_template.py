#!/usr/bin/env python3
"""
STANDARDIZED TRAINING TEMPLATE
All experiments must use this template for fair comparison
Seed: 42 (reproducible)
"""

import os
os.chdir("/workspace/Hermes-YOLO")

from ultralytics import YOLO
import json
import torch
import random
import numpy as np

def set_seed(seed=42):
    """Set all seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# STANDARD CONFIGURATION (Same for ALL experiments)
STANDARD_CONFIG = {
    # Core
    "epochs": 50,
    "imgsz": 640,
    "batch": 16,
    "workers": 4,
    "device": 0,
    "seed": 42,
    "patience": 15,
    
    # LR & Optimizer
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "cos_lr": True,
    
    # Augmentation (Baseline)
    "augment": True,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    
    # Loss
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    
    # Other
    "close_mosaic": 10,
    "amp": True,
    "deterministic": True,
    "exist_ok": True,
    "plots": False,
    "val": True,
    "verbose": False,
}

def run_standard_experiment(exp_id, name, **overrides):
    """
    Run experiment with standard config + specific overrides
    
    Args:
        exp_id: Experiment ID (e.g., "BREAK_122")
        name: Experiment name
        **overrides: Specific settings that differ from standard
    """
    print(f"🚀 {exp_id}: {name}")
    print("="*60)
    
    # Set seed for reproducibility
    set_seed(42)
    print("✅ Seed 42 set for reproducibility")
    
    # Start with standard config
    config = STANDARD_CONFIG.copy()
    
    # Apply overrides (experiment-specific changes)
    config.update(overrides)
    
    print(f"📊 Configuration:")
    for key, value in overrides.items():
        print(f"   {key}: {value} (OVERRIDE)")
    
    # Load model
    model = YOLO('yolov8n.pt')
    
    # Run training with standard + overrides
    results = model.train(
        data='/workspace/Hermes-YOLO/dataset_absolute.yaml',
        project=f'experiments/runs/{exp_id}',
        name='train',
        **config
    )
    
    # Save results
    metrics = results.results_dict if hasattr(results, 'results_dict') else {}
    result_data = {
        'experiment_id': exp_id,
        'name': name,
        'map50': float(metrics.get('metrics/mAP50(B)', 0)),
        'map75': float(metrics.get('metrics/mAP50-95(B)', 0)),
        'precision': float(metrics.get('metrics/precision(B)', 0)),
        'recall': float(metrics.get('metrics/recall(B)', 0)),
        'seed': 42,
        'config': {k: v for k, v in config.items() if k in overrides or k in ['epochs', 'imgsz', 'batch', 'seed']}
    }
    
    with open(f'experiments/runs/{exp_id}/results.json', 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n✅ {exp_id} COMPLETE!")
    print(f"   mAP50: {result_data['map50']:.4f}")
    print(f"   Recall: {result_data['recall']:.4f}")
    print(f"   Precision: {result_data['precision']:.4f}")
    
    return result_data

if __name__ == "__main__":
    # Example usage
    print("📋 Standardized Training Template")
    print("Use run_standard_experiment() for all experiments")
    print("\nExample:")
    print("  run_standard_experiment('BREAK_122', 'High_Resolution', imgsz=768)")
