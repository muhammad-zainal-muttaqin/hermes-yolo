#!/usr/bin/env python3
"""
BREAK_143: No_Augmentation
STANDARDIZED - Uses STANDARD_CONFIG with specific overrides only
Seed: 42 (reproducible)
"""

import os
import sys
sys.path.insert(0, '/workspace/Hermes-YOLO')

from train_standard_template import run_standard_experiment

if __name__ == "__main__":
    result = run_standard_experiment(
        "BREAK_143",
        "No_Augmentation",
        augment=False,
        mosaic=0.0,
    )
    print(f"\n🎉 BREAK_143 COMPLETE: mAP50={result['map50']:.4f}")
