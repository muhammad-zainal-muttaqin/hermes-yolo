#!/usr/bin/env python3
"""
BREAK_144: Max_Augmentation
STANDARDIZED - Uses STANDARD_CONFIG with specific overrides only
Seed: 42 (reproducible)
"""

import os
import sys
sys.path.insert(0, '/workspace/Hermes-YOLO')

from train_standard_template import run_standard_experiment

if __name__ == "__main__":
    result = run_standard_experiment(
        "BREAK_144",
        "Max_Augmentation",
        hsv_h=0.1,
        hsv_s=0.9,
        hsv_v=0.9,
        degrees=10.0,
        shear=5.0,
    )
    print(f"\n🎉 BREAK_144 COMPLETE: mAP50={result['map50']:.4f}")
