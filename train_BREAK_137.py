#!/usr/bin/env python3
"""
BREAK_137: Optimizer_AdamW
STANDARDIZED - Uses STANDARD_CONFIG with specific overrides only
Seed: 42 (reproducible)
"""

import os
import sys
sys.path.insert(0, '/workspace/Hermes-YOLO')

from train_standard_template import run_standard_experiment

if __name__ == "__main__":
    result = run_standard_experiment(
        "BREAK_137",
        "Optimizer_AdamW",
        optimizer='AdamW',
        lr0=0.001,
    )
    print(f"\n🎉 BREAK_137 COMPLETE: mAP50={result['map50']:.4f}")
