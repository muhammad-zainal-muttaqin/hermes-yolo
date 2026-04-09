#!/usr/bin/env python3
"""
BREAK_132: Epochs_25
STANDARDIZED - Uses STANDARD_CONFIG with specific overrides only
Seed: 42 (reproducible)
"""

import os
import sys
sys.path.insert(0, '/workspace/Hermes-YOLO')

from train_standard_template import run_standard_experiment

if __name__ == "__main__":
    result = run_standard_experiment(
        "BREAK_132",
        "Epochs_25",
        epochs=25,
    )
    print(f"\n🎉 BREAK_132 COMPLETE: mAP50={result['map50']:.4f}")
