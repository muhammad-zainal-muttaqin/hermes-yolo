#!/usr/bin/env python3
"""
BREAK_150: Final_Best_Config
STANDARDIZED - Uses STANDARD_CONFIG with specific overrides only
Seed: 42 (reproducible)
"""

import os
import sys
sys.path.insert(0, '/workspace/Hermes-YOLO')

from train_standard_template import run_standard_experiment

if __name__ == "__main__":
    result = run_standard_experiment(
        "BREAK_150",
        "Final_Best_Config",
        imgsz=768,
        epochs=100,
        batch=16,
        mixup=0.1,
        copy_paste=0.3,
        mosaic=1.0,
    )
    print(f"\n🎉 BREAK_150 COMPLETE: mAP50={result['map50']:.4f}")
