#!/usr/bin/env python3
"""
BREAK_142: With_CopyPaste
STANDARDIZED - Uses STANDARD_CONFIG with specific overrides only
Seed: 42 (reproducible)
"""

import os
import sys
sys.path.insert(0, '/workspace/Hermes-YOLO')

from train_standard_template import run_standard_experiment

if __name__ == "__main__":
    result = run_standard_experiment(
        "BREAK_142",
        "With_CopyPaste",
        copy_paste=0.3,
    )
    print(f"\n🎉 BREAK_142 COMPLETE: mAP50={result['map50']:.4f}")
