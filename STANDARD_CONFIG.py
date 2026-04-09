# 🎯 HERMES-YOLO: STANDARD CONFIGURATION
# Reproducible, Fair Comparison Settings
# Seed: 42 (for reproducibility)
# All experiments must use this base config

STANDARD_CONFIG = {
    # Core Settings (REQUIRED - must be same for all)
    "epochs": 50,              # Standard: 50 epochs
    "imgsz": 640,              # Standard: 640px
    "batch": 16,               # Standard: batch size 16
    "workers": 4,              # Standard: 4 workers (optimized)
    "device": 0,               # GPU device
    "seed": 42,                # CRITICAL: Seed 42 for reproducibility
    
    # Training Settings (REQUIRED - must be same for all)
    "patience": 15,            # Early stopping patience
    "lr0": 0.01,               # Initial learning rate
    "lrf": 0.01,               # Final learning rate factor
    "momentum": 0.937,         # SGD momentum
    "weight_decay": 0.0005,    # Weight decay
    "warmup_epochs": 3.0,      # Warmup epochs
    "warmup_momentum": 0.8,    # Warmup momentum
    "warmup_bias_lr": 0.1,     # Warmup bias LR
    
    # Augmentation (Standard baseline)
    "augment": True,           # Enable augmentation
    "mosaic": 1.0,             # Mosaic augmentation
    "mixup": 0.0,              # Standard: no mixup
    "copy_paste": 0.0,         # Standard: no copy-paste
    "degrees": 0.0,            # No rotation
    "translate": 0.1,          # Translation
    "scale": 0.5,              # Scaling
    "shear": 0.0,              # No shear
    "perspective": 0.0,        # No perspective
    "flipud": 0.0,             # No vertical flip
    "fliplr": 0.5,             # Horizontal flip
    "bgr": 0.0,                # No BGR
    "hsv_h": 0.015,            # HSV hue
    "hsv_s": 0.7,              # HSV saturation
    "hsv_v": 0.4,              # HSV value
    
    # Loss weights (Standard)
    "box": 7.5,                # Box loss weight
    "cls": 0.5,                # Class loss weight
    "dfl": 1.5,                # Distribution focal loss
    
    # Other (Standard)
    "cos_lr": True,            # Cosine LR schedule
    "close_mosaic": 10,        # Close mosaic in last 10 epochs
    "amp": True,               # Automatic Mixed Precision
    "deterministic": True,     # Deterministic mode
    "exist_ok": True,          # Overwrite existing
    "plots": False,            # No plots (save CPU)
    "val": True,               # Validate after training
    "verbose": False,          # Minimal output
}

# Variable Settings (for specific experiments)
# These can change per experiment:
VARIABLE_SETTINGS = {
    "experiment_id": "REQUIRED",
    "name": "REQUIRED",
    "project": "experiments/runs/{experiment_id}",
    # Can vary: optimizer, resolution, augmentation intensity, etc.
}

# Reproducibility Checklist:
# ✅ Seed 42 set
# ✅ Deterministic mode on
# ✅ Workers fixed at 4
# ✅ Standard config documented
# ✅ All experiments use same base
