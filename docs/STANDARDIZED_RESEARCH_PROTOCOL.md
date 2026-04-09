# 🎯 STANDARDIZED RESEARCH PROTOCOL

**Date**: April 9, 2025  
**Purpose**: Ensure fair, reproducible, comparable experiments  
**Seed**: 42 (for all experiments)  

---

## 📋 STANDARD CONFIGURATION (BASELINE)

All experiments use this as the **starting point**. Only specific parameters are changed per experiment.

### Core Settings
| Parameter | Value | Notes |
|:----------|:------|:------|
| `epochs` | 50 | Standard training length |
| `imgsz` | 640 | Standard resolution |
| `batch` | 16 | Standard batch size |
| `workers` | 4 | Optimized for GPU/CPU balance |
| `device` | 0 | GPU device |
| `seed` | **42** | **CRITICAL: Reproducibility** |
| `patience` | 15 | Early stopping |

### Learning Rate & Optimizer
| Parameter | Value | Notes |
|:----------|:------|:------|
| `optimizer` | auto | YOLOv8 default (SGD with momentum) |
| `lr0` | 0.01 | Initial learning rate |
| `lrf` | 0.01 | Final LR factor |
| `momentum` | 0.937 | SGD momentum |
| `weight_decay` | 0.0005 | Regularization |
| `warmup_epochs` | 3.0 | Warmup period |
| `cos_lr` | True | Cosine annealing |

### Augmentation (Standard)
| Parameter | Value | Notes |
|:----------|:------|:------|
| `augment` | True | Enable augmentation |
| `mosaic` | 1.0 | Mosaic enabled (standard) |
| `mixup` | 0.0 | **Disabled by default** |
| `copy_paste` | 0.0 | **Disabled by default** |
| `degrees` | 0.0 | No rotation |
| `translate` | 0.1 | Translation allowed |
| `scale` | 0.5 | Scaling allowed |
| `shear` | 0.0 | No shear |
| `flipud` | 0.0 | No vertical flip |
| `fliplr` | 0.5 | Horizontal flip OK |
| `hsv_h` | 0.015 | Small hue shift |
| `hsv_s` | 0.7 | Saturation OK |
| `hsv_v` | 0.4 | Value shift OK |

### Other Settings
| Parameter | Value | Notes |
|:----------|:------|:------|
| `box` | 7.5 | Box loss weight |
| `cls` | 0.5 | Class loss weight |
| `dfl` | 1.5 | DFL weight |
| `amp` | True | Mixed precision |
| `deterministic` | True | **Reproducibility** |
| `cos_lr` | True | Cosine schedule |
| `close_mosaic` | 10 | Close mosaic last 10 epochs |

---

## 🔬 EXPERIMENT MATRIX

### BREAK_122-126: Resolution Study
| Exp | imgsz | batch | Notes |
|:----|:------|:------|:------|
| BREAK_122 | Baseline | 16 | Standard (640px) |
| BREAK_124 | 320 | 16 | Low resolution |
| BREAK_125 | 480 | 16 | Medium-low |
| BREAK_126 | 640 | 16 | Standard |
| BREAK_127 | 768 | 16 | High resolution |
| BREAK_128 | 960 | 8 | Very high (reduce batch) |

**Variable**: Only `imgsz` (and batch for memory)  
**Question**: Which resolution is optimal?

---

### BREAK_129-131: Batch Size Study
| Exp | batch | Notes |
|:----|:------|:------|
| BREAK_129 | 8 | Small batch |
| BREAK_130 | 16 | Standard |
| BREAK_131 | 32 | Large batch |

**Variable**: Only `batch`  
**Question**: Does batch size affect final accuracy?

---

### BREAK_132-134: Training Duration
| Exp | epochs | patience | Notes |
|:----|:-------|:---------|:------|
| BREAK_132 | 25 | 10 | Quick training |
| BREAK_133 | 50 | 15 | Standard |
| BREAK_134 | 100 | 20 | Extended training |

**Variable**: Only `epochs` and `patience`  
**Question**: Do more epochs help?

---

### BREAK_135-137: Optimizer Study
| Exp | optimizer | lr0 | Notes |
|:----|:----------|:----|:------|
| BREAK_135 | SGD | 0.01 | Standard |
| BREAK_136 | Adam | 0.001 | Alternative |
| BREAK_137 | AdamW | 0.001 | Modern optimizer |

**Variable**: Only `optimizer` and `lr0`  
**Question**: Which optimizer works best?

---

### BREAK_138-144: Augmentation Study
| Exp | mosaic | mixup | copy_paste | augment | Notes |
|:----|:-------|:------|:-----------|:--------|:------|
| BREAK_138 | 0.0 | 0.0 | 0.0 | True | No mosaic |
| BREAK_139 | 1.0 | 0.0 | 0.0 | True | High mosaic |
| BREAK_140 | 1.0 | 0.1 | 0.0 | True | + Mixup |
| BREAK_141 | 1.0 | 0.2 | 0.0 | True | ++ Mixup |
| BREAK_142 | 1.0 | 0.0 | 0.3 | True | + CopyPaste |
| BREAK_143 | 0.0 | 0.0 | 0.0 | False | **No aug** |
| BREAK_144 | 1.0 | 0.0 | 0.0 | True | + More HSV/rotation |

**Variable**: Augmentation parameters  
**Question**: Which augmentation combination is best?

---

### BREAK_145-147: Learning Rate Study
| Exp | lr0 | Notes |
|:----|:----|:------|
| BREAK_145 | 0.01 | Standard |
| BREAK_146 | 0.001 | 10x lower |
| BREAK_147 | 0.0001 | 100x lower |

**Variable**: Only `lr0`  
**Question**: Is default LR optimal?

---

### BREAK_148-150: Combined Best Practices
| Exp | Key Changes | Notes |
|:----|:------------|:------|
| BREAK_148 | 768px, 75ep, +mixup, +copy_paste | Best practice A |
| BREAK_149 | 768px, 100ep, AdamW, lr0=0.001 | Best practice B |
| BREAK_150 | 768px, 100ep, +all best aug | **Final best config** |

**Variable**: Multiple optimized settings  
**Question**: Can we break 0.55 mAP50?

---

## ✅ REPRODUCIBILITY CHECKLIST

- ✅ **Seed 42** set in all experiments
- ✅ **Deterministic mode** enabled
- ✅ **Standard config** as baseline
- ✅ **Only one variable** changed per experiment group
- ✅ **All settings documented** in results.json
- ✅ **Git commits** for every experiment

---

## 📊 COMPARISON RULES

1. **Fair Comparison**: Only change ONE parameter group at a time
2. **Baseline**: BREAK_122 (pure standard config)
3. **Control**: BREAK_123 (seed=43) to test reproducibility
4. **Metric**: mAP50 is primary comparison metric
5. **Significance**: Difference >0.01 considered meaningful

---

## 🚀 EXECUTION ORDER

1. **First**: BREAK_122 (baseline) - establishes reference
2. **Second**: BREAK_123 (seed test) - verifies reproducibility  
3. **Then**: All others in any order (independent)
4. **Finally**: BREAK_150 (best combined config)

---

## 📝 TEMPLATE USAGE

All experiments use `train_standard_template.py`:

```python
from train_standard_template import run_standard_experiment

result = run_standard_experiment(
    "BREAK_XXX",           # Experiment ID
    "Experiment_Name",    # Name
    imgsz=768,            # ONLY changed parameters
    epochs=100,           # Rest uses STANDARD_CONFIG
)
```

---

**Status**: ✅ Standardized protocol established  
**Next**: Execute all 29 experiments sequentially  
**Expected**: Identify best configuration fairly

---

*This ensures all 150+ experiments are comparable and reproducible*
