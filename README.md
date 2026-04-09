# Hermes-YOLO

> Autonomous hyperparameter & strategy search for TBS (Tandan Buah Segar) oil palm ripeness detection using YOLO.  
> Inspired by [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

---

## Problem

Detect 4 ripeness classes of oil palm fresh fruit bunches (TBS):

| Class | Color / Shape | Biological Position | Maturity |
|:-----:|:-------------|:--------------------|:---------|
| **B1** | Merah, besar, **bulat**, posisi **paling bawah** tandan | Outermost fruitlets | **Paling matang (Ripe)** |
| **B2** | Hitam → transisi merah, besar, bulat, di atas B1 | Mid-outer fruitlets | **Transisi (Semi-ripe)** |
| **B3** | Full hitam, **berduri**, **lonjong**, di atas B2 | Mid-inner fruitlets | **Belum matang (Unripe)** |
| **B4** | Terkecil, **terdalam** dalam tandan, duri banyak, hitam → hijau | Innermost fruitlets | **Paling belum matang (Least ripe)** |

**Ordinal biological ordering**: B1 → B2 → B3 → B4 = most ripe → least ripe

**Core challenges**:
- **B2/B3 confusion** — adjacent in the ripening sequence, visually similar color/texture at class boundary; misclassification by 1 step is biologically acceptable
- **B1/B4 rare classes** — B1/B4 are underrepresented in the dataset; B4 fruitlets are very small and deeply embedded
- **B1↔B4 errors are costly** — skipping 3 ordinal steps; should be penalized more heavily than B2↔B3

> Color physics insight: the CIE L\*a\*b\* **a\* channel** (green–red axis) physically separates B1 (a\* strongly positive, red) from B3 (a\* near zero, black) from B4 (a\* negative, green-tinted), providing a natural discriminative feature.

---

## Results

![Progress Chart](experiments/visualizations/progress_map50.png)

| Metric | Value |
|:-------|:------|
| Baseline mAP50 | 0.504 (STRUCT_000) |
| **Best single model** | **0.5269** (NOVEL_021, 768px + Label Smoothing 0.15) |
| **Best ensemble** | **0.5298** (BREAK_037, Top-5 Ensemble) |
| Total experiments | **153+** (131 BREAK + 22 NOVEL) |
| Improvement over baseline | **+4.5%** |
| Target | > 0.70 |
| SOTA reference | 0.842 (Mansour 2022) |

See [LEADERBOARD.md](LEADERBOARD.md) for full rankings.

---

## NOVEL Series — Top 10

| Rank | Experiment | mAP50 | Strategy | Epochs |
|:----:|:-----------|:-----:|:---------|:------:|
| 1 | **NOVEL_021** | **0.5269** | 768px + Label Smoothing 0.15 | 20 |
| 2 | NOVEL_013 | 0.5232 | Pseudo-label SSOD | 15 |
| 3 | NOVEL_020 | 0.5231 | 768px + SORD sigma=0.5 | 20 |
| 4 | NOVEL_005 | 0.5219 | Higher Resolution 768px | 15 |
| 4 | NOVEL_012 | 0.5219 | 768px + LabelSmoothing 0.05 + CosLR | 15 |
| 6 | NOVEL_015 | 0.5192 | Strong Aug Warmup / SimCLR proxy | 20 |
| 7 | NOVEL_001 | 0.5185 | Label Smoothing 0.15 + CosLR | 15 |
| 7 | NOVEL_011 | 0.5185 | Three-Phase Curriculum | 15 |
| 7 | NOVEL_018 | 0.5185 | Aspect Ratio Aux Loss | 15 |
| 10 | NOVEL_010 | 0.5076 | SORD sigma=0.5 | 15 |

---

## Key Findings

**What works:**

| Strategy | Best mAP50 | Notes |
|:---------|:----------:|:------|
| 768px + Label Smoothing 0.15 | 0.5269 | Best single model (NOVEL_021) |
| Pseudo-label SSOD | 0.5232 | NOVEL_005 as teacher (NOVEL_013) |
| 768px + SORD sigma=0.5 | 0.5231 | Ordinal + resolution combo (NOVEL_020) |
| Higher resolution (768px) | 0.5219 | Single strongest individual change (NOVEL_005) |
| Top-K Ensemble | 0.5298 | Historical best (BREAK_037) |

**What doesn't work:**

| Strategy | mAP50 | Lesson |
|:---------|:-----:|:-------|
| EDL (Evidential Deep Learning) | 0.1034 | Dirichlet loss destabilizes YOLO detection head |
| Full combo all-at-once (NOVEL_009) | 0.3251 | Too many simultaneous changes |
| Born Again Networks KD | 0.3488 | Soft matrix needs much longer training |
| Extended 60-epoch training | 0.5065 | Overfitting — worse than 15-20 epoch runs |
| P2 Detection Head | 0.4333 | Adds complexity; dataset is the constraint |

---

## Experiment Tiers (NOVEL Series)

| Tier | Focus | Experiments | Status |
|:-----|:------|:-----------|:-------|
| **TIER 1** | Zero inference cost, training-only | NOVEL_001–010 | ✅ Done |
| **TIER 2** | High ROI, low-medium effort | NOVEL_011–014 | ✅ Done |
| **TIER 3** | Strong potential, medium effort | NOVEL_015–016 | ✅ Done |
| **TIER 4** | Uncertainty quantification | NOVEL_017 | ✅ Done |
| **TIER 5** | Experimental | NOVEL_018–019 | ✅ Done |
| **Combos** | Best strategies combined | NOVEL_020–022 | ✅ Done |

See [IDEA.md](IDEA.md) for full strategy descriptions and per-experiment analysis.

---

## Training Setup

| Component | Detail |
|:----------|:-------|
| **Model** | YOLOv8n (Nano) — `yolov8n.pt` (6MB) |
| **Framework** | Ultralytics 8.4.35 + PyTorch 2.4.1 |
| **GPU** | NVIDIA RTX A4500 (20GB VRAM) |
| **CUDA** | 12.4 |
| **Platform** | RunPod cloud instance |

### Default Training Config

```yaml
model: yolov8n.pt
imgsz: 640          # or 768 for resolution experiments
batch: 16           # or 8 for 768px (VRAM constraint)
epochs: 15          # scout runs; 20 for combos, 60 for extended
seed: 42
optimizer: auto     # AdamW (Ultralytics default)
lr0: 0.01
lrf: 0.01           # final LR factor
cos_lr: false       # true for label smoothing experiments
label_smoothing: 0.0
```

### Pre-trained Weights

Top model weights are saved in [`weights/`](weights/) for direct inference:

```bash
# Run inference with best model
yolo detect predict model=weights/NOVEL_021_best.pt source=your_image.jpg

# Validate on test set
yolo detect val model=weights/NOVEL_021_best.pt data=dataset_novel.yaml split=test
```

| Weight File | mAP50 | Strategy |
|:------------|:-----:|:---------|
| `NOVEL_021_best.pt` | 0.5269 | 768px + Label Smoothing 0.15 |
| `NOVEL_013_best.pt` | 0.5232 | Pseudo-label SSOD |
| `NOVEL_020_best.pt` | 0.5231 | 768px + SORD sigma=0.5 |
| `NOVEL_005_best.pt` | 0.5219 | Higher Resolution 768px |
| `NOVEL_015_best.pt` | 0.5192 | SimCLR proxy |
| `BREAK_101_best.pt` | 0.5250 | Extended 52-epoch baseline |

---

## Repository Structure

```
Hermes-YOLO/
├── novel_runner.py              # Experiment runner (TIER 1-5, all 22 experiments)
├── dataset_novel.yaml           # Dataset config (RGB, 2764 train / 604 val / 624 test)
├── dataset_novel_lab.yaml       # Dataset config (L*a*b* color space variant)
├── weights/                     # Top model weights (best.pt) for inference
├── IDEA.md                      # Strategy tracker — all TIER 1-5 ideas + results
├── LEADERBOARD.md               # Full experiment rankings
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── experiments/
│   └── visualizations/          # progress_map50.png — auto-updated after each run
└── runs/detect/experiments/
    └── experiments/runs/
        └── NOVEL_*/             # Per-experiment results.csv, args.yaml
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Run all NOVEL experiments (TIER 1-5), sequentially
python novel_runner.py
```

---

## Dataset

- **4 classes**: B1 (ripe), B2 (transitioning), B3 (unripe), B4 (least ripe)
- **Split**: 2,764 train / 604 val / 624 test images
- **Sources**: DAMIMAS + LONSUM oil palm plantations
- **Class imbalance**: B1/B4 underrepresented; B2/B3 dominant
- **L\*a\*b\* variant**: pre-converted dataset for color-space experiments (NOVEL_002)

---

## Reproducibility

All experiments use `seed=42`.

```bash
# Check a specific experiment config
cat runs/detect/experiments/experiments/runs/NOVEL_021/args.yaml

# Re-run the best single-model experiment
python -c "
from novel_runner import EXPERIMENTS, run_experiment
cfg = next(e for e in EXPERIMENTS if e['id'] == 'NOVEL_021')
run_experiment(cfg)
"
```

---

## References

- Díaz & Marathe, *Soft Labels for Ordinal Regression* (CVPR 2019) — SORD loss
- Furlanello et al., *Born Again Networks* (ICML 2018) — Knowledge Distillation
- Han et al., *Co-Teaching* (NeurIPS 2018) — Noisy label training
- Sensoy et al., *Evidential Deep Learning* (NeurIPS 2018) — EDL uncertainty
- Lin et al., *Focal Loss* (ICCV 2017) — Hard-example mining
- Mansour et al., 2022 — SOTA 0.842 mAP on TBS detection
- Septiarini et al., 2021 — L\*a\*b\* color for oil palm ripeness (98.3% accuracy)
