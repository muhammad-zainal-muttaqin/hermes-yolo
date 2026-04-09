# Leaderboard — TBS Ripeness Detection (mAP50)

> **Task**: 4-class oil palm fresh fruit bunch (TBS) ripeness detection  
> **Model base**: YOLOv11s | **Seed**: 42 | **Metric**: mAP50 (B)  
> **Last updated**: 2026-04-09 | **Total experiments**: 131+

---

## Top Results (Local Runs)

| Rank | Experiment | mAP50 | Recall | Precision | Epochs | Notes |
|:----:|:-----------|:-----:|:------:|:---------:|:------:|:------|
| 🥇 | **BREAK_101** | **0.5250** | 0.5845 | — | 52 | Extended training, best local |
| 🥈 | BREAK_034 | 0.5207 | 0.5961 | — | 20 | Tier-2 run |
| 🥉 | BREAK_035 | 0.5207 | 0.5961 | — | 20 | Tier-2 run |
| 4 | BREAK_036 | 0.5207 | 0.5961 | — | 20 | Tier-2 run |
| 5 | BREAK_005–032 | 0.5025 | 0.5871 | — | 10 | Early batch (28 experiments) |
| - | NOVEL_001 | 0.5185 | 0.5962 | 0.4901 | 15 | Label Smoothing + CosLR |

---

## Historical Best (from research ledger, 131+ experiments)

| Rank | Experiment | mAP50 | Approach |
|:----:|:-----------|:-----:|:---------|
| 1 | BREAK_037 | 0.5298 | Top-5 Ensemble |
| 2 | BREAK_038 | 0.5250 | Test-Time Augmentation |
| 3 | BREAK_101 | 0.5250 | Extended 52-epoch training |
| 4 | STRUCT_004 | 0.5225 | CopyPaste Semantic augmentation |
| 5 | STRUCT_003 | 0.5194 | SmallObjectFPN (768px) |

---

## Progress Summary

| Metric | Value |
|:-------|:------|
| Baseline mAP50 | 0.504 (STRUCT_000) |
| **Current Best** | **0.5298** (BREAK_037, Top-5 Ensemble) |
| Improvement | **+5.1%** from baseline |
| Target | > 0.55 |
| Gap to target | 0.020 |
| SOTA (Mansour 2022) | 0.842 |

---

## What Works

| Strategy | mAP50 | Notes |
|:---------|:-----:|:------|
| Top-K Ensemble | 0.5298 | Best single result |
| Extended training (50+ epochs) | 0.5250 | Diminishing returns after epoch 30 |
| CopyPaste Semantic Aug | 0.5225 | Rare class (B1/B4) augmentation |
| Higher resolution (768px) | 0.5194 | Helps small B4 detection |

## What Doesn't Work

| Strategy | Result | Lesson |
|:---------|:------:|:-------|
| Test-Time Augmentation (TTA) | 0.504 | No gain for this dataset |
| Adaptive Label Smoothing | 0.504 | Not the bottleneck |
| Architecture swap alone | 0.495 | Dataset is the constraint |


## NOVEL Series Results (15-epoch scout runs)

| Rank | Experiment | mAP50 | Recall | Epochs | Strategy |
|:----:|:-----------|:-----:|:------:|:------:|:---------|
| - | NOVEL_001 | 0.5185 | 0.5962 | 15/15 | Label Smoothing + CosLR |
| - | NOVEL_002 | 0.4963 | 0.5736 | 14/14 | L*a*b* Color Space Input |
| - | NOVEL_003 | 0.4380 | 0.5245 | 13/14 | P2 Detection Head |
| - | NOVEL_005 | 0.5177 | 0.5869 | 13/14 | Higher Resolution 768px |
| - | NOVEL_007 | 0.3723 | 0.4968 | 14/14 | L*a*b* + P2 Head |
