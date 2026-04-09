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
| - | NOVEL_012 | 0.5219 | 0.5979 | 0.4886 | 15 | 768px + LabelSmoothing 0.05 + CosLR (T2-003 variant) |
| - | NOVEL_022 | 0.5316 | 0.5954 | 0.4941 | 33 | Extended Training: NOVEL_005 @ 60 epochs |
| - | NOVEL_021 | 0.5269 | 0.5882 | 0.5081 | 20 | 768px + Label Smoothing 0.15 (Top-2 Combo) |
| - | NOVEL_020 | 0.5250 | 0.6054 | 0.4888 | 19 | 768px + SORD σ=0.5 (Winning Combo) |
| - | NOVEL_016 | 0.5021 | 0.5631 | 0.4861 | 10 | Co-Teaching for Noisy Labels (T3-002) |
| - | NOVEL_015 | 0.5197 | 0.5811 | 0.4957 | 18 | Strong Aug Warmup / SimCLR proxy (T3-004) |
| - | NOVEL_013 | 0.5252 | 0.5970 | 0.4780 | 14 | Pseudo-label SSOD (T2-001 Efficient Teacher proxy) |
| - | NOVEL_011 | 0.5185 | 0.5818 | 0.4813 | 15 | Three-Phase Curriculum Learning |
| - | NOVEL_010 | 0.5076 | 0.5926 | 0.4790 | 15 | SORD sigma=0.5 (tighter ordinal) |
| - | NOVEL_005 | 0.5219 | 0.5979 | 0.4886 | 15 | Higher Resolution 768px |
| - | NOVEL_002 | 0.5003 | 0.5741 | 0.4781 | 15 | L*a*b* Color Space Input |
| - | NOVEL_001 | 0.5185 | 0.5962 | 0.4901 | 15 | Label Smoothing + CosLR |
| - | **NOVEL_005** | **0.5219** | 0.5979 | 0.4886 | 15 | Higher Resolution 768px ⭐ |
| - | NOVEL_001 | 0.5185 | 0.5962 | 0.4901 | 15 | Label Smoothing + CosLR |
| - | NOVEL_010 | 0.5076 | 0.5926 | 0.4790 | 15 | SORD σ=0.5 (tighter ordinal) |
| - | NOVEL_002 | 0.5003 | 0.5741 | 0.4781 | 15 | L*a*b* Color Space Input |
| - | NOVEL_004 | 0.4640 | 0.5802 | 0.4342 | 15 | SORD Ordinal Loss (σ=0.8) |
| - | NOVEL_006 | 0.4640 | 0.5802 | 0.4342 | 15 | SORD + Label Smoothing |
| - | NOVEL_003 | 0.4380 | 0.5245 | 0.4374 | 13 | P2 Detection Head |
| - | NOVEL_008 | 0.3827 | 0.4922 | 0.3896 | 14 | SORD + P2 Head |
| - | NOVEL_007 | 0.3723 | 0.4968 | 0.3704 | 14 | L*a*b* + P2 Head Combo |
| - | NOVEL_009 | 0.3251 | 0.4815 | 0.3282 | 15 | Full Tier 1: LAB+SORD+P2 |

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
| **Current Best** | **0.5316** (NOVEL_022) |
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


## TIER 2-5 Extended Experiments (running...)

| Rank | Experiment | mAP50 | Recall | Precision | Epochs | Strategy | Status |
|:----:|:-----------|:-----:|:------:|:---------:|:------:|:---------|:-------|
| — | NOVEL_011 | — | — | — | 0/15 | Three-Phase Curriculum | 🔄 |
| — | NOVEL_012 | — | — | — | 0/15 | Focal Loss 768px (T2-003) | 🔄 |
| — | NOVEL_013 | — | — | — | 0/15 | Pseudo-label SSOD (T2-001) | 🔄 |
| — | NOVEL_014 | — | — | — | 0/15 | Born Again Networks KD (T2-002) | 🔄 |
| — | NOVEL_015 | — | — | — | 0/20 | Strong Aug Warmup/SimCLR (T3-004) | 🔄 |
| — | NOVEL_016 | — | — | — | 0/15 | Co-Teaching (T3-002) | 🔄 |
| — | NOVEL_017 | — | — | — | 0/15 | Evidential Deep Learning (T4-001) | 🔄 |
| — | NOVEL_018 | — | — | — | 0/15 | Aspect Ratio Aux Loss (T5-002) | 🔄 |
| — | NOVEL_019 | — | — | — | 0/15 | CLIP Soft Labels (T5-003) | 🔄 |
| — | NOVEL_020 | — | — | — | 0/20 | 768px + SORD σ=0.5 combo | 🔄 |
| — | NOVEL_021 | — | — | — | 0/20 | 768px + Label Smoothing combo | 🔄 |
| — | NOVEL_022 | — | — | — | 0/60 | Extended 60-epoch (T1-005) | 🔄 |

---

## NOVEL Series Results (15-epoch scout runs)

| Rank | Experiment | mAP50 | Recall | Precision | Epochs | Strategy | Status |
|:----:|:-----------|:-----:|:------:|:---------:|:------:|:---------|:-------|
| 1 | NOVEL_005 | **0.5219** | 0.5979 | 0.4886 | 15/15 | Higher Resolution 768px | ✅ |
| 2 | NOVEL_001 | 0.5185 | 0.5962 | 0.4901 | 15/15 | Label Smoothing + CosLR | ✅ |
| 3 | NOVEL_010 | **0.5076** | 0.5926 | 0.4790 | 15/15 | SORD σ=0.5 (tighter ordinal) | ✅ |
| 4 | NOVEL_002 | 0.5003 | 0.5741 | 0.4781 | 15/15 | L*a*b* Color Space Input | ✅ |
| 5= | NOVEL_004 | 0.4640 | 0.5802 | 0.4342 | 15/15 | SORD Ordinal Loss (σ=0.8) | ✅ |
| 5= | NOVEL_006 | 0.4640 | 0.5802 | 0.4342 | 15/15 | SORD + Label Smoothing | ✅ |
| 7 | NOVEL_003 | 0.4380 | 0.5245 | 0.4374 | 13/15 | P2 Detection Head | ✅ |
| 8 | NOVEL_008 | 0.3827 | 0.4922 | 0.3896 | 14/15 | SORD + P2 Head Combo | ✅ |
| 9 | NOVEL_007 | 0.3723 | 0.4968 | 0.3704 | 14/15 | L*a*b* + P2 Head Combo | ✅ |
| 10 | NOVEL_009 | 0.3251 | 0.4815 | 0.3282 | 15/15 | Full Tier 1: LAB+SORD+P2 | ✅ |
