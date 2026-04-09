# Leaderboard — TBS Ripeness Detection (mAP50)

> **Task**: 4-class oil palm fresh fruit bunch (TBS) ripeness detection  
> **Model base**: YOLOv8n | **Seed**: 42 | **Metric**: mAP50 (B)  
> **Last updated**: 2026-04-09 | **Total experiments**: 153+ (131 BREAK + 22 NOVEL)

---

## NOVEL Series — Final Rankings

| Rank | Experiment | mAP50 | Recall | Precision | Epochs | Strategy |
|:----:|:-----------|:-----:|:------:|:---------:|:------:|:---------|
| 1 | **NOVEL_021** | **0.5269** | 0.5882 | 0.5081 | 20 | 768px + Label Smoothing 0.15 |
| 2 | NOVEL_013 | 0.5232 | 0.5805 | 0.4921 | 15 | Pseudo-label SSOD |
| 3 | NOVEL_020 | 0.5231 | 0.6020 | 0.4912 | 20 | 768px + SORD sigma=0.5 |
| 4 | NOVEL_005 | 0.5219 | 0.5979 | 0.4886 | 15 | Higher Resolution 768px |
| 4 | NOVEL_012 | 0.5219 | 0.5979 | 0.4886 | 15 | 768px + LabelSmoothing 0.05 + CosLR |
| 6 | NOVEL_015 | 0.5192 | 0.5817 | 0.4951 | 20 | Strong Aug Warmup / SimCLR proxy |
| 7 | NOVEL_001 | 0.5185 | 0.5962 | 0.4901 | 15 | Label Smoothing 0.15 + CosLR |
| 7 | NOVEL_011 | 0.5185 | 0.5818 | 0.4813 | 15 | Three-Phase Curriculum |
| 7 | NOVEL_018 | 0.5185 | 0.5818 | 0.4813 | 15 | Aspect Ratio Aux Loss |
| 10 | NOVEL_010 | 0.5076 | 0.5926 | 0.4790 | 15 | SORD sigma=0.5 |
| 11 | NOVEL_022 | 0.5065 | 0.5812 | 0.4762 | 60 | Extended 60-epoch @ 768px |
| 12 | NOVEL_016 | 0.5021 | 0.5631 | 0.4861 | 10 | Co-Teaching (Model B) |
| 13 | NOVEL_002 | 0.5003 | 0.5741 | 0.4781 | 15 | L\*a\*b\* Color Space |
| 14 | NOVEL_006 | 0.4640 | 0.5802 | 0.4342 | 15 | SORD + Label Smoothing |
| 14 | NOVEL_004 | 0.4640 | 0.5802 | 0.4342 | 15 | SORD Ordinal (sigma=0.8) |
| 16 | NOVEL_019 | 0.4439 | 0.5832 | 0.4199 | 15 | CLIP Soft Labels |
| 17 | NOVEL_003 | 0.4333 | 0.5488 | 0.4213 | 15 | P2 Detection Head |
| 18 | NOVEL_008 | 0.3833 | 0.5017 | 0.3833 | 15 | SORD + P2 Head |
| 19 | NOVEL_007 | 0.3723 | 0.4968 | 0.3704 | 14 | L\*a\*b\* + P2 Head |
| 20 | NOVEL_014 | 0.3488 | 0.5281 | 0.2797 | 15 | Born Again Networks KD |
| 21 | NOVEL_009 | 0.3251 | 0.4815 | 0.3282 | 15 | Full Tier 1: LAB+SORD+P2 |
| 22 | NOVEL_017 | 0.1034 | 0.7838 | 0.0124 | 15 | Evidential Deep Learning |

---

## Historical Best (BREAK series, 131 experiments)

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
| **Best single model** | **0.5269** (NOVEL_021) |
| **Best ensemble** | **0.5298** (BREAK_037) |
| Improvement over baseline | **+4.5%** |
| Target | > 0.70 |
| SOTA (Mansour 2022) | 0.842 |

---

## What Works

| Strategy | Best mAP50 | Experiment |
|:---------|:----------:|:-----------|
| 768px + Label Smoothing 0.15 | 0.5269 | NOVEL_021 |
| Pseudo-label SSOD | 0.5232 | NOVEL_013 |
| 768px + SORD sigma=0.5 | 0.5231 | NOVEL_020 |
| Higher resolution (768px) | 0.5219 | NOVEL_005 |
| Three-Phase Curriculum | 0.5185 | NOVEL_011 |
| Top-K Ensemble | 0.5298 | BREAK_037 (historical) |

## What Doesn't Work

| Strategy | mAP50 | Lesson |
|:---------|:-----:|:-------|
| EDL (Evidential Deep Learning) | 0.1034 | Dirichlet loss destabilizes YOLO detection head |
| Full Tier 1 combo (LAB+SORD+P2) | 0.3251 | Too many simultaneous changes |
| Born Again Networks KD | 0.3488 | Teacher soft matrix needs longer training |
| Extended 60-epoch training | 0.5065 | Overfitting — worse than 15-20 epoch runs |
| Architecture swap (P2 head) | 0.4333 | Dataset quality is the constraint |
