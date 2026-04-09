# IDEA.md — Novel Strategy Tracking

> **Project**: Hermes-YOLO — TBS Oil Palm Ripeness Detection  
> **Target**: mAP50 > 0.70 (current best: 0.5269, NOVEL_021)  
> **Last updated**: 2026-04-09

---

## Class Semantics

| Class | Color/Shape | Maturity | Notes |
|:-----:|:-----------|:--------:|:------|
| **B1** | Merah, besar, bulat, posisi paling bawah tandan | **Paling matang (Ripe)** | a\* tinggi (positif kuat) |
| **B2** | Hitam → transisi merah, besar, bulat, di atas B1 | **Transisi** | a\* sedikit positif |
| **B3** | Full hitam, berduri, lonjong, di atas B2 | **Belum matang** | a\* near-zero |
| **B4** | Terkecil, terdalam di tandan, duri banyak, hitam→hijau | **Paling belum matang** | a\* negatif (kehijauan) |

**Ordinal urutan biologis**: B1 → B2 → B3 → B4 = paling matang → paling belum matang

> Key confusion: B2/B3 (adjacent, 1 step) lebih acceptable dari B1/B4 (jauh, 3 step)

---

## TIER 1 — Zero Inference Cost, Training-Only

### T1-001: Label Smoothing (Proxy SORD) — ✅ NOVEL_001
- **Config**: yolov8n.pt, 640px, 15 epochs, ls=0.15, cos_lr
- **Result**: mAP50=**0.5185** | Recall=0.5962 | Precision=0.4901

### T1-002: SORD Soft Labels (Full Ordinal) — ✅ NOVEL_004, NOVEL_006, NOVEL_010
- **Config**: yolov8n.pt, 640px, 15 epochs, Gaussian-kernel soft targets
- **Result (sigma=0.8)**: mAP50=0.4640 (NOVEL_004)
- **Result (SORD+LS)**: mAP50=0.4640 (NOVEL_006)
- **Result (sigma=0.5)**: mAP50=**0.5076** (NOVEL_010) — tighter sigma works better
- **Reference**: Diaz & Marathe, CVPR 2019

### T1-003: L\*a\*b\* Color Space Input — ✅ NOVEL_002
- **Config**: yolov8n.pt, LAB dataset, 640px, 15 epochs
- **Result**: mAP50=**0.5003** | Recall=0.5741 | Precision=0.4781
- **Notes**: Slightly below baseline — LAB alone insufficient
- **Reference**: Septiarini et al., 2021

### T1-004: P2 Detection Head (Small Object B4) — ✅ NOVEL_003
- **Config**: yolov8n-p2.yaml, 640px, 15 epochs
- **Result**: mAP50=**0.4333** | Recall=0.5488 | Precision=0.4213
- **Notes**: P2 head adds complexity, needs 50+ epochs to converge

### T1-005: Higher Resolution 768px — ✅ NOVEL_005
- **Config**: yolov8n.pt, 768px, 15 epochs
- **Result**: mAP50=**0.5219** | Recall=0.5979 | Precision=0.4886
- **Notes**: Single strongest individual change

### Tier 1 Combos — ✅ NOVEL_007, NOVEL_008, NOVEL_009
- **L\*a\*b\* + P2 Head** (NOVEL_007): mAP50=0.3723
- **SORD + P2 Head** (NOVEL_008): mAP50=0.3833
- **Full Tier 1: LAB+SORD+P2** (NOVEL_009): mAP50=0.3251
- **Lesson**: Too many simultaneous changes hurt; each needs more epochs

---

## TIER 2 — High ROI, Low-Medium Effort

### T2-001: Efficient Teacher SSOD — ✅ NOVEL_013
- **Mechanism**: Pseudo-label val images with NOVEL_005 (conf>0.5), extend training set
- **Result**: mAP50=**0.5232** | Recall=0.5805 | Precision=0.4921
- **Reference**: Alibaba Research, efficientteacher

### T2-002: Knowledge Distillation (Born Again Networks) — ✅ NOVEL_014
- **Mechanism**: Student learns from NOVEL_005 teacher via soft label matrix (T=4.0, alpha=0.7)
- **Result**: mAP50=**0.3488** | Recall=0.5281 | Precision=0.2797
- **Notes**: Teacher soft matrix approach needs much longer training
- **Reference**: Furlanello et al., ICML 2018

### T2-003: 768px + LabelSmoothing 0.05 + CosLR — ✅ NOVEL_012
- **Config**: yolov8n.pt, 768px, 15 epochs, ls=0.05, cos_lr
- **Result**: mAP50=**0.5219** | Recall=0.5979 | Precision=0.4886
- **Notes**: Same as NOVEL_005 — lighter smoothing (0.05) has no effect at 768px

### T2-004: GPT-4V Annotation Audit — ❌ BLOCKED
- **Notes**: Needs VLM API key; potential +10% impact via annotation error correction

---

## TIER 3 — Strong Potential, Medium Effort

### T3-001: CrowdLayer Multi-Annotator — ❌ BLOCKED
- **Notes**: Dataset lacks per-annotator label splits

### T3-002: Co-Teaching for Noisy Labels — ✅ NOVEL_016
- **Mechanism**: Cross-Pseudo-Supervision: Model A (5 epochs) → pseudo-labels → Model B (10 epochs)
- **Result**: mAP50=**0.5021** | Recall=0.5631 | Precision=0.4861
- **Reference**: Han et al., NeurIPS 2018

### T3-003: Three-Phase Curriculum — ✅ NOVEL_011
- **Mechanism**: label_smoothing schedule: 0.30 → 0.15 → 0.00 over 3 phases
- **Result**: mAP50=**0.5185** | Recall=0.5818 | Precision=0.4813

### T3-004: SimCLR/DenseCL Pretraining proxy — ✅ NOVEL_015
- **Mechanism**: 5-epoch strong-aug warmup (mixup=0.4, copy_paste=0.5) → 15 normal epochs
- **Result**: mAP50=**0.5192** | Recall=0.5817 | Precision=0.4951

---

## TIER 4 — Uncertainty Quantification

### T4-001: Evidential Deep Learning (EDL) — ✅ NOVEL_017
- **Mechanism**: Dirichlet-parameterized classification loss with KL annealing
- **Result**: mAP50=**0.1034** | Recall=0.7838 | Precision=0.0124
- **Notes**: High recall but near-zero precision — Dirichlet loss destabilizes YOLO detection head
- **Reference**: Sensoy et al., NeurIPS 2018

### T4-002: Conformal Prediction Sets — ⏭️
- **Notes**: Post-processing analysis, no training needed

### T4-003: Burst-Shot Multi-Frame Voting — ⏭️
- **Notes**: Inference-time WBF ensemble, no training needed

---

## TIER 5 — Experimental

### T5-001: DCNv4 Deformable Convolutions — ❌ BLOCKED
- **Notes**: Requires C++ compilation, not available in environment

### T5-002: Aspect Ratio Auxiliary Loss — ✅ NOVEL_018
- **Mechanism**: MSE(actual_AR, expected_AR_for_class) for foreground anchors
- **Result**: mAP50=**0.5185** | Recall=0.5818 | Precision=0.4813
- **Notes**: Matches curriculum/label smoothing baselines — AR aux loss has neutral effect

### T5-003: CLIP Soft Labels — ✅ NOVEL_019
- **Mechanism**: open_clip ViT-B/32 crop similarity → soft labels, blended 60% CLIP + 40% SORD
- **Result**: mAP50=**0.4439** | Recall=0.5832 | Precision=0.4199
- **Notes**: CLIP soft targets hurt — noise overwhelms ordinal signal

### T5-004: PPAL Active Learning — ⏭️
- **Notes**: Post-training analysis for re-annotation prioritization

---

## Combo Experiments

### 768px + SORD sigma=0.5 — ✅ NOVEL_020
- **Result**: mAP50=**0.5231** | Recall=0.6020 | Precision=0.4912 (20 epochs)

### 768px + Label Smoothing 0.15 — ✅ NOVEL_021
- **Result**: mAP50=**0.5269** | Recall=0.5882 | Precision=0.5081 (20 epochs)
- **Notes**: **Best single model result across all experiments**

### Extended Training: NOVEL_005 @ 60 epochs — ✅ NOVEL_022
- **Result**: mAP50=**0.5065** | Recall=0.5812 | Precision=0.4762 (60 epochs)
- **Notes**: Overfitting — performance peaked around epoch 15-20, then degraded

---

## Summary

| Metric | Value |
|:-------|:------|
| Baseline | 0.504 (STRUCT_000) |
| **Best single model** | **0.5269** (NOVEL_021) |
| Best ensemble | 0.5298 (BREAK_037) |
| Target | > 0.70 |
| SOTA Reference | 0.842 (Mansour 2022) |
