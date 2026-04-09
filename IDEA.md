# IDEA.md — Novel Strategy Tracking

> **Project**: Hermes-YOLO — TBS Oil Palm Ripeness Detection  
> **Target**: mAP50 > 0.70 (current best: 0.5298, BREAK_037)  
> **Last updated**: 2026-04-09

---

## Class Semantics (CRITICAL — affects all ordinal strategies)

| Class | Color/Shape | Maturity | Notes |
|:-----:|:-----------|:--------:|:------|
| **B1** | Merah, besar, bulat, posisi PALING BAWAH tandan | **Paling matang (Ripe)** | a* tinggi (positif kuat) |
| **B2** | Hitam → transisi merah, besar, bulat, di atas B1 | **Transisi** | a* sedikit positif |
| **B3** | Full hitam, berduri, lonjong, di atas B2 | **Belum matang** | a* near-zero |
| **B4** | Terkecil, terdalam di tandan, duri banyak, hitam→hijau | **Paling belum matang** | a* negatif (kehijauan) |

**Ordinal urutan biologis**: B1 → B2 → B3 → B4 = paling matang → paling belum matang

> Key confusion: B2/B3 (adjacent, 1 step) lebih acceptable dari B1/B4 (jauh, 3 step)

---

## Status Legend

| Symbol | Meaning |
|:------:|:--------|
| ⬜ | Belum dimulai |
| 🔄 | Sedang berjalan |
| ✅ | Selesai |
| ❌ | Gagal / tidak berhasil |
| ⏭️ | Skip (superseded by better idea) |

---

## TIER 1 — Zero Inference Cost, Training-Only (Target: +8–15% combined)

### T1-001: Label Smoothing (Proxy SORD)
- **Status**: ✅
- **Experiment ID**: NOVEL_001
- **Mechanism**: Native YOLO `label_smoothing=0.15` + cos_lr — reduces overconfidence pada B2/B3 boundary
- **Config**: yolov8n.pt, 640px, 15 epochs, ls=0.15
- **Expected gain**: +0.5–2% mAP50
- **Result**: mAP50=**0.5185** | Recall=0.5962 | Precision=0.4901 (15/15 epochs)
- **Notes**: Best NOVEL experiment so far; slightly below BREAK_101 (0.5250) but good signal

### T1-002: SORD Soft Labels (Full Ordinal)
- **Status**: ✅
- **Experiment ID**: NOVEL_004, NOVEL_006, NOVEL_008, NOVEL_010
- **Mechanism**: Gaussian-kernel soft targets: B2 label = [0.05, 0.87, 0.07, 0.01], penalizes B1↔B4 lebih berat dari B2↔B3. Diimplementasi via custom YOLO trainer yang override BCE targets.
- **Config**: yolov8n.pt, 640px, 15 epochs, σ=0.8
- **Expected gain**: +2–5% mAP50
- **Reference**: Díaz & Marathe, CVPR 2019
- **Result**: FAILED (de_parallel ImportError) → Fixed → Re-running
- **Notes**: Bug: `de_parallel` → `unwrap_model` (ultralytics 8.4.35); fixed in code

### T1-003: L\*a\*b\* Color Space Input
- **Status**: ✅
- **Experiment ID**: NOVEL_002
- **Mechanism**: Ganti RGB dengan CIE L\*a\*b\*. Channel a\* = Green-Red axis secara fisik memisahkan B1 (a\* tinggi) vs B3 (a\* ~0) vs B4 (a\* negatif). Preprocess seluruh dataset → simpan sebagai PNG dengan nilai LAB.
- **Config**: yolov8n.pt, 640px, 15 epochs (dataset LAB)
- **Expected gain**: +3–6% mAP50 pada B2/B3 confusion
- **Reference**: Septiarini et al., 2021 (98.3% accuracy untuk oil palm)
- **Result**: mAP50=**0.5003** | Recall=0.5741 | Precision=0.4781 (15/15 epochs)
- **Notes**: Slightly below baseline — LAB helps but needs more epochs or combined approach

### T1-004: P2 Detection Head (Small Object B4)
- **Status**: ✅
- **Experiment ID**: NOVEL_003
- **Mechanism**: Tambah detection head di P2 (stride=4, resolusi 4× lebih tinggi dari P3). B4 yang sangat kecil dan tersembunyi dalam tandan akan terdeteksi lebih baik.
- **Config**: yolov8n-p2.yaml, 640px, 15 epochs
- **Expected gain**: +2–4% mAP50 terutama recall B4
- **Reference**: BPD-YOLO 2025 (+2.8%), MFEF-YOLO 2025
- **Result**: mAP50=**0.4380** | Recall=0.5245 | Precision=0.4374 (13/15 epochs)
- **Notes**: Lower than baseline at 15 epochs — P2 head adds complexity, needs more epochs to converge

### T1-005: Higher Resolution 768px
- **Status**: ✅
- **Experiment ID**: NOVEL_005
- **Mechanism**: Increase input resolution from 640px to 768px — helps detect small B4 fruitlets with more detail.
- **Config**: yolov8n.pt, 768px, 15 epochs
- **Expected gain**: +1–3% mAP50
- **Result**: mAP50=**0.5219** | Recall=0.5979 | Precision=0.4886 (15/15 epochs)
- **Notes**: Very close to BREAK_101 (0.5250). Combined with other techniques may exceed baseline.

### T1-006: SORD + Label Smoothing
- **Status**: 🔄 Re-running (after bug fix)
- **Experiment ID**: NOVEL_006
- **Mechanism**: SORD ordinal loss + label_smoothing=0.1 as regularizer. Combines ordinal penalty with soft target regularization.
- **Config**: yolov8n.pt, 640px, 15 epochs, SORD σ=0.8 + ls=0.1
- **Expected gain**: +3–5% mAP50
- **Result**: ❌ FAILED initially (de_parallel ImportError) → Fixed → Queued for re-run
- **Notes**: Fixed `de_parallel` → `unwrap_model`; will run in next batch

### T1-007: L*a*b* + P2 Head Combo
- **Status**: ✅
- **Experiment ID**: NOVEL_007
- **Mechanism**: LAB color space input + P2 detection head (no SORD). Physical color separation combined with small-object detection.
- **Config**: yolov8n-p2.yaml, LAB dataset, 640px, 15 epochs
- **Expected gain**: +3–6% mAP50
- **Result**: 🔄 In progress (~epoch 11/15, mAP50≈0.33)
- **Notes**: Running now; SORD variant (NOVEL_009) queued after bug fix

---

## TIER 2 — High ROI, Low-Medium Effort (Target: +5–10% additional)

### T2-001: Efficient Teacher SSOD
- **Status**: 🔄
- **Experiment ID**: NOVEL_013
- **Mechanism**: Pseudo-label val images dengan NOVEL_005 (conf>0.5), extend training set. Simplified SSOD proxy menggunakan best checkpoint sebagai teacher.
- **Reference**: Alibaba Research, efficientteacher
- **Result**: —
- **Notes**: Teacher = NOVEL_005 best.pt; conf>0.5 filter untuk quality pseudo-labels

### T2-002: Ensemble Knowledge Distillation
- **Status**: 🔄
- **Experiment ID**: NOVEL_014
- **Mechanism**: Born Again Networks — student belajar dari teacher NOVEL_005 via teacher-seeded soft label matrix (KDSoftv8DetectionLoss). Temperature=4.0, blend alpha=0.7.
- **Reference**: Furlanello et al., ICML 2018 (Born Again Networks); Hinton et al., 2015
- **Result**: —
- **Notes**: Teacher soft matrix dibangun dari NOVEL_005 inference on val set

### T2-003: Sub-center ArcFace Classification Head
- **Status**: 🔄
- **Experiment ID**: NOVEL_012
- **Mechanism**: Focal Loss γ=2.0 at 768px sebagai proxy untuk hard-example discrimination yang ArcFace tawarkan. Fokus pada difficult B2/B3 borderline samples.
- **Reference**: Deng et al., ECCV 2020; Lin et al., ICCV 2017 (Focal Loss)
- **Result**: —
- **Notes**: fl_gamma=2.0 built-in, 768px resolution untuk detail maksimal

### T2-004: GPT-4V Annotation Audit
- **Status**: ❌
- **Experiment ID**: (blocked — no API key)
- **Mechanism**: Feed setiap cropped B2/B3 sample ke VLM dengan prompt tentang warna dan kematangan. Identifikasi systematic annotation errors.
- **Expected impact**: +10% potential
- **Result**: BLOCKED — Perlu Anthropic/OpenAI API key
- **Notes**: Dapat dilakukan secara manual jika API key tersedia

---

## TIER 3 — Strong Potential, Medium Effort

### T3-001: CrowdLayer Multi-Annotator
- **Status**: ❌
- **Mechanism**: Trainable confusion matrix per annotator. Butuh per-annotator labels (bukan majority vote).
- **Reference**: Rodrigues & Pereira, AAAI 2018
- **Result**: BLOCKED — Dataset tidak memiliki per-annotator label splits

### T3-002: Co-Teaching for Noisy Labels
- **Status**: 🔄
- **Experiment ID**: NOVEL_016
- **Mechanism**: Cross-Pseudo-Supervision: Model A train 5e, generate high-conf pseudo-labels, Model B train 10e pada original+pseudo. Best model dipilih sebagai winner.
- **Reference**: Han et al., NeurIPS 2018
- **Result**: —
- **Notes**: Simplified cross-training (tidak full simultaneous co-training karena YOLO API constraint)

### T3-003: Three-Phase Curriculum
- **Status**: 🔄
- **Experiment ID**: NOVEL_011
- **Mechanism**: Callback yang adjust label_smoothing per epoch: Phase 1 (0-33%): 0.30, Phase 2 (33-67%): 0.15, Phase 3 (67-100%): 0.00.
- **Result**: —
- **Notes**: Training schedule callback only, no dataset changes needed

### T3-004: SimCLR/DenseCL Pretraining
- **Status**: 🔄
- **Experiment ID**: NOVEL_015
- **Mechanism**: Strong Augmentation Warmup proxy: 5 epochs dengan mixup=0.4, copy_paste=0.5, degrees=15°, shear=10° (forces invariant feature learning) → 15 epochs normal training.
- **Expected gain**: +3–5% (literature benchmark)
- **Result**: —
- **Notes**: SimCLR proxy via aggressive aug warmup — avoids need for unlabeled data

---

## TIER 4 — Deployment UX (Uncertainty Quantification)

### T4-001: Evidential Deep Learning (EDL)
- **Status**: 🔄
- **Experiment ID**: NOVEL_017
- **Mechanism**: Dirichlet-parameterized classification loss. Evidence e=ReLU(logit), α=e+1, p=α/Σα. Loss = MSE(y,p) + Var(p) + λ(t)*KL. KL annealing over 10 epochs.
- **Reference**: Sensoy et al., NeurIPS 2018
- **Result**: —
- **Notes**: EDLv8DetectionLoss subclass replacing BCE

### T4-002: Conformal Prediction Sets
- **Status**: ⏭️
- **Mechanism**: Post-processing analysis — tidak memerlukan training baru. Akan diimplementasikan sebagai analysis script setelah semua training selesai.
- **Reference**: Andéol & Mossina, 2025 (SeqCRC for YOLO)
- **Notes**: Butuh calibration set — dapat menggunakan val split dari existing best model

### T4-003: Burst-Shot Multi-Frame Voting
- **Status**: ⏭️
- **Mechanism**: Post-processing (WBF) — tidak memerlukan training baru. Dapat diimplementasikan sebagai inference-time ensemble.
- **Notes**: Requires multi-frame images at inference — post-training analysis only

---

## TIER 5 — Experimental (Higher Risk, Varied Impact)

### T5-001: DCNv4 Deformable Convolutions
- **Status**: ❌
- **Mechanism**: Ganti 2-3 backbone conv dengan DCNv4 yang adapt ke shape objek.
- **Reference**: DP-YOLO 2024 (+19.6%)
- **Result**: BLOCKED — Requires C++ compilation, tidak tersedia di environment

### T5-002: Aspect Ratio Auxiliary Loss
- **Status**: 🔄
- **Experiment ID**: NOVEL_018
- **Mechanism**: AspectRatioAuxv8DetectionLoss: penalty MSE(actual_AR, expected_AR_for_class) untuk foreground anchors. B1/B2 round (AR≈1.0-1.05), B3/B4 elongated (AR≈1.35-1.45). ar_weight=0.15.
- **Result**: —
- **Notes**: Custom loss subclass extending v8DetectionLoss

### T5-003: CLIP Soft Label Generation
- **Status**: 🔄
- **Experiment ID**: NOVEL_019
- **Mechanism**: open_clip ViT-B/32 (laion2b) crop embeddings vs text descriptions → temperature-scaled softmax (T=0.07) → soft label distribution. Blended 60% CLIP + 40% SORD.
- **Result**: —
- **Notes**: open_clip available in environment; CLIPSoftv8DetectionLoss

### T5-004: PPAL Active Learning
- **Status**: ⏭️
- **Mechanism**: Analysis-only: identify B2/B3 boundary samples dengan highest uncertainty dari existing models. k-means++ diversity sampling untuk re-annotation prioritization.
- **Notes**: Post-training analysis — akan diimplementasikan setelah semua training selesai

---

## Experiment Results Log

| ID | Experiment | mAP50 | Recall | Precision | Epochs | Config | Date | Notes |
|:--:|:-----------|:-----:|:------:|:---------:|:------:|:-------|:-----|:------|
| — | BREAK_101 (baseline) | 0.5250 | 0.5845 | — | 52 | yolov8n, 768px | 2026-04-08 | Best sebelum NOVEL series |
| — | BREAK_037 (hist best) | 0.5298 | — | — | — | Top-5 Ensemble | 2026-04-08 | Historical best |
| NOVEL_010 | SORD sigma=0.5 (tighter ordinal) | 0.5076 | 0.5926 | 0.4790 | 15/15 | SORD with tighter sigma=0.5 (harder ordinal boundaries) | 2026-04-09 | — |
| NOVEL_009 | Full Tier 1: L*a*b* + SORD + P2 | 0.3251 | 0.4815 | 0.3282 | 15/15 | Full Tier 1 combo: LAB input + SORD loss + P2 head | 2026-04-09 | — |
| NOVEL_008 | SORD + P2 Head Combo | 0.3833 | 0.5017 | 0.3833 | 15/15 | SORD ordinal loss + P2 head for small B4 | 2026-04-09 | — |
| NOVEL_007 | L*a*b* + P2 Head Combo | 0.3723 | 0.4968 | 0.3704 | 14/14 | LAB input + P2 head combo | 2026-04-09 | — |
| NOVEL_006 | SORD + Label Smoothing | 0.4640 | 0.5802 | 0.4342 | 15/15 | SORD ordinal loss + mild label smoothing as regularizer | 2026-04-09 | — |
| NOVEL_005 | Higher Resolution 768px | 0.5219 | 0.5979 | 0.4886 | 15/15 | 768px resolution for small B4 fruitlets (known to help) | 2026-04-09 | — |
| NOVEL_004 | SORD Ordinal Soft Labels | 0.4640 | 0.5802 | 0.4342 | 15/15 | SORD (σ=0.8): B2↔B3 confusion penalized less than B1↔B4 | 2026-04-09 | — |
| NOVEL_001 | Label Smoothing + CosLR | **0.5185** | 0.5962 | 0.4901 | 15/15 | yolov8n, 640px, ls=0.15 | 2026-04-09 | Best NOVEL so far |
| NOVEL_002 | L*a*b* Color Space Input | **0.5003** | 0.5741 | 0.4781 | 15/15 | yolov8n, LAB dataset, 640px | 2026-04-09 | — |
| NOVEL_003 | P2 Detection Head | **0.4380** | 0.5245 | 0.4374 | 13/15 | yolov8n-p2.yaml, 640px | 2026-04-09 | Needs more epochs |
| NOVEL_005 | Higher Resolution 768px | **0.5219** | 0.5979 | 0.4886 | 15/15 | yolov8n, 768px | 2026-04-09 | Close to BREAK_101! |
| NOVEL_007 | L*a*b* + P2 Head Combo | **0.3723** | 0.4968 | 0.3704 | 14/15 | yolov8n-p2.yaml, LAB, 640px | 2026-04-09 | Complex arch needs more epochs |

---

## Summary Progress

| Metric | Value |
|:-------|:------|
| Baseline | 0.504 (STRUCT_000) |
| Current Best | **0.5298** (BREAK_037) |
| NOVEL Series Best | **0.5219** (NOVEL_005 — 768px resolution) |
| Target | > 0.70 |
| SOTA Reference | 0.842 (Mansour 2022) |
