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
- **Status**: ✅ (partial — NOVEL_004/006/008/010 are re-runs after bug fix)
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
- **Status**: 🔄 Running (NOVEL_007)
- **Experiment ID**: NOVEL_007
- **Mechanism**: LAB color space input + P2 detection head (no SORD). Physical color separation combined with small-object detection.
- **Config**: yolov8n-p2.yaml, LAB dataset, 640px, 15 epochs
- **Expected gain**: +3–6% mAP50
- **Result**: 🔄 In progress (~epoch 11/15, mAP50≈0.33)
- **Notes**: Running now; SORD variant (NOVEL_009) queued after bug fix

---

## TIER 2 — High ROI, Low-Medium Effort (Target: +5–10% additional)

### T2-001: Efficient Teacher SSOD
- **Status**: ⬜
- **Experiment ID**: NOVEL_008
- **Mechanism**: Semi-supervised: teacher generate pseudo-labels untuk unlabeled plantation images, student belajar dari keduanya. Spesifik untuk YOLO.
- **Reference**: Alibaba Research, efficientteacher
- **Result**: —
- **Notes**: Perlu unlabeled images — bisa dari validation set atau collect baru

### T2-002: Ensemble Knowledge Distillation
- **Status**: ⬜
- **Experiment ID**: NOVEL_009
- **Mechanism**: Train 5 diverse YOLO models (seed berbeda), average soft predictions → student belajar dari consensus. Untuk ambiguous B2/B3, ensemble menghasilkan distribution halus yang encode uncertainty.
- **Reference**: Hinton et al., 2015 (dark knowledge)
- **Result**: —
- **Notes**: Butuh 5× training time untuk teacher generation

### T2-003: Sub-center ArcFace Classification Head
- **Status**: ⬜
- **Experiment ID**: NOVEL_010
- **Mechanism**: Ganti BCE classification loss dengan Sub-center ArcFace (K=2 sub-centers per class). Sub-center menyerap noisy B2/B3 ambiguous samples secara otomatis.
- **Reference**: Deng et al., ECCV 2020
- **Result**: —
- **Notes**: Lebih complex daripada SupCon (yang sudah dicoba)

### T2-004: GPT-4V Annotation Audit
- **Status**: ⬜
- **Experiment ID**: (preprocessing, bukan training run)
- **Mechanism**: Feed setiap cropped B2/B3 sample ke VLM dengan prompt tentang warna dan kematangan. Identifikasi systematic annotation errors. Generate confidence-weighted labels.
- **Expected impact**: Bisa jadi single highest-leverage fix (+10% potential)
- **Cost**: ~$0.01-0.03 per image via API
- **Result**: —
- **Notes**: Perlu Anthropic/OpenAI API key

---

## TIER 3 — Strong Potential, Medium Effort

### T3-001: CrowdLayer Multi-Annotator
- **Status**: ⬜
- **Mechanism**: Trainable confusion matrix per annotator. Dihapus saat inference. Butuh per-annotator labels (bukan majority vote).
- **Reference**: Rodrigues & Pereira, AAAI 2018
- **Notes**: Perlu akses per-annotator labels dari dataset

### T3-002: Co-Teaching for Noisy Labels
- **Status**: ⬜
- **Mechanism**: 2 network train bersamaan, saling pilih small-loss samples. Mislabeled B2/B3 disaring secara otomatis.
- **Reference**: Han et al., NeurIPS 2018
- **Notes**: 2× training time, 1 network di deploy

### T3-003: Three-Phase Curriculum
- **Status**: ⬜
- **Mechanism**: Phase 1: binary ripe(B1) vs unripe(B2+B3+B4) → Phase 2: 3-class → Phase 3: full 4-class + SORD
- **Notes**: Training schedule change only

### T3-004: SimCLR/DenseCL Pretraining
- **Status**: ⬜
- **Mechanism**: SSL pretraining pada 2000+ unlabeled plantation images, lalu fine-tune
- **Expected gain**: +3–5% (literature benchmark)

---

## TIER 4 — Deployment UX (Uncertainty Quantification)

### T4-001: Evidential Deep Learning (EDL)
- **Status**: ⬜
- **Mechanism**: Dirichlet-parameterized output, uncertainty per sample dalam satu forward pass
- **Reference**: Sensoy et al., NeurIPS 2018

### T4-002: Conformal Prediction Sets
- **Status**: ⬜
- **Mechanism**: Output {B2, B3} saat uncertain, formal coverage guarantee 95%
- **Reference**: Andéol & Mossina, 2025 (SeqCRC for YOLO)

### T4-003: Burst-Shot Multi-Frame Voting
- **Status**: ⬜
- **Mechanism**: 3-5 frames burst mode, YOLO per frame, Weighted Box Fusion (WBF). High entropy → uncertain.

---

## TIER 5 — Experimental (Higher Risk, Varied Impact)

### T5-001: DCNv4 Deformable Convolutions
- **Status**: ⬜
- **Mechanism**: Ganti 2-3 backbone conv dengan DCNv4 yang adapt ke shape objek. B3 lonjong vs B1/B2 bulat.
- **Reference**: DP-YOLO 2024 (+19.6%)

### T5-002: Aspect Ratio Auxiliary Loss
- **Status**: ⬜
- **Mechanism**: Auxiliary head prediksi {elongated, round} dari bbox features. Label dari aspect ratio GT tanpa extra annotation.

### T5-003: CLIP Soft Label Generation
- **Status**: ⬜
- **Mechanism**: CLIP similarity scores antara cropped image vs text descriptions → soft labels via KL-divergence

### T5-004: PPAL Active Learning
- **Status**: ⬜
- **Mechanism**: Prioritize B2/B3 boundary samples untuk re-annotation. k-means++ diversity sampling.

---

## Experiment Results Log

| ID | Experiment | mAP50 | Recall | Precision | Epochs | Config | Date | Notes |
|:--:|:-----------|:-----:|:------:|:---------:|:------:|:-------|:-----|:------|
| — | BREAK_101 (baseline) | 0.5250 | 0.5845 | — | 52 | yolov8n, 768px | 2026-04-08 | Best sebelum NOVEL series |
| — | BREAK_037 (hist best) | 0.5298 | — | — | — | Top-5 Ensemble | 2026-04-08 | Historical best |
| NOVEL_002 | L*a*b* Color Space Input | 0.5003 | 0.5741 | 0.4781 | 15/15 | L*a*b* input — a* channel separates B1(red) vs B3(black) vs B4(green) | 2026-04-09 | — |
| NOVEL_001 | Label Smoothing + CosLR | 0.5185 | 0.5962 | 0.4901 | 15/15 | Ordinal proxy via label_smoothing=0.15 + cosine LR decay | 2026-04-09 | — |
| NOVEL_001 | Label Smoothing + CosLR | **0.5185** | 0.5962 | 0.4901 | 15/15 | auto-tracked | 2026-04-09 | — |
| NOVEL_002 | L*a*b* Color Space Input | **0.4963** | 0.5736 | 0.4827 | 14/14 | auto-tracked | 2026-04-09 | — |
| NOVEL_003 | P2 Detection Head | **0.4380** | 0.5245 | 0.4374 | 13/14 | auto-tracked | 2026-04-09 | — |
| NOVEL_004 | SORD Ordinal Soft Labels | ❌ FAILED | — | — | — | SORD σ=0.8 | 2026-04-09 | de_parallel bug (fixed) |
| NOVEL_005 | Higher Resolution 768px | **0.5177** | 0.5869 | 0.4867 | 13/14 | auto-tracked | 2026-04-09 | — |
| NOVEL_006 | SORD + Label Smoothing | ❌ FAILED | — | — | — | SORD σ=0.8 + ls=0.1 | 2026-04-09 | de_parallel bug (fixed) |
| NOVEL_007 | L*a*b* + P2 Head | **0.3723** | 0.4968 | 0.3704 | 14/14 | auto-tracked | 2026-04-09 | — |
| NOVEL_008 | SORD + P2 Head Combo | ⬜ Queued | — | — | — | yolov8n-p2.yaml, SORD σ=0.8 | — | — |
| NOVEL_009 | Full Tier 1: LAB+SORD+P2 | ⬜ Queued | — | — | — | yolov8n-p2.yaml, LAB, SORD | — | — |
| NOVEL_010 | SORD σ=0.5 (tighter ordinal) | ⬜ Queued | — | — | — | yolov8n, SORD σ=0.5 | — | — |

---

## Summary Progress

| Metric | Value |
|:-------|:------|
| Baseline | 0.504 (STRUCT_000) |
| Current Best | **0.5298** (BREAK_037) |
| NOVEL Series Best | **0.5003** (NOVEL_002) |
| Target | > 0.70 |
| SOTA Reference | 0.842 (Mansour 2022) |
