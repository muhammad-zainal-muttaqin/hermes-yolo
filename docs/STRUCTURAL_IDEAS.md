# 💡 Structural Research Ideas

> **Following CONTEXT.md strictly - NO parameter tinkering!**
> 
> Context reminder: "Recipe tuning biasa sudah diminishing returns... bottleneck utama kemungkinan ada pada struktur sinyal pembelajaran dan formulasi task."

---

## 🚫 CLOSED (Jangan Gunakan)

Sesuai CONTEXT.md Section 6:
- ❌ Learning rate tweaks (`lr0`, `lrf`)
- ❌ Loss weight adjustments (`BOX`, `CLS`, `DFL`)
- ❌ Augmentation parameter tuning (`mosaic`, `scale`, `degrees`)
- ❌ Epoch/patience changes
- ❌ Batch size tuning
- ❌ Optimizer swaps (SGD vs AdamW)
- ❌ Simple architecture swaps (YOLOv9e, RT-DETR)

## ✅ OPEN (Gunakan Ini!)

### 1. Test-Time Augmentation (TTA) dengan Uncertainty Fusion
**Status**: 🔄 STRUCT_000 (Running)

**Structural Change**: Multi-scale inference wrapper, bukan training parameter!

**Rationale**: 
- Bukan mengubah cara model belajar, tapi cara kita menggunakan model
- Ensemble predictions dari multiple augmentations
- Weight by inverse uncertainty untuk consensus

**Target Problem**: B2/B3 ambiguity → higher confidence through ensemble consensus

**Expected Impact**: 2-5% mAP50 boost tanpa mengubah training sama sekali

---

### 2. Hard Negative Mining (Online)
**Status**: ⏳ Queued

**Structural Change**: Custom data loader dengan loss-based sampling

**Rationale**:
- Bukan mengubau hyperparameter, tapi mengubah WHICH samples yang dilihat model
- Prioritize ambiguous B2/B3 pairs
- Dynamic batch composition berdasarkan difficulty

**Target Problem**: B2/B3 confusion - model perlu lebih banyak melihat cases yang sulit

**Different from CLOSED**: Bukan oversampling sederhana, tapi intelligent selection berdasarkan loss magnitude

---

### 3. Adaptive Label Smoothing (Per-Class)
**Status**: ⏳ Queued

**Structural Change**: Custom loss function dengan class-specific coefficients

**Rationale**:
- B1: Low smoothing (0.01) - easy class, biarkan confident
- B2/B3: High smoothing (0.15) - confused classes, reduce overconfidence
- B4: Medium smoothing (0.05) - small object challenge

**Target Problem**: B2/B3 overconfidence di wrong predictions

**Different from CLOSED**: Bukan global smoothing 0.1, tapi structural change di loss formulation

---

### 4. Enhanced FPN dengan P2 Detection Head
**Status**: ⏳ Queued

**Structural Change**: Architecture modification - add finer resolution detection layer

**Rationale**:
- B4 adalah smallest object (median 94x96 px)
- Standard FPN (P3-P5) might miss very small objects
- Add P2 pathway untuk 2x upsampled features

**Target Problem**: B4 small object detection

**Different from CLOSED**: Bukan sekadar imgsz=1024 (sudah di-CLOSED), tapi architectural change

---

### 5. Semantic-Aware Copy-Paste
**Status**: ⏳ Queued

**Structural Change**: Advanced augmentation dengan context preservation

**Rationale**:
- Copy-paste naif sudah di-CLOSED (Section 6.2)
- Tapi SMART copy-paste (context-aware) adalah structural
- Hanya paste B1/B4 ke backgrounds yang compatible

**Target Problem**: B1/B4 rare class underrepresentation

**Different from CLOSED**: Intelligence di placement, bukan sekadar quantity

---

### 6. Domain-Stratified Sampling
**Status**: ⏳ Queued

**Structural Change**: Data loader dengan explicit domain balancing

**Rationale**:
- DAMIMAS: 90.1%, LONSUM: 9.9% - severe imbalance
- Force 60/40 balance per batch regardless of dataset composition
- Address domain shift explicitly

**Target Problem**: Domain imbalance and generalization

**Different from CLOSED**: Bukan sekadar oversampling, tapi structural batch composition

---

### 7. Ensemble Diverse Architectures
**Status**: ⏳ Queued

**Structural Change**: Multi-model system dengan disagreement-based fusion

**Rationale**:
- YOLOv8 dan YOLO11 punya inductive biases berbeda
- Weighted Box Fusion (WBF) dari multiple detectors
- Disagreement indicates uncertainty → lower weight

**Target Problem**: Single model limitations, robustness

**Different from CLOSED**: Bukan model soup (CLOSED Section 6.1), tapi ensemble dengan intelligent fusion

---

### 8. Uncertainty-Guided Training (MC Dropout)
**Status**: ⏳ Queued

**Structural Change**: Bayesian approach dengan dropout during training

**Rationale**:
- Monte Carlo Dropout untuk uncertainty quantification
- High uncertainty = ambiguous sample (B2/B3)
- Focus training on high-uncertainty regions

**Target Problem**: B2/B3 ambiguity detection

**Different from CLOSED**: Bukan sekadar tweaking, tapi fundamental probabilistic approach

---

## 📊 Analysis Framework

Untuk setiap eksperimen structural, kita analisis:

1. **Did structural change help?** Compare to baseline
2. **Which problem improved?** Per-class breakdown
3. **What structural approach next?** Based on remaining problems

---

## 🎯 Target CONTEXT.md Problem Statement

> "Sistem 4-kelas RGB-only tampak mentok karena kombinasi ambiguity B2/B3, kesulitan small-object pada B4, domain imbalance DAMIMAS/LONSUM, dan kualitas/ketelitian bbox pada kasus kecil/ambigu, sehingga recipe tuning biasa tidak lagi memberi lompatan berarti."

**Structural approaches address**:
- ✅ B2/B3 ambiguity: TTA, Hard mining, Adaptive smoothing, Uncertainty guidance
- ✅ B4 small object: Enhanced FPN, Semantic copy-paste
- ✅ Domain imbalance: Domain-stratified sampling
- ✅ Overall robustness: Ensemble diverse

---

*Generated following CONTEXT.md Section 7: "Hal yang Masih Layak Dianggap Hidup"*
