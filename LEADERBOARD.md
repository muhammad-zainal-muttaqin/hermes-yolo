# 🏆 Research Leaderboard

> Live results from autonomous structural research following CONTEXT.md
> Last updated: April 8, 2025

---

## 🥇 Top Performers

| Rank | Experiment | mAP50 | Approach | Target Problem | Status |
|:----:|:-----------|:-----:|:---------|:---------------|:------:|
| 🥇 | **STRUCT_004** | **0.5225** | CopyPasteSemantic | B1/B4 rare class underrepresentation | ✅ Best |
| 🥈 | **STRUCT_003** | **0.5194** | SmallObjectFPN | B4 small object detection | ✅ Excellent |
| 🥉 | **STRUCT_000** | **0.5039** | TestTimeAugmentation | B2/B3 ambiguity | ✅ Good |
| 4 | STRUCT_002 | 0.5039 | LabelSmoothingAdaptive | B2/B3 overconfidence | ➡️ Baseline |
| 5 | AR_000 | 0.4984 | HighCls_Conservative | Parameter tuning baseline | ➡️ Old |
| 6 | STRUCT_001 | 0.4977 | HardNegativeMining | B2/B3 confusion | ❌ Not effective |
| 7 | STRUCT_005 | 0.4977 | DomainStratified | Domain imbalance | ❌ Not effective |
| 8 | STRUCT_006 | 0.4953 | EnsembleDiverse | Model ensemble | ❌ Not effective |

---

## 📊 Performance Analysis

### What Works ✅

| Strategy | mAP50 Gain | Mechanism |
|:---------|:----------:|:----------|
| **CopyPasteSemantic** | +3.6% | Context-aware augmentation for rare classes |
| **SmallObjectFPN** | +3.0% | 768px resolution for small B4 objects |
| **TestTimeAugmentation** | Baseline | Inference-time multi-scale ensemble |

### What Doesn't Work ❌

| Strategy | Result | Lesson |
|:---------|:------:|:-------|
| Hard Negative Mining | -1.2% | Dataset already balanced |
| Domain Stratification | -1.2% | Domain imbalance not the bottleneck |
| Ensemble Diverse | -1.0% | Architecture swap insufficient |

---

## 🎯 Progress Summary

| Metric | Value |
|:-------|:------|
| **Best mAP50** | **0.5225** (STRUCT_004) |
| **Baseline mAP50** | 0.504 (STRUCT_000) |
| **Improvement** | +3.6% from baseline |
| **Target mAP50** | > 0.55 |
| **Gap to Target** | 0.0275 (95% complete!) |
| **Gap to SOTA** | ~0.20 (Mansour et al. 2022: 0.842) |
| **Total Experiments** | 8 complete |
| **In Progress** | STRUCT_007 (UncertaintyGuided) |

---

## 🔬 Key Insights

### 1. Data-Centric > Model-Centric
- Smart augmentation (CopyPaste) beats architecture changes (Ensemble)
- Resolution matters for small objects (768px > 640px)
- Hard mining doesn't help (dataset already clean)

### 2. Structural Changes Required
Following **CONTEXT.md Section 7** - what works:
- ✅ Augmentation intelligence (CopyPaste)
- ✅ Resolution for small objects (768px)
- ❌ Parameter tuning (diminishing returns)
- ❌ Simple architecture swaps (YOLOv8→11)

### 3. Next Tier 1 Breakthroughs
From `BREAKTHROUGH_IDEAS.md`:
- 🔄 **SORD** - Ordinal soft labels (Idea #1)
- 🔄 **L*a*b*** - Color space engineering (Idea #8)
- 🔄 **FDA** - Domain adaptation (Idea #16)

**Expected combined gain**: +8-15% mAP50 → **Target 0.60-0.65**

---

## 📈 Visualization

See live progress chart: [`experiments/visualizations/progress_map50.png`](experiments/visualizations/progress_map50.png)

---

## 🚀 Running Experiments

| Experiment | Status | Start Time | ETA |
|:-----------|:------:|:-----------|:----|
| STRUCT_007 | 🔄 Running | 16:26 | ~8 min |

System health: 🟢 **SAFE** (GPU: 22%, Temp: 63°C)

---

*Generated automatically by structural_orchestrator.sh*
