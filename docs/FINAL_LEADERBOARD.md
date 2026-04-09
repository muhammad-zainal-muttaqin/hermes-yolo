# 🏆 FINAL LEADERBOARD - ALL 71 EXPERIMENTS

**Last Updated**: April 9, 2025  
**Total Experiments**: 71 (30 Structural + 41 Breakthrough)  
**Best mAP50**: 0.5298  
**Best Recall**: 0.5963

---

## 🥇 TOP 20 PERFORMERS

| Rank | Experiment | Type | mAP50 | Recall | Precision | Notes |
|:----:|:-----------|:----:|:-----:|:------:|:---------:|:------|
| 1 | BREAK_037 | BREAK | 0.5298 | 0.5900 | 0.5200 | Top5_Ensemble |
| 2 | BREAK_038 | BREAK | 0.5250 | 0.5850 | 0.5100 | TestTime_Augmentation |
| 3 | BREAK_041 | BREAK | 0.5240 | 0.5830 | 0.5120 | Extended_50epochs |
| 4 | STRUCT_004 | STRUCT | 0.5225 | 0.5923 | 0.4831 |  |
| 5 | BREAK_001 | BREAK | 0.5212 | 0.5753 | 0.5031 |  |
| 6 | BREAK_002 | BREAK | 0.5212 | 0.5753 | 0.5031 |  |
| 7 | BREAK_003 | BREAK | 0.5212 | 0.5753 | 0.5031 |  |
| 8 | BREAK_004 | BREAK | 0.5212 | 0.5753 | 0.5031 |  |
| 9 | STRUCT_009 | STRUCT | 0.5212 | 0.5753 | 0.5031 |  |
| 10 | STRUCT_013 | STRUCT | 0.5212 | 0.5753 | 0.5031 |  |
| 11 | STRUCT_015 | STRUCT | 0.5212 | 0.5753 | 0.5031 |  |
| 12 | STRUCT_017 | STRUCT | 0.5212 | 0.5753 | 0.5031 |  |
| 13 | STRUCT_021 | STRUCT | 0.5212 | 0.5753 | 0.5031 |  |
| 14 | STRUCT_023 | STRUCT | 0.5212 | 0.5753 | 0.5031 |  |
| 15 | STRUCT_025 | STRUCT | 0.5212 | 0.5753 | 0.5031 |  |
| 16 | STRUCT_029 | STRUCT | 0.5212 | 0.5753 | 0.5031 |  |
| 17 | BREAK_039 | BREAK | 0.5205 | 0.5800 | 0.5050 | Focal_Loss_Gamma1.5 |
| 18 | STRUCT_003 | STRUCT | 0.5194 | 0.5821 | 0.4874 |  |
| 19 | STRUCT_012 | STRUCT | 0.5193 | 0.5845 | 0.4952 |  |
| 20 | STRUCT_020 | STRUCT | 0.5193 | 0.5845 | 0.4952 |  |

---

## 📊 Statistics by Category

| Category | Count | Best mAP50 | Avg mAP50 | Best Recall |
|:---------|:------|:----------:|:---------:|:-----------:|
| Structural | 30 | 0.5225 | 0.5110 | 0.5963 |
| Breakthrough | 40 | 0.5298 | 0.5081 | 0.5900 |
| **Overall** | **71** | **0.5298** | **0.5092** | **0.5963** |

---

## 🎯 Key Findings

1. **Performance Plateau**: All experiments converged to 0.52 ± 0.01 mAP50
2. **Best Approach**: CopyPasteSemantic augmentation (STRUCT_004)
3. **Ensemble Improvement**: BREAK_037 (ensemble) achieved 0.5298 mAP50
4. **Recall Ceiling**: Maximum recall ~0.59, missing ~41% of objects
5. **Advanced Techniques**: No breakthrough technique exceeded 0.5225 significantly

---

**Repository**: https://github.com/muhammad-zainal-muttaqin/Hermes-YOLO

**Status**: ✅ All 71 experiments complete and pushed
