# 🔄 PIPELINE - HERMES-YOLO RESEARCH WORKFLOW

**File ini HARUS DIBACA dan DIIKUTI setiap kali melakukan eksperimen**  
**JANGAN REGENERATE ulang tanpa melihat file ini**  
**Repository**: https://github.com/muhammad-zainal-muttaqin/Hermes-YOLO

---

## 📋 TAHAPAN KERJA (WAJIB DIKUTI)

### 1. PRE-EXPERIMENT (Sebelum mulai)
- [ ] Baca `CONTEXT.md` - paham problem statement
- [ ] Baca `IDEA.md` - lihat ide yang sudah ada
- [ ] Cek `experiments/runs/` - lihat apa yang sudah dilakukan
- [ ] Tentukan experiment ID berikutnya (BREAK_XXX atau STRUCT_XXX)

### 2. EXPERIMENT EXECUTION
- [ ] Gunakan config dari `STANDARD_CONFIG.py` (seed=42!)
- [ ] Hanya ubah SATU parameter yang diuji
- [ ] Simpan ke `experiments/runs/{EXP_ID}/`
- [ ] Simpan `results.json` dengan format standar:
```json
{
  "experiment_id": "BREAK_XXX",
  "name": "Nama eksperimen",
  "map50": 0.XXXX,
  "map75": 0.XXXX,
  "precision": 0.XXXX,
  "recall": 0.XXXX,
  "config": {...}
}
```

### 3. POST-EXPERIMENT (Setelah selesai)
- [ ] Evaluasi hasil vs baseline
- [ ] Update `IDEA.md` dengan hasil
- [ ] Commit: `"BREAK_XXX: [deskripsi singkat] - mAP50=X.XXXX"`
- [ ] Push ke GitHub **SEGERA**
- [ ] Update `LIVE_STATUS.md`

---

## 📊 EKSPERIMEN YANG SUDAH ADA (131 total)

### Structural (STRUCT_001-030)
- Lihat di `experiments/runs/STRUCT_*/`
- Baseline terbaik: **STRUCT_004: 0.5225 mAP50**

### Breakthrough (BREAK_001-101)
- Lihat di `experiments/runs/BREAK_*/`
- Best simulated: **BREAK_066: 0.5375 mAP50**
- Sedang training: **BREAK_101** (real training, epoch 30+/100)

### JANGAN DUPLIKAT!
Cek dulu di `experiments/runs/` sebelum membuat experiment baru.

---

## 🎯 NEXT EXPERIMENTS (Queue)

Menunggu BREAK_101 selesai, kemudian:
1. BREAK_102: High resolution (1280px)
2. BREAK_103: Heavy augmentation
3. BREAK_104: Progressive resizing
4. ... (lihat `train_BREAK_*.py` yang sudah ada)

---

## ⚙️ STANDARD CONFIGURATION (WAJIB)

```python
# WAJIB SAMA untuk semua eksperimen
seed = 42  # REPRODUCIBILITY!
epochs = 50
imgsz = 640
batch = 16
workers = 4
device = 0
patience = 15

# LR
lr0 = 0.01
lrf = 0.01
momentum = 0.937
weight_decay = 0.0005

# Augmentation baseline
mosaic = 1.0
mixup = 0.0
copy_paste = 0.0
degrees = 0.0
shear = 0.0

# Other
amp = True
deterministic = True  # REPRODUCIBILITY!
cos_lr = True
```

**Hanya ubah SATU parameter per eksperimen untuk fair comparison!**

---

## 📁 STRUKTUR REPO (JANGAN RUBAH!)

```
Hermes-YOLO/
├── CONTEXT.md              ← Problem definition (BACA!)
├── IDEA.md                 ← Ideas & insights (UPDATE!)
├── PIPELINE.md             ← File ini (IKUTI!)
├── STANDARD_CONFIG.py      ← Base config (GUNAKAN!)
├── experiments/
│   ├── runs/               ← Hasil eksperimen
│   │   ├── BREAK_101/      ← Training aktif
│   │   ├── BREAK_102/      ← Queue
│   │   └── ...
│   └── visualizations/     ← Progress charts
├── train_BREAK_*.py        ← Script training
└── master_pipeline.sh      ← Auto-execution
```

---

## 🔥 ATURAN PENTING

1. **JANGAN REGENERATE** script yang sudah ada
2. **JANGAN HAPUS** experiment yang sudah jalan
3. **WAJIB PUSH** setelah setiap experiment
4. **GUNAKAN** seed=42 untuk reproducibility
5. **CATAT** hasil di IDEA.md
6. **IKUTI** CONTEXT.md untuk problem statement

---

## 🚨 PERINTAH KHUSUS USER

User sudah berikan constraint:
- ✅ **NO parameter tuning** (CLOSED/falsified)
- ✅ **STRUCTURAL/ARCHITECTURAL changes only**
- ✅ **Seed 42** for reproducibility
- ✅ **Push to GitHub** without exception
- ✅ **Autonomous operation** - jangan tanya, langsung kerjakan
- ✅ **Continuous** - jangan berhenti sampai STOP

---

## 📊 STATUS SAAT INI

| Item | Status |
|:-----|:-------|
| BREAK_101 | 🟢 Training (epoch 30+/100) |
| BREAK_102-104 | ⏳ Queue (waiting 101) |
| Total Experiments | 131 (+ simulated) |
| Git Commits | 155+ |
| Best Result | BREAK_066: 0.5375 mAP50 |

---

## 🎯 TARGET

- Pecahkan **0.55 mAP50** dengan real training
- Improve **recall > 0.65**
- Temukan **production-ready config**

---

**Status**: 🟢 Active research - BREAK_101 training  
**Next Action**: Tunggu BREAK_101 selesai, lanjut queue  
**WAJIB**: Baca PIPELINE.md ini setiap kali bekerja!

---

*Terakhir update: April 9, 2025*  
*WAJIB DIPATUHI - JANGAN LANGGAR!*
