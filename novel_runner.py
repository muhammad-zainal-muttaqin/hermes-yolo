#!/usr/bin/env python3
"""
Novel Strategy Training Runner — Hermes-YOLO
Implements Tier 1 ideas from IDEA.md and runs them sequentially.

Experiments:
  NOVEL_001: Label Smoothing + CosLR (baseline++)
  NOVEL_002: L*a*b* Color Space Input
  NOVEL_003: P2 Head (yolov8n-p2.yaml)
  NOVEL_004: SORD Ordinal Soft Labels
  NOVEL_005: L*a*b* + P2 Head Combo
  NOVEL_006: SORD + P2 Head Combo
  NOVEL_007: L*a*b* + P2 Head + SORD (Full Tier 1)

After each run:
  - Parse mAP50 from results.csv
  - Update IDEA.md with result
  - Regenerate progress chart
  - Git push to GitHub
"""

import os
import sys
import subprocess
import csv
import json
import re
import shutil
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss, make_anchors

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/workspace/Hermes-YOLO")
DATASET_DIR = Path("/workspace/dataset-sawit")
DATASET_LAB_DIR = Path("/workspace/dataset-sawit-lab")
RUNS_DIR = BASE_DIR / "runs/detect/experiments/runs"
CHART_DIR = BASE_DIR / "experiments/visualizations"
IDEA_MD = BASE_DIR / "IDEA.md"

# ─── Config ───────────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
SEED = 42

EXPERIMENTS = [
    # ── Tier 1: Individual strategies (15 epochs each for quick comparison) ──
    {
        "id": "NOVEL_001",
        "name": "Label Smoothing + CosLR",
        "idea_id": "T1-001",
        "epochs": 15,
        "imgsz": 640,
        "batch": 16,
        "model": "yolov8n.pt",
        "dataset": str(BASE_DIR / "dataset_novel.yaml"),
        "extra_kwargs": {
            "label_smoothing": 0.15,
            "cos_lr": True,
            "lr0": 0.01,
            "lrf": 0.005,
        },
        "description": "Ordinal proxy via label_smoothing=0.15 + cosine LR decay",
    },
    {
        "id": "NOVEL_002",
        "name": "L*a*b* Color Space Input",
        "idea_id": "T1-003",
        "epochs": 15,
        "imgsz": 640,
        "batch": 16,
        "model": "yolov8n.pt",
        "dataset": str(BASE_DIR / "dataset_novel_lab.yaml"),
        "extra_kwargs": {
            "label_smoothing": 0.1,
            "cos_lr": True,
        },
        "description": "L*a*b* input — a* channel separates B1(red) vs B3(black) vs B4(green)",
        "requires_lab": True,
    },
    {
        "id": "NOVEL_003",
        "name": "P2 Detection Head (Small Object B4)",
        "idea_id": "T1-004",
        "epochs": 15,
        "imgsz": 640,
        "batch": 12,
        "model": "yolov8n-p2.yaml",
        "dataset": str(BASE_DIR / "dataset_novel.yaml"),
        "extra_kwargs": {
            "label_smoothing": 0.1,
            "cos_lr": True,
        },
        "description": "P2 head (stride=4) for tiny B4 detection",
    },
    {
        "id": "NOVEL_004",
        "name": "SORD Ordinal Soft Labels",
        "idea_id": "T1-002",
        "epochs": 15,
        "imgsz": 640,
        "batch": 16,
        "model": "yolov8n.pt",
        "dataset": str(BASE_DIR / "dataset_novel.yaml"),
        "extra_kwargs": {
            "cos_lr": True,
        },
        "description": "SORD (σ=0.8): B2↔B3 confusion penalized less than B1↔B4",
        "use_sord": True,
        "sord_sigma": 0.8,
    },
    {
        "id": "NOVEL_005",
        "name": "Higher Resolution 768px",
        "idea_id": "T1-005",
        "epochs": 15,
        "imgsz": 768,
        "batch": 8,
        "model": "yolov8n.pt",
        "dataset": str(BASE_DIR / "dataset_novel.yaml"),
        "extra_kwargs": {
            "label_smoothing": 0.1,
            "cos_lr": True,
        },
        "description": "768px resolution for small B4 fruitlets (known to help)",
    },
    {
        "id": "NOVEL_006",
        "name": "SORD + Label Smoothing",
        "idea_id": "T1-002+T1-001",
        "epochs": 15,
        "imgsz": 640,
        "batch": 16,
        "model": "yolov8n.pt",
        "dataset": str(BASE_DIR / "dataset_novel.yaml"),
        "extra_kwargs": {
            "label_smoothing": 0.1,
            "cos_lr": True,
        },
        "description": "SORD ordinal loss + mild label smoothing as regularizer",
        "use_sord": True,
        "sord_sigma": 0.8,
    },
    {
        "id": "NOVEL_007",
        "name": "L*a*b* + P2 Head Combo",
        "idea_id": "T1-003+T1-004",
        "epochs": 15,
        "imgsz": 640,
        "batch": 12,
        "model": "yolov8n-p2.yaml",
        "dataset": str(BASE_DIR / "dataset_novel_lab.yaml"),
        "extra_kwargs": {
            "label_smoothing": 0.1,
            "cos_lr": True,
        },
        "description": "LAB input + P2 head combo",
        "requires_lab": True,
    },
    {
        "id": "NOVEL_008",
        "name": "SORD + P2 Head Combo",
        "idea_id": "T1-002+T1-004",
        "epochs": 15,
        "imgsz": 640,
        "batch": 12,
        "model": "yolov8n-p2.yaml",
        "dataset": str(BASE_DIR / "dataset_novel.yaml"),
        "extra_kwargs": {
            "cos_lr": True,
        },
        "description": "SORD ordinal loss + P2 head for small B4",
        "use_sord": True,
        "sord_sigma": 0.8,
    },
    {
        "id": "NOVEL_009",
        "name": "Full Tier 1: L*a*b* + SORD + P2",
        "idea_id": "T1-007",
        "epochs": 15,
        "imgsz": 640,
        "batch": 12,
        "model": "yolov8n-p2.yaml",
        "dataset": str(BASE_DIR / "dataset_novel_lab.yaml"),
        "extra_kwargs": {
            "cos_lr": True,
        },
        "description": "Full Tier 1 combo: LAB input + SORD loss + P2 head",
        "requires_lab": True,
        "use_sord": True,
        "sord_sigma": 0.8,
    },
    {
        "id": "NOVEL_010",
        "name": "SORD sigma=0.5 (tighter ordinal)",
        "idea_id": "T1-002-v2",
        "epochs": 15,
        "imgsz": 640,
        "batch": 16,
        "model": "yolov8n.pt",
        "dataset": str(BASE_DIR / "dataset_novel.yaml"),
        "extra_kwargs": {
            "cos_lr": True,
        },
        "description": "SORD with tighter sigma=0.5 (harder ordinal boundaries)",
        "use_sord": True,
        "sord_sigma": 0.5,
    },
]


# ─── SORD Loss Implementation ──────────────────────────────────────────────────

class SORDv8DetectionLoss(v8DetectionLoss):
    """
    v8DetectionLoss with SORD (Soft Ordinal Labels for Detection).

    Replaces one-hot classification targets with Gaussian-kernel ordinal
    soft labels. For B2/B3 confusion: adjacent step error gets reduced penalty.

    Class ordinal order: B1(0, ripe) → B2(1, transition) → B3(2, unripe) → B4(3, least_ripe)

    Reference: Díaz & Marathe, CVPR 2019
    """

    def __init__(self, model, sigma=0.8):
        super().__init__(model)
        self.sord_sigma = sigma

        # Get nc
        nc = model.nc if hasattr(model, 'nc') else 4

        # Build soft label matrix: soft_matrix[true_class, j] = P(j | true_class)
        positions = np.arange(nc, dtype=np.float32)
        matrix = np.zeros((nc, nc), dtype=np.float32)
        for i in range(nc):
            dists = (positions - i) ** 2
            weights = np.exp(-dists / (2 * sigma ** 2))
            matrix[i] = weights / weights.sum()

        device = next(model.parameters()).device
        self.soft_matrix = torch.tensor(matrix, device=device, dtype=torch.float32)

        print(f"[SORD] sigma={sigma}, soft label matrix:")
        class_names = ["B1(ripe)", "B2(trans)", "B3(unripe)", "B4(least)"]
        for i, name in enumerate(class_names):
            row = [f"{v:.3f}" for v in matrix[i]]
            print(f"  {name}: [{', '.join(row)}]")

    def get_assigned_targets_and_loss(self, preds, batch):
        """Override to apply SORD soft labels to classification targets."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(
            preds["feats"][0].shape[2:], device=self.device, dtype=dtype
        ) * self.stride[0]

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # ── SORD modification ──────────────────────────────────────────────────
        # target_scores: (batch, num_anchors, nc) — soft one-hot with IoU scaling
        # For foreground anchors, replace one-hot with SORD soft distribution
        if fg_mask.any():
            fg_scores = target_scores[fg_mask]          # (num_fg, nc)

            # IoU scale = max value in the row (could be <1.0 due to IoU weighting)
            iou_scale = fg_scores.max(dim=-1, keepdim=True)[0]  # (num_fg, 1)

            # Assigned class = argmax of score row
            assigned_cls = fg_scores.argmax(dim=-1)     # (num_fg,)

            # Look up SORD soft labels
            soft_labels = self.soft_matrix[assigned_cls]  # (num_fg, nc)

            # Apply same IoU scaling as original
            target_scores[fg_mask] = soft_labels * iou_scale
        # ── End SORD ──────────────────────────────────────────────────────────

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss (BCE with SORD targets)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )


def make_sord_trainer(sigma=0.8):
    """Factory: return a YOLO model with SORD criterion injected via callback."""

    def on_train_start(trainer):
        """Called after model and criterion are initialized — replace with SORD."""
        from ultralytics.utils.torch_utils import de_parallel
        model = de_parallel(trainer.model)
        sord_loss = SORDv8DetectionLoss(model, sigma=sigma)
        model.criterion = sord_loss
        print(f"[SORD] Criterion replaced on model (sigma={sigma})")

    return on_train_start


# ─── L*a*b* Dataset Preprocessing ─────────────────────────────────────────────

def convert_to_lab(src_img_dir: Path, dst_img_dir: Path):
    """Convert all images from RGB to L*a*b* and save as uint8 PNG.

    Encoding:
      L:  [0, 100]   → [0, 255]   (multiply by 2.55)
      a*: [-128, 127] → [0, 255]  (add 128)
      b*: [-128, 127] → [0, 255]  (add 128)
    """
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    img_paths = list(src_img_dir.glob("*.jpg")) + list(src_img_dir.glob("*.jpeg")) + \
                list(src_img_dir.glob("*.png")) + list(src_img_dir.glob("*.JPG"))

    print(f"  Converting {len(img_paths)} images: {src_img_dir.name} → LAB")
    converted = 0

    for img_path in img_paths:
        dst_path = dst_img_dir / (img_path.stem + ".jpg")
        if dst_path.exists():
            converted += 1
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  WARNING: could not read {img_path}")
            continue

        # Convert BGR → L*a*b*
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

        # lab already uint8 in OpenCV: L=[0,255], a=[0,255], b=[0,255]
        # (OpenCV shifts: L*2.55, a+128, b+128)

        cv2.imwrite(str(dst_path), lab, [cv2.IMWRITE_JPEG_QUALITY, 95])
        converted += 1

    print(f"  Done: {converted}/{len(img_paths)} images")
    return converted


def prepare_lab_dataset():
    """Prepare the full L*a*b* dataset."""
    print("\n[LAB] Preparing L*a*b* dataset...")

    for split in ["train", "val", "test"]:
        src_img = DATASET_DIR / "images" / split
        dst_img = DATASET_LAB_DIR / "images" / split

        # Convert images
        convert_to_lab(src_img, dst_img)

        # Symlink labels (same format, no change needed)
        dst_lbl = DATASET_LAB_DIR / "labels" / split
        src_lbl = DATASET_DIR / "labels" / split
        if not dst_lbl.exists():
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            try:
                dst_lbl.symlink_to(src_lbl.resolve())
            except Exception:
                # Copy if symlink fails
                shutil.copytree(str(src_lbl), str(dst_lbl))

    # Write dataset YAML
    lab_yaml = BASE_DIR / "dataset_novel_lab.yaml"
    lab_yaml.write_text(f"""path: {DATASET_LAB_DIR}
train: images/train
val: images/val
test: images/test

nc: 4
names:
  0: B1
  1: B2
  2: B3
  3: B4

# L*a*b* encoded: OpenCV LAB format (L=[0,255], a=[0,255], b=[0,255])
# a* channel: B1(red,high) > B2(transition) > B3(black,~128) > B4(greenish,low)
""")
    print(f"[LAB] Dataset YAML written: {lab_yaml}")
    return True


# ─── Result Parsing ────────────────────────────────────────────────────────────

def parse_results(run_dir: Path) -> dict:
    """Parse training results from results.csv."""
    results_csv = run_dir / "train" / "results.csv"
    if not results_csv.exists():
        return {}

    best_map50 = 0.0
    best_recall = 0.0
    best_precision = 0.0
    best_epoch = 0

    with open(results_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for i, row in enumerate(rows):
        try:
            # Strip whitespace from keys
            row = {k.strip(): v.strip() for k, v in row.items()}
            map50_key = [k for k in row.keys() if "map50" in k.lower() and "95" not in k.lower()]
            recall_key = [k for k in row.keys() if "recall" in k.lower()]
            prec_key = [k for k in row.keys() if "precision" in k.lower()]

            if map50_key:
                val = float(row[map50_key[0]])
                if val > best_map50:
                    best_map50 = val
                    best_epoch = i + 1
                    if recall_key:
                        best_recall = float(row[recall_key[0]])
                    if prec_key:
                        best_precision = float(row[prec_key[0]])
        except (ValueError, KeyError):
            continue

    return {
        "map50": best_map50,
        "recall": best_recall,
        "precision": best_precision,
        "best_epoch": best_epoch,
        "total_epochs": len(rows),
    }


# ─── IDEA.md Updater ───────────────────────────────────────────────────────────

def update_idea_md(exp_id: str, idea_id: str, result: dict, config: dict):
    """Update IDEA.md with experiment result."""
    content = IDEA_MD.read_text()

    # Update status: ⬜ → ✅
    idea_pattern = f"### {idea_id}:"
    # Find the section and update status
    lines = content.split("\n")
    in_section = False
    updated_lines = []

    for i, line in enumerate(lines):
        if idea_id in line and "###" in line:
            in_section = True

        if in_section and "**Status**:" in line:
            line = f"- **Status**: ✅"
            in_section = False  # only update first occurrence

        if in_section and "**Result**:" in line:
            r = result
            line = f"- **Result**: mAP50={r.get('map50', 0):.4f} | Recall={r.get('recall', 0):.4f} | BestEpoch={r.get('best_epoch', 0)}"

        if in_section and "**Experiment ID**:" in line:
            line = f"- **Experiment ID**: {exp_id}"

        updated_lines.append(line)

    content = "\n".join(updated_lines)

    # Update results table
    date_str = datetime.now().strftime("%Y-%m-%d")
    r = result
    new_row = f"| {exp_id} | {config['name']} | {r.get('map50', 0):.4f} | {r.get('recall', 0):.4f} | {r.get('precision', 0):.4f} | {r.get('best_epoch', 0)}/{r.get('total_epochs', 0)} | {config.get('description', '')} | {date_str} | — |"

    # Find results table and append
    if "| BREAK_101 (baseline)" in content:
        content = content.replace(
            "| — | BREAK_037 (hist best) | 0.5298 | — | — | — | Top-5 Ensemble | 2026-04-08 | Historical best |",
            f"| — | BREAK_037 (hist best) | 0.5298 | — | — | — | Top-5 Ensemble | 2026-04-08 | Historical best |\n{new_row}"
        )

    # Update NOVEL Series Best in summary
    current_best_match = re.search(r"NOVEL Series Best \| (.+?) \|", content)
    current_best = 0.0
    if current_best_match:
        try:
            current_best = float(current_best_match.group(1).strip().split(" ")[0])
        except:
            pass

    if r.get("map50", 0) > current_best:
        content = re.sub(
            r"NOVEL Series Best \| .+? \|",
            f"NOVEL Series Best | **{r.get('map50', 0):.4f}** ({exp_id}) |",
            content
        )

    IDEA_MD.write_text(content)
    print(f"[IDEA.md] Updated with {exp_id}: mAP50={r.get('map50', 0):.4f}")


# ─── Chart Generator ───────────────────────────────────────────────────────────

def generate_chart():
    """Regenerate progress_map50.png from all available results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CHART_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all results
    experiments = []

    # Historical runs
    for run_dir in sorted(RUNS_DIR.glob("*")):
        if run_dir.is_dir():
            result = parse_results(run_dir)
            if result and result["map50"] > 0:
                experiments.append({
                    "id": run_dir.name,
                    "map50": result["map50"],
                    "series": "NOVEL" if run_dir.name.startswith("NOVEL") else "BREAK",
                })

    if not experiments:
        print("[Chart] No results to plot")
        return

    # Colors
    def get_color(exp_id):
        if exp_id.startswith("NOVEL"):
            num = int(exp_id.split("_")[-1]) if "_" in exp_id else 0
            tier1_colors = ["#00ff88", "#00cc66", "#009944", "#00ffaa",
                           "#44ffaa", "#66ffcc", "#88ffdd", "#aaffee"]
            return tier1_colors[num % len(tier1_colors)]
        elif "101" in exp_id:
            return "#ff4444"
        elif any(x in exp_id for x in ["034", "035", "036"]):
            return "#ff8800"
        else:
            return "#4488ff"

    ids = [e["id"] for e in experiments]
    vals = [e["map50"] for e in experiments]
    colors = [get_color(e["id"]) for e in experiments]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10),
                                    facecolor="#0d1117")
    fig.suptitle("Hermes-YOLO — TBS Oil Palm Ripeness Detection Progress",
                 color="white", fontsize=15, fontweight="bold", y=0.98)

    # Panel 1: All experiments bar chart
    ax1.set_facecolor("#0d1117")
    ax1.spines["bottom"].set_color("#30363d")
    ax1.spines["left"].set_color("#30363d")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(colors="#8b949e")

    bars = ax1.bar(range(len(ids)), vals, color=colors, width=0.7, alpha=0.9)
    ax1.axhline(y=0.5298, color="#ff4444", linestyle="--", alpha=0.7, linewidth=1.5,
                label="Hist Best (BREAK_037=0.5298)")
    ax1.axhline(y=0.504, color="#888888", linestyle=":", alpha=0.7, linewidth=1,
                label="Baseline (0.504)")
    ax1.axhline(y=0.70, color="#ffff00", linestyle="--", alpha=0.5, linewidth=1,
                label="Target (0.70)")

    best_val = max(vals)
    best_idx = vals.index(best_val)
    ax1.annotate(f"Best: {best_val:.4f}", xy=(best_idx, best_val),
                xytext=(best_idx, best_val + 0.02),
                color="white", fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="white", lw=1))

    ax1.set_xticks(range(len(ids)))
    ax1.set_xticklabels(ids, rotation=45, ha="right", fontsize=7, color="#8b949e")
    ax1.set_ylabel("mAP50 (B)", color="#8b949e", fontsize=10)
    ax1.set_ylim(0.45, max(max(vals) + 0.05, 0.72))
    ax1.legend(facecolor="#161b22", labelcolor="white", fontsize=8, loc="upper left")
    ax1.set_title("All Experiments — mAP50 Comparison", color="#e6edf3", fontsize=11)

    # Panel 2: NOVEL series detail
    novel_exps = [e for e in experiments if e["id"].startswith("NOVEL")]
    break_exps = [e for e in experiments if not e["id"].startswith("NOVEL")]

    ax2.set_facecolor("#0d1117")
    ax2.spines["bottom"].set_color("#30363d")
    ax2.spines["left"].set_color("#30363d")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(colors="#8b949e")

    if novel_exps:
        novel_ids = [e["id"] for e in novel_exps]
        novel_vals = [e["map50"] for e in novel_exps]
        ax2.bar(range(len(novel_ids)), novel_vals, color="#00ff88", width=0.6, alpha=0.9, label="NOVEL series")
        ax2.set_xticks(range(len(novel_ids)))
        ax2.set_xticklabels(novel_ids, rotation=30, ha="right", fontsize=9, color="#8b949e")
    else:
        ax2.text(0.5, 0.5, "NOVEL experiments running...",
                ha="center", va="center", color="#8b949e", fontsize=12,
                transform=ax2.transAxes)

    ax2.axhline(y=0.5250, color="#ff4444", linestyle="--", alpha=0.7, linewidth=1.5,
                label="BREAK_101 local best (0.5250)")
    ax2.axhline(y=0.5298, color="#ff8800", linestyle="--", alpha=0.6, linewidth=1.2,
                label="BREAK_037 hist best (0.5298)")
    ax2.axhline(y=0.70, color="#ffff00", linestyle="--", alpha=0.5, linewidth=1,
                label="Target (0.70)")
    ax2.set_ylabel("mAP50 (B)", color="#8b949e", fontsize=10)
    ax2.set_ylim(0.45, 0.75)
    ax2.legend(facecolor="#161b22", labelcolor="white", fontsize=8, loc="upper right")
    ax2.set_title("NOVEL Series — Novel Strategy Results", color="#e6edf3", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = CHART_DIR / "progress_map50.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close()
    print(f"[Chart] Saved: {out_path}")


# ─── Git Push ─────────────────────────────────────────────────────────────────

def git_push(message: str):
    """Stage tracked files and push to GitHub."""
    token = os.environ.get("GITHUB_TOKEN", GITHUB_TOKEN)
    if not token:
        print("[Git] No GITHUB_TOKEN, skipping push")
        return False

    remote_url = f"https://{token}@github.com/muhammad-zainal-muttaqin/Hermes-YOLO.git"

    try:
        subprocess.run(
            ["git", "remote", "set-url", "origin", remote_url],
            cwd=str(BASE_DIR), check=True, capture_output=True
        )
        subprocess.run(
            ["git", "add", "IDEA.md", "LEADERBOARD.md", "README.md",
             "experiments/visualizations/progress_map50.png",
             "dataset_novel.yaml", "dataset_novel_lab.yaml"],
            cwd=str(BASE_DIR), check=False, capture_output=True
        )
        # Add any NOVEL run results (small files only)
        subprocess.run(
            ["git", "add", "runs/detect/experiments/runs/NOVEL_*/train/results.csv",
             "runs/detect/experiments/runs/NOVEL_*/train/args.yaml"],
            cwd=str(BASE_DIR), check=False, capture_output=True
        )
        result = subprocess.run(
            ["git", "commit", "-m", message, "--allow-empty"],
            cwd=str(BASE_DIR), capture_output=True, text=True
        )
        if "nothing to commit" in result.stdout:
            print("[Git] Nothing new to commit")
            return True
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=str(BASE_DIR), check=True, capture_output=True
        )
        print(f"[Git] Pushed: {message}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Git] Push failed: {e}")
        return False


# ─── Update LEADERBOARD ────────────────────────────────────────────────────────

def update_leaderboard(exp_id: str, result: dict, config: dict):
    """Update LEADERBOARD.md with new result if it's top-5."""
    lb_path = BASE_DIR / "LEADERBOARD.md"
    content = lb_path.read_text()

    # Simple: just append a line in historical best section if map50 is notable
    r = result
    if r.get("map50", 0) > 0.50:
        new_row = f"| - | {exp_id} | {r['map50']:.4f} | {r.get('recall', 0):.4f} | {r.get('precision', 0):.4f} | {r.get('best_epoch', 0)} | {config['name']} |"

        # Insert before closing of the local runs table
        if "| 5 | BREAK_005" in content:
            content = content.replace(
                "| 5 | BREAK_005–032 | 0.5025 | 0.5871 | — | 10 | Early batch (28 experiments) |",
                f"| 5 | BREAK_005–032 | 0.5025 | 0.5871 | — | 10 | Early batch (28 experiments) |\n{new_row}"
            )

        lb_path.write_text(content)

    # Update current best line
    best_section = re.search(r"\*\*Current Best\*\* \| .+? \|", content)
    if best_section:
        current = re.search(r"(\d+\.\d+)", best_section.group())
        if current and r.get("map50", 0) > float(current.group()):
            content = re.sub(
                r"\*\*Current Best\*\* \| .+? \|",
                f"**Current Best** | **{r['map50']:.4f}** ({exp_id}) |",
                content
            )
            lb_path.write_text(content)


# ─── Main Training Loop ────────────────────────────────────────────────────────

def run_experiment(config: dict) -> dict:
    """Run a single experiment and return results."""
    exp_id = config["id"]
    run_dir = RUNS_DIR / exp_id
    results_csv = run_dir / "train" / "results.csv"

    print(f"\n{'='*60}")
    print(f"[{exp_id}] {config['name']}")
    print(f"  Description: {config['description']}")
    print(f"  Model: {config['model']}, Epochs: {config['epochs']}, Img: {config['imgsz']}px")
    print(f"{'='*60}")

    # Skip if already completed with good results
    if results_csv.exists():
        existing = parse_results(run_dir)
        if existing and existing["map50"] > 0.1:
            print(f"[{exp_id}] Already completed: mAP50={existing['map50']:.4f}. Skipping.")
            return existing

    # Check if LAB dataset required
    if config.get("requires_lab"):
        lab_yaml = BASE_DIR / "dataset_novel_lab.yaml"
        if not lab_yaml.exists() or not (DATASET_LAB_DIR / "images" / "train").exists():
            prepare_lab_dataset()

    # Training kwargs
    train_kwargs = {
        "data": config["dataset"],
        "epochs": config["epochs"],
        "imgsz": config["imgsz"],
        "batch": config["batch"],
        "seed": SEED,
        "device": "0",
        "workers": 8,
        "project": str(RUNS_DIR.parent),
        "name": f"experiments/runs/{exp_id}",
        "exist_ok": True,
        "verbose": False,
        "plots": True,
        "save": True,
        # Standard augmentations (same as BREAK_101)
        "copy_paste": 0.3,
        "mixup": 0.1,
        "degrees": 5.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "fliplr": 0.5,
        "mosaic": 1.0,
    }
    train_kwargs.update(config.get("extra_kwargs", {}))

    try:
        model = YOLO(config["model"])

        if config.get("use_sord"):
            sigma = config.get("sord_sigma", 0.8)
            print(f"[{exp_id}] Registering SORD callback (sigma={sigma})")
            model.add_callback("on_train_start", make_sord_trainer(sigma))

        model.train(**train_kwargs)
    except Exception as e:
        print(f"[{exp_id}] TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {"map50": 0.0, "error": str(e)}

    # Parse results
    result = parse_results(run_dir)
    print(f"\n[{exp_id}] RESULT: mAP50={result.get('map50', 0):.4f} | "
          f"Recall={result.get('recall', 0):.4f} | "
          f"BestEpoch={result.get('best_epoch', 0)}/{result.get('total_epochs', 0)}")

    return result


def main():
    print("=" * 60)
    print("Hermes-YOLO Novel Strategy Runner")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Experiments: {len(EXPERIMENTS)}")
    print(f"Dataset: {DATASET_DIR}")
    print("=" * 60)

    # Ensure dataset is accessible
    assert (DATASET_DIR / "images" / "train").exists(), f"Dataset not found: {DATASET_DIR}"
    assert (DATASET_DIR / "labels" / "train").exists(), f"Labels not found!"
    print(f"Dataset OK: {len(list((DATASET_DIR / 'images' / 'train').glob('*.jpg')))} train images")

    # Ensure runs dir exists
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-generate LAB dataset (do it once before all experiments)
    lab_yaml = BASE_DIR / "dataset_novel_lab.yaml"
    if not lab_yaml.exists():
        prepare_lab_dataset()

    # Initial push
    # Token read from env (set externally before running: export GITHUB_TOKEN=...)
    generate_chart()
    git_push("feat: add IDEA.md, novel_runner.py, dataset_novel.yaml — begin NOVEL series")

    # Run experiments
    all_results = {}

    for config in EXPERIMENTS:
        exp_id = config["id"]

        # Train
        result = run_experiment(config)
        all_results[exp_id] = result

        # Update tracking
        update_idea_md(exp_id, config["idea_id"], result, config)
        update_leaderboard(exp_id, result, config)

        # Regenerate chart
        generate_chart()

        # Push after each experiment
        map50 = result.get("map50", 0)
        msg = f"experiment: {exp_id} — mAP50={map50:.4f} ({config['name']})"
        git_push(msg)

        print(f"\n[Progress] Completed {exp_id}: mAP50={map50:.4f}")

    # Final summary
    print("\n" + "=" * 60)
    print("NOVEL Series Complete!")
    print("=" * 60)
    for exp_id, result in all_results.items():
        print(f"  {exp_id}: mAP50={result.get('map50', 0):.4f}")

    best_id = max(all_results, key=lambda k: all_results[k].get("map50", 0))
    best_val = all_results[best_id].get("map50", 0)
    print(f"\nBest NOVEL: {best_id} → mAP50={best_val:.4f}")

    # Final push
    generate_chart()
    git_push(f"final: NOVEL series complete — best={best_id} mAP50={best_val:.4f}")


if __name__ == "__main__":
    main()
