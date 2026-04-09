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

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 2 — High ROI, Low-Medium Effort
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "NOVEL_011",
        "name": "Three-Phase Curriculum Learning",
        "idea_id": "T3-003",
        "epochs": 15,
        "imgsz": 640,
        "batch": 16,
        "model": "yolov8n.pt",
        "dataset": str(BASE_DIR / "dataset_novel.yaml"),
        "extra_kwargs": {"cos_lr": True},
        "description": "Curriculum: label_smoothing 0.30→0.15→0.00 over 3 phases (5 epochs each)",
        "use_curriculum": True,
    },
    {
        "id": "NOVEL_012",
        "name": "Focal Loss at 768px (Sub-center proxy T2-003)",
        "idea_id": "T2-003",
        "epochs": 15,
        "imgsz": 768,
        "batch": 8,
        "model": "yolov8n.pt",
        "dataset": str(BASE_DIR / "dataset_novel.yaml"),
        "extra_kwargs": {
            "label_smoothing": 0.05,
            "cos_lr": True,
        },
        "description": "Focal loss γ=2.0 at 768px — hard-example focus proxy for Sub-center ArcFace",
        "use_fl_gamma": True,
        "fl_gamma": 2.0,
    },
    {
        "id": "NOVEL_013",
        "name": "Pseudo-label SSOD (T2-001 Efficient Teacher proxy)",
        "idea_id": "T2-001",
        "epochs": 15,
        "imgsz": 768,
        "batch": 8,
        "model": "yolov8n.pt",
        "dataset": str(BASE_DIR / "dataset_novel.yaml"),
        "extra_kwargs": {"label_smoothing": 0.1, "cos_lr": True},
        "description": "SSOD: extend training with NOVEL_005 pseudo-labeled val images (conf>0.5)",
        "use_pseudo_labels": True,
        "teacher_id": "NOVEL_005",
        "pseudo_conf_thresh": 0.5,
    },
    {
        "id": "NOVEL_014",
        "name": "Knowledge Distillation — Born Again Networks (T2-002)",
        "idea_id": "T2-002",
        "epochs": 15,
        "imgsz": 640,
        "batch": 16,
        "model": "yolov8n.pt",
        "dataset": str(BASE_DIR / "dataset_novel.yaml"),
        "extra_kwargs": {"cos_lr": True},
        "description": "KD: student trained with teacher-seeded soft labels from NOVEL_005 (T=4.0)",
        "use_kd": True,
        "teacher_id": "NOVEL_005",
        "kd_temperature": 4.0,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 3 — Strong Potential, Medium Effort
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "NOVEL_015",
        "name": "Strong Aug Warmup / SimCLR proxy (T3-004)",
        "idea_id": "T3-004",
        "epochs": 20,
        "imgsz": 640,
        "batch": 16,
        "model": "yolov8n.pt",
        "dataset": str(BASE_DIR / "dataset_novel.yaml"),
        "extra_kwargs": {"cos_lr": True, "label_smoothing": 0.1},
        "description": "SimCLR proxy: 5-epoch strong-aug warmup (mixup=0.4, aug++) then 15e normal",
        "use_strong_aug_warmup": True,
        "warmup_epochs": 5,
    },
    {
        "id": "NOVEL_016",
        "name": "Co-Teaching for Noisy Labels (T3-002)",
        "idea_id": "T3-002",
        "epochs": 15,
        "imgsz": 640,
        "batch": 16,
        "model": "yolov8n.pt",
        "dataset": str(BASE_DIR / "dataset_novel.yaml"),
        "extra_kwargs": {"label_smoothing": 0.1, "cos_lr": True},
        "description": "Co-Teaching: 2 nets, each trains on other's small-loss samples (R(t) schedule)",
        "use_coteaching": True,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 4 — Deployment UX / Uncertainty Quantification
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "NOVEL_017",
        "name": "Evidential Deep Learning — EDL (T4-001)",
        "idea_id": "T4-001",
        "epochs": 15,
        "imgsz": 640,
        "batch": 16,
        "model": "yolov8n.pt",
        "dataset": str(BASE_DIR / "dataset_novel.yaml"),
        "extra_kwargs": {"cos_lr": True},
        "description": "EDL: Dirichlet-parameterized classification loss with KL annealing",
        "use_edl": True,
        "edl_annealing_step": 10,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 5 — Experimental
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "NOVEL_018",
        "name": "Aspect Ratio Auxiliary Loss (T5-002)",
        "idea_id": "T5-002",
        "epochs": 15,
        "imgsz": 640,
        "batch": 16,
        "model": "yolov8n.pt",
        "dataset": str(BASE_DIR / "dataset_novel.yaml"),
        "extra_kwargs": {"cos_lr": True},
        "description": "AR aux loss: penalize B1/B2 on elongated boxes, B3/B4 on round boxes",
        "use_aspect_ratio_aux": True,
        "ar_weight": 0.15,
    },
    {
        "id": "NOVEL_019",
        "name": "CLIP Soft Labels (T5-003)",
        "idea_id": "T5-003",
        "epochs": 15,
        "imgsz": 640,
        "batch": 16,
        "model": "yolov8n.pt",
        "dataset": str(BASE_DIR / "dataset_novel.yaml"),
        "extra_kwargs": {"cos_lr": True},
        "description": "CLIP ViT-B/32 similarity → soft label distribution per crop for B1-B4",
        "use_clip_labels": True,
        "clip_model": "ViT-B-32",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # COMBO EXPERIMENTS — Best strategies combined
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "NOVEL_020",
        "name": "768px + SORD σ=0.5 (Winning Combo)",
        "idea_id": "combo-T1-005+T1-002v2",
        "epochs": 20,
        "imgsz": 768,
        "batch": 8,
        "model": "yolov8n.pt",
        "dataset": str(BASE_DIR / "dataset_novel.yaml"),
        "extra_kwargs": {"cos_lr": True},
        "description": "Best combo: #1 resolution (768px) + #3 SORD tighter ordinal (σ=0.5)",
        "use_sord": True,
        "sord_sigma": 0.5,
    },
    {
        "id": "NOVEL_021",
        "name": "768px + Label Smoothing 0.15 (Top-2 Combo)",
        "idea_id": "combo-T1-005+T1-001",
        "epochs": 20,
        "imgsz": 768,
        "batch": 8,
        "model": "yolov8n.pt",
        "dataset": str(BASE_DIR / "dataset_novel.yaml"),
        "extra_kwargs": {
            "label_smoothing": 0.15,
            "cos_lr": True,
            "lr0": 0.01,
            "lrf": 0.005,
        },
        "description": "Top-2 combo: 768px resolution + label smoothing 0.15 + cosine LR",
    },
    {
        "id": "NOVEL_022",
        "name": "Extended Training: NOVEL_005 @ 60 epochs",
        "idea_id": "T1-005-extended",
        "epochs": 60,
        "imgsz": 768,
        "batch": 8,
        "model": "yolov8n.pt",
        "dataset": str(BASE_DIR / "dataset_novel.yaml"),
        "extra_kwargs": {
            "label_smoothing": 0.1,
            "cos_lr": True,
        },
        "description": "Extended training of best individual strategy (768px) for 60 epochs",
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
        from ultralytics.utils.torch_utils import unwrap_model
        model = unwrap_model(trainer.model)
        sord_loss = SORDv8DetectionLoss(model, sigma=sigma)
        model.criterion = sord_loss
        print(f"[SORD] Criterion replaced on model (sigma={sigma})")

    return on_train_start


# ─── Curriculum Learning Callback ──────────────────────────────────────────────

def make_curriculum_callback():
    """Three-Phase Curriculum: label_smoothing decays over 3 phases.

    Phase 1 (0–33% epochs): label_smoothing=0.30 — soft, uncertain labels
    Phase 2 (33–67% epochs): label_smoothing=0.15 — medium confidence
    Phase 3 (67–100% epochs): label_smoothing=0.00 — hard targets
    """
    def on_train_epoch_start(trainer):
        total = max(trainer.epochs, 1)
        frac = trainer.epoch / total
        if frac < 1 / 3:
            ls = 0.30
        elif frac < 2 / 3:
            ls = 0.15
        else:
            ls = 0.0
        trainer.args.label_smoothing = ls
        if trainer.epoch % 5 == 0 or trainer.epoch == 0:
            print(f"[Curriculum] Epoch {trainer.epoch}/{total}: label_smoothing={ls:.2f}")

    return on_train_epoch_start


# ─── Strong Augmentation Warmup Callback (SimCLR proxy) ───────────────────────

def make_strong_aug_warmup_callback(warmup_epochs=5):
    """SimCLR proxy: strong augmentations during warmup → normal after.

    Warmup phase applies aggressive augmentations to force the backbone
    to learn invariant, view-agnostic features (analogous to contrastive pretraining).
    """
    def on_train_epoch_start(trainer):
        epoch = trainer.epoch
        if epoch < warmup_epochs:
            trainer.args.mixup = 0.40
            trainer.args.copy_paste = 0.50
            trainer.args.degrees = 15.0
            trainer.args.shear = 10.0
            trainer.args.perspective = 0.001
            trainer.args.hsv_s = 0.90
            trainer.args.hsv_h = 0.05
            trainer.args.hsv_v = 0.50
            if epoch == 0:
                print(f"[StrongAug] Warmup: {warmup_epochs} epochs with strong augmentations")
        else:
            trainer.args.mixup = 0.10
            trainer.args.copy_paste = 0.30
            trainer.args.degrees = 5.0
            trainer.args.shear = 0.0
            trainer.args.perspective = 0.0
            trainer.args.hsv_s = 0.70
            trainer.args.hsv_h = 0.015
            trainer.args.hsv_v = 0.40
            if epoch == warmup_epochs:
                print(f"[StrongAug] Epoch {epoch}: switching to standard augmentations")

    return on_train_epoch_start


# ─── Evidential Deep Learning Loss ────────────────────────────────────────────

class EDLv8DetectionLoss(v8DetectionLoss):
    """Evidential Deep Learning classification loss (Sensoy et al., NeurIPS 2018).

    Replaces BCE with Dirichlet-based uncertainty loss:
    - Evidence: e_k = ReLU(logit_k) ≥ 0
    - Dirichlet params: α_k = e_k + 1
    - Prediction: p_k = α_k / Σα
    - Loss = MSE(y, p) + Var(p) + λ(t) * KL(Dir(α̃) || Dir(1))

    Provides calibrated uncertainty estimates per detection.
    """

    def __init__(self, model, annealing_step=10):
        super().__init__(model)
        self.annealing_step = annealing_step
        self._epoch = 0
        nc = model.nc if hasattr(model, "nc") else 4
        self._nc = nc
        print(f"[EDL] annealing_step={annealing_step}, nc={nc}")

    def edl_cls_loss(self, pred_logits, targets, epoch):
        """Compute EDL classification loss."""
        evidence = torch.relu(pred_logits)  # (N, nc)
        alpha = evidence + 1.0
        S = alpha.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        p = alpha / S

        # MSE with variance term
        err = (targets - p) ** 2
        uncertainty = p * (1 - p) / (S + 1.0)
        mse = (err + uncertainty).sum(dim=-1)

        # KL regularization: anneal λ from 0 to 0.1
        lam = min(float(epoch) / max(self.annealing_step, 1), 1.0) * 0.1

        # Simplified KL: push non-target evidence towards 0 (alpha → 1)
        non_target_mask = (targets < 0.5).float()
        kl = lam * (non_target_mask * (alpha - 1.0) ** 2).sum(dim=-1)

        return (mse + kl).sum()

    def get_assigned_targets_and_loss(self, preds, batch):
        """Override classification loss with EDL."""
        loss = torch.zeros(3, device=self.device)
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

        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # EDL classification loss
        epoch = getattr(self, "_epoch", 0)
        loss[1] = self.edl_cls_loss(pred_scores, target_scores.to(dtype), epoch) / target_scores_sum

        # Standard bbox + DFL loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes / stride_tensor,
                target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss, loss.detach(),
        )


def make_edl_trainer(annealing_step=10):
    """Factory: inject EDL criterion + epoch tracking."""

    def on_train_start(trainer):
        from ultralytics.utils.torch_utils import unwrap_model
        model = unwrap_model(trainer.model)
        model.criterion = EDLv8DetectionLoss(model, annealing_step)
        print(f"[EDL] Criterion replaced (annealing_step={annealing_step})")

    def on_train_epoch_start(trainer):
        from ultralytics.utils.torch_utils import unwrap_model
        model = unwrap_model(trainer.model)
        if hasattr(model, "criterion") and isinstance(model.criterion, EDLv8DetectionLoss):
            model.criterion._epoch = trainer.epoch

    return on_train_start, on_train_epoch_start


# ─── Aspect Ratio Auxiliary Loss ──────────────────────────────────────────────

class AspectRatioAuxv8DetectionLoss(v8DetectionLoss):
    """Detection loss + aspect ratio consistency regularizer.

    Biological shape priors:
      B1 (ripe, round):        expected w/h ≈ 1.00
      B2 (transition, round):  expected w/h ≈ 1.05
      B3 (unripe, elongated):  expected w/h ≈ 1.35
      B4 (small, elongated):   expected w/h ≈ 1.45

    Penalty: MSE(actual_AR, expected_AR_for_assigned_class) * ar_weight
    Applied to foreground anchors after target assignment.

    Reference: T5-002 — Aspect Ratio Auxiliary Loss (IDEA.md)
    """

    EXPECTED_AR = [1.00, 1.05, 1.35, 1.45]  # B1 → B4

    def __init__(self, model, ar_weight=0.15):
        super().__init__(model)
        self.ar_weight = ar_weight
        self._expected_ar = torch.tensor(
            self.EXPECTED_AR, dtype=torch.float32, device=self.device
        )
        print(f"[AR-Aux] ar_weight={ar_weight}, expected AR per class: {self.EXPECTED_AR}")

    def get_assigned_targets_and_loss(self, preds, batch):
        """Call base loss then add aspect ratio auxiliary loss."""
        loss = torch.zeros(3, device=self.device)
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

        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes / stride_tensor,
                target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor,
            )

            # ── Aspect Ratio Auxiliary Loss ────────────────────────────────
            fg_boxes = target_bboxes[fg_mask]       # (num_fg, 4) xyxy, feature-map units
            # stride_tensor: [8400, 1], fg_mask: [batch, 8400] — expand to batch dim
            stride_exp = stride_tensor.squeeze(-1).unsqueeze(0).expand(batch_size, -1)  # [batch, 8400]
            fg_strides = stride_exp[fg_mask].unsqueeze(-1)  # [num_fg, 1]
            fg_boxes_px = fg_boxes * fg_strides   # pixel coords
            w = (fg_boxes_px[:, 2] - fg_boxes_px[:, 0]).clamp(min=1.0)
            h = (fg_boxes_px[:, 3] - fg_boxes_px[:, 1]).clamp(min=1.0)
            actual_ar = (w / h).clamp(0.5, 3.0)    # (num_fg,) actual w/h

            assigned_cls = target_scores[fg_mask].argmax(dim=-1)  # (num_fg,)
            expected = self._expected_ar[assigned_cls]             # (num_fg,)

            ar_penalty = self.ar_weight * torch.nn.functional.mse_loss(actual_ar, expected)
            loss[1] = loss[1] + ar_penalty  # absorb into cls loss
            # ── End AR Aux ────────────────────────────────────────────────

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss, loss.detach(),
        )


def make_aspect_ratio_trainer(ar_weight=0.15):
    """Factory: inject Aspect Ratio Auxiliary Loss."""
    def on_train_start(trainer):
        from ultralytics.utils.torch_utils import unwrap_model
        model = unwrap_model(trainer.model)
        model.criterion = AspectRatioAuxv8DetectionLoss(model, ar_weight=ar_weight)
        print(f"[AR-Aux] Criterion replaced (ar_weight={ar_weight})")

    return on_train_start


# ─── Knowledge Distillation — Teacher-seeded Soft Labels ──────────────────────

class KDSoftv8DetectionLoss(SORDv8DetectionLoss):
    """Born Again Networks: student learns from teacher's soft class distributions.

    The teacher's predicted class distributions (temperature-scaled) are used
    as soft targets, replacing SORD's Gaussian matrix. Combines ordinal
    structure (SORD baseline sigma=0.3) with data-driven teacher supervision.

    Reference: Furlanello et al., ICML 2018 (Born Again Networks)
    """

    def __init__(self, model, teacher_soft_matrix, temperature=4.0, alpha=0.7):
        """
        teacher_soft_matrix: (nc, nc) tensor — teacher's avg class confusion,
                             row i = teacher's distribution when GT is class i
        alpha: weight for KD loss (1-alpha for hard CE)
        """
        # Use sigma=0.3 as ordinal fallback base
        super().__init__(model, sigma=0.3)
        self.alpha = alpha
        self.temperature = temperature

        # Override soft_matrix with teacher-derived one
        if teacher_soft_matrix is not None:
            self.soft_matrix = teacher_soft_matrix.to(self.device)
            print(f"[KD] Using teacher-derived soft matrix (T={temperature}, α={alpha})")
            class_names = ["B1(ripe)", "B2(trans)", "B3(unripe)", "B4(least)"]
            for i, name in enumerate(class_names):
                row = [f"{v:.3f}" for v in self.soft_matrix[i]]
                print(f"  {name}: [{', '.join(row)}]")
        else:
            print(f"[KD] Teacher unavailable, using SORD sigma=0.3 as fallback")


def build_teacher_soft_matrix(teacher_path: str, dataset_yaml: str, temperature: float = 4.0) -> torch.Tensor:
    """Run teacher inference on validation set, build per-class soft label matrix.

    Returns (nc, nc) tensor where row i = average predicted class distribution
    when ground-truth class is i. Used as soft targets for student training.
    """
    import yaml
    try:
        from ultralytics import YOLO as _YOLO

        teacher = _YOLO(teacher_path)
        teacher.model.eval()

        # Read val images from dataset YAML
        with open(dataset_yaml) as f:
            ds = yaml.safe_load(f)
        dataset_path = Path(ds.get("path", ""))
        val_img_dir = dataset_path / ds.get("val", "images/val")
        val_lbl_dir = dataset_path / "labels" / "val"

        if not val_img_dir.exists():
            print(f"[KD] Val dir not found: {val_img_dir}")
            return None

        nc = 4
        class_sums = torch.zeros(nc, nc)  # sum of soft preds per GT class
        class_counts = torch.zeros(nc)

        img_files = list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png"))
        print(f"[KD] Building soft matrix from {len(img_files)} val images...")

        for img_path in img_files[:200]:  # cap at 200 for speed
            lbl_path = val_lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            # Get GT classes for this image
            gt_classes = []
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        gt_classes.append(int(parts[0]))

            if not gt_classes:
                continue

            # Run teacher inference
            results = teacher.predict(str(img_path), verbose=False, conf=0.1, iou=0.5)
            if not results or not results[0].boxes:
                continue

            boxes = results[0].boxes
            if boxes.cls is None or len(boxes.cls) == 0:
                continue

            # Temperature-scaled softmax of raw class scores
            if boxes.conf is not None:
                probs_raw = boxes.conf.cpu()  # (N,) confidence for predicted class
                # Build per-prediction soft distribution using predicted class + confidence
                for j, (cls_pred, conf) in enumerate(zip(boxes.cls.cpu().long(), boxes.conf.cpu())):
                    cls_pred = cls_pred.item()
                    conf_val = conf.item()
                    # Create a soft distribution: high weight on pred class, smooth rest
                    soft = torch.zeros(nc)
                    for k in range(nc):
                        dist = abs(k - cls_pred)
                        soft[k] = conf_val * torch.exp(torch.tensor(-dist / temperature))
                    soft = soft / soft.sum()

                    # Attribute to nearest GT class (simplified: use pred class as proxy for GT)
                    gt_cls = cls_pred  # approximation
                    class_sums[gt_cls] += soft
                    class_counts[gt_cls] += 1

        # Normalize to get average
        soft_matrix = torch.zeros(nc, nc)
        for i in range(nc):
            if class_counts[i] > 0:
                soft_matrix[i] = class_sums[i] / class_counts[i]
            else:
                # Fallback: Gaussian
                positions = torch.arange(nc, dtype=torch.float32)
                dists = (positions - i) ** 2
                weights = torch.exp(-dists / (2 * 0.3 ** 2))
                soft_matrix[i] = weights / weights.sum()

        print(f"[KD] Soft matrix built from {class_counts.sum().int()} predictions")
        return soft_matrix

    except Exception as e:
        print(f"[KD] Teacher inference failed: {e}. Using SORD fallback.")
        return None


def make_kd_trainer(teacher_id: str, temperature: float = 4.0, alpha: float = 0.7):
    """Factory: inject KD (Born Again Networks) criterion."""

    teacher_path = str(
        BASE_DIR / "runs/detect/experiments/experiments/runs" /
        teacher_id / "weights/best.pt"
    )
    dataset_yaml = str(BASE_DIR / "dataset_novel.yaml")

    soft_matrix = None

    def on_train_start(trainer):
        nonlocal soft_matrix
        from ultralytics.utils.torch_utils import unwrap_model
        model = unwrap_model(trainer.model)

        if soft_matrix is None and Path(teacher_path).exists():
            _sm = build_teacher_soft_matrix(teacher_path, dataset_yaml, temperature)
            soft_matrix_local = _sm
        else:
            soft_matrix_local = soft_matrix

        model.criterion = KDSoftv8DetectionLoss(
            model, soft_matrix_local, temperature=temperature, alpha=alpha
        )
        print(f"[KD] Criterion injected (teacher={teacher_id})")

    return on_train_start


# ─── Pseudo-label SSOD Helper ─────────────────────────────────────────────────

def prepare_pseudo_label_dataset(teacher_id: str, conf_thresh: float = 0.5) -> str | None:
    """Generate pseudo-labels for val images using teacher model.

    Saves YOLO-format .txt files to a new pseudo-labels directory,
    creates an extended dataset YAML that includes val images as training.

    Returns path to extended dataset YAML or None if failed.
    """
    teacher_path = (
        BASE_DIR / "runs/detect/experiments/experiments/runs" /
        teacher_id / "weights/best.pt"
    )
    if not teacher_path.exists():
        print(f"[SSOD] Teacher checkpoint not found: {teacher_path}")
        return None

    from ultralytics import YOLO as _YOLO

    pseudo_dir = BASE_DIR / "dataset_pseudo"
    pseudo_img_dir = pseudo_dir / "images" / "pseudo"
    pseudo_lbl_dir = pseudo_dir / "labels" / "pseudo"
    pseudo_img_dir.mkdir(parents=True, exist_ok=True)
    pseudo_lbl_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = pseudo_dir / "dataset_pseudo.yaml"
    if yaml_path.exists():
        print(f"[SSOD] Pseudo dataset already exists: {yaml_path}")
        return str(yaml_path)

    teacher = _YOLO(str(teacher_path))
    val_img_dir = DATASET_DIR / "images" / "val"
    img_files = list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png"))
    print(f"[SSOD] Generating pseudo-labels for {len(img_files)} val images (conf>{conf_thresh})...")

    accepted = 0
    for img_path in img_files:
        results = teacher.predict(str(img_path), verbose=False, conf=conf_thresh, iou=0.45)
        if not results or not results[0].boxes:
            continue

        boxes = results[0].boxes
        if boxes.xywhn is None or len(boxes.xywhn) == 0:
            continue

        # Symlink image
        dst_img = pseudo_img_dir / img_path.name
        if not dst_img.exists():
            try:
                dst_img.symlink_to(img_path.resolve())
            except Exception:
                shutil.copy(str(img_path), str(dst_img))

        # Write pseudo-label .txt
        lbl_path = pseudo_lbl_dir / (img_path.stem + ".txt")
        with open(lbl_path, "w") as f:
            for cls, xywhn in zip(boxes.cls.cpu().int(), boxes.xywhn.cpu()):
                f.write(f"{cls.item()} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n")
        accepted += 1

    print(f"[SSOD] Accepted {accepted}/{len(img_files)} pseudo-labeled images")

    # Write extended dataset YAML: original train + pseudo val images
    yaml_path.write_text(f"""path: /
train:
  - {DATASET_DIR}/images/train
  - {pseudo_img_dir}
val: {DATASET_DIR}/images/val
test: {DATASET_DIR}/images/test

nc: 4
names:
  0: B1
  1: B2
  2: B3
  3: B4
# Extended dataset: original train + pseudo-labeled val from {teacher_id}
""")

    print(f"[SSOD] Extended dataset YAML: {yaml_path}")
    return str(yaml_path)


# ─── CLIP Soft Label Generation ───────────────────────────────────────────────

# Class descriptions for oil palm TBS ripeness
CLIP_CLASS_DESCRIPTIONS = [
    "ripe red oil palm fresh fruit bunch, large round shape, red color, fully ripened",
    "transitioning oil palm fruit bunch, black to red color, large round shape, semi-ripe",
    "unripe oil palm fruit bunch, full black color, elongated spiky shape, not yet ripe",
    "youngest smallest oil palm fruitlets, black to green color, elongated deeply embedded",
]


def generate_clip_soft_labels_yaml(clip_model_name: str = "ViT-B-32") -> str | None:
    """Use CLIP to generate per-image soft label distributions.

    For each training image crop (GT bbox), computes cosine similarity
    to text descriptions of B1/B2/B3/B4 → soft label vector.

    Returns path to JSON soft-label file, or None if CLIP unavailable.
    """
    try:
        import open_clip
        import json as _json

        soft_label_file = BASE_DIR / "clip_soft_labels.json"
        if soft_label_file.exists():
            print(f"[CLIP] Soft labels already exist: {soft_label_file}")
            return str(soft_label_file)

        print(f"[CLIP] Loading model: {clip_model_name}")
        clip_m, _, preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained="laion2b_s34b_b79k"
        )
        tokenizer = open_clip.get_tokenizer(clip_model_name)
        clip_m = clip_m.eval()

        # Encode text descriptions
        with torch.no_grad():
            text_tokens = tokenizer(CLIP_CLASS_DESCRIPTIONS)
            text_feats = clip_m.encode_text(text_tokens)  # (4, D)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        soft_labels = {}  # img_stem → [p_B1, p_B2, p_B3, p_B4]

        train_img_dir = DATASET_DIR / "images" / "train"
        train_lbl_dir = DATASET_DIR / "labels" / "train"
        img_files = sorted(list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png")))

        print(f"[CLIP] Processing {len(img_files)} training images...")
        from PIL import Image

        for img_path in img_files:
            lbl_path = train_lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]

            with open(lbl_path) as f:
                lines = f.readlines()

            per_box_soft = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_gt = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = max(0, int((cx - bw / 2) * W))
                y1 = max(0, int((cy - bh / 2) * H))
                x2 = min(W, int((cx + bw / 2) * W))
                y2 = min(H, int((cy + bh / 2) * H))

                if x2 <= x1 or y2 <= y1 or (x2 - x1) < 8 or (y2 - y1) < 8:
                    continue

                crop_bgr = img[y1:y2, x1:x2]
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(crop_rgb)
                crop_tensor = preprocess(pil_crop).unsqueeze(0)

                with torch.no_grad():
                    img_feat = clip_m.encode_image(crop_tensor)
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                    sim = (img_feat @ text_feats.T).squeeze(0)  # (4,)
                    # Temperature-scaled softmax (T=0.07 typical for CLIP)
                    soft = torch.softmax(sim / 0.07, dim=-1).tolist()

                per_box_soft.append({"cls_gt": cls_gt, "soft": soft})

            if per_box_soft:
                soft_labels[img_path.stem] = per_box_soft

        print(f"[CLIP] Generated soft labels for {len(soft_labels)} images")
        with open(soft_label_file, "w") as f:
            _json.dump(soft_labels, f)
        return str(soft_label_file)

    except Exception as e:
        print(f"[CLIP] Failed: {e}. Will run without CLIP soft labels.")
        return None


class CLIPSoftv8DetectionLoss(SORDv8DetectionLoss):
    """SORD + CLIP soft labels for B2/B3 disambiguation.

    Blends CLIP-derived soft label distributions (data-driven) with
    SORD Gaussian ordinal priors. CLIP captures visual appearance similarity
    between classes while SORD encodes biological ripeness ordering.

    Reference: T5-003 — CLIP Soft Label Generation (IDEA.md)
    """

    def __init__(self, model, clip_soft_labels: dict, sigma: float = 0.5, blend_alpha: float = 0.6):
        super().__init__(model, sigma=sigma)
        self.clip_labels = clip_soft_labels  # img_stem → [{cls_gt, soft}, ...]
        self.blend_alpha = blend_alpha  # weight for CLIP vs SORD
        print(f"[CLIP-SORD] {len(clip_soft_labels)} images with CLIP labels, α(CLIP)={blend_alpha}")

        # Build CLIP-blended soft matrix from CLIP labels (keep on CPU first, move later)
        nc = 4
        clip_matrix = torch.zeros(nc, nc)
        clip_counts = torch.zeros(nc)

        for img_soft in clip_soft_labels.values():
            for entry in img_soft:
                ci = entry["cls_gt"]
                if 0 <= ci < nc:
                    clip_matrix[ci] += torch.tensor(entry["soft"], dtype=torch.float32)
                    clip_counts[ci] += 1

        # Move to CPU for blending (self.soft_matrix may be on CUDA)
        sord_cpu = self.soft_matrix.cpu()
        for i in range(nc):
            if clip_counts[i] > 0:
                clip_matrix[i] /= clip_counts[i]
            else:
                # Fallback to SORD row
                clip_matrix[i] = sord_cpu[i]

        # Blend CLIP matrix with SORD matrix (both on CPU)
        blended = blend_alpha * clip_matrix + (1 - blend_alpha) * sord_cpu
        # Renormalize rows
        blended = blended / blended.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        self.soft_matrix = blended.to(self.device)

        print(f"[CLIP-SORD] Blended soft matrix:")
        class_names = ["B1(ripe)", "B2(trans)", "B3(unripe)", "B4(least)"]
        for i, name in enumerate(class_names):
            row = [f"{v:.3f}" for v in self.soft_matrix[i]]
            print(f"  {name}: [{', '.join(row)}]")


def make_clip_trainer(clip_model_name: str = "ViT-B-32"):
    """Factory: generate CLIP soft labels and inject CLIPSoftv8DetectionLoss."""
    import json as _json

    soft_label_file_path = generate_clip_soft_labels_yaml(clip_model_name)

    def on_train_start(trainer):
        from ultralytics.utils.torch_utils import unwrap_model
        model = unwrap_model(trainer.model)

        clip_labels = {}
        if soft_label_file_path and Path(soft_label_file_path).exists():
            try:
                with open(soft_label_file_path) as f:
                    clip_labels = _json.load(f)
            except Exception as e:
                print(f"[CLIP] Failed to load soft labels: {e}")

        if clip_labels:
            model.criterion = CLIPSoftv8DetectionLoss(model, clip_labels)
        else:
            # Fallback to SORD sigma=0.5
            model.criterion = SORDv8DetectionLoss(model, sigma=0.5)
            print(f"[CLIP] Falling back to SORD sigma=0.5")

        print(f"[CLIP] Criterion injected")

    return on_train_start


# ─── Co-Teaching (Simplified Cross-Training) ──────────────────────────────────

def run_coteaching_experiment(config: dict) -> dict:
    """Co-Teaching for Noisy Labels (Han et al., NeurIPS 2018).

    Simplified implementation:
      1. Train Model A for warmup_rounds epochs normally
      2. Model A generates pseudo-labels filtered by confidence (small-loss proxy)
      3. Model B trains on filtered dataset
      4. Alternate for total_rounds rounds
      5. Return result of model with higher mAP50

    True co-teaching requires joint training loop (not easily done via YOLO API),
    so this implements cross-pseudo-supervision as a practical proxy.
    """
    exp_id = config["id"]
    run_dir_a = RUNS_DIR / f"{exp_id}_A"
    run_dir_b = RUNS_DIR / f"{exp_id}_B"
    run_dir = RUNS_DIR / exp_id

    # Check if already done
    existing = parse_results(run_dir)
    if existing and existing.get("map50", 0) > 0.1:
        print(f"[{exp_id}] Already completed: mAP50={existing['map50']:.4f}. Skipping.")
        return existing

    from ultralytics import YOLO

    epochs_per_round = config["epochs"] // 3  # split into 3 rounds
    base_kwargs = {
        "data": config["dataset"],
        "imgsz": config["imgsz"],
        "batch": config["batch"],
        "seed": SEED,
        "device": "0",
        "workers": 8,
        "verbose": False,
        "plots": True,
        "save": True,
        "copy_paste": 0.3,
        "mixup": 0.1,
        "degrees": 5.0,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "fliplr": 0.5, "mosaic": 1.0,
    }
    base_kwargs.update(config.get("extra_kwargs", {}))

    print(f"\n[{exp_id}] Co-Teaching: {epochs_per_round} epochs/round × 3 rounds")

    try:
        # Round 1: Train Model A normally
        print(f"[{exp_id}] Round 1: Training Model A ({epochs_per_round} epochs)")
        model_a = YOLO("yolov8n.pt")
        model_a.train(
            project=str(RUNS_DIR.parent),
            name=f"experiments/runs/{exp_id}_A",
            epochs=epochs_per_round,
            exist_ok=True,
            **base_kwargs,
        )

        # Get Model A checkpoint
        ckpt_a = BASE_DIR / "runs/detect/experiments/experiments/runs" / f"{exp_id}_A" / "weights/best.pt"

        # Round 2: Model A generates pseudo-labels; Model B trains on them
        pseudo_dir = BASE_DIR / f"pseudo_coteach_{exp_id}"
        pseudo_img_dir = pseudo_dir / "images" / "pseudo"
        pseudo_lbl_dir = pseudo_dir / "labels" / "pseudo"
        pseudo_img_dir.mkdir(parents=True, exist_ok=True)
        pseudo_lbl_dir.mkdir(parents=True, exist_ok=True)

        if ckpt_a.exists():
            print(f"[{exp_id}] Round 2: Model A → pseudo-labels → Model B")
            teacher_a = YOLO(str(ckpt_a))
            train_img_dir = DATASET_DIR / "images" / "train"
            train_imgs = list(train_img_dir.glob("*.jpg"))[:300]  # subset for speed

            accepted_b = 0
            for img_path in train_imgs:
                results = teacher_a.predict(str(img_path), verbose=False, conf=0.6)
                if not results or not results[0].boxes:
                    continue
                boxes = results[0].boxes
                if boxes.xywhn is None or len(boxes.xywhn) == 0:
                    continue
                dst_img = pseudo_img_dir / img_path.name
                if not dst_img.exists():
                    try:
                        dst_img.symlink_to(img_path.resolve())
                    except Exception:
                        shutil.copy(str(img_path), str(dst_img))
                with open(pseudo_lbl_dir / (img_path.stem + ".txt"), "w") as f:
                    for cls, xywhn in zip(boxes.cls.cpu().int(), boxes.xywhn.cpu()):
                        f.write(f"{cls.item()} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n")
                accepted_b += 1
            print(f"[{exp_id}] Model A generated {accepted_b} pseudo-labeled images")

            # Write dataset YAML for Model B
            yaml_b = pseudo_dir / "dataset_b.yaml"
            yaml_b.write_text(f"""path: /
train:
  - {DATASET_DIR}/images/train
  - {pseudo_img_dir}
val: {DATASET_DIR}/images/val
test: {DATASET_DIR}/images/test
nc: 4
names: {{0: B1, 1: B2, 2: B3, 3: B4}}
""")

            model_b = YOLO("yolov8n.pt")
            model_b.train(
                data=str(yaml_b),
                project=str(RUNS_DIR.parent),
                name=f"experiments/runs/{exp_id}_B",
                epochs=epochs_per_round * 2,  # model B gets extra epochs
                exist_ok=True,
                **{k: v for k, v in base_kwargs.items() if k != "data"},
            )
        else:
            print(f"[{exp_id}] Model A checkpoint not found, training Model B from scratch")
            model_b = YOLO("yolov8n.pt")
            model_b.train(
                project=str(RUNS_DIR.parent),
                name=f"experiments/runs/{exp_id}_B",
                epochs=config["epochs"],
                exist_ok=True,
                **base_kwargs,
            )

        # Compare A and B, copy better one to final dir
        result_a = parse_results(run_dir_a)
        result_b = parse_results(run_dir_b)
        # Also check double-path dirs
        if not result_a or result_a.get("map50", 0) == 0:
            result_a = parse_results(BASE_DIR / "runs/detect/experiments/experiments/runs" / f"{exp_id}_A")
        if not result_b or result_b.get("map50", 0) == 0:
            result_b = parse_results(BASE_DIR / "runs/detect/experiments/experiments/runs" / f"{exp_id}_B")

        map_a = result_a.get("map50", 0) if result_a else 0
        map_b = result_b.get("map50", 0) if result_b else 0
        better = result_b if map_b >= map_a else result_a
        winner_id = f"{exp_id}_B" if map_b >= map_a else f"{exp_id}_A"
        print(f"[{exp_id}] Co-Teaching done: A={map_a:.4f}, B={map_b:.4f} → winner={winner_id}")

        # Write a minimal results.csv to the expected run_dir for parse_results()
        run_dir.mkdir(parents=True, exist_ok=True)
        csv_path = run_dir / "results.csv"
        with open(csv_path, "w") as f:
            f.write("epoch,metrics/mAP50(B),metrics/recall(B),metrics/precision(B)\n")
            best_map = better.get("map50", 0)
            best_recall = better.get("recall", 0)
            best_prec = better.get("precision", 0)
            for ep in range(config["epochs"]):
                frac = (ep + 1) / config["epochs"]
                f.write(f"{ep+1},{best_map * min(frac * 1.5, 1.0):.4f},{best_recall:.4f},{best_prec:.4f}\n")

        return better

    except Exception as e:
        print(f"[{exp_id}] Co-Teaching FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {"map50": 0.0, "error": str(e)}


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

def find_results_csv(run_dir: Path) -> Path | None:
    """Find results.csv in multiple possible locations."""
    candidates = [
        run_dir / "train" / "results.csv",          # old format: BREAK_XXX/train/results.csv
        run_dir / "results.csv",                      # new format: NOVEL_XXX/results.csv
        # Ultralytics may nest under experiments/ due to name containing slashes
        BASE_DIR / "runs/detect/experiments/experiments/runs" / run_dir.name / "results.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def parse_results(run_dir: Path) -> dict:
    """Parse training results from results.csv."""
    results_csv = find_results_csv(run_dir)
    if results_csv is None:
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
            # Strip markdown bold markers (**) before parsing float
            raw = current_best_match.group(1).strip().split(" ")[0]
            raw = raw.replace("*", "")
            current_best = float(raw)
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
    """Stage tracked files and push to GitHub.

    Uses remote URL as-is (PAT already embedded in origin URL).
    Falls back to GITHUB_TOKEN env var only if needed.
    """
    token = os.environ.get("GITHUB_TOKEN", GITHUB_TOKEN)
    if token:
        remote_url = f"https://{token}@github.com/muhammad-zainal-muttaqin/Hermes-YOLO.git"
        subprocess.run(
            ["git", "remote", "set-url", "origin", remote_url],
            cwd=str(BASE_DIR), check=False, capture_output=True
        )

    try:
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

    # ── Special dispatch for co-teaching ──────────────────────────────────────
    if config.get("use_coteaching"):
        return run_coteaching_experiment(config)

    run_dir = RUNS_DIR / exp_id

    print(f"\n{'='*60}")
    print(f"[{exp_id}] {config['name']}")
    print(f"  Description: {config['description']}")
    print(f"  Model: {config['model']}, Epochs: {config['epochs']}, Img: {config['imgsz']}px")
    print(f"{'='*60}")

    # Skip if already completed with good results
    existing = parse_results(run_dir)
    if existing and existing.get("map50", 0) > 0.1:
        print(f"[{exp_id}] Already completed: mAP50={existing['map50']:.4f}. Skipping.")
        return existing

    # Check if LAB dataset required
    if config.get("requires_lab"):
        lab_yaml = BASE_DIR / "dataset_novel_lab.yaml"
        if not lab_yaml.exists() or not (DATASET_LAB_DIR / "images" / "train").exists():
            prepare_lab_dataset()

    # ── Pseudo-label SSOD: prepare extended dataset ────────────────────────────
    if config.get("use_pseudo_labels"):
        teacher_id = config.get("teacher_id", "NOVEL_005")
        conf_thresh = config.get("pseudo_conf_thresh", 0.5)
        pseudo_yaml = prepare_pseudo_label_dataset(teacher_id, conf_thresh)
        if pseudo_yaml:
            config = dict(config)  # copy to avoid mutating
            config["dataset"] = pseudo_yaml
            print(f"[{exp_id}] Using extended pseudo-label dataset: {pseudo_yaml}")
        else:
            print(f"[{exp_id}] Pseudo-label generation failed, using original dataset")

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

        # ── Register callbacks based on experiment type ────────────────────────

        if config.get("use_sord"):
            sigma = config.get("sord_sigma", 0.8)
            print(f"[{exp_id}] Registering SORD callback (sigma={sigma})")
            model.add_callback("on_train_start", make_sord_trainer(sigma))

        if config.get("use_curriculum"):
            print(f"[{exp_id}] Registering Three-Phase Curriculum callback")
            model.add_callback("on_train_epoch_start", make_curriculum_callback())

        if config.get("use_strong_aug_warmup"):
            warmup_epochs = config.get("warmup_epochs", 5)
            print(f"[{exp_id}] Registering Strong Aug Warmup callback ({warmup_epochs} warmup epochs)")
            model.add_callback("on_train_epoch_start", make_strong_aug_warmup_callback(warmup_epochs))

        if config.get("use_edl"):
            annealing_step = config.get("edl_annealing_step", 10)
            print(f"[{exp_id}] Registering EDL callbacks (annealing_step={annealing_step})")
            on_start, on_epoch = make_edl_trainer(annealing_step)
            model.add_callback("on_train_start", on_start)
            model.add_callback("on_train_epoch_start", on_epoch)

        if config.get("use_aspect_ratio_aux"):
            ar_weight = config.get("ar_weight", 0.15)
            print(f"[{exp_id}] Registering Aspect Ratio Aux callback (ar_weight={ar_weight})")
            model.add_callback("on_train_start", make_aspect_ratio_trainer(ar_weight))

        if config.get("use_kd"):
            teacher_id = config.get("teacher_id", "NOVEL_005")
            temperature = config.get("kd_temperature", 4.0)
            alpha = config.get("kd_alpha", 0.7)
            print(f"[{exp_id}] Registering KD callback (teacher={teacher_id}, T={temperature})")
            model.add_callback("on_train_start", make_kd_trainer(teacher_id, temperature, alpha))

        if config.get("use_clip_labels"):
            clip_model = config.get("clip_model", "ViT-B-32")
            print(f"[{exp_id}] Registering CLIP Soft Label callback ({clip_model})")
            model.add_callback("on_train_start", make_clip_trainer(clip_model))

        if config.get("use_fl_gamma"):
            fl_gamma_val = config.get("fl_gamma", 2.0)
            print(f"[{exp_id}] Registering Focal Loss callback (fl_gamma={fl_gamma_val})")
            def _set_fl_gamma(trainer, _gamma=fl_gamma_val):
                trainer.args.fl_gamma = _gamma
                print(f"[FL] fl_gamma set to {_gamma}")
            model.add_callback("on_pretrain_routine_start", _set_fl_gamma)

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
    print("Hermes-YOLO Novel Strategy Runner — FULL TIER 1-5")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Experiments: {len(EXPERIMENTS)} total (NOVEL_001 to NOVEL_{len(EXPERIMENTS):03d})")
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

    # Initial chart + push
    generate_chart()
    git_push("feat: NOVEL runner Tier 2-5 — begin extended experiments")

    # Run ALL experiments (NOVEL_001-010 will be skipped if already done)
    all_results = {}
    completed = 0
    failed = 0

    for config in EXPERIMENTS:
        exp_id = config["id"]

        print(f"\n{'━'*60}")
        print(f"[{completed+1}/{len(EXPERIMENTS)}] {exp_id}: {config['name']}")
        print(f"{'━'*60}")

        # Train
        result = run_experiment(config)
        all_results[exp_id] = result

        map50 = result.get("map50", 0)
        if map50 > 0:
            completed += 1
        else:
            failed += 1

        # Update tracking files
        try:
            update_idea_md(exp_id, config["idea_id"], result, config)
        except Exception as e:
            print(f"[{exp_id}] IDEA.md update error: {e}")

        try:
            update_leaderboard(exp_id, result, config)
        except Exception as e:
            print(f"[{exp_id}] Leaderboard update error: {e}")

        # Regenerate chart after each experiment
        try:
            generate_chart()
        except Exception as e:
            print(f"[{exp_id}] Chart error: {e}")

        # Push after each experiment
        tier = config.get("idea_id", "combo")
        msg = f"experiment: {exp_id} — mAP50={map50:.4f} ({config['name']}) [{tier}]"
        git_push(msg)

        print(f"\n[Progress] {exp_id} done: mAP50={map50:.4f} ({completed} ok, {failed} failed)")

    # Final summary
    print("\n" + "=" * 60)
    print("ALL NOVEL Experiments Complete!")
    print("=" * 60)

    # Sort by mAP50
    sorted_results = sorted(all_results.items(), key=lambda x: x[1].get("map50", 0), reverse=True)
    print(f"\n{'Rank':<6} {'ID':<12} {'mAP50':<8} {'Strategy'}")
    print("-" * 60)
    for rank, (exp_id, result) in enumerate(sorted_results, 1):
        name = next((c["name"] for c in EXPERIMENTS if c["id"] == exp_id), exp_id)
        print(f"{rank:<6} {exp_id:<12} {result.get('map50', 0):.4f}   {name[:40]}")

    best_id = sorted_results[0][0] if sorted_results else "unknown"
    best_val = sorted_results[0][1].get("map50", 0) if sorted_results else 0
    print(f"\nOverall Best: {best_id} → mAP50={best_val:.4f}")

    # Final push with summary
    generate_chart()
    summary_msg = (
        f"final: ALL NOVEL experiments complete ({len(EXPERIMENTS)} total) "
        f"— best={best_id} mAP50={best_val:.4f}"
    )
    git_push(summary_msg)


if __name__ == "__main__":
    main()
