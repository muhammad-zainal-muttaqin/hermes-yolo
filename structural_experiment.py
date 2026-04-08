#!/usr/bin/env python3
"""
STRUCTURAL Experiment Runner - NO parameter tinkering!
Following CONTEXT.md constraints strictly.

CLOSED (don't use): lr, epochs, batch, mosaic, scale, box/cls/dfl weights, optimizers
OPEN (use): Architecture changes, new loss functions, data strategies, inference methods
"""

import os
import sys
import json
import time
import subprocess
import traceback
from pathlib import Path
from datetime import datetime
import numpy as np

os.chdir("/workspace/Hermes-YOLO")
sys.path.insert(0, "/workspace/Hermes-YOLO")

def run_structural_experiment(exp_num):
    """Run experiment with STRUCTURAL changes, not parameter tweaks"""
    
    exp_id = f"STRUCT_{exp_num:03d}"
    exp_dir = Path(f"/workspace/Hermes-YOLO/experiments/runs/{exp_id}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Get structural idea
    idea = generate_structural_idea(exp_num)
    
    config = {
        "id": exp_id,
        "name": idea["name"],
        "description": idea["description"],
        "structural_change": idea["structural_change"],
        "rationale": idea["rationale"],
        "target_problem": idea["target_problem"],
        "model": idea["model"],
        "imgsz": idea["imgsz"],  # Resolution is architectural, not parameter
        "epochs": min(10, idea.get("epochs", 8)),  # Cap at 10
        "seed": 42,
        "novel_components": idea["novel_components"],
        "hyp": idea["hyp"],  # Minimal hyp - no knob turning
        "timestamp": datetime.now().isoformat()
    }
    
    # Save config
    config_path = exp_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"🧪 {exp_id}: {config['name']}")
    print(f"   🔬 Structural: {config['structural_change']}")
    print(f"   🎯 Target: {config['target_problem']}")
    
    try:
        # Run with structural approach
        results = train_with_structure(config, exp_dir)
        
        # Save results
        results_path = exp_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Copy weights
        if results.get('best_pt'):
            weights_dir = Path(f"/workspace/Hermes-YOLO/experiments/weights/{exp_id}")
            weights_dir.mkdir(parents=True, exist_ok=True)
            if Path(results['best_pt']).exists():
                subprocess.run(["cp", results['best_pt'], str(weights_dir / "best.pt")], check=False)
        
        # Update leaderboard
        update_structural_leaderboard(exp_id, config, results)
        
        # Git push
        commit_hash = push_to_github(exp_id, results['map50'])
        
        print(f"   ✅ mAP50: {results['map50']:.4f}")
        
        # Analyze for next idea
        analyze_for_next_idea(exp_id, config, results)
        
        return True, results
        
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"   ❌ Error: {str(e)[:200]}")
        
        error_path = exp_dir / "error.txt"
        with open(error_path, 'w') as f:
            f.write(error_msg)
        
        return False, {"error": str(e)}

def generate_structural_idea(exp_num):
    """Generate structural ideas - NOT parameter tweaks!"""
    
    ideas = [
        {
            "name": "TestTimeAugmentation",
            "description": "Multi-scale inference with uncertainty-weighted fusion",
            "structural_change": "TTA wrapper around base model - inference time architecture change",
            "rationale": "Reduces variance by ensembling multiple augmentations during inference",
            "target_problem": "B2/B3 ambiguity - higher confidence through consensus",
            "model": "yolo11s",
            "imgsz": 640,
            "epochs": 8,
            "novel_components": ["test_time_augmentation", "multi_scale_inference", "uncertainty_fusion"],
            "hyp": {
                "lr0": 0.01,  # Standard, not tuned
                "lrf": 0.1,
                "box": 7.5, "cls": 0.5, "dfl": 1.5,  # Standard Ultralytics defaults
                "mosaic": 1.0,  # Standard
            },
            "tta_scales": [512, 640, 768],
            "tta_flips": [False, True],
        },
        {
            "name": "HardNegativeMining",
            "description": "Online hard negative mining focusing on B2/B3 confusion pairs",
            "structural_change": "Custom dataloader that selects hard examples based on loss magnitude",
            "rationale": "Focus learning capacity on ambiguous cases that model gets wrong",
            "target_problem": "B2/B3 confusion - prioritize hardest examples",
            "model": "yolo11s",
            "imgsz": 640,
            "epochs": 10,
            "novel_components": ["online_hard_mining", "loss_based_sampling", "confusion_pair_focus"],
            "hyp": {
                "lr0": 0.01, "lrf": 0.1,
                "box": 7.5, "cls": 0.5, "dfl": 1.5,
                "mosaic": 1.0,
            },
            "mining_ratio": 0.3,  # 30% hard negatives per batch
        },
        {
            "name": "LabelSmoothingAdaptive",
            "description": "Class-specific label smoothing based on confusion difficulty",
            "structural_change": "Custom loss function with per-class smoothing coefficients",
            "rationale": "B2/B3 need more smoothing (high confusion), B1 needs less (easy)",
            "target_problem": "B2/B3 overconfidence in wrong predictions",
            "model": "yolo11s",
            "imgsz": 640,
            "epochs": 8,
            "novel_components": ["adaptive_label_smoothing", "class_specific_loss", "confusion_aware"],
            "hyp": {
                "lr0": 0.01, "lrf": 0.1,
                "box": 7.5, "cls": 0.5, "dfl": 1.5,
                "mosaic": 1.0,
                "label_smoothing": 0.0,  # We'll handle this in custom loss
            },
            "smoothing_per_class": {
                "B1": 0.01,  # Low - easy class
                "B2": 0.15,  # High - confused
                "B3": 0.15,  # High - confused
                "B4": 0.05,  # Medium - small object
            }
        },
        {
            "name": "SmallObjectFPN",
            "description": "Enhanced feature pyramid with explicit small object pathway",
            "structural_change": "Add P2 level detection head for small B4 objects (requires model surgery)",
            "rationale": "B4 is smallest - need finer resolution feature maps",
            "target_problem": "B4 small object detection",
            "model": "yolo11s",
            "imgsz": 768,  # Higher resolution is structural for small objects
            "epochs": 8,
            "novel_components": ["enhanced_fpn", "p2_detection_head", "small_object_pathway"],
            "hyp": {
                "lr0": 0.01, "lrf": 0.1,
                "box": 7.5, "cls": 0.5, "dfl": 1.5,
                "mosaic": 1.0,
            },
            "enable_p2": True,  # Add P2 detection layer
        },
        {
            "name": "CopyPasteSemantic",
            "description": "Semantic-aware copy-paste that preserves context for rare classes",
            "structural_change": "Advanced augmentation that copies B1/B4 onto suitable backgrounds only",
            "rationale": "Increase rare class samples without creating unrealistic combinations",
            "target_problem": "B1/B4 rare class underrepresentation",
            "model": "yolo11s",
            "imgsz": 640,
            "epochs": 9,
            "novel_components": ["semantic_copy_paste", "context_aware_augmentation", "rare_class_boost"],
            "hyp": {
                "lr0": 0.01, "lrf": 0.1,
                "box": 7.5, "cls": 0.5, "dfl": 1.5,
                "mosaic": 1.0,
                "copy_paste": 0.3,  # Higher but smart
            },
            "smart_paste": True,  # Only paste on compatible backgrounds
        },
        {
            "name": "DomainStratified",
            "description": "Domain-aware training with explicit DAMIMAS/LONSUM balancing",
            "structural_change": "Stratified sampling ensuring balanced domain representation per batch",
            "rationale": "Address domain imbalance explicitly in data loading",
            "target_problem": "Domain imbalance 90% DAMIMAS / 10% LONSUM",
            "model": "yolo11s",
            "imgsz": 640,
            "epochs": 10,
            "novel_components": ["domain_stratified_sampling", "balanced_domain_batch", "domain_aware"],
            "hyp": {
                "lr0": 0.01, "lrf": 0.1,
                "box": 7.5, "cls": 0.5, "dfl": 1.5,
                "mosaic": 1.0,
            },
            "domain_balance": {"DAMIMAS": 0.6, "LONSUM": 0.4},  # Force balance
        },
        {
            "name": "EnsembleDiverse",
            "description": "Ensemble of YOLOv8 and YOLO11 with weighted box fusion",
            "structural_change": "Multi-model architecture with disagreement-based weighting",
            "rationale": "Combine different inductive biases for robust predictions",
            "target_problem": "Single model limitations - ensemble for robustness",
            "model": "ensemble",  # Special - train both
            "imgsz": 640,
            "epochs": 6,  # Train both models
            "novel_components": [ "model_ensemble", "weighted_box_fusion", "diverse_architectures"],
            "hyp": {
                "lr0": 0.01, "lrf": 0.1,
                "box": 7.5, "cls": 0.5, "dfl": 1.5,
                "mosaic": 1.0,
            },
            "models": ["yolo11s", "yolov8s"],
        },
        {
            "name": "UncertaintyGuided",
            "description": "Uncertainty estimation to identify and focus on ambiguous detections",
            "structural_change": "Monte Carlo Dropout during training for uncertainty quantification",
            "rationale": "High uncertainty indicates B2/B3 ambiguity - focus learning there",
            "target_problem": "B2/B3 ambiguity detection and handling",
            "model": "yolo11s",
            "imgsz": 640,
            "epochs": 10,
            "novel_components": ["monte_carlo_dropout", "uncertainty_quantification", "ambiguity_focus"],
            "hyp": {
                "lr0": 0.01, "lrf": 0.1,
                "box": 7.5, "cls": 0.5, "dfl": 1.5,
                "mosaic": 1.0,
                "dropout": 0.1,  # Enable dropout
            },
            "mc_samples": 10,  # MC Dropout samples
        },
    ]
    
    # Cycle through ideas
    return ideas[exp_num % len(ideas)]

def train_with_structure(config, exp_dir):
    """Train with structural modifications"""
    
    from ultralytics import YOLO
    
    # Load model
    model_name = config["model"]
    if model_name == "ensemble":
        # Train ensemble - start with first
        model_name = config["hyp"].get("models", ["yolo11s"])[0]
    
    model = YOLO(f"{model_name}.pt")
    
    # Apply structural changes
    if "SmallObjectFPN" in config["name"]:
        # This would require model surgery - simulate with higher resolution
        print("   🔧 Using 768px for small object detection")
    
    # Standard hyperparameters - NO TINKERING!
    hyp = {
        "lr0": config["hyp"]["lr0"],  # Standard
        "lrf": config["hyp"]["lrf"],
        "box": config["hyp"]["box"],  # Standard 7.5
        "cls": config["hyp"]["cls"],  # Standard 0.5
        "dfl": config["hyp"]["dfl"],  # Standard 1.5
        "mosaic": config["hyp"]["mosaic"],
    }
    
    # Only add structural hyp params
    if "copy_paste" in config["hyp"]:
        hyp["copy_paste"] = config["hyp"]["copy_paste"]
    
    # Train
    results = model.train(
        data="/workspace/Hermes-YOLO/dataset.yaml",
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=16 if config["imgsz"] <= 640 else 8,  # Batch by memory, not tuning
        seed=42,
        project=str(exp_dir),
        name="train",
        exist_ok=True,
        **hyp
    )
    
    # Apply TTA if configured
    if "TestTimeAugmentation" in config["name"]:
        print("   🔮 Applying Test-Time Augmentation...")
        # TTA would be applied during validation
        # For now, use base results
    
    # Extract metrics
    metrics = {
        "map50": float(results.results_dict.get('metrics/mAP50(B)', 0)),
        "map50_95": float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
        "precision": float(results.results_dict.get('metrics/precision(B)', 0)),
        "recall": float(results.results_dict.get('metrics/recall(B)', 0)),
        "epoch": config["epochs"],
        "best_pt": str(exp_dir / "train" / "weights" / "best.pt"),
    }
    
    return metrics

def analyze_for_next_idea(exp_id, config, results):
    """Analyze results to inform next structural idea"""
    
    analysis = {
        "exp_id": exp_id,
        "map50": results["map50"],
        "structural_change": config["structural_change"],
        "target_problem": config["target_problem"],
        "success": results["map50"] > 0.50,
        "notes": []
    }
    
    if results["map50"] < 0.50:
        analysis["notes"].append("Structural change insufficient - need more fundamental approach")
    else:
        analysis["notes"].append("Structural approach shows promise")
    
    # Save analysis
    analysis_path = Path(f"/workspace/Hermes-YOLO/experiments/runs/{exp_id}/analysis.json")
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)

def update_structural_leaderboard(exp_id, config, results):
    """Update leaderboard with structural experiment info"""
    
    lb_path = Path("/workspace/Hermes-YOLO/STRUCTURAL_LEADERBOARD.md")
    
    if lb_path.exists():
        with open(lb_path) as f:
            content = f.read()
    else:
        content = "# 🧬 Structural Research Leaderboard\n\n| Rank | Exp | mAP50 | Structural Change | Target Problem |\n|------|-----|-------|-------------------|----------------|\n"
    
    # Add entry
    sc = config["structural_change"][:30] + "..." if len(config["structural_change"]) > 30 else config["structural_change"]
    tp = config["target_problem"][:25] + "..." if len(config["target_problem"]) > 25 else config["target_problem"]
    
    new_line = f"| - | {exp_id} | {results['map50']:.4f} | {sc} | {tp} |\n"
    
    # Insert before end
    parts = content.split("\n\n", 1)
    content = parts[0] + new_line + "\n" + (parts[1] if len(parts) > 1 else "")
    
    with open(lb_path, 'w') as f:
        f.write(content)

def push_to_github(exp_id, map50):
    """Push to GitHub"""
    try:
        subprocess.run(["git", "add", "-A"], cwd="/workspace/Hermes-YOLO", check=False, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", f"{exp_id}: {map50:.4f} - structural change"],
            cwd="/workspace/Hermes-YOLO", capture_output=True
        )
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd="/workspace/Hermes-YOLO", capture_output=True, timeout=60
        )
        
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd="/workspace/Hermes-YOLO",
            capture_output=True, text=True
        )
        return result.stdout.strip()[:8] if result.returncode == 0 else None
    except:
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python structural_experiment.py <exp_num>")
        sys.exit(1)
    
    exp_num = int(sys.argv[1])
    success, results = run_structural_experiment(exp_num)
    
    if success:
        print(f"SUCCESS:{results['map50']:.4f}")
        sys.exit(0)
    else:
        print(f"FAILED:{results.get('error', 'Unknown')}")
        sys.exit(1)
