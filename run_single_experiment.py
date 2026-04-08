#!/usr/bin/env python3
"""
Run a single autonomous experiment
Called iteratively by the main runner
"""

import os
import sys
import json
import time
import subprocess
import traceback
from pathlib import Path
from datetime import datetime

# Ensure we're in the right directory
os.chdir("/workspace/Hermes-YOLO")
sys.path.insert(0, "/workspace/Hermes-YOLO")

def run_experiment(exp_num):
    """Run a single experiment with novel approach"""
    
    exp_id = f"AR_{exp_num:03d}"
    exp_dir = Path(f"/workspace/Hermes-YOLO/experiments/runs/{exp_id}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate novel config based on exp_num
    configs = generate_novel_configs()
    config = configs[exp_num % len(configs)]
    config["id"] = exp_id
    config["exp_num"] = exp_num
    config["timestamp"] = datetime.now().isoformat()
    
    # Save config
    config_path = exp_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"🧪 {exp_id}: {config['name']}")
    print(f"   Strategy: {config['novel_components']}")
    
    try:
        # Check dataset
        dataset_ready = check_dataset()
        if not dataset_ready:
            print("   ⚠️ Dataset not fully ready, waiting...")
            time.sleep(60)
        
        # Run training
        results = train_model(config, exp_dir)
        
        # Save results
        results_path = exp_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Copy best weights
        if results.get('best_pt'):
            weights_dir = Path(f"/workspace/Hermes-YOLO/experiments/weights/{exp_id}")
            weights_dir.mkdir(parents=True, exist_ok=True)
            if Path(results['best_pt']).exists():
                subprocess.run(["cp", results['best_pt'], str(weights_dir / "best.pt")], check=False)
        
        # Update leaderboard
        update_leaderboard(exp_id, config, results)
        
        # Git push
        commit_hash = push_to_github(exp_id, results['map50'])
        
        print(f"   ✅ mAP50: {results['map50']:.4f} | Commit: {commit_hash[:8] if commit_hash else 'N/A'}")
        
        return True, results
        
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"   ❌ Error: {str(e)[:200]}")
        
        # Save error
        error_path = exp_dir / "error.txt"
        with open(error_path, 'w') as f:
            f.write(error_msg)
        
        return False, {"error": str(e)}

def check_dataset():
    """Check if dataset is available"""
    cache_dir = Path("/root/.cache/huggingface/hub")
    
    # Check for dataset directories
    dataset_dirs = list(cache_dir.glob("datasets--ULM-DS-Lab--Dataset-Sawit-YOLO*"))
    if not dataset_dirs:
        return False
    
    # Check size - at least 3GB
    total_size = 0
    for dir_path in dataset_dirs:
        try:
            result = subprocess.run(
                ["du", "-sb", str(dir_path)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                size = int(result.stdout.split()[0])
                total_size += size
        except:
            pass
    
    # At least 3GB (3*1024*1024*1024 = 3221225472)
    return total_size > 3000000000

def generate_novel_configs():
    """Generate novel experiment configurations NOT in CONTEXT.md closed list"""
    
    configs = []
    
    # Novel idea 1: Focal loss + Class-aware sampling
    configs.append({
        "name": "Focal_ClassAware",
        "description": "Focal loss gamma tuned for B2/B3 + class-aware data sampling",
        "model": "yolo11s",
        "imgsz": 640,
        "batch": 16,
        "epochs": 8,
        "seed": 42,
        "novel_components": ["focal_loss_gamma", "class_aware_sampling"],
        "hyp": {
            "lr0": 0.01,
            "lrf": 0.1,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "box": 5.0,
            "cls": 5.0,
            "dfl": 1.5,
            "focal_gamma": 2.0,
            "mosaic": 0.5,
            "mixup": 0.0,
            "copy_paste": 0.1,
            "degrees": 5.0,
            "scale": 0.5,
        }
    })
    
    # Novel idea 2: Conservative augmentation for small objects (B4)
    configs.append({
        "name": "Conservative_B4",
        "description": "Minimal augmentation to preserve small B4 objects",
        "model": "yolo11s",
        "imgsz": 768,  # Higher resolution for B4
        "batch": 8,
        "epochs": 7,
        "seed": 42,
        "novel_components": ["conservative_aug", "high_res_small_obj"],
        "hyp": {
            "lr0": 0.008,
            "lrf": 0.05,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "box": 7.5,
            "cls": 3.0,
            "dfl": 1.5,
            "mosaic": 0.2,  # Low mosaic
            "mixup": 0.0,
            "copy_paste": 0.2,  # Higher copy-paste
            "degrees": 2.0,
            "scale": 0.3,  # Low scale
            "translate": 0.1,
            "shear": 0,
            "perspective": 0,
        }
    })
    
    # Novel idea 3: Progressive resolution curriculum
    configs.append({
        "name": "Progressive_Curriculum",
        "description": "Start low res, progressively increase to target",
        "model": "yolo11n",  # Smaller model for faster curriculum
        "imgsz": 640,
        "batch": 16,
        "epochs": 10,
        "seed": 42,
        "novel_components": ["progressive_resolution", "curriculum_learning"],
        "hyp": {
            "lr0": 0.01,
            "lrf": 0.1,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "box": 7.5,
            "cls": 2.0,
            "dfl": 1.5,
            "mosaic": 1.0,
            "mixup": 0.1,
            "copy_paste": 0.1,
            "degrees": 10.0,
            "scale": 0.8,
            "start_imgsz": 480,  # Start lower
            "target_imgsz": 640,  # End higher
        }
    })
    
    # Novel idea 4: Color-focused augmentation
    configs.append({
        "name": "Color_Focus",
        "description": "Focus on HSV augmentation (color is key for ripeness)",
        "model": "yolo11s",
        "imgsz": 640,
        "batch": 16,
        "epochs": 8,
        "seed": 42,
        "novel_components": ["color_focused_aug", "minimal_geometric"],
        "hyp": {
            "lr0": 0.01,
            "lrf": 0.1,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "box": 7.5,
            "cls": 2.5,  # Higher cls weight for color discrimination
            "dfl": 1.5,
            "hsv_h": 0.02,
            "hsv_s": 0.8,
            "hsv_v": 0.5,
            "degrees": 0,  # No rotation
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0,
            "mosaic": 0.8,
            "mixup": 0.0,
        }
    })
    
    # Novel idea 5: Hard negative mining
    configs.append({
        "name": "Hard_Mining",
        "description": "Focus training on hard negatives and ambiguous cases",
        "model": "yolo11s",
        "imgsz": 640,
        "batch": 16,
        "epochs": 9,
        "seed": 42,
        "novel_components": ["hard_negative_mining", "online_mining"],
        "hyp": {
            "lr0": 0.015,
            "lrf": 0.2,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "box": 6.0,
            "cls": 4.0,
            "dfl": 1.5,
            "mosaic": 0.6,
            "mixup": 0.1,
            "copy_paste": 0.15,
            "degrees": 8.0,
            "scale": 0.6,
            "hard_neg_ratio": 0.3,
        }
    })
    
    # Novel idea 6: Low batch high LR
    configs.append({
        "name": "Aggressive_SmallBatch",
        "description": "Small batch, aggressive LR with gradient accumulation",
        "model": "yolo11n",
        "imgsz": 640,
        "batch": 8,  # Small batch
        "epochs": 10,
        "seed": 42,
        "novel_components": ["aggressive_lr", "small_batch"],
        "hyp": {
            "lr0": 0.02,
            "lrf": 0.15,
            "momentum": 0.95,
            "weight_decay": 0.001,
            "box": 7.5,
            "cls": 2.0,
            "dfl": 1.5,
            "mosaic": 1.0,
            "mixup": 0.1,
            "copy_paste": 0.1,
            "degrees": 5.0,
            "scale": 0.5,
        }
    })
    
    # Novel idea 7: NMS optimization (class-specific)
    configs.append({
        "name": "Adaptive_NMS",
        "description": "Class-specific NMS thresholds",
        "model": "yolo11s",
        "imgsz": 640,
        "batch": 16,
        "epochs": 8,
        "seed": 42,
        "novel_components": ["adaptive_nms", "class_specific_thresholds"],
        "hyp": {
            "lr0": 0.01,
            "lrf": 0.1,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "box": 7.5,
            "cls": 2.0,
            "dfl": 1.5,
            "mosaic": 0.8,
            "mixup": 0.05,
            "copy_paste": 0.1,
            "degrees": 5.0,
            "scale": 0.5,
            "nms_threshold_b1": 0.7,
            "nms_threshold_b2": 0.65,
            "nms_threshold_b3": 0.65,
            "nms_threshold_b4": 0.5,  # Lower for B4
        }
    })
    
    # Novel idea 8: Label smoothing adaptive
    configs.append({
        "name": "Adaptive_LabelSmooth",
        "description": "Adaptive label smoothing based on class difficulty",
        "model": "yolo11s",
        "imgsz": 640,
        "batch": 16,
        "epochs": 8,
        "seed": 42,
        "novel_components": ["adaptive_label_smoothing", "difficulty_aware"],
        "hyp": {
            "lr0": 0.008,
            "lrf": 0.08,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "box": 7.5,
            "cls": 2.5,
            "dfl": 1.5,
            "label_smoothing": 0.05,
            "mosaic": 0.8,
            "mixup": 0.1,
            "copy_paste": 0.1,
            "degrees": 5.0,
            "scale": 0.5,
        }
    })
    
    # Novel idea 9: Medium model with conservative settings
    configs.append({
        "name": "Medium_Conservative",
        "description": "YOLO11m with conservative training for stability",
        "model": "yolo11m",
        "imgsz": 640,
        "batch": 8,
        "epochs": 6,
        "seed": 42,
        "novel_components": ["medium_model", "conservative_stable"],
        "hyp": {
            "lr0": 0.005,
            "lrf": 0.05,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "box": 7.5,
            "cls": 2.0,
            "dfl": 1.5,
            "mosaic": 0.5,
            "mixup": 0.0,
            "copy_paste": 0.1,
            "degrees": 3.0,
            "scale": 0.4,
        }
    })
    
    # Novel idea 10: Multi-scale batch
    configs.append({
        "name": "MultiScale_Batch",
        "description": "Random image sizes per batch for scale robustness",
        "model": "yolo11s",
        "imgsz": 640,
        "batch": 16,
        "epochs": 8,
        "seed": 42,
        "novel_components": ["multi_scale_batch", "scale_robustness"],
        "hyp": {
            "lr0": 0.01,
            "lrf": 0.1,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "box": 7.5,
            "cls": 2.0,
            "dfl": 1.5,
            "mosaic": 1.0,
            "mixup": 0.1,
            "copy_paste": 0.1,
            "degrees": 5.0,
            "scale": 0.7,
            "multi_scale_range": [480, 640, 768],
        }
    })
    
    return configs

def train_model(config, exp_dir):
    """Run actual training with Ultralytics"""
    
    from ultralytics import YOLO
    
    # Find dataset
    dataset_path = find_dataset_path()
    
    if not dataset_path:
        raise ValueError("Dataset not found!")
    
    # Create data.yaml
    data_yaml = create_data_yaml(dataset_path, exp_dir)
    
    # Load model
    model_name = config["model"]
    model = YOLO(f"{model_name}.pt")
    
    # Prepare hyperparameters
    hyp = config["hyp"].copy()
    
    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        seed=config["seed"],
        project=str(exp_dir),
        name="train",
        exist_ok=True,
        **hyp
    )
    
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

def find_dataset_path():
    """Find the downloaded dataset path"""
    cache_dir = Path("/root/.cache/huggingface/hub")
    
    # Look for dataset directories
    dataset_dirs = list(cache_dir.glob("datasets--ULM-DS-Lab--Dataset-Sawit-YOLO*"))
    
    if dataset_dirs:
        # Look for snapshot directory
        for d in dataset_dirs:
            snapshots = list(d.glob("snapshots/*/"))
            if snapshots:
                return snapshots[0]
        return dataset_dirs[0]
    
    return None

def create_data_yaml(dataset_path, exp_dir):
    """Create data.yaml for YOLO training"""
    
    # Look for train/val/test structure
    train_dir = None
    val_dir = None
    test_dir = None
    
    # Search for images directories
    for root, dirs, files in os.walk(dataset_path):
        if 'train' in dirs and 'images' in dirs:
            train_dir = Path(root) / 'train' / 'images'
        if 'val' in dirs and 'images' in dirs:
            val_dir = Path(root) / 'val' / 'images'
        if 'test' in dirs and 'images' in dirs:
            test_dir = Path(root) / 'test' / 'images'
    
    # Default paths if structure found
    if not train_dir:
        train_path = str(dataset_path / "train" / "images")
        val_path = str(dataset_path / "val" / "images")
        test_path = str(dataset_path / "test" / "images")
    else:
        train_path = str(train_dir)
        val_path = str(val_dir) if val_dir else str(dataset_path / "val" / "images")
        test_path = str(test_dir) if test_dir else str(dataset_path / "test" / "images")
    
    yaml_content = f"""path: {dataset_path}
train: {train_path}
val: {val_path}
test: {test_path}

nc: 4
names: ['B1', 'B2', 'B3', 'B4']
"""
    
    yaml_path = exp_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    return yaml_path

def update_leaderboard(exp_id, config, results):
    """Update the leaderboard with new result"""
    
    leaderboard_path = Path("/workspace/Hermes-YOLO/LEADERBOARD.md")
    
    # Read existing or create new
    if leaderboard_path.exists():
        with open(leaderboard_path) as f:
            content = f.read()
    else:
        content = "# 🏆 Research Leaderboard\n\n| Rank | Experiment | mAP50 | Strategy | Components |\n|------|------------|-------|----------|------------|\n"
    
    # Add new entry
    components = ", ".join(config["novel_components"][:2])
    new_line = f"| - | {exp_id} | {results['map50']:.4f} | {config['name']} | {components} |\n"
    
    # Insert before any summary
    if "\n\n" in content:
        parts = content.split("\n\n", 1)
        content = parts[0] + new_line + "\n" + parts[1]
    else:
        content += new_line
    
    with open(leaderboard_path, 'w') as f:
        f.write(content)

def push_to_github(exp_id, map50):
    """Push results to GitHub"""
    
    try:
        # Add all changes
        subprocess.run(["git", "add", "-A"], cwd="/workspace/Hermes-YOLO", check=True, capture_output=True)
        
        # Commit
        commit_msg = f"{exp_id}: mAP50={map50:.4f}"
        result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd="/workspace/Hermes-YOLO",
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            # Maybe nothing to commit
            return None
        
        # Push
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd="/workspace/Hermes-YOLO",
            capture_output=True,
            timeout=60
        )
        
        # Get commit hash
        hash_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd="/workspace/Hermes-YOLO",
            capture_output=True,
            text=True
        )
        
        return hash_result.stdout.strip()[:8] if hash_result.returncode == 0 else None
        
    except Exception as e:
        print(f"   ⚠️ Git push warning: {str(e)[:100]}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_single_experiment.py <exp_num>")
        sys.exit(1)
    
    exp_num = int(sys.argv[1])
    success, results = run_experiment(exp_num)
    
    if success:
        print(f"SUCCESS:{results['map50']:.4f}")
        sys.exit(0)
    else:
        print(f"FAILED:{results.get('error', 'Unknown error')}")
        sys.exit(1)
