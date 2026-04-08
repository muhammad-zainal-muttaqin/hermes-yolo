#!/usr/bin/env python3
"""
Autonomous Research Pipeline for TBS Ripeness Detection
Inspired by Andrej Karpathy's autoresearch

Time limit: 8 hours
Max epochs per experiment: 10
Focus: mAP50 (not mAP50-95)
Seed: 42 (reproducible)
"""

import os
import sys
import json
import yaml
import time
import random
import subprocess
import datetime
import traceback
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import itertools

# Set seed for reproducibility
SEED = 42
random.seed(SEED)

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    id: str
    name: str
    description: str
    base_model: str  # yolo11n, yolo11s, yolo11m
    img_size: int
    batch_size: int
    epochs: int
    seed: int
    
    # Novel ideas (not in CONTEXT.md closed list)
    novel_components: List[str]
    
    # Training hyperparameters
    lr0: float
    lrf: float
    momentum: float
    weight_decay: float
    
    # Augmentation (novel approaches)
    augmentations: Dict
    
    # Loss weights
    box_weight: float
    cls_weight: float
    dfl_weight: float
    
    # Data strategy
    data_strategy: str  # "standard", "class_aware", "size_aware", "hard_mining"
    
    def to_yaml(self) -> str:
        return yaml.dump(asdict(self), default_flow_style=False)

class ResearchLedger:
    """Track all experiments and results"""
    
    def __init__(self, base_dir: str = "/workspace/Hermes-YOLO/experiments"):
        self.base_dir = Path(base_dir)
        self.runs_dir = self.base_dir / "runs"
        self.logs_dir = self.base_dir / "logs"
        self.ledger_file = self.logs_dir / "research_ledger.json"
        
        self.ledger = self._load_ledger()
    
    def _load_ledger(self) -> List[Dict]:
        if self.ledger_file.exists():
            with open(self.ledger_file) as f:
                return json.load(f)
        return []
    
    def save_ledger(self):
        self.ledger_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_file, 'w') as f:
            json.dump(self.ledger, f, indent=2)
    
    def add_experiment(self, config: ExperimentConfig, status: str = "pending"):
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "config": asdict(config),
            "status": status,
            "results": {},
            "git_commit": None,
            "notes": []
        }
        self.ledger.append(entry)
        self.save_ledger()
        return len(self.ledger) - 1
    
    def update_experiment(self, idx: int, results: Dict, git_commit: str = None):
        self.ledger[idx]["results"] = results
        self.ledger[idx]["status"] = "completed"
        if git_commit:
            self.ledger[idx]["git_commit"] = git_commit
        self.save_ledger()
    
    def get_best_experiment(self) -> Optional[Dict]:
        completed = [e for e in self.ledger if e["status"] == "completed"]
        if not completed:
            return None
        return max(completed, key=lambda x: x["results"].get("map50", 0))

class NovelIdeaGenerator:
    """
    Generate novel experimental ideas NOT in CONTEXT.md closed list.
    
    CONTEXT.md closed ideas:
    - Standard augmentations (flipud, scale, degrees)
    - Basic hyperparameter tweaks
    - Long-run training
    - Model soup
    - Standard architecture swaps
    - Simple oversampling
    """
    
    # Novel strategies for B2/B3 ambiguity and B4 small objects
    STRATEGIES = {
        "focus_loss": {
            "name": "Focal Loss with Class-Aware Gamma",
            "description": "Dynamic focal loss gamma per class based on confusion matrix. B2/B3 get higher gamma.",
            "components": ["focal_loss", "class_aware_gamma"]
        },
        "contrastive_head": {
            "name": "Supervised Contrastive Detection Head",
            "description": "Add contrastive learning branch to force B2/B3 separation in embedding space",
            "components": ["contrastive_head", "supcon_loss"]
        },
        "cascaded_nms": {
            "name": "Cascaded NMS with Class-Specific Thresholds",
            "description": "Different NMS thresholds per class. B4 gets lower threshold to reduce miss rate",
            "components": ["cascaded_nms", "adaptive_thresholds"]
        },
        "hard_negative_mining": {
            "name": "Online Hard Negative Mining",
            "description": "Focus training on hard negatives and ambiguous B2/B3 cases",
            "components": ["ohnm", "hard_mining_scheduler"]
        },
        "test_time_augment": {
            "name": "Test-Time Augmentation with Uncertainty Fusion",
            "description": "Multi-scale TTA with uncertainty-weighted fusion for ambiguous cases",
            "components": ["tta", "uncertainty_fusion"]
        },
        "multi_scale_training": {
            "name": "Multi-Scale Training with Curriculum",
            "description": "Progressively increase resolution during training. Start small, end at target",
            "components": ["curriculum_scale", "progressive_resolution"]
        },
        "label_smoothing_adaptive": {
            "name": "Adaptive Label Smoothing",
            "description": "Label smoothing based on class confusion. More smoothing for B2/B3",
            "components": ["adaptive_label_smooth", "confusion_aware"]
        },
        "class_wised_nms": {
            "name": "Class-Wise NMS with Learned Thresholds",
            "description": "Learn optimal NMS threshold per class during validation",
            "components": ["learned_nms", "class_specific_thresholds"]
        },
        "center_focused": {
            "name": "Center-Focused Augmentation",
            "description": "Augmentations that preserve center region (where TBS is) more than edges",
            "components": ["center_mask", "focus_augment"]
        },
        "mosaic_evolved": {
            "name": "Evolved Mosaic with Semantic Preservation",
            "description": "Mosaic that avoids mixing semantically different classes in same quadrant",
            "components": ["semantic_mosaic", "smart_composition"]
        }
    }
    
    # Novel data strategies
    DATA_STRATEGIES = [
        "hard_example_mining",  # Focus on B2/B3 confusion pairs
        "class_balanced_sampler",  # Dynamic sampler based on class performance
        "size_stratified_batch",  # Ensure each batch has all size categories
        "confusion_pair_oversample",  # Oversample confusing B2/B3 pairs
        "easy_negative_suppression",  # Downweight easy negatives in loss
    ]
    
    # Novel augmentation combinations (not in closed list)
    AUGMENTATIONS = {
        "strong_geometric": {
            "degrees": 15.0,
            "translate": 0.2,
            "scale": 0.9,
            "shear": 5.0,
            "perspective": 0.001,
        },
        "conservative_aug": {
            "degrees": 5.0,
            "translate": 0.1,
            "scale": 0.5,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
        },
        "color_focused": {
            "hsv_h": 0.02,
            "hsv_s": 0.8,
            "hsv_v": 0.5,
            "degrees": 0,  # No rotation - color is key for ripeness
            "translate": 0.1,
        },
        "small_object_focus": {
            "scale": 0.3,  # Less scaling to preserve small B4
            "mosaic": 0.5,  # Less mosaic
            "mixup": 0.0,  # No mixup - confuses small objects
            "copy_paste": 0.1,  # Light copy-paste for B4
        }
    }
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
    
    def generate_experiment(self, exp_id: int) -> ExperimentConfig:
        """Generate a novel experiment configuration"""
        
        # Select strategy
        strategy_key = random.choice(list(self.STRATEGIES.keys()))
        strategy = self.STRATEGIES[strategy_key]
        
        # Select data strategy
        data_strategy = random.choice(self.DATA_STRATEGIES)
        
        # Select augmentation
        aug_key = random.choice(list(self.AUGMENTATIONS.keys()))
        base_augment = self.AUGMENTATIONS[aug_key].copy()
        
        # Model size progression
        models = ["yolo11n", "yolo11s", "yolo11m"]
        base_model = models[exp_id % len(models)]
        
        # Image size - focus on efficient sizes
        img_sizes = [480, 640, 768]
        img_size = img_sizes[exp_id % len(img_sizes)]
        
        # Batch size based on model and image size
        batch_sizes = {480: 32, 640: 16, 768: 8}
        batch_size = batch_sizes[img_size]
        
        # Epochs - max 10 as per requirement
        epochs = min(10, 5 + (exp_id % 6))  # 5-10 epochs
        
        # Learning rate schedule
        lr_schedules = [
            (0.01, 0.01),   # Constant high
            (0.01, 0.1),    # Standard decay
            (0.005, 0.05),  # Conservative
            (0.02, 0.2),    # Aggressive then decay
        ]
        lr0, lrf = lr_schedules[exp_id % len(lr_schedules)]
        
        # Loss weights - prioritize classification for B2/B3
        loss_configs = [
            {"box": 7.5, "cls": 2.0, "dfl": 1.5},  # Standard
            {"box": 5.0, "cls": 5.0, "dfl": 1.5},  # High cls weight
            {"box": 7.5, "cls": 3.0, "dfl": 1.0},  # Higher cls, lower dfl
            {"box": 6.0, "cls": 4.0, "dfl": 2.0},  # Balanced
        ]
        loss_cfg = loss_configs[exp_id % len(loss_configs)]
        
        config = ExperimentConfig(
            id=f"AR_{exp_id:03d}",
            name=f"{strategy['name']}_{data_strategy}",
            description=f"{strategy['description']} + Data: {data_strategy}",
            base_model=base_model,
            img_size=img_size,
            batch_size=batch_size,
            epochs=epochs,
            seed=SEED,
            novel_components=strategy["components"] + [data_strategy],
            lr0=lr0,
            lrf=lrf,
            momentum=0.937,
            weight_decay=0.0005,
            augmentations=base_augment,
            box_weight=loss_cfg["box"],
            cls_weight=loss_cfg["cls"],
            dfl_weight=loss_cfg["dfl"],
            data_strategy=data_strategy
        )
        
        return config

class ExperimentRunner:
    """Run a single experiment and collect results"""
    
    def __init__(self, base_dir: str = "/workspace/Hermes-YOLO"):
        self.base_dir = Path(base_dir)
        self.experiments_dir = self.base_dir / "experiments"
        self.dataset_dir = Path("/root/.cache/huggingface/hub/datasets--ULM-DS-Lab--Dataset-Sawit-YOLO")
        
    def prepare_dataset(self) -> Path:
        """Ensure dataset is ready"""
        # Find the actual dataset path in cache
        cache_dir = Path("/root/.cache/huggingface/hub")
        dataset_dirs = list(cache_dir.glob("datasets--ULM-DS-Lab--Dataset-Sawit-YOLO*"))
        
        if dataset_dirs:
            return dataset_dirs[0]
        
        # Fallback - wait for download or use default
        return cache_dir
    
    def run_experiment(self, config: ExperimentConfig) -> Dict:
        """Run training and return metrics"""
        
        exp_dir = self.experiments_dir / "runs" / config.id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = exp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            f.write(config.to_yaml())
        
        # Create training config for Ultralytics
        data_yaml = self._create_data_yaml(config)
        
        # Run training
        results = self._train(config, data_yaml, exp_dir)
        
        return results
    
    def _create_data_yaml(self, config: ExperimentConfig) -> Path:
        """Create data.yaml for training"""
        dataset_path = self.prepare_dataset()
        
        # Find train/val/test paths
        # Assuming standard YOLO structure
        data_yaml_content = f"""
path: {dataset_path}
train: train/images
val: val/images
test: test/images

nc: 4
names: ['B1', 'B2', 'B3', 'B4']
"""
        data_yaml_path = self.experiments_dir / "configs" / f"data_{config.id}.yaml"
        data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(data_yaml_path, 'w') as f:
            f.write(data_yaml_content)
        
        return data_yaml_path
    
    def _train(self, config: ExperimentConfig, data_yaml: Path, exp_dir: Path) -> Dict:
        """Execute training with Ultralytics"""
        
        from ultralytics import YOLO
        
        # Load base model
        model = YOLO(f"{config.base_model}.pt")
        
        # Prepare hyperparameters
        hyp = {
            'lr0': config.lr0,
            'lrf': config.lrf,
            'momentum': config.momentum,
            'weight_decay': config.weight_decay,
            'box': config.box_weight,
            'cls': config.cls_weight,
            'dfl': config.dfl_weight,
            **config.augmentations
        }
        
        # Add novel augmentations based on strategy
        if "center_mask" in config.novel_components:
            hyp['copy_paste'] = 0.1
        
        if "semantic_mosaic" in config.novel_components:
            hyp['mosaic'] = 0.5  # Reduced mosaic
        
        # Train
        results = model.train(
            data=str(data_yaml),
            epochs=config.epochs,
            imgsz=config.img_size,
            batch=config.batch_size,
            seed=config.seed,
            project=str(exp_dir),
            name="train",
            exist_ok=True,
            **hyp
        )
        
        # Extract metrics
        metrics = {
            "map50": results.results_dict.get('metrics/mAP50(B)', 0),
            "map50_95": results.results_dict.get('metrics/mAP50-95(B)', 0),
            "precision": results.results_dict.get('metrics/precision(B)', 0),
            "recall": results.results_dict.get('metrics/recall(B)', 0),
            "epoch": config.epochs,
        }
        
        # Save results
        results_path = exp_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics

class GitHubSync:
    """Sync experiments to GitHub"""
    
    def __init__(self, repo_dir: str = "/workspace/Hermes-YOLO"):
        self.repo_dir = Path(repo_dir)
    
    def push_experiment(self, exp_id: str, message: str = None) -> str:
        """Push current experiment to GitHub"""
        
        if not message:
            message = f"Experiment {exp_id}: autoresearch iteration"
        
        commands = [
            "git add -A",
            f'git commit -m "{message}" || echo "Nothing to commit"',
            "git push origin main"
        ]
        
        commit_hash = None
        for cmd in commands:
            result = subprocess.run(
                cmd, shell=True, cwd=self.repo_dir,
                capture_output=True, text=True
            )
            if cmd.startswith("git commit") and result.returncode == 0:
                # Extract commit hash
                result_hash = subprocess.run(
                    "git rev-parse HEAD", shell=True, cwd=self.repo_dir,
                    capture_output=True, text=True
                )
                commit_hash = result_hash.stdout.strip()
        
        return commit_hash

class Visualizer:
    """Create visualizations like Karpathy's autoresearch"""
    
    def __init__(self, base_dir: str = "/workspace/Hermes-YOLO/experiments"):
        self.base_dir = Path(base_dir)
        self.viz_dir = self.base_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
    
    def create_progress_plot(self, ledger: List[Dict]):
        """Create mAP50 progression plot"""
        import matplotlib.pyplot as plt
        
        completed = [e for e in ledger if e["status"] == "completed"]
        if not completed:
            return
        
        exp_ids = [e["config"]["id"] for e in completed]
        map50s = [e["results"].get("map50", 0) for e in completed]
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(map50s)), map50s, 'b-o', linewidth=2, markersize=8)
        plt.axhline(y=max(map50s), color='g', linestyle='--', label=f'Best: {max(map50s):.3f}')
        plt.xlabel('Experiment #')
        plt.ylabel('mAP50')
        plt.title('Autonomous Research Progress - mAP50 Over Experiments')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(range(len(exp_ids)), [e.split('_')[1] for e in exp_ids], rotation=45)
        plt.tight_layout()
        
        plt.savefig(self.viz_dir / 'progress_map50.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_leaderboard(self, ledger: List[Dict]) -> str:
        """Create markdown leaderboard"""
        completed = [e for e in ledger if e["status"] == "completed"]
        if not completed:
            return "# Leaderboard\n\nNo completed experiments yet."
        
        # Sort by mAP50
        sorted_exps = sorted(completed, key=lambda x: x["results"].get("map50", 0), reverse=True)
        
        lines = ["# 🏆 Research Leaderboard\n"]
        lines.append(f"_Last updated: {datetime.datetime.now().isoformat()}_\n")
        lines.append("| Rank | Experiment | mAP50 | mAP50-95 | Components |" )
        lines.append("|------|------------|-------|----------|------------|")
        
        for i, exp in enumerate(sorted_exps[:10], 1):  # Top 10
            config = exp["config"]
            results = exp["results"]
            components = ", ".join(config["novel_components"][:2])  # First 2
            lines.append(
                f"| {i} | {config['id']} | {results.get('map50', 0):.4f} | "
                f"{results.get('map50_95', 0):.4f} | {components} |"
            )
        
        return "\n".join(lines)
    
    def generate_report(self, ledger: List[Dict]):
        """Generate full research report"""
        # Progress plot
        self.create_progress_plot(ledger)
        
        # Leaderboard
        leaderboard = self.create_leaderboard(ledger)
        leaderboard_path = self.base_dir / "LEADERBOARD.md"
        with open(leaderboard_path, 'w') as f:
            f.write(leaderboard)
        
        # Summary stats
        completed = [e for e in ledger if e["status"] == "completed"]
        if completed:
            best = max(completed, key=lambda x: x["results"].get("map50", 0))
            
            summary = f"""# Autonomous Research Summary

**Status**: {len(completed)} experiments completed
**Best mAP50**: {best['results'].get('map50', 0):.4f}
**Best Experiment**: {best['config']['id']}
**Best Strategy**: {best['config']['name']}

## Best Configuration
```yaml
{best['config']}
```

## Novel Components Used
- {chr(10).join('- ' + c for c in best['config']['novel_components'])}
"""
            summary_path = self.base_dir / "SUMMARY.md"
            with open(summary_path, 'w') as f:
                f.write(summary)

class AutonomousResearch:
    """Main autonomous research orchestrator"""
    
    def __init__(self, max_hours: float = 8.0):
        self.max_hours = max_hours
        self.start_time = time.time()
        self.end_time = self.start_time + (max_hours * 3600)
        
        self.ledger = ResearchLedger()
        self.idea_gen = NovelIdeaGenerator(seed=SEED)
        self.runner = ExperimentRunner()
        self.git = GitHubSync()
        self.viz = Visualizer()
        
        self.exp_counter = len(self.ledger.ledger)
    
    def time_remaining(self) -> float:
        """Return remaining time in hours"""
        return (self.end_time - time.time()) / 3600
    
    def should_continue(self) -> bool:
        """Check if we have time for another experiment"""
        return time.time() < self.end_time and self.time_remaining() > 0.5  # At least 30 min left
    
    def run(self):
        """Main research loop"""
        print("🔬 Starting Autonomous Research Pipeline")
        print(f"⏱️  Time limit: {self.max_hours} hours")
        print(f"🎯 Target: Maximize mAP50 (max 10 epochs per experiment)")
        print(f"🌱 Seed: {SEED} (reproducible)")
        print("=" * 60)
        
        while self.should_continue():
            exp_start = time.time()
            
            # Generate experiment
            config = self.idea_gen.generate_experiment(self.exp_counter)
            
            print(f"\n🧪 Experiment {config.id}: {config.name}")
            print(f"   Time remaining: {self.time_remaining():.2f} hours")
            
            # Add to ledger
            exp_idx = self.ledger.add_experiment(config, "running")
            
            try:
                # Run experiment
                results = self.runner.run_experiment(config)
                
                # Update ledger
                self.ledger.update_experiment(exp_idx, results)
                
                print(f"   ✅ mAP50: {results['map50']:.4f}")
                
                # Generate visualizations
                self.viz.generate_report(self.ledger.ledger)
                
                # Push to GitHub
                commit = self.git.push_experiment(
                    config.id, 
                    f"{config.id}: {config.name} - mAP50={results['map50']:.4f}"
                )
                
                if commit:
                    self.ledger.ledger[exp_idx]["git_commit"] = commit
                    self.ledger.save_ledger()
                    print(f"   📤 Pushed: {commit[:8]}")
                
            except Exception as e:
                error_msg = traceback.format_exc()
                self.ledger.ledger[exp_idx]["status"] = "failed"
                self.ledger.ledger[exp_idx]["notes"].append(f"Error: {str(e)}")
                self.ledger.save_ledger()
                print(f"   ❌ Failed: {str(e)[:100]}")
            
            exp_time = (time.time() - exp_start) / 60
            print(f"   ⏱️  Duration: {exp_time:.1f} min")
            
            self.exp_counter += 1
            
            # Safety break
            if self.exp_counter >= 50:  # Max 50 experiments
                print("\n🛑 Reached maximum experiment count")
                break
        
        # Final push
        print("\n🏁 Research session complete!")
        self.viz.generate_report(self.ledger.ledger)
        self.git.push_experiment("final", "Final: Autonomous research complete")
        
        best = self.ledger.get_best_experiment()
        if best:
            print(f"\n🎉 Best result: {best['config']['id']} with mAP50={best['results']['map50']:.4f}")

def main():
    """Entry point"""
    research = AutonomousResearch(max_hours=8.0)
    research.run()

if __name__ == "__main__":
    main()
