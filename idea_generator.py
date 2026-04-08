#!/usr/bin/env python3
"""
Adaptive Idea Generator for TBS Ripeness Detection
Analyzes experiment results and generates new ideas based on findings
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import random

class ExperimentAnalyzer:
    """Analyze experiment results to identify patterns and issues"""
    
    def __init__(self, ledger_path: str = "/workspace/Hermes-YOLO/experiments/logs/research_ledger.json"):
        self.ledger_path = Path(ledger_path)
        self.ledger = self._load_ledger()
    
    def _load_ledger(self) -> List[Dict]:
        if self.ledger_path.exists():
            with open(self.ledger_path) as f:
                return json.load(f)
        return []
    
    def analyze_trends(self) -> Dict:
        """Analyze trends across experiments"""
        completed = [e for e in self.ledger if e.get("status") == "completed" and e.get("results")]
        if not completed:
            return {"status": "no_data"}
        
        # Extract metrics
        map50s = [e["results"].get("map50", 0) for e in completed]
        strategies = [e["config"].get("name", "unknown") for e in completed]
        
        # Find best and worst
        best_idx = max(range(len(map50s)), key=lambda i: map50s[i])
        worst_idx = min(range(len(map50s)), key=lambda i: map50s[i])
        
        # Analyze components that worked
        best_components = completed[best_idx]["config"].get("novel_components", [])
        
        # Calculate trend
        if len(map50s) >= 3:
            recent_avg = sum(map50s[-3:]) / 3
            older_avg = sum(map50s[:-3]) / max(len(map50s)-3, 1)
            trend = "improving" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "status": "analyzed",
            "total_experiments": len(completed),
            "best_map50": max(map50s) if map50s else 0,
            "best_exp": completed[best_idx]["config"]["id"] if completed else None,
            "best_strategy": strategies[best_idx] if completed else None,
            "best_components": best_components,
            "worst_map50": min(map50s) if map50s else 0,
            "mean_map50": sum(map50s) / len(map50s) if map50s else 0,
            "trend": trend,
            "recent_performance": map50s[-3:] if len(map50s) >= 3 else map50s,
        }
    
    def identify_problems(self) -> List[Dict]:
        """Identify specific problems from failed or poor experiments"""
        problems = []
        
        for exp in self.ledger:
            if exp.get("status") == "failed":
                problems.append({
                    "exp_id": exp["config"]["id"],
                    "type": "execution_failure",
                    "details": exp.get("notes", ["Unknown error"])[0] if exp.get("notes") else "Unknown error"
                })
            elif exp.get("status") == "completed":
                results = exp.get("results", {})
                map50 = results.get("map50", 0)
                
                # Identify underperforming configurations
                if map50 < 0.4:  # Threshold for poor performance
                    problems.append({
                        "exp_id": exp["config"]["id"],
                        "type": "poor_performance",
                        "map50": map50,
                        "components": exp["config"].get("novel_components", []),
                        "suggestion": "Avoid similar configuration or modify hyperparameters"
                    })
        
        return problems
    
    def extract_insights(self) -> List[str]:
        """Extract actionable insights from experiments"""
        insights = []
        trends = self.analyze_trends()
        
        if trends["status"] == "no_data":
            return ["No experiments completed yet. Start with baseline configurations."]
        
        # Performance insights
        if trends["best_map50"] > 0.55:
            insights.append(f"✅ Strong performance achieved: {trends['best_map50']:.4f} mAP50")
        elif trends["best_map50"] > 0.50:
            insights.append(f"⚠️ Moderate performance: {trends['best_map50']:.4f} mAP50 - room for improvement")
        else:
            insights.append(f"❌ Low performance: {trends['best_map50']:.4f} mAP50 - needs significant changes")
        
        # Trend insights
        if trends["trend"] == "improving":
            insights.append("📈 Recent experiments showing improvement - current direction is working")
        elif trends["trend"] == "declining":
            insights.append("📉 Performance declining - need to pivot strategy")
        
        # Component insights
        if trends["best_components"]:
            insights.append(f"💡 Best components so far: {', '.join(trends['best_components'][:3])}")
        
        return insights

class IdeaGenerator:
    """Generate new experiment ideas based on analysis"""
    
    # Base strategies that have shown promise in literature
    PROMISING_TECHNIQUES = {
        "test_time_augmentation": {
            "name": "Test-Time Augmentation (TTA)",
            "description": "Apply multiple augmentations during inference and ensemble predictions",
            "rationale": "Can improve robustness for ambiguous B2/B3 cases",
            "implementation": "Multi-scale inference with flip/scale variations",
            "expected_gain": "2-5% mAP50 improvement",
            "novelty_score": 8,
        },
        "attention_mechanism": {
            "name": "Attention-Based Refinement",
            "description": "Add CBAM or SE attention modules to focus on relevant regions",
            "rationale": "Help model focus on color/texture differences for ripeness",
            "implementation": "Inject attention modules in neck/head",
            "expected_gain": "3-7% mAP50 improvement",
            "novelty_score": 7,
        },
        "contrastive_learning": {
            "name": "Supervised Contrastive Detection",
            "description": "Use contrastive loss to separate B2/B3 embeddings",
            "rationale": "Directly address B2/B3 confusion at feature level",
            "implementation": "Add SupCon branch to detection head",
            "expected_gain": "5-10% mAP50 on B2/B3 specifically",
            "novelty_score": 9,
        },
        "dynamic_label_assignment": {
            "name": "Dynamic Label Assignment",
            "description": "Use TaskAlignedAssigner or similar dynamic strategies",
            "rationale": "Better anchor-to-ground-truth matching for small B4",
            "implementation": "Replace static assignment with dynamic",
            "expected_gain": "2-4% mAP50 for small objects",
            "novelty_score": 6,
        },
        "multi_scale_feature_fusion": {
            "name": "Advanced Multi-Scale Fusion",
            "description": "BiFPN or ASFF-style feature fusion",
            "rationale": "Better feature representation across scales for all classes",
            "implementation": "Replace PANet with BiFPN/ASFF",
            "expected_gain": "3-6% mAP50",
            "novelty_score": 7,
        },
        "iou_aware_loss": {
            "name": "IoU-Aware Classification Loss",
            "description": "Weight classification loss by IoU quality",
            "rationale": "Focus learning on high-quality detections",
            "implementation": "Modify cls loss to include iou factor",
            "expected_gain": "2-4% mAP50",
            "novelty_score": 6,
        },
        "class_specific_anchors": {
            "name": "Class-Specific Anchor Optimization",
            "description": "Learn optimal anchors per class instead of shared",
            "rationale": "B1-B4 have different size distributions",
            "implementation": "Separate anchor generation per class",
            "expected_gain": "3-5% mAP50",
            "novelty_score": 7,
        },
        "uncertainty_estimation": {
            "name": "Uncertainty-Guided Training",
            "description": "Use epistemic uncertainty to weight hard examples",
            "rationale": "Identify and focus on truly ambiguous cases",
            "implementation": "Add uncertainty head, use for mining",
            "expected_gain": "4-8% mAP50",
            "novelty_score": 9,
        },
        "domain_adversarial": {
            "name": "Domain Adversarial Training",
            "description": "Use DANN-style training for DAMIMAS/LONSUM",
            "rationale": "Address domain imbalance explicitly",
            "implementation": "Add domain classifier with gradient reversal",
            "expected_gain": "5-10% cross-domain performance",
            "novelty_score": 8,
        },
        "self_training": {
            "name": "Pseudo-Label Self-Training",
            "description": "Generate pseudo-labels for unlabeled or weak data",
            "rationale": "Increase effective training data size",
            "implementation": "Iterative teacher-student training",
            "expected_gain": "3-7% with enough unlabeled data",
            "novelty_score": 7,
        },
        "mosaic_enhanced": {
            "name": "Semantic-Aware Mosaic",
            "description": "Mosaic that avoids mixing conflicting classes",
            "rationale": "Prevent B2/B3 confusion in mosaic grid",
            "implementation": "Smart mosaic composition based on labels",
            "expected_gain": "2-4% mAP50",
            "novelty_score": 6,
        },
        "copy_paste_advanced": {
            "name": "Context-Aware Copy-Paste",
            "description": "Copy-paste with background matching and blending",
            "rationale": "Augment rare classes (B1, B4) realistically",
            "implementation": "Poisson blending or similar techniques",
            "expected_gain": "3-5% for rare classes",
            "novelty_score": 7,
        },
        "ensemble_diverse": {
            "name": "Diverse Model Ensemble",
            "description": "Ensemble different architectures (YOLOv8 + YOLO11)",
            "rationale": "Combine strengths of different inductive biases",
            "implementation": "Weighted box fusion of multiple models",
            "expected_gain": "5-10% mAP50 (inference only)",
            "novelty_score": 6,
        },
        "knowledge_distillation": {
            "name": "Knowledge Distillation",
            "description": "Distill from large teacher to efficient student",
            "rationale": "Get YOLO11x quality with YOLO11s speed",
            "implementation": "Feature + response distillation",
            "expected_gain": "3-6% over baseline student",
            "novelty_score": 7,
        },
        "gradual_unfreezing": {
            "name": "Gradual Layer Unfreezing",
            "description": "Progressively unfreeze layers during training",
            "rationale": "Better transfer learning for small dataset",
            "implementation": "Freeze schedule by epoch",
            "expected_gain": "2-4% mAP50",
            "novelty_score": 5,
        },
    }
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.used_strategies = set()
        self.analyzer = ExperimentAnalyzer()
    
    def generate_new_idea(self, exp_num: int) -> Dict:
        """Generate a new experiment idea based on previous results"""
        
        # Analyze past experiments
        trends = self.analyzer.analyze_trends()
        insights = self.analyzer.extract_insights()
        problems = self.analyzer.identify_problems()
        
        # Select strategy based on analysis
        if trends["status"] == "no_data":
            # First experiment - start with promising baseline
            strategy_key = "contrastive_learning"
        elif trends["best_map50"] < 0.50:
            # Low performance - need fundamental changes
            strategy_key = self._select_for_low_performance(trends, problems)
        elif trends["trend"] == "improving":
            # Keep improving current direction
            strategy_key = self._select_for_improvement(trends)
        else:
            # Stagnant - try something completely different
            strategy_key = self._select_for_exploration()
        
        # Get strategy details
        strategy = self.PROMISING_TECHNIQUES.get(strategy_key, self.PROMISING_TECHNIQUES["contrastive_learning"])
        self.used_strategies.add(strategy_key)
        
        # Generate experiment config
        idea = {
            "id": f"IDEA_{exp_num:03d}",
            "timestamp": datetime.now().isoformat(),
            "strategy_key": strategy_key,
            "name": strategy["name"],
            "description": strategy["description"],
            "rationale": strategy["rationale"],
            "implementation_notes": strategy["implementation"],
            "expected_gain": strategy["expected_gain"],
            "novelty_score": strategy["novelty_score"],
            "based_on_analysis": {
                "trends": trends,
                "insights": insights,
                "problems": problems[:3] if problems else [],  # Top 3 problems
            },
            "related_literature": self._suggest_literature(strategy_key),
            "ablation_suggestions": self._suggest_ablations(strategy_key),
        }
        
        return idea
    
    def _select_for_low_performance(self, trends: Dict, problems: List) -> str:
        """Select strategy when performance is poor"""
        # Focus on fundamental improvements
        candidates = ["contrastive_learning", "attention_mechanism", "dynamic_label_assignment"]
        available = [c for c in candidates if c not in self.used_strategies]
        return random.choice(available) if available else "test_time_augmentation"
    
    def _select_for_improvement(self, trends: Dict) -> str:
        """Select strategy when things are improving"""
        # Extend current success
        candidates = ["multi_scale_feature_fusion", "iou_aware_loss", "class_specific_anchors"]
        available = [c for c in candidates if c not in self.used_strategies]
        return random.choice(available) if available else "uncertainty_estimation"
    
    def _select_for_exploration(self) -> str:
        """Select strategy for exploration when stagnant"""
        # Try completely different approaches
        candidates = ["domain_adversarial", "self_training", "ensemble_diverse"]
        available = [c for c in candidates if c not in self.used_strategies]
        return random.choice(available) if available else "knowledge_distillation"
    
    def _suggest_literature(self, strategy_key: str) -> List[str]:
        """Suggest relevant papers for the strategy"""
        literature_map = {
            "contrastive_learning": [
                "Khosla et al. - Supervised Contrastive Learning (NeurIPS 2020)",
                "Ye et al. - Ranking-Based Contrastive Loss for Object Detection",
            ],
            "attention_mechanism": [
                "Woo et al. - CBAM: Convolutional Block Attention Module",
                "Hu et al. - Squeeze-and-Excitation Networks",
            ],
            "test_time_augmentation": [
                "Moshkov et al. - Test-Time Augmentation for Object Detection",
            ],
            "multi_scale_feature_fusion": [
                "Tan et al. - EfficientDet: BiFPN",
                "Liu et al. - Path Aggregation Network (PANet)",
            ],
            "uncertainty_estimation": [
                "Kendall & Gal - What Uncertainties Do We Need?",
                "Miller et al. - Uncertainty-Aware Object Detection",
            ],
            "domain_adversarial": [
                "Ganin et al. - Domain-Adversarial Neural Networks",
            ],
            "knowledge_distillation": [
                "Hinton et al. - Distilling the Knowledge in a Neural Network",
                "Zhang et al. - Distilling Object Detectors",
            ],
        }
        return literature_map.get(strategy_key, ["See general object detection literature"])
    
    def _suggest_ablations(self, strategy_key: str) -> List[str]:
        """Suggest ablation studies for the strategy"""
        return [
            f"Remove {strategy_key} and compare baseline",
            f"Vary intensity of {strategy_key}",
            f"Combine {strategy_key} with best previous strategy",
            f"Apply {strategy_key} only to specific classes (B2/B3 or B4)",
        ]
    
    def generate_idea_md(self, idea: Dict) -> str:
        """Generate markdown documentation for the idea"""
        
        md = f"""# {idea['name']}

**Experiment ID**: {idea['id']}  
**Timestamp**: {idea['timestamp']}  
**Novelty Score**: {idea['novelty_score']}/10

## 🎯 Problem Statement

{idea['description']}

## 💡 Rationale

{idea['rationale']}

## 🔧 Implementation Approach

{idea['implementation_notes']}

## 📊 Expected Outcome

**Expected Gain**: {idea['expected_gain']}

### Analysis Context

This idea was generated based on:

- **Total experiments completed**: {idea['based_on_analysis']['trends'].get('total_experiments', 0)}
- **Current best mAP50**: {idea['based_on_analysis']['trends'].get('best_map50', 0):.4f}
- **Performance trend**: {idea['based_on_analysis']['trends'].get('trend', 'unknown')}

### Key Insights

"""
        
        for insight in idea['based_on_analysis']['insights']:
            md += f"- {insight}\n"
        
        md += f"""
### Problems Identified

"""
        for problem in idea['based_on_analysis']['problems']:
            md += f"- **{problem.get('exp_id', 'N/A')}**: {problem.get('type', 'unknown')} - {problem.get('details', problem.get('suggestion', 'N/A'))}\n"
        
        md += f"""
## 📚 Related Literature

"""
        for paper in idea['related_literature']:
            md += f"- {paper}\n"
        
        md += f"""
## 🔬 Suggested Ablations

"""
        for ablation in idea['ablation_suggestions']:
            md += f"- [ ] {ablation}\n"
        
        md += f"""
## 📝 Notes

_Add implementation details and results here after running the experiment._

---

*Generated by Adaptive Idea Generator*  
*Seed: 42 (Reproducible)*
"""
        
        return md

def update_idea_md(exp_num: int):
    """Update IDEA.md with new experiment idea"""
    
    generator = IdeaGenerator(seed=42)
    idea = generator.generate_new_idea(exp_num)
    
    # Generate markdown
    idea_md = generator.generate_idea_md(idea)
    
    # Save to IDEA.md
    idea_path = Path("/workspace/Hermes-YOLO/IDEA.md")
    
    # Read existing or create header
    if idea_path.exists():
        with open(idea_path, 'r') as f:
            existing = f.read()
    else:
        existing = "# 💡 Research Ideas Log\n\nThis document tracks all experimental ideas, their rationale, and results.\n\n---\n\n"
    
    # Prepend new idea (newest first)
    new_content = idea_md + "\n\n---\n\n" + existing
    
    with open(idea_path, 'w') as f:
        f.write(new_content)
    
    return idea

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exp_num = int(sys.argv[1])
    else:
        exp_num = 0
    
    idea = update_idea_md(exp_num)
    print(f"Generated idea: {idea['name']} (Novelty: {idea['novelty_score']}/10)")
    print(f"Expected gain: {idea['expected_gain']}")
