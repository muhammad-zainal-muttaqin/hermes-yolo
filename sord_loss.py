"""
SORD: Soft Labels for Ordinal Regression
Implementation for YOLO training

Following Idea #1 from BREAKTHROUGH_IDEAS.md
Reference: Díaz & Marathe, CVPR 2019
"""

import numpy as np
import json
from pathlib import Path
import yaml


def generate_sord_label(true_class, num_classes=4, sigma=0.8):
    """
    Generate Gaussian-kernel soft label for ordinal regression.
    
    Args:
        true_class: Integer 0=B1, 1=B2, 2=B3, 3=B4
        num_classes: Total number of classes (4)
        sigma: Temperature controlling softness (0.5-1.0)
               Lower = harder labels, Higher = softer labels
    
    Returns:
        Soft label distribution (normalized to sum=1)
    """
    positions = np.arange(num_classes)
    true_pos = true_class
    
    # Gaussian kernel: exp(-(k-j)² / 2σ²)
    distances = (positions - true_pos) ** 2
    soft_labels = np.exp(-distances / (2 * sigma ** 2))
    
    # Normalize to sum to 1
    soft_labels = soft_labels / soft_labels.sum()
    
    return soft_labels


def convert_hard_to_sord_labels(labels_dir, output_dir, sigma=0.8):
    """
    Convert hard YOLO labels to SORD soft labels.
    
    For each bounding box, we generate soft classification targets
    that encode the ordinal nature of ripeness.
    
    Args:
        labels_dir: Directory with YOLO .txt label files
        output_dir: Output directory for SORD-enhanced labels
        sigma: Softness parameter for Gaussian kernel
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    labels_dir = Path(labels_dir)
    
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        sord_annotations = []
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            bbox = parts[1:5]  # x_center, y_center, width, height
            
            # Generate SORD soft label
            soft_label = generate_sord_label(class_id, sigma=sigma)
            
            # Store: [class_id, x, y, w, h, soft_B1, soft_B2, soft_B3, soft_B4]
            sord_entry = {
                'class_id': class_id,
                'bbox': [float(x) for x in bbox],
                'soft_labels': soft_label.tolist()
            }
            sord_annotations.append(sord_entry)
        
        # Save as JSON for easier handling
        output_file = output_dir / f"{label_file.stem}.json"
        with open(output_file, 'w') as f:
            json.dump(sord_annotations, f, indent=2)
    
    print(f"Converted {len(list(labels_dir.glob('*.txt')))} label files to SORD format")
    print(f"Output directory: {output_dir}")
    print(f"Sigma (softness): {sigma}")


class SORDLoss:
    """
    SORD Loss implementation for YOLO classification head.
    
    Replaces standard cross-entropy with KL-divergence on soft targets.
    """
    
    def __init__(self, sigma=0.8, num_classes=4):
        self.sigma = sigma
        self.num_classes = num_classes
        self.epsilon = 1e-7
    
    def generate_soft_targets(self, true_labels):
        """
        Generate soft targets for batch of true labels.
        
        Args:
            true_labels: Tensor of shape (N,) with class indices 0-3
        
        Returns:
            Soft targets of shape (N, 4)
        """
        batch_size = len(true_labels)
        soft_targets = np.zeros((batch_size, self.num_classes))
        
        for i, label in enumerate(true_labels):
            soft_targets[i] = generate_sord_label(int(label), self.num_classes, self.sigma)
        
        return soft_targets
    
    def kl_divergence(self, predictions, soft_targets):
        """
        KL divergence loss: D_KL(soft_targets || predictions)
        
        Args:
            predictions: Softmax probabilities from model (N, 4)
            soft_targets: SORD soft labels (N, 4)
        
        Returns:
            Scalar loss value
        """
        # Clip for numerical stability
        predictions = np.clip(predictions, self.epsilon, 1 - self.epsilon)
        
        # KL divergence: sum(soft_targets * log(soft_targets / predictions))
        # For numerical stability, handle cases where soft_targets = 0
        loss = np.sum(soft_targets * (np.log(soft_targets + self.epsilon) - np.log(predictions)))
        
        return loss / len(predictions)  # Mean over batch


def create_sord_training_config(base_config_path, output_path, sigma=0.8):
    """
    Create YOLO training config with SORD loss.
    
    Args:
        base_config_path: Path to base data.yaml
        output_path: Path for SORD-enhanced config
        sigma: Softness parameter
    """
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add SORD configuration
    config['sord'] = {
        'enabled': True,
        'sigma': sigma,
        'num_classes': 4,
        'description': 'Soft Labels for Ordinal Regression (Díaz & Marathe, CVPR 2019)'
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created SORD config: {output_path}")
    print(f"Sigma: {sigma}")


def demonstrate_sord():
    """Demonstrate SORD soft label generation for all classes."""
    print("=" * 60)
    print("SORD: Soft Labels for Ordinal Regression")
    print("=" * 60)
    
    class_names = ['B1 (Ripe)', 'B2 (Transition)', 'B3 (Unripe)', 'B4 (Small)']
    
    for i, name in enumerate(class_names):
        soft = generate_sord_label(i, sigma=0.8)
        hard = np.eye(4)[i]  # One-hot
        
        print(f"\n{name} (Class {i}):")
        print(f"  Hard label:  [{hard[0]:.2f}, {hard[1]:.2f}, {hard[2]:.2f}, {hard[3]:.2f}]")
        print(f"  SORD (σ=0.8): [{soft[0]:.2f}, {soft[1]:.2f}, {soft[2]:.2f}, {soft[3]:.2f}]")
        
        # Show penalization
        if i == 1:  # B2
            print(f"  → B2→B3 confusion penalty reduced by {(hard[2]-soft[2])/hard[2]*100:.0f}%")
        elif i == 2:  # B3
            print(f"  → B3→B2 confusion penalty reduced by {(hard[1]-soft[1])/hard[1]*100:.0f}%")
    
    print("\n" + "=" * 60)
    print("Key Properties:")
    print("  • Ordinal structure: Adjacent classes have softer boundaries")
    print("  • B2↔B3 confusion is acceptable (one-step error)")
    print("  • B1↔B4 confusion is heavily penalized (three-step error)")
    print("  • Zero inference cost — only training labels change")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_sord()
