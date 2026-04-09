"""
L*a*b* Color Space Preprocessing for Oil Palm Maturity Detection
Implementation for YOLO training and inference

Following Idea #8 from BREAKTHROUGH_IDEAS.md
Reference: León et al., 2006; Septiarini et al., 2021 (98.3% accuracy)

Key insight:
- a* channel directly encodes red-green axis
- B2 (black-to-red transition) = positive a* values
- B3 (pure black) = near-zero a*
"""

import cv2
import numpy as np
from pathlib import Path
import multiprocessing as mp
from functools import partial
import argparse


def rgb_to_lab(image_rgb):
    """
    Convert RGB image to CIE L*a*b* color space.
    
    Args:
        image_rgb: numpy array (H, W, 3) in RGB format
    
    Returns:
        lab_image: numpy array (H, W, 3) in L*a*b* format
        - L: Lightness (0-100)
        - a: Green-Red axis (-128 to +127)
        - b: Blue-Yellow axis (-128 to +127)
    """
    # OpenCV uses BGR by default, so we need careful conversion
    if image_rgb.dtype != np.uint8:
        image_rgb = (image_rgb * 255).astype(np.uint8) if image_rgb.max() <= 1.0 else image_rgb.astype(np.uint8)
    
    # Convert RGB to BGR (OpenCV format)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # Convert BGR to L*a*b*
    lab_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    
    return lab_image


def rgb_to_lab_normalized(image_rgb):
    """
    Convert RGB to L*a*b* with normalized channels for neural network input.
    
    Normalization:
    - L: [0, 100] → [0, 1]
    - a: [-128, 127] → [0, 1]
    - b: [-128, 127] → [0, 1]
    
    Returns normalized L*a*b* image ready for YOLO input.
    """
    lab = rgb_to_lab(image_rgb)
    
    # Normalize each channel
    L_norm = lab[:, :, 0] / 100.0  # L: 0-100 → 0-1
    a_norm = (lab[:, :, 1] + 128) / 255.0  # a: -128-127 → 0-1
    b_norm = (lab[:, :, 2] + 128) / 255.0  # b: -128-127 → 0-1
    
    # Stack back
    lab_normalized = np.stack([L_norm, a_norm, b_norm], axis=-1)
    
    return lab_normalized


def create_multichannel_input(image_rgb, include_redness=True):
    """
    Create multi-channel input combining RGB, L*a*b*, and redness index.
    
    Channels:
    - 0-2: RGB
    - 3: L (Lightness)
    - 4: a* (Green-Red axis) ⭐ KEY for B2/B3 separation
    - 5: b* (Blue-Yellow axis)
    - 6: redness_index (R-G)/(R+G) ⭐ Additional maturity signal
    
    Returns 6 or 7 channel image (H, W, 6 or 7)
    """
    # Normalize RGB to [0, 1]
    if image_rgb.dtype == np.uint8:
        rgb_norm = image_rgb / 255.0
    else:
        rgb_norm = image_rgb
    
    # Get L*a*b*
    lab = rgb_to_lab_normalized(image_rgb)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    
    if include_redness:
        # Compute redness index: (R - G) / (R + G + epsilon)
        R = rgb_norm[:, :, 0]
        G = rgb_norm[:, :, 1]
        epsilon = 1e-7
        redness_index = (R - G) / (R + G + epsilon)
        
        # Stack: RGB + L + a* + b* + redness = 7 channels
        multichannel = np.stack([
            rgb_norm[:, :, 0],  # R
            rgb_norm[:, :, 1],  # G
            rgb_norm[:, :, 2],  # B
            L,                   # Lightness
            a,                   # a* (Green-Red) ⭐ KEY CHANNEL
            b,                   # b* (Blue-Yellow)
            redness_index        # (R-G)/(R+G) ⭐ MATURITY SIGNAL
        ], axis=-1)
    else:
        # Stack: RGB + L + a* + b* = 6 channels
        multichannel = np.stack([
            rgb_norm[:, :, 0],
            rgb_norm[:, :, 1],
            rgb_norm[:, :, 2],
            L,
            a,
            b
        ], axis=-1)
    
    return multichannel


def process_image(image_path, output_dir, color_space='lab', include_redness=True):
    """
    Process a single image and save in specified color space.
    
    Args:
        image_path: Path to input image
        output_dir: Directory for output
        color_space: 'lab', 'rgb_lab', or 'multichannel'
        include_redness: Whether to include redness index
    """
    try:
        # Read image
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            return False, f"Failed to read {image_path}"
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Process based on color space
        if color_space == 'lab':
            # Pure L*a*b* (3 channels)
            output = rgb_to_lab_normalized(image_rgb)
        elif color_space == 'rgb_lab':
            # RGB + L*a*b* (6 channels)
            lab = rgb_to_lab_normalized(image_rgb)
            rgb_norm = image_rgb / 255.0
            output = np.concatenate([rgb_norm, lab], axis=-1)
        elif color_space == 'multichannel':
            # Full multi-channel (6 or 7 channels)
            output = create_multichannel_input(image_rgb, include_redness)
        else:
            return False, f"Unknown color space: {color_space}"
        
        # Save as numpy array (compressed)
        output_path = Path(output_dir) / f"{Path(image_path).stem}.npy"
        np.save(output_path, output)
        
        return True, None
    except Exception as e:
        return False, str(e)


def preprocess_dataset(input_dir, output_dir, color_space='lab', num_workers=8):
    """
    Preprocess entire dataset to L*a*b* or multi-channel format.
    
    Args:
        input_dir: Directory with input images
        output_dir: Directory for preprocessed outputs
        color_space: 'lab', 'rgb_lab', or 'multichannel'
        num_workers: Number of parallel workers
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_paths.extend(input_dir.glob(ext))
    
    print(f"Found {len(image_paths)} images in {input_dir}")
    print(f"Color space: {color_space}")
    print(f"Output directory: {output_dir}")
    
    # Process in parallel
    process_func = partial(process_image, output_dir=output_dir, color_space=color_space)
    
    with mp.Pool(num_workers) as pool:
        results = pool.map(process_func, image_paths)
    
    # Report results
    success_count = sum(1 for success, _ in results if success)
    error_count = len(results) - success_count
    
    print(f"✅ Successfully processed: {success_count}")
    print(f"❌ Failed: {error_count}")
    
    return success_count, error_count


def demonstrate_lab_channels():
    """Demonstrate the L*a*b* channels and their relevance to oil palm maturity."""
    print("=" * 70)
    print("L*a*b* Color Space for Oil Palm Maturity Detection")
    print("=" * 70)
    
    print("\n📊 Channel Definitions:")
    print("  L* (Lightness):     0 (black) to 100 (white)")
    print("  a* (Green-Red):   -128 (green) to +127 (red) ⭐ KEY CHANNEL")
    print("  b* (Blue-Yellow): -128 (blue) to +127 (yellow)")
    
    print("\n🌴 Oil Palm Maturity Signatures:")
    print("  B1 (Ripe):        High a* (red undertones), Medium L*")
    print("  B2 (Transition):  Positive a* (red appearing), Lower L*")
    print("  B3 (Unripe):      Near-zero a* (pure black), Low L*")
    print("  B4 (Small):       Varies by individual fruitlet maturity")
    
    print("\n💡 Why L*a*b* works for B2/B3 separation:")
    print("  • a* channel directly separates red (B2) from black (B3)")
    print("  • RGB conflates luminance with chrominance")
    print("  • L*a*b* is perceptually uniform")
    print("  • Reference: Septiarini et al., 2021 (98.3% accuracy)")
    
    print("\n🎯 Implementation Strategy:")
    print("  1. Preprocess dataset to L*a*b* (3 channels)")
    print("  2. Or: Multi-channel (RGB + L + a* + b* + redness = 6-7 channels)")
    print("  3. Modify YOLO YAML: ch: 6 or ch: 7")
    print("  4. First layer loses ImageNet pretraining, rest keeps pretraining")
    print("=" * 70)


def visualize_lab_analysis(image_path):
    """
    Analyze a sample image showing L*a*b* channel distributions.
    Useful for debugging and validation.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Read and convert
    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    lab = rgb_to_lab(image_rgb)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original RGB
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original RGB')
    axes[0, 0].axis('off')
    
    # R, G, B channels
    axes[0, 1].imshow(image_rgb[:, :, 0], cmap='Reds')
    axes[0, 1].set_title('R Channel')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(image_rgb[:, :, 1], cmap='Greens')
    axes[0, 2].set_title('G Channel')
    axes[0, 2].axis('off')
    
    # L*, a*, b* channels
    axes[1, 0].imshow(lab[:, :, 0], cmap='gray')
    axes[1, 0].set_title('L* (Lightness)')
    axes[1, 0].axis('off')
    
    # a* channel: shift to positive for visualization
    a_display = lab[:, :, 1].astype(np.float32)
    axes[1, 1].imshow(a_display, cmap='RdYlGn_r', vmin=-128, vmax=127)
    axes[1, 1].set_title('a* (Green-Red)\nNegative=Green, Positive=Red')
    axes[1, 1].axis('off')
    
    # b* channel
    b_display = lab[:, :, 2].astype(np.float32)
    axes[1, 2].imshow(b_display, cmap='YlGnBu', vmin=-128, vmax=127)
    axes[1, 2].set_title('b* (Blue-Yellow)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    output_path = Path(image_path).parent / f"{Path(image_path).stem}_lab_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved L*a*b* analysis to: {output_path}")
    
    # Print statistics
    print(f"\n📊 Channel Statistics:")
    print(f"  L*:  mean={lab[:, :, 0].mean():.1f}, std={lab[:, :, 0].std():.1f}")
    print(f"  a*:  mean={lab[:, :, 1].mean():.1f}, std={lab[:, :, 1].std():.1f}")
    print(f"  b*:  mean={lab[:, :, 2].mean():.1f}, std={lab[:, :, 2].std():.1f}")
    
    # Interpretation
    a_mean = lab[:, :, 1].mean()
    if a_mean > 20:
        maturity_guess = "Likely B1 (Ripe) - Strong red signal"
    elif a_mean > 0:
        maturity_guess = "Likely B2 (Transition) - Emerging red"
    elif a_mean > -20:
        maturity_guess = "Likely B3 (Unripe) - Neutral/Dark"
    else:
        maturity_guess = "Likely B4 (Small) or very unripe"
    
    print(f"\n🎯 Maturity Estimate from a* channel: {maturity_guess}")


if __name__ == "__main__":
    demonstrate_lab_channels()
