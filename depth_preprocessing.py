"""
Enhanced Depth Processing Pipeline for TAPIP3D
Reduces noise and improves 3D trajectory quality
"""

import numpy as np
import cv2
import torch
from scipy.ndimage import median_filter

def preprocess_depth_pipeline(depth, intrinsic=None, method='comprehensive'):
    """
    Comprehensive depth preprocessing to reduce noise.
    
    Args:
        depth: (H, W) depth map
        intrinsic: (3, 3) camera intrinsic (optional, for geometric filtering)
        method: 'minimal', 'standard', or 'comprehensive'
    
    Returns:
        depth_clean: Denoised depth map
    """
    depth = depth.copy()
    
    if method == 'minimal':
        # Just bilateral filter (fastest)
        return bilateral_filter_depth(depth)
    
    elif method == 'standard':
        # Bilateral + outlier removal
        depth = bilateral_filter_depth(depth)
        depth = remove_depth_outliers(depth)
        return depth
    
    elif method == 'comprehensive':
        # Full pipeline (best quality)
        depth = remove_depth_outliers(depth)           # Step 1: Remove spikes
        depth = bilateral_filter_depth(depth)          # Step 2: Smooth while preserving edges
        # depth = temporal_median_filter(depth)          # Step 3: Temporal consistency (if multiple frames)
        depth = fill_small_holes(depth)                # Step 4: Fill small invalid regions
        return depth
    
    return depth


# ============================================================================
# Individual Processing Steps
# ============================================================================

def bilateral_filter_depth(depth, d=5, sigma_color=0.1, sigma_space=5):
    """
    Bilateral filter: Smooths while preserving edges.
    
    - Reduces noise in flat regions
    - Keeps object boundaries sharp
    - Best for monocular depth artifacts
    
    Parameters:
        d: Kernel diameter (5-9 recommended)
        sigma_color: Color space sigma (0.05-0.2)
        sigma_space: Coordinate space sigma (5-10)
    """
    depth_filtered = cv2.bilateralFilter(
        depth.astype(np.float32),
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )
    return depth_filtered


def remove_depth_outliers(depth, window_size=5, threshold=2.5):
    """
    Remove depth spikes using local statistics.
    
    - Fixes sudden depth jumps
    - Removes isolated noisy pixels
    - Uses median absolute deviation (MAD)
    
    Parameters:
        window_size: Local neighborhood size (5-11)
        threshold: Sigma threshold for outlier (2.0-3.0)
    """
    # Compute local median
    depth_median = median_filter(depth, size=window_size)
    
    # Compute median absolute deviation (MAD)
    mad = np.median(np.abs(depth - depth_median))
    
    # Mark outliers
    outlier_mask = np.abs(depth - depth_median) > threshold * mad
    
    # Replace outliers with median
    depth_clean = depth.copy()
    depth_clean[outlier_mask] = depth_median[outlier_mask]
    
    # print(f"Removed {outlier_mask.sum()} outlier pixels ({outlier_mask.sum()/depth.size*100:.2f}%)")
    return depth_clean


def temporal_median_filter(depth_sequence, window=3):
    """
    Temporal filtering across video frames.
    
    USE THIS if processing a video sequence!
    
    Args:
        depth_sequence: (T, H, W) or list of (H, W) depth maps
        window: Temporal window size (3-5 frames)
    
    Returns:
        depth_filtered: Temporally smoothed depths
    """
    if isinstance(depth_sequence, list):
        depth_sequence = np.array(depth_sequence)
    
    T, H, W = depth_sequence.shape
    depth_filtered = depth_sequence.copy()
    
    half_window = window // 2
    
    for t in range(T):
        t_start = max(0, t - half_window)
        t_end = min(T, t + half_window + 1)
        
        # Median across time
        depth_filtered[t] = np.median(depth_sequence[t_start:t_end], axis=0)
    
    return depth_filtered


def fill_small_holes(depth, max_hole_size=50):
    """
    Fill small invalid regions in depth map.
    
    - Interpolates over holes
    - Keeps large invalid regions unchanged
    
    Parameters:
        max_hole_size: Maximum hole area to fill (pixels)
    """
    # Find invalid regions
    invalid_mask = (depth == 0) | np.isnan(depth) | np.isinf(depth)
    
    if invalid_mask.sum() == 0:
        return depth
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(invalid_mask.astype(np.uint8))
    
    depth_filled = depth.copy()
    
    for label in range(1, num_labels):
        component_mask = (labels == label)
        component_size = component_mask.sum()
        
        if component_size <= max_hole_size:
            # Fill small holes using inpainting
            depth_filled = cv2.inpaint(
                depth_filled.astype(np.float32),
                component_mask.astype(np.uint8),
                inpaintRadius=3,
                flags=cv2.INPAINT_TELEA
            )
    
    return depth_filled


def depth_consistency_check(depth, intrinsic, threshold=0.02):
    """
    Check for geometric consistency (optional, more advanced).
    
    - Ensures depth gradients match surface normals
    - Removes physically impossible depth values
    
    Parameters:
        threshold: Max allowed depth gradient
    """
    # Compute depth gradients
    grad_x = np.gradient(depth, axis=1)
    grad_y = np.gradient(depth, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize by depth (closer objects = larger gradients OK)
    grad_normalized = grad_mag / (depth + 1e-6)
    
    # Mark inconsistent regions
    inconsistent = grad_normalized > threshold
    
    # Smooth inconsistent regions
    depth_clean = depth.copy()
    depth_clean[inconsistent] = cv2.GaussianBlur(depth, (5, 5), 0)[inconsistent]
    
    return depth_clean


# ============================================================================
# Integration with Your Code
# ============================================================================

def process_depth_sequence(depths, intrinsics=None, method='comprehensive', 
                          use_temporal=True, num_threads=8):
    """
    Process entire depth sequence with parallelization.
    
    Args:
        depths: (T, H, W) numpy array or list of depth maps
        intrinsics: (T, 3, 3) camera intrinsics (optional)
        method: 'minimal', 'standard', or 'comprehensive'
        use_temporal: Apply temporal filtering (recommended for videos)
        num_threads: Number of parallel threads
    
    Returns:
        depths_clean: Processed depth maps
    """
    from concurrent.futures import ThreadPoolExecutor
    
    if isinstance(depths, list):
        depths = np.array(depths)
    
    T, H, W = depths.shape
    
    # Step 1: Spatial filtering (parallel)
    print(f"Processing {T} depth maps with method='{method}'...")
    
    with ThreadPoolExecutor(num_threads) as executor:
        futures = [
            executor.submit(preprocess_depth_pipeline, depths[t], 
                          intrinsics[t] if intrinsics is not None else None, 
                          method)
            for t in range(T)
        ]
        depths_spatial = np.array([f.result() for f in futures])
    
    # Step 2: Temporal filtering (optional but recommended)
    if use_temporal and T > 3:
        print("Applying temporal median filter...")
        depths_clean = temporal_median_filter(depths_spatial, window=3)
    else:
        depths_clean = depths_spatial
    
    print("✓ Depth preprocessing complete")
    return depths_clean

