# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)

from concurrent.futures import ThreadPoolExecutor
import shlex
import tap
import torch
from typing import Optional, Tuple
from pathlib import Path
from datetime import datetime
from einops import repeat
from utils.common_utils import setup_logger
import logging
from annotation.megasam import MegaSAMAnnotator
import numpy as np
import cv2
import json
import imageio.v2 as imageio
from datasets.data_ops import _filter_one_depth

from utils.inference_utils import load_model, read_video, inference, get_grid_queries, resize_depth_bilinear
from scipy.signal import savgol_filter

from depth_preprocessing import process_depth_sequence

logger = logging.getLogger(__name__)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_QUERY_GRID_SIZE = 16

# NeRF (OpenGL) to OpenCV 
NERF_TO_OPENCV = np.array([
    [ 1,  0,  0,  0],
    [ 0, -1,  0,  0],
    [ 0,  0, -1,  0],
    [ 0,  0,  0,  1]
], dtype=np.float32)


def filter_significant_motion(coords, visibs, vis_thresh=0.5, motion_thresh=0.01):
    """
    Identifies tracks with significant motion to filter out stationary/noisy points.
    
    Args:
        coords: np.ndarray (T, N, 3)
        visibs: np.ndarray (T, N)
        vis_thresh: visibility threshold
        motion_thresh: minimum motion required (in coordinate units)
    
    Returns:
        significant_tracks: np.ndarray (N,) boolean mask
        motion_stats: dict with motion statistics per track
    """
    T, N, _ = coords.shape
    significant_tracks = np.zeros(N, dtype=bool)
    motion_stats = {}
    
    for n in range(N):
        valid = visibs[:, n] > vis_thresh
        if valid.sum() < 2:
            continue
        
        valid_coords = coords[valid, n, :]  # (T_valid, 3)
        
        # Strategy 1: Total displacement (first to last valid frame)
        total_displacement = np.linalg.norm(valid_coords[-1] - valid_coords[0])
        
        # Strategy 2: Cumulative path length
        diffs = np.diff(valid_coords, axis=0)
        path_length = np.sum(np.linalg.norm(diffs, axis=1))
        
        # Strategy 3: Maximum coordinate range across all dimensions
        coord_range = np.max(valid_coords, axis=0) - np.min(valid_coords, axis=0)
        max_range = np.max(coord_range)
        
        # Strategy 4: Standard deviation (spread of motion)
        std_dev = np.std(valid_coords, axis=0).max()
        
        motion_stats[n] = {
            'total_displacement': total_displacement,
            'path_length': path_length,
            'max_range': max_range,
            'std_dev': std_dev,
            'num_valid_frames': valid.sum()
        }
        
        # Mark as significant if any metric exceeds threshold
        if (total_displacement > motion_thresh or 
            path_length > motion_thresh or 
            max_range > motion_thresh):
            significant_tracks[n] = True
    
    return significant_tracks, motion_stats


def smooth_3d_tracks_with_motion_filter(coords, visibs, window=7, poly=2, 
                                         vis_thresh=0.5, motion_thresh=0.01):
    """
    Smooths only tracks with significant motion to avoid over-smoothing noise.
    """
    T, N, _ = coords.shape
    coords_smooth = coords.copy()
    
    # First, identify tracks with significant motion
    significant_tracks, motion_stats = filter_significant_motion(
        coords, visibs, vis_thresh, motion_thresh
    )
    
    tracks_smoothed = 0
    tracks_skipped_motion = 0
    tracks_skipped_frames = 0
    
    print(f"Motion filtering: {significant_tracks.sum()}/{N} tracks have motion > {motion_thresh}")
    
    for n in range(N):
        valid = visibs[:, n] > vis_thresh
        num_valid = valid.sum()
        
        # Skip if insufficient motion
        if not significant_tracks[n]:
            tracks_skipped_motion += 1
            continue
        
        # Skip if insufficient frames
        if num_valid < window:
            # print(f"Track {n}: Has motion but only {num_valid} valid frames (need {window})")
            tracks_skipped_frames += 1
            continue
        
        # Smooth each coordinate channel
        max_change = 0.0
        for dim in range(3):
            y = coords[valid, n, dim]
            y_smooth = savgol_filter(y, window, poly, mode='interp')
            coords_smooth[valid, n, dim] = y_smooth
            change = np.abs(y - y_smooth).max()
            max_change = max(max_change, change)
        
        stats = motion_stats[n]
        # print(f"Track {n}: Smoothed {num_valid} frames | "
        #       f"motion={stats['total_displacement']:.4f} | "
        #       f"max_change={max_change:.6f}")
        tracks_smoothed += 1
    
    print(f"\nSummary:")
    print(f"  {tracks_smoothed} tracks smoothed")
    print(f"  {tracks_skipped_motion} skipped (insufficient motion)")
    print(f"  {tracks_skipped_frames} skipped (insufficient frames)")
    
    return coords_smooth, significant_tracks, motion_stats




def get_queries_from_mask(
    mask: np.ndarray,  # (H, W) binary mask
    frame_idx: int,
    depth: torch.Tensor,  # (H, W)
    intrinsic: torch.Tensor,  # (3, 3)
    extrinsic: torch.Tensor,  # (4, 4)
    max_points: int = 256
) -> torch.Tensor:
    """
    Convert a 2D mask to 3D query points.
    
    Returns:
        query_points: (N, 4) tensor of [frame_idx, x_world, y_world, z_world]
    """
    # Get pixel coordinates where mask is True
    y_coords, x_coords = np.where(mask > 0)
    
    if len(y_coords) == 0:
        raise ValueError("Mask is empty - no pixels to query")
    
    # Subsample if too many points
    if len(y_coords) > max_points:
        indices = np.random.choice(len(y_coords), size=max_points, replace=False)
        y_coords = y_coords[indices]
        x_coords = x_coords[indices]
    
    # Convert to torch
    xy = torch.tensor(np.stack([x_coords, y_coords], axis=-1), 
                      dtype=torch.float32, device=depth.device)  # (N, 2)
    
    # Get depth values at these points
    ji = xy.long()
    d = depth[ji[:, 1], ji[:, 0]]  # (N,)
    
    # Filter out invalid depth
    mask_valid = d > 0
    xy = xy[mask_valid]
    d = d[mask_valid]
    
    if len(d) == 0:
        raise ValueError("No valid depth values found in mask region")
    
    # Backproject to 3D world coordinates
    inv_intrinsic = torch.linalg.inv(intrinsic)
    inv_extrinsic = torch.linalg.inv(extrinsic)
    
    # To homogeneous coordinates
    xy_homo = torch.cat([xy, torch.ones(len(xy), 1, device=xy.device)], dim=-1)  # (N, 3)
    
    # To camera coordinates
    local_coords = torch.einsum('ij,nj->ni', inv_intrinsic, xy_homo) * d[:, None]  # (N, 3)
    local_coords_homo = torch.cat([local_coords, torch.ones(len(local_coords), 1, device=xy.device)], dim=-1)  # (N, 4)
    
    # To world coordinates
    world_coords = torch.einsum('ij,nj->ni', inv_extrinsic, local_coords_homo)[:, :3]  # (N, 3)
    
    # Add frame index as first column
    frame_indices = torch.full((len(world_coords), 1), frame_idx, 
                               dtype=torch.float32, device=world_coords.device)
    query_points = torch.cat([frame_indices, world_coords], dim=-1)  # (N, 4)
    
    # Return (N, 4) - NO batch dimension, just like get_grid_queries
    return query_points

class Arguments(tap.Tap):
    input_path: str  # Can be .mp4, .npz, or transforms.json
    device: str = "cuda"
    num_iters: int = 12
    support_grid_size: int = 16
    num_threads: int = 8
    resolution_factor: float = 0.5
    vis_threshold: Optional[float] = 0.95
    checkpoint: Optional[str] = "/local/home/pmishra/cvg/TAPIP3D/checkpoints/tapip3d_final.pth"
    output_dir: str = "outputs/inference"
    depth_model: str = "moge"
    mask_path: Optional[str] = None  # Path to mask image
    mask_frame: int = 0  # Which frame the mask corresponds to
    max_query_points: int = 512  # Max number of points to sample from mask


def load_from_nerf_transforms(json_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load video, depths, intrinsics, and extrinsics from NeRF transforms.json
    
    Returns:
        video: (T, H, W, 3) uint8 array
        depths: (T, H, W) float32 array
        intrinsics: (T, 3, 3) float32 array
        extrinsics: (T, 4, 4) float32 array (world-to-camera in OpenCV convention)
    """
    with open(json_path, "r") as f:
        meta = json.load(f)
    
    fx, fy, cx, cy = meta["fl_x"], meta["fl_y"], meta["cx"], meta["cy"]
    W, H = meta["w"], meta["h"]
    frames = meta["frames"]
    T = len(frames)
    
    logger.info(f"Loading {T} frames from NeRF transforms: {json_path}")
    logger.info(f"Image size: {W}x{H}, Intrinsics: fx={fx:.2f}, fy={fy:.2f}")
    
    base_dir = json_path.parent
    video = []
    depths = []
    extrinsics = []
    
    for idx, fdata in enumerate(frames):
        # Load image
        img_path = (base_dir / fdata["file_path"]).resolve()
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = imageio.imread(img_path).astype(np.uint8)
        video.append(img)
        
        # Load depth
        depth_file = fdata.get("depth_npy_file_path", fdata.get("depth_file_path"))
        if depth_file is None:
            raise ValueError(f"No depth path for frame {idx}")
        depth_path = (base_dir / depth_file).resolve()
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth file not found: {depth_path}")
        
        if depth_path.suffix == ".npy":
            depth = np.load(depth_path)
        else:
            depth = imageio.imread(depth_path).astype(np.float32)
            depth = depth / 1000.0 # assuming depth in mm
        depths.append(depth)
        
        c2w_nerf = np.array(fdata["transform_matrix"], dtype=np.float32)
        
        # Convert from NeRF (OpenGL) to OpenCV convention
        c2w_opencv =  c2w_nerf @ NERF_TO_OPENCV
        
        w2c_opencv = np.linalg.inv(c2w_opencv)
        
        extrinsics.append(w2c_opencv)
    
    video = np.stack(video, axis=0)
    depths = np.stack(depths, axis=0)
    extrinsics = np.stack(extrinsics, axis=0)
    
    # Build intrinsics matrix
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)
    intrinsics = np.tile(K[None, :, :], (T, 1, 1))
    
    logger.info(f"Loaded {T} frames with camera motion (translation std): {extrinsics[:, :3, 3].std(axis=0)}")
    
    return video, depths, intrinsics, extrinsics


def prepare_inputs(
    input_path: str, 
    inference_res: Tuple[int, int], 
    support_grid_size: int, 
    num_threads: int = 8, 
    device: str = "cpu",
    depth_model: str = "moge"
):
    """
    Prepare inputs for TAPIP3D inference.
    Supports .mp4, .npz, or transforms.json input formats.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise ValueError(f"Input file not found: {input_path}")
    
    video, depths, intrinsics, extrinsics, query_point = None, None, None, None, None
    
    # Determine input type and load accordingly
    if input_path.suffix in [".mp4", ".avi", ".mov", ".webm"]:
        video = read_video(str(input_path))
        
    elif input_path.suffix == ".npz":
        data = np.load(input_path)
        video = data['video']
        assert video.ndim == 4 and video.shape[-1] == 3 and video.dtype == np.uint8, \
            f"Invalid video shape or dtype: {video.shape}, {video.dtype}"
        depths = data.get('depths', None)
        intrinsics = data.get('intrinsics', None)
        extrinsics = data.get('extrinsics', None)
        query_point = data.get('query_point', None)
        
    elif input_path.suffix == ".json" or input_path.name.startswith("transforms"):
        # Load from NeRF transforms.json
        video, depths, intrinsics, extrinsics = load_from_nerf_transforms(input_path)
        
    else:
        raise ValueError(
            f"Unsupported input type: {input_path}. "
            f"Supported formats: .mp4, .npz, or transforms*.json"
        )
    
    # If no depth provided, run depth estimation
    if depths is None:
        logger.info(f"No depth provided, running depth estimation with {depth_model}")
        megasam = MegaSAMAnnotator(
            script_path=Path(__file__).parent / "third_party" / "megasam" / "inference.py",
            depth_model=depth_model,
            resolution=inference_res[0] * inference_res[1]
        )
        megasam.to(device)
        depths, intrinsics, extrinsics = megasam.process_video(
            video, 
            gt_intrinsics=intrinsics, 
            return_raw_depths=True
        )
        _original_res = video.shape[1:3]
    else:
        _original_res = depths.shape[1:3]

    depths = process_depth_sequence(
        depths,
        intrinsics=intrinsics,
        method='comprehensive',
        use_temporal=True,
        num_threads=num_threads
    )
    
    # Validate inputs
    if intrinsics is None:
        raise ValueError("Intrinsics must be provided if depth is provided")
    
    if extrinsics is None:
        logger.info(f"No extrinsics provided, using identity matrix for all frames")
        extrinsics = repeat(np.eye(4), "i j -> t i j", t=len(video))
    
    # Scale intrinsics to match inference resolution
    intrinsics[:, 0, :] *= (inference_res[1] - 1) / (_original_res[1] - 1)
    intrinsics[:, 1, :] *= (inference_res[0] - 1) / (_original_res[0] - 1)
    
    # Resize video and depths
    with ThreadPoolExecutor(num_threads) as executor:
        video_futures = [
            executor.submit(cv2.resize, rgb, (inference_res[1], inference_res[0]), 
                          interpolation=cv2.INTER_LINEAR) 
            for rgb in video
        ]
        depths_futures = [
            executor.submit(resize_depth_bilinear, depth, (inference_res[1], inference_res[0])) 
            for depth in depths
        ]
        
        video = np.stack([future.result() for future in video_futures])
        depths = np.stack([future.result() for future in depths_futures])
        
        # Filter depth edges
        depths_futures = [
            executor.submit(_filter_one_depth, depth, 0.15, 10, intrinsic) 
            for depth, intrinsic in zip(depths, intrinsics)
        ]
        depths = np.stack([future.result() for future in depths_futures])
    
    # Convert to torch tensors
    video = (torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0).to(device)
    depths = torch.from_numpy(depths).float().to(device)
    intrinsics = torch.from_numpy(intrinsics).float().to(device)
    extrinsics = torch.from_numpy(extrinsics).float().to(device)
    
    # Generate query points if not provided
    if query_point is None:
        support_grid_size = 0
        query_point = get_grid_queries(
            grid_size=DEFAULT_QUERY_GRID_SIZE, 
            depths=depths, 
            intrinsics=intrinsics, 
            extrinsics=extrinsics
        )
        logger.info(f"No queries provided, using {DEFAULT_QUERY_GRID_SIZE}x{DEFAULT_QUERY_GRID_SIZE} grid at first frame")
    else:
        query_point = torch.from_numpy(query_point).float().to(device)

    return video, depths, intrinsics, extrinsics, query_point, support_grid_size

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if __name__ == "__main__":
    setup_logger()
    args = Arguments().parse_args()
    
    # output_dir = Path(args.output_dir) / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.input_path = str(Path(args.input_path).resolve())
    args.mask_path = str(Path(args.mask_path).resolve())
    args.output_dir = str(output_dir.resolve())

    # Load model
    model = load_model(args.checkpoint)
    model.to(args.device)
    
    inference_res = (
        int(model.image_size[0] * np.sqrt(args.resolution_factor)), 
        int(model.image_size[1] * np.sqrt(args.resolution_factor))
    )
    model.set_image_size(inference_res)
    
    video, depths, intrinsics, extrinsics, query_point, support_grid_size = prepare_inputs(
        input_path=args.input_path, 
        inference_res=inference_res, 
        support_grid_size=args.support_grid_size,
        num_threads=args.num_threads,
        device=args.device,
        depth_model=args.depth_model
    )

    # Add this section to handle mask
    if args.mask_path is not None:
        logger.info(f"Loading mask from {args.mask_path}")
        mask = imageio.imread(args.mask_path)
        
        # Convert to binary mask
        if mask.ndim == 3:
            mask = mask[..., 0]  # Take first channel if RGB/RGBA
        mask = (mask > 127).astype(bool)
            
        mask_resized = cv2.resize(
            mask.astype(np.uint8), 
            (depths.shape[2], depths.shape[1]),  # cv2.resize expects (width, height)
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        
        query_point = get_queries_from_mask(
            mask=mask_resized,
            frame_idx=args.mask_frame,
            depth=depths[args.mask_frame],
            intrinsic=intrinsics[args.mask_frame],
            extrinsic=extrinsics[args.mask_frame],
            max_points=args.max_query_points
        )

        print(f"Query point shape: {query_point.shape}")  # Should be [N, 4]
                        
        support_grid_size = 0  
        logger.info(f"Generated {query_point.shape[0]} query points from mask at frame {args.mask_frame}")

    # Run inference
    logger.info("Running TAPIP3D inference...")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        coords, visibs = inference(
            model=model,
            video=video,
            depths=depths,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            query_point=query_point,
            num_iters=args.num_iters,
            grid_size=support_grid_size,
        )


    
    # Save results
    video = video.cpu().numpy()
    depths = depths.cpu().numpy()
    intrinsics = intrinsics.cpu().numpy()
    extrinsics = extrinsics.cpu().numpy()
    coords = coords.cpu().numpy()
    visibs = visibs.cpu().numpy()
    query_point = query_point.cpu().numpy()


    # Filter 3d tracks
    print("\nFiltering tracks with significant motion...")
    coords_smooth, sig_tracks, stats = smooth_3d_tracks_with_motion_filter(
        coords, visibs,
        window=9,
        poly=3,
        motion_thresh=0.02 
    )

    # Apply the same filtering to all related arrays
    coords_filtered = coords_smooth[:, sig_tracks, :]
    visibs_filtered = visibs[:, sig_tracks]
    query_point_filtered = query_point[sig_tracks, :]

    print(f"\nFiltered from {coords.shape[1]} to {coords_filtered.shape[1]} tracks")
    print(f"Kept {coords_filtered.shape[1]/coords.shape[1]*100:.1f}% of tracks")

    # Save filtered results
    input_name = Path(args.input_path).stem
    if input_name.startswith("transforms"):
        input_name = Path(args.input_path).parent.name

    
    npz_path = output_dir / "tapip3d_trajectory.npz"
    npz_path.parent.mkdir(exist_ok=True, parents=True)
    
    np.savez(
        npz_path,
        video=video,
        depths=depths,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        coords=coords_filtered,
        visibs=visibs_filtered,
        query_points=query_point_filtered,
    )
    
    logger.info(f"Results saved to {npz_path.resolve()}.")
    logger.info(f"To visualize, run: [bold yellow]python visualize.py {shlex.quote(str(npz_path.resolve()))}[/bold yellow]")