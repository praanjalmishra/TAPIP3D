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

logger = logging.getLogger(__name__)

DEFAULT_QUERY_GRID_SIZE = 8

NERF_TO_OPENCV = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1]
], dtype=np.float32)


class Arguments(tap.Tap):
    input_path: str  # Can be .mp4, .npz, or transforms.json
    device: str = "cuda"
    num_iters: int = 6
    support_grid_size: int = 16
    num_threads: int = 8
    resolution_factor: int = 1
    vis_threshold: Optional[float] = 0.9
    checkpoint: Optional[str] = "checkpoints/tapip3d_final.pth"
    output_dir: str = "outputs/inference"
    depth_model: str = "moge"
def load_from_nerf_transforms(json_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load video, depths, intrinsics, and extrinsics from NeRF transforms.json
    
    Returns:
        video: (T, H, W, 3) uint8 array
        depths: (T, H, W) float32 array
        intrinsics: (T, 3, 3) float32 array
        extrinsics: (T, 4, 4) float32 array (camera-to-world in OpenCV convention)
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
            # if depth.max() > 0:
            #     depth /= depth.max()
        depths.append(depth)
        
        c2w_nerf = np.array(fdata["transform_matrix"], dtype=np.float32)

        # Convert to OpenCV convention (still c2w)
        c2w_opencv = NERF_TO_OPENCV @ c2w_nerf @ NERF_TO_OPENCV.T

        w2c_opencv = np.linalg.inv(c2w_opencv)
        extrinsics.append(w2c_opencv)

    video = np.stack(video, axis=0)
    depths = np.stack(depths, axis=0)
    extrinsics = np.stack(extrinsics, axis=0)
    

    # === Debug verification block ===
    logger.info("\n=== Verification summary ===")
    logger.info(f"Extrinsics shape: {extrinsics.shape}")

    # Check first and last frame
    first = np.linalg.inv(extrinsics[0])  # back to c2w
    last = np.linalg.inv(extrinsics[-1])
    print_frame_debug(first, extrinsics[0], 0)
    print_frame_debug(last, extrinsics[-1], len(extrinsics) - 1)

    # Translation sanity
    trans = np.stack([E[:3, 3] for E in extrinsics])
    logger.info(f"Translation range per axis: X[{trans[:,0].min():.3f},{trans[:,0].max():.3f}], "
                f"Y[{trans[:,1].min():.3f},{trans[:,1].max():.3f}], "
                f"Z[{trans[:,2].min():.3f},{trans[:,2].max():.3f}]")

    # Rotation determinant check
    determinants = [np.linalg.det(E[:3, :3]) for E in extrinsics]
    logger.info(f"Rotation det stats: mean={np.mean(determinants):.6f}, "
                f"std={np.std(determinants):.6e}, min={np.min(determinants):.6f}, max={np.max(determinants):.6f}")

    # Optional quick world-stationary test (approximate)
    if "coords" in meta:
        coords = np.array(meta["coords"])
        std_val = np.std(coords, axis=0).mean()
        logger.info(f"World motion check: coord std = {std_val:.6f}")
        if std_val > 0.05:
            logger.warning("⚠️ World likely moving — double-check extrinsics convention!")
    else:
        logger.info("No coords field for world-motion test.")

    logger.info("Verification complete.\n")

    # Build intrinsics matrix
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)
    intrinsics = np.tile(K[None, :, :], (T, 1, 1))
    
    logger.info(f"Loaded {T} frames with camera motion (translation std): {extrinsics[:, :3, 3].std(axis=0)}")
    
    return video, depths, intrinsics, extrinsics


import numpy.linalg as LA

def assert_rotation_valid(R: np.ndarray):
    det = np.linalg.det(R)
    if not np.isclose(det, 1.0, atol=1e-3):
        logger.warning(f"[Rotation check] ⚠️ Determinant not 1.0 → {det:.6f}")
    ortho_err = LA.norm(R.T @ R - np.eye(3))
    if ortho_err > 1e-3:
        logger.warning(f"[Rotation check] ⚠️ Non-orthogonal rotation (error={ortho_err:.4e})")

def print_frame_debug(c2w: np.ndarray, w2c: np.ndarray, idx: int):
    # Forward, up, right vectors (OpenCV convention)
    fwd = -c2w[:3, 2]
    up = c2w[:3, 1]
    right = c2w[:3, 0]
    logger.info(f"Frame {idx}:")
    logger.info(f"  c2w translation = {c2w[:3, 3]}")
    logger.info(f"  w2c translation = {w2c[:3, 3]}")
    logger.info(f"  det(R) = {np.linalg.det(c2w[:3,:3]):.6f}")
    logger.info(f"  Forward (OpenCV -Z) = {fwd}")
    logger.info(f"  Up (+Y) = {up}")
    logger.info(f"  Right (+X) = {right}")


def is_c2w(matrix):
    # Forward axis (OpenCV convention: camera looks along -Z)
    forward = -matrix[:3, 2]
    up = matrix[:3, 1]
    # Should point roughly opposite directions between conventions
    return forward[2] < 0 and abs(up[1]) > 0.5

def assert_rotation_valid(R):
    det = np.linalg.det(R)
    # assert np.isclose(det, 1.0, atol=1e-3), f"Bad rotation det={det}"
    # assert np.allclose(R.T @ R, np.eye(3), atol=1e-3), "Rotation not orthogonal"


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
            executor.submit(_filter_one_depth, depth, 0.08, 15, intrinsic) 
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


if __name__ == "__main__":
    setup_logger()
    args = Arguments().parse_args()
    
    output_dir = Path(args.output_dir) / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    input_name = Path(args.input_path).stem
    if input_name.startswith("transforms"):
        input_name = Path(args.input_path).parent.name
    
    npz_path = (output_dir / input_name).with_suffix(".result.npz")
    npz_path.parent.mkdir(exist_ok=True, parents=True)
    
    np.savez(
        npz_path,
        video=video,
        depths=depths,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        coords=coords,
        visibs=visibs,
        query_points=query_point,
    )

    # coords = np.load("...result.npz")["coords"]
    print(np.std(coords, axis=0).mean())

    logger.info(f"Results saved to {npz_path.resolve()}.")
    logger.info(f"To visualize, run: [bold yellow]python visualize.py {shlex.quote(str(npz_path.resolve()))}[/bold yellow]")