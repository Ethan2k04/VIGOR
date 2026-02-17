import os
import tempfile
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import torch
import cv2
from tqdm import tqdm

from .base import BaseEvaluator


_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_VGGT_MODEL = None


def _get_amp_dtype() -> torch.dtype:
    """Get appropriate AMP dtype based on GPU capability."""
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def get_vggt_model():
    """
    Load VGGT once and reuse (singleton pattern).
    """
    global _VGGT_MODEL
    if _VGGT_MODEL is not None:
        return _VGGT_MODEL

    from ..third_party.vggt.models.vggt import VGGT
    
    model = VGGT()
    url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(_DEVICE)
    _VGGT_MODEL = model
    return _VGGT_MODEL


print("=" * 80)
print("[INFO] Loading VGGT model for vanilla reprojection...")
model = get_vggt_model()
print(f"[INFO] Device: {_DEVICE}")
print("[INFO] ✓ VGGT model loaded successfully")
print("=" * 80)


def extract_frames_uniform(video_path: str, max_frames: int, target_short_side: int) -> List[np.ndarray]:
    """
    Extract up to max_frames uniformly from the video. Return frames in BGR uint8.
    Resize so that short side equals target_short_side (keep aspect).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        idxs = None
    else:
        max_frames = min(int(max_frames), total)
        idxs = np.linspace(0, total - 1, max_frames).round().astype(int).tolist()

    frames = []
    if idxs is None:
        while len(frames) < max_frames:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
    else:
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, f = cap.read()
            if ok:
                frames.append(f)

    cap.release()
    if len(frames) == 0:
        raise ValueError("No frames extracted.")

    if target_short_side and int(target_short_side) > 0:
        resized = []
        ts = int(target_short_side)
        for f in frames:
            h, w = f.shape[:2]
            short = min(h, w)
            if short == ts:
                resized.append(f)
                continue
            scale = ts / float(short)
            nh = int(round(h * scale))
            nw = int(round(w * scale))
            resized.append(cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA))
        frames = resized

    return frames


def _normalize_depth_to_SHW1(depth_np: np.ndarray) -> np.ndarray:
    """Normalize depth array to (S,H,W,1) shape."""
    d = np.asarray(depth_np)
    if d.ndim == 5 and d.shape[0] == 1:
        d = d[0]
    if d.ndim == 4:
        if d.shape[-1] != 1:
            d = d[..., None]
        return d
    if d.ndim == 3:
        return d[..., None]
    raise ValueError(f"Unexpected depth shape: {d.shape}")


def run_vggt_once(frames_bgr: List[np.ndarray]) -> Dict[str, Any]:
    """
    Run VGGT on S frames, returning:
      images_rgb_0_1: (S,H,W,3) float32
      world_points:   (S,H,W,3) float32
      extrinsic:      (S,3,4) float32, world->cam
      intrinsic:      (S,3,3) float32
      depth:          (S,H,W) float32
      point_cloud:    (N,3) float32 - flattened 3D points
      colors:         (N,3) float32 - corresponding RGB colors
    """
    from ..third_party.vggt.utils.load_fn import load_and_preprocess_images
    from ..third_party.vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from ..third_party.vggt.utils.geometry import unproject_depth_map_to_point_map
    
    # Save frames to temp directory for VGGT preprocessing
    tmpdir = tempfile.mkdtemp(prefix="vggt_vanilla_")
    paths = []
    for k, bgr in enumerate(frames_bgr):
        p = os.path.join(tmpdir, f"{k:06d}.png")
        cv2.imwrite(p, bgr)
        paths.append(p)

    # Load and preprocess images
    images = load_and_preprocess_images(paths).to(_DEVICE)  # (S,3,H,W), 0..1
    amp_dtype = _get_amp_dtype()

    # Run VGGT inference
    model = get_vggt_model()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=amp_dtype):
            preds = model(images)

    # Extract camera parameters
    extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])

    # Extract depth
    depth_raw = preds["depth"].detach().float().cpu().numpy()
    depth = _normalize_depth_to_SHW1(depth_raw).astype(np.float32)  # (S,H,W,1)

    # Convert camera params to numpy
    extr = extrinsic.detach().float().cpu().numpy()
    intr = intrinsic.detach().float().cpu().numpy()
    if extr.ndim == 4 and extr.shape[0] == 1:
        extr = extr[0]
    if intr.ndim == 4 and intr.shape[0] == 1:
        intr = intr[0]
    extr = extr.astype(np.float32)
    intr = intr.astype(np.float32)

    # Unproject depth to 3D world points
    world_points = unproject_depth_map_to_point_map(depth, extr, intr).astype(np.float32)

    # Extract RGB images
    imgs = preds["images"].detach().float().cpu().numpy()
    if imgs.ndim == 5 and imgs.shape[0] == 1:
        imgs = imgs[0]
    imgs = np.transpose(imgs, (0, 2, 3, 1)).astype(np.float32)  # (S,H,W,3)

    depth_2d = depth[..., 0].astype(np.float32)

    # Create flattened point cloud (for VideoGPA-style rendering)
    S, H, W, _ = world_points.shape
    point_cloud = world_points.reshape(-1, 3)  # (S*H*W, 3)
    colors = imgs.reshape(-1, 3)  # (S*H*W, 3)
    
    # Filter invalid points (depth <= 0 or not finite)
    valid_mask = np.isfinite(point_cloud).all(axis=1) & (depth_2d.reshape(-1) > 1e-6)
    point_cloud = point_cloud[valid_mask]
    colors = colors[valid_mask]

    # Clean up temp directory
    try:
        import shutil
        shutil.rmtree(tmpdir)
    except Exception:
        pass

    return {
        "images_rgb_0_1": imgs,
        "world_points": world_points,
        "extrinsic": extr,
        "intrinsic": intr,
        "depth": depth_2d,
        "point_cloud": point_cloud,
        "colors": colors,
    }


def _init_lpips():
    """Initialize LPIPS perceptual loss."""
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').to(_DEVICE)
        return lpips_fn
    except ImportError:
        print("Warning: LPIPS not available, using MSE only")
        return None


class ReprojectionVanillaEvaluator(BaseEvaluator):
    """
    Vanilla reprojection-based 3D consistency evaluator following VideoGPA.
    
    This implementation follows the VideoGPA paper approach:
    1. Use VGGT to extract 3D structure and camera poses
    2. Reproject 3D points back to original frames using painter's algorithm
    3. Compute MSE + LPIPS between reprojected and original images
    
    Key differences from custom reprojection.py:
    - No geometry-aware sampling (processes all pixels)
    - Direct image-space reprojection (not correspondence-based)
    - Computes reconstruction error in pixel space (MSE + LPIPS)
    """
    
    def __init__(
        self,
        sampling_rate: int = 1,         # kept for compatibility with base class
        max_frames: int = 10,
        short_side: int = 448,
        use_lpips: bool = True,
        seed: int = 0,
    ):
        """
        Initialize the vanilla reprojection evaluator.
        
        Args:
            sampling_rate: Process every Nth frame (kept for compatibility, not used)
            max_frames: Number of frames to sample for 3D reconstruction (T in paper)
            short_side: Target short side resolution for processing
            use_lpips: Whether to use LPIPS loss (if False, MSE only)
            seed: Random seed for reproducibility
        """
        print("\n" + "=" * 80)
        print("[INFO] Initializing ReprojectionVanillaEvaluator...")
        super().__init__(sampling_rate)
        
        self.max_frames = int(max_frames)
        self.short_side = int(short_side)
        self.use_lpips = bool(use_lpips)
        self.seed = int(seed)
        
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        # Initialize LPIPS
        self.lpips_fn = None
        if self.use_lpips:
            self.lpips_fn = _init_lpips()
        
        print(f"[INFO] Max frames: {self.max_frames}")
        print(f"[INFO] Target short side: {self.short_side}")
        print(f"[INFO] Use LPIPS: {self.use_lpips}")
        print(f"[INFO] LPIPS available: {self.lpips_fn is not None}")
        print(f"[INFO] Random seed: {self.seed}")
        print("=" * 80 + "\n")
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ReprojectionVanillaEvaluator":
        """
        Create evaluator from configuration dictionary.
        
        Args:
            config: Configuration dictionary with parameters
            
        Returns:
            ReprojectionVanillaEvaluator instance
        """
        return cls(
            sampling_rate=config.get('sampling_rate', 1),
            max_frames=config.get('max_frames', 10),
            short_side=config.get('short_side', 448),
            use_lpips=config.get('use_lpips', True),
            seed=config.get('seed', 0),
        )
    
    def _reproject_to_frame(
        self,
        point_cloud: np.ndarray,
        colors: np.ndarray,
        extr: np.ndarray,
        intr: np.ndarray,
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Reproject 3D points to 2D frame using painter's algorithm.
        
        Following VideoGPA approach (Section 3.3):
        - Transform world points to camera frame
        - Project to image plane using intrinsics
        - Render using back-to-front ordering (painter's algorithm)
        
        Args:
            point_cloud: 3D points (N, 3) in world coordinates
            colors: RGB colors (N, 3) in [0, 1]
            extr: Camera extrinsic (3, 4), world->cam transform
            intr: Camera intrinsic (3, 3)
            target_shape: (height, width) of target image
            
        Returns:
            Reprojected image (H, W, 3) in [0, 255] uint8
        """
        H, W = target_shape
        
        # Transform points to camera frame
        # Xc = R @ Xw + t
        R = extr[:, :3]
        t = extr[:, 3]
        points_cam = (point_cloud @ R.T) + t[None, :]
        
        # Filter points behind camera
        valid_mask = points_cam[:, 2] > 1e-6
        if not np.any(valid_mask):
            return np.zeros((H, W, 3), dtype=np.uint8)
        
        points_cam = points_cam[valid_mask]
        colors_valid = colors[valid_mask]
        
        # Project to image plane
        fx, fy = float(intr[0, 0]), float(intr[1, 1])
        cx, cy = float(intr[0, 2]), float(intr[1, 2])
        
        z_inv = 1.0 / points_cam[:, 2]
        u = fx * (points_cam[:, 0] * z_inv) + cx
        v = fy * (points_cam[:, 1] * z_inv) + cy
        
        # Filter out-of-bounds points
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(in_bounds):
            return np.zeros((H, W, 3), dtype=np.uint8)
        
        u = u[in_bounds]
        v = v[in_bounds]
        depths = points_cam[in_bounds, 2]
        colors_valid = colors_valid[in_bounds]
        
        # Painter's algorithm: sort by depth (far to near)
        # This ensures correct occlusion handling
        sorted_indices = np.argsort(-depths)
        
        # Render points to image
        reprojected = np.zeros((H, W, 3), dtype=np.float32)
        
        for idx in sorted_indices:
            x_int = int(np.round(u[idx]))
            y_int = int(np.round(v[idx]))
            
            # Double-check bounds (should already be filtered)
            if 0 <= x_int < W and 0 <= y_int < H:
                reprojected[y_int, x_int] = colors_valid[idx]
        
        # Convert to uint8 [0, 255]
        reprojected = (reprojected * 255.0).clip(0, 255).astype(np.uint8)
        
        return reprojected
    
    def _compute_reconstruction_loss(
        self,
        original: np.ndarray,
        reprojected: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute reconstruction loss between original and reprojected images.
        
        Following VideoGPA Eq. (12):
        E_Recon = MSE(I_hat, I) + LPIPS(I_hat, I)
        
        Args:
            original: Original frame (H, W, 3) uint8 [0, 255]
            reprojected: Reprojected frame (H', W', 3) uint8 [0, 255]
            
        Returns:
            Dictionary with 'mse', 'lpips', and 'total' losses
        """
        # Ensure both images have the same shape
        if original.shape != reprojected.shape:
            H, W = original.shape[:2]
            # H_rep, W_rep = reprojected.shape[:2]
            # print(f"      Warning: Size mismatch - Original: {W}x{H}, Reprojected: {W_rep}x{H_rep}")
            # print(f"      Resizing reprojected to match original...")
            reprojected = cv2.resize(reprojected, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # Compute MSE (normalized to [0, 1] range)
        mse = np.mean((original.astype(float) / 255.0 - reprojected.astype(float) / 255.0) ** 2)
        
        # Compute LPIPS if available and enabled
        lpips_val = 0.0
        if self.use_lpips and self.lpips_fn is not None:
            # Convert to torch tensors and normalize to [-1, 1]
            orig_tensor = torch.from_numpy(original).permute(2, 0, 1).float() / 127.5 - 1.0
            repr_tensor = torch.from_numpy(reprojected).permute(2, 0, 1).float() / 127.5 - 1.0
            
            orig_tensor = orig_tensor.unsqueeze(0).to(_DEVICE)
            repr_tensor = repr_tensor.unsqueeze(0).to(_DEVICE)
            
            with torch.no_grad():
                lpips_val = self.lpips_fn(orig_tensor, repr_tensor).item()
        
        # Following VideoGPA Eq. (12): average MSE and LPIPS
        total = mse + lpips_val
        
        return {
            'mse': float(mse),
            'lpips': float(lpips_val),
            'total': float(total),
        }
    
    def compute_metrics(self, frame):
        """
        Not used (multi-frame metric). Present only to satisfy BaseEvaluator abstract API.
        """
        raise NotImplementedError("ReprojectionVanillaEvaluator overrides evaluate_video().")
    
    def aggregate_metrics(self, frame_metrics) -> Tuple[float, Dict[str, Any]]:
        """
        Not used. Present only to satisfy BaseEvaluator abstract API.
        """
        raise NotImplementedError("ReprojectionVanillaEvaluator overrides evaluate_video().")
    
    def evaluate_video(self, video_path: str) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate video using vanilla reprojection method.
        
        Following VideoGPA approach (Section 3.3):
        1. Sample T frames uniformly from video
        2. Use VGGT to reconstruct 3D scene (point cloud + camera poses)
        3. For each frame: reproject 3D points back to image plane
        4. Compute MSE + LPIPS between reprojected and original
        5. Average reconstruction error across all frames
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (3d_consistency_score, all_metrics)
            Lower score = better 3D consistency
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print(f"\n{'='*80}")
        print(f"Evaluating: {os.path.basename(video_path)}")
        print(f"{'='*80}")
        
        # Step 1: Extract frames uniformly
        print(f"[1/4] Extracting {self.max_frames} frames...")
        frames_bgr = extract_frames_uniform(
            video_path=video_path,
            max_frames=self.max_frames,
            target_short_side=self.short_side,
        )
        S = len(frames_bgr)
        H, W = frames_bgr[0].shape[:2]
        print(f"      Extracted {S} frames at {W}x{H}")
        
        # Step 2: Run VGGT to reconstruct 3D scene
        print(f"[2/4] Running VGGT reconstruction...")
        reconstruction = run_vggt_once(frames_bgr)
        
        point_cloud = reconstruction["point_cloud"]
        colors = reconstruction["colors"]
        extr = reconstruction["extrinsic"]
        intr = reconstruction["intrinsic"]
        original_images = reconstruction["images_rgb_0_1"]
        
        print(f"      Reconstructed {len(point_cloud)} 3D points")
        
        # Step 3: Reproject to each frame and compute errors
        print(f"[3/4] Computing reprojection errors...")
        frame_losses = []
        
        for i in range(S):
            # Convert original image to uint8 [0, 255]
            original = (original_images[i] * 255.0).clip(0, 255).astype(np.uint8)
            
            # Reproject 3D points to this frame
            reprojected = self._reproject_to_frame(
                point_cloud=point_cloud,
                colors=colors,
                extr=extr[i],
                intr=intr[i],
                target_shape=(H, W)
            )
            
            # Compute reconstruction loss
            loss = self._compute_reconstruction_loss(original, reprojected)
            frame_losses.append(loss)
        
        # Step 4: Aggregate metrics across frames
        print(f"[4/4] Aggregating results...")
        avg_mse = float(np.mean([l['mse'] for l in frame_losses]))
        avg_lpips = float(np.mean([l['lpips'] for l in frame_losses]))
        avg_total = float(np.mean([l['total'] for l in frame_losses]))
        
        # 3D Consistency Score (lower is better)
        # Following VideoGPA Eq. (12)
        consistency_score = avg_total
        
        all_metrics = {
            'reconstruction_error': {
                'mse': avg_mse,
                'lpips': avg_lpips,
                'total': avg_total,
            },
            '3d_consistency_score': consistency_score,
            'num_frames': S,
            'resolution': [W, H],
            'num_3d_points': int(len(point_cloud)),
            'use_lpips': self.use_lpips,
            'per_frame_losses': frame_losses,
        }
        
        print(f"\n{'='*80}")
        print(f"Results:")
        print(f"  3D Consistency Score: {consistency_score:.4f}")
        print(f"  MSE:   {avg_mse:.6f}")
        if self.use_lpips:
            print(f"  LPIPS: {avg_lpips:.4f}")
        print(f"  Frames: {S}")
        print(f"  3D Points: {len(point_cloud):,}")
        print(f"{'='*80}\n")
        
        return consistency_score, all_metrics
    
    @property
    def name(self) -> str:
        """Return the name of this evaluator."""
        return "reprojection_vanilla"