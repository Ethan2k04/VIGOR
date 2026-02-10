import os
import math
import tempfile
import logging
from typing import Dict, Any, Tuple, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseEvaluator

from ..third_party.vggt.models.vggt import VGGT
from ..third_party.vggt.utils.load_fn import load_and_preprocess_images
from ..third_party.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from ..third_party.vggt.utils.geometry import unproject_depth_map_to_point_map


_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_VGGT_MODEL: Optional[VGGT] = None


def _get_amp_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def get_vggt_model() -> VGGT:
    """
    Load VGGT once and reuse (singleton).
    """
    global _VGGT_MODEL
    if _VGGT_MODEL is not None:
        return _VGGT_MODEL

    model = VGGT()
    url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(_DEVICE)
    _VGGT_MODEL = model
    return _VGGT_MODEL


print("=" * 60)
print("[INFO] Loading VGGT model...")
model = get_vggt_model()
print(f"[INFO] Device: {_DEVICE}")
print("[INFO] ✓ VGGT model loaded successfully")
print("=" * 60)


class SkySegONNX:
    """
    ONNXRuntime sky segmentation wrapper.

    Output:
      - is_sky: bool (H,W), True means sky pixel
    """

    def __init__(self, onnx_path: str):
        self.onnx_path = onnx_path
        self.session = None

        try:
            import onnxruntime  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "onnxruntime is required for skyseg.onnx sky filtering. "
                "Please `pip install onnxruntime` (or onnxruntime-gpu)."
            ) from e

        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"skyseg.onnx not found at: {self.onnx_path}")

        print(f"[INFO] onnxruntime is available (version: {onnxruntime.__version__})")
        print(f"[INFO] ✓ Found skyseg.onnx at: {self.onnx_path}")
        file_size = os.path.getsize(self.onnx_path) / (1024 * 1024)
        print(f"[INFO] skyseg.onnx file size: {file_size:.2f} MB")

    def _ensure_session(self):
        if self.session is None:
            import onnxruntime
            self.session = onnxruntime.InferenceSession(self.onnx_path)

    @staticmethod
    def _run_skyseg(session, input_size_hw, image_bgr_u8: np.ndarray) -> np.ndarray:
        """
        Returns uint8 map in [0,255], shape (h,w).
        """
        h_in, w_in = int(input_size_hw[0]), int(input_size_hw[1])
        resize_image = cv2.resize(image_bgr_u8, dsize=(w_in, h_in))
        x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB).astype(np.float32)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x / 255.0 - mean) / std
        x = x.transpose(2, 0, 1)[None].astype("float32")  # (1,3,H,W)

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        out = session.run([output_name], {input_name: x})
        out = np.array(out).squeeze()

        minv = float(np.min(out))
        maxv = float(np.max(out))
        denom = (maxv - minv) if (maxv > minv) else 1.0
        out01 = (out - minv) / denom
        out_u8 = (out01 * 255.0).clip(0, 255).astype(np.uint8)
        return out_u8

    def segment_sky(self, frame_bgr_u8: np.ndarray, sky_threshold: int = 32) -> np.ndarray:
        """
        Returns:
          is_sky: bool (H,W), True means sky

        Convention used here:
          is_sky = (score_map < sky_threshold)
        """
        self._ensure_session()
        sky_threshold = int(np.clip(sky_threshold, 0, 255))

        H, W = frame_bgr_u8.shape[:2]
        score = self._run_skyseg(self.session, (320, 320), frame_bgr_u8)
        score = cv2.resize(score, (W, H), interpolation=cv2.INTER_LINEAR)

        is_sky = score >= sky_threshold
        return is_sky


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
      images_tensor:  torch tensor (S,3,H,W) on GPU
    """
    tmpdir = tempfile.mkdtemp(prefix="vggt_eval_")
    paths = []
    for k, bgr in enumerate(frames_bgr):
        p = os.path.join(tmpdir, f"{k:06d}.png")
        cv2.imwrite(p, bgr)
        paths.append(p)

    images = load_and_preprocess_images(paths).to(_DEVICE)  # (S,3,H,W), 0..1
    amp_dtype = _get_amp_dtype()

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=amp_dtype):
            preds = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])

    depth_raw = preds["depth"].detach().float().cpu().numpy()
    depth = _normalize_depth_to_SHW1(depth_raw).astype(np.float32)  # (S,H,W,1)

    extr = extrinsic.detach().float().cpu().numpy()
    intr = intrinsic.detach().float().cpu().numpy()
    if extr.ndim == 4 and extr.shape[0] == 1:
        extr = extr[0]
    if intr.ndim == 4 and intr.shape[0] == 1:
        intr = intr[0]
    extr = extr.astype(np.float32)
    intr = intr.astype(np.float32)

    world_points = unproject_depth_map_to_point_map(depth, extr, intr).astype(np.float32)

    imgs = preds["images"].detach().float().cpu().numpy()
    if imgs.ndim == 5 and imgs.shape[0] == 1:
        imgs = imgs[0]
    imgs = np.transpose(imgs, (0, 2, 3, 1)).astype(np.float32)  # (S,H,W,3)

    depth_2d = depth[..., 0].astype(np.float32)

    return {
        "images_rgb_0_1": imgs,
        "world_points": world_points,
        "extrinsic": extr,
        "intrinsic": intr,
        "depth": depth_2d,
        "images_tensor": images,
    }


def patch_centers(H: int, W: int, p: int) -> Tuple[np.ndarray, int, int]:
    """
    Return patch grid centers in pixel coords.
    centers: (Hp*Wp,2)
    """
    p = int(p)
    Hp = int(np.ceil(H / p))
    Wp = int(np.ceil(W / p))
    us = (np.arange(Wp, dtype=np.float32) + 0.5) * p
    vs = (np.arange(Hp, dtype=np.float32) + 0.5) * p
    uu, vv = np.meshgrid(us, vs)
    uu = np.clip(uu, 0.0, float(W - 1))
    vv = np.clip(vv, 0.0, float(H - 1))
    centers = np.stack([uu, vv], axis=-1).reshape(-1, 2).astype(np.float32)
    return centers, Hp, Wp


def project_world_to_pixel(Xw: np.ndarray, extr_3x4: np.ndarray, K_3x3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project world points (N,3) into target camera pixel coords.
    Returns:
      uv_hat: (N,2)
      z: (N,)
    """
    R = extr_3x4[:, :3]
    t = extr_3x4[:, 3]
    Xc = (Xw @ R.T) + t[None, :]
    z = Xc[:, 2].astype(np.float32)

    fx = float(K_3x3[0, 0])
    fy = float(K_3x3[1, 1])
    cx = float(K_3x3[0, 2])
    cy = float(K_3x3[1, 2])

    invz = 1.0 / np.maximum(z, 1e-8)
    u = fx * (Xc[:, 0] * invz) + cx
    v = fy * (Xc[:, 1] * invz) + cy
    uv_hat = np.stack([u, v], axis=-1).astype(np.float32)
    return uv_hat, z


def run_track_for_reference_i(images_tensor_SCHW: torch.Tensor, query_uv: np.ndarray, ref_i: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Track query points defined on frame ref_i by reordering the sequence so ref_i becomes the first frame.
    Returns tracks_uv (S,N,2) and conf (S,N) in original frame order.
    """
    S, _, _, _ = images_tensor_SCHW.shape
    ref_i = int(ref_i)

    perm = [ref_i] + [k for k in range(S) if k != ref_i]
    inv = np.zeros((S,), dtype=np.int32)
    for new_idx, orig_idx in enumerate(perm):
        inv[orig_idx] = new_idx

    images_perm = images_tensor_SCHW[perm]
    qp = torch.from_numpy(query_uv.astype(np.float32)).to(_DEVICE)

    model = get_vggt_model()
    amp_dtype = _get_amp_dtype()

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=amp_dtype):
            preds = model(images_perm, query_points=qp)

    track = preds["track"].detach().float().cpu().numpy()[0]  # (S,N,2) perm order
    conf = preds["conf"].detach().float().cpu().numpy()[0]    # (S,N) perm order

    track_orig = np.zeros_like(track, dtype=np.float32)
    conf_orig = np.zeros_like(conf, dtype=np.float32)
    for orig_idx in range(S):
        new_idx = inv[orig_idx]
        track_orig[orig_idx] = track[new_idx]
        conf_orig[orig_idx] = conf[new_idx]

    return track_orig, conf_orig


def compute_reference_attention_maps(
    images_tensor_SCHW: torch.Tensor,
    attn_layer: int = 0,
    reduce_mode: str = "max",  # "max" or "topk"
    topk: int = 8,
) -> np.ndarray:
    """
    Compute per-reference attention heatmap (S,H,W) in [0,1] float32.

    This follows your hook-based design:
      - capture q_norm/k_norm of aggregator.global_blocks[attn_layer].attn
      - per ref i: aggregate cross-attn i->j for all j!=i
      - per query token: max or topk-mean over keys
      - average heads, sum over j
      - upsample to H,W and normalize to [0,1] per frame
    """
    model = get_vggt_model()
    model.eval()

    if images_tensor_SCHW.ndim != 4:
        raise ValueError(f"Expected images_tensor_SCHW as (S,3,H,W), got {tuple(images_tensor_SCHW.shape)}")

    S, _, H, W = images_tensor_SCHW.shape
    images_bs = images_tensor_SCHW.unsqueeze(0)  # (1,S,3,H,W)

    q_out: Dict[int, torch.Tensor] = {}
    k_out: Dict[int, torch.Tensor] = {}
    handles = []

    def _make_hook(store_dict, idx: int):
        def _hook(_module, _inp, out):
            store_dict[idx] = out.detach()
        return _hook

    blk = model.aggregator.global_blocks[int(attn_layer)].attn
    handles.append(blk.q_norm.register_forward_hook(_make_hook(q_out, int(attn_layer))))
    handles.append(blk.k_norm.register_forward_hook(_make_hook(k_out, int(attn_layer))))

    amp_dtype = _get_amp_dtype()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=amp_dtype):
            _, patch_start_idx = model.aggregator(images_bs)

    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    if int(attn_layer) not in q_out or int(attn_layer) not in k_out:
        raise RuntimeError("Failed to capture Q/K from hooks. Check attn_layer index and VGGT aggregator structure.")

    Q = q_out[int(attn_layer)].to(dtype=torch.float32)  # (B, heads, T, d)
    K = k_out[int(attn_layer)].to(dtype=torch.float32)  # (B, heads, T, d)

    patch_size = int(model.aggregator.patch_size)
    h_v = int(H // patch_size)
    w_v = int(W // patch_size)
    num_patch_tokens = int(h_v * w_v)
    tokens_per_image = int(patch_start_idx + num_patch_tokens)

    T_total = int(K.shape[-2])
    num_images_in_seq = int(T_total // tokens_per_image)
    S_eff = min(S, num_images_in_seq)

    def _slice_patch_tokens(x: torch.Tensor, img_idx: int) -> torch.Tensor:
        start = img_idx * tokens_per_image + patch_start_idx
        end = start + num_patch_tokens
        return x[:, :, start:end, :]

    scale = 1.0 / math.sqrt(float(Q.shape[-1]))
    attn_token_grid = np.zeros((S, h_v, w_v), dtype=np.float32)

    for i in range(S_eff):
        q_i = _slice_patch_tokens(Q, i)  # (1, heads, Nq, d)
        if q_i.shape[-2] != num_patch_tokens:
            continue

        agg_map_i = torch.zeros((num_patch_tokens,), dtype=torch.float32, device=Q.device)

        for j in range(S_eff):
            if j == i:
                continue
            k_j = _slice_patch_tokens(K, j)  # (1, heads, Nk, d)
            if k_j.shape[-2] != num_patch_tokens:
                continue

            logits = torch.einsum("bhqd,bhkd->bhqk", q_i, k_j) * scale
            probs = torch.softmax(logits, dim=-1)

            if reduce_mode == "topk":
                kk = max(1, int(topk))
                kk = min(kk, probs.shape[-1])
                score_q = torch.topk(probs, k=kk, dim=-1).values.mean(dim=-1)
            else:
                score_q = probs.max(dim=-1).values

            score_q = score_q.mean(dim=1).squeeze(0)  # (Nq,)
            agg_map_i += score_q

        attn_token_grid[i] = agg_map_i.view(h_v, w_v).detach().cpu().numpy().astype(np.float32)

    # Upsample to H,W and normalize per-frame to [0,1]
    attn_hw_01 = np.zeros((S, H, W), dtype=np.float32)
    for i in range(S):
        m = attn_token_grid[i]
        if not np.isfinite(m).all():
            m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)

        t = torch.from_numpy(m)[None, None]  # (1,1,h_v,w_v)
        up = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

        mn = float(up.min())
        mx = float(up.max())
        if mx > mn:
            attn_hw_01[i] = ((up - mn) / (mx - mn)).astype(np.float32)
        else:
            attn_hw_01[i] = np.zeros_like(up, dtype=np.float32)

    return attn_hw_01


def attention_to_patch_values(
    attn_hw_01_i: np.ndarray,
    patch_size: int,
    sky_mask_i: np.ndarray,
    depth_i: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int, int, np.ndarray]:
    """
    Convert attention heatmap (H,W) -> patch values (Hp*Wp).
    Patch value = mean attention within patch window.
    Valid if:
      - not sky at patch center
      - depth finite & > 0 at patch center
    """
    H, W = attn_hw_01_i.shape[:2]
    p = int(patch_size)
    centers_uv, Hp, Wp = patch_centers(H, W, p)
    N = Hp * Wp

    patch_vals = np.zeros((N,), dtype=np.float64)
    valid = np.zeros((N,), dtype=bool)

    for py in range(Hp):
        y0 = py * p
        y1 = min(H, (py + 1) * p)
        if y0 >= H:
            continue
        for px in range(Wp):
            x0 = px * p
            x1 = min(W, (px + 1) * p)
            if x0 >= W:
                continue

            cy = int(min(H - 1, py * p + p * 0.5))
            cx = int(min(W - 1, px * p + p * 0.5))

            if sky_mask_i[cy, cx]:
                continue

            d = float(depth_i[cy, cx])
            if (not np.isfinite(d)) or (d <= 1e-6):
                continue

            patch_attn = attn_hw_01_i[y0:y1, x0:x1]
            if patch_attn.size == 0:
                continue

            a = float(np.mean(patch_attn))
            idx = py * Wp + px
            patch_vals[idx] = max(0.0, a)
            valid[idx] = True

    return patch_vals, valid, Hp, Wp, centers_uv


def select_patches_top_percent(
    patch_vals: np.ndarray,
    valid: np.ndarray,
    top_percent: float,
    max_query_points: int,
    rng: np.random.RandomState,
    filter_before_select: bool = True,  # 新增参数
) -> Tuple[np.ndarray, float, int, int]:
    """
    Select top-K patches by attention value.
    
    Args:
        patch_vals: (N,) attention values per patch
        valid: (N,) bool mask, True if patch is valid
        top_percent: percentage of patches to select
        max_query_points: maximum number of points to return
        rng: random number generator
        filter_before_select: if True, select from valid patches only (old behavior)
                             if False, select from all patches, then filter invalid (new behavior)
    
    Returns:
        sel_idx: indices of selected patches
        threshold: attention threshold used
        candidate_count: number of candidates before selection
        kept_count: number of patches after selection
    """
    N = int(patch_vals.shape[0])
    
    if filter_before_select:
        # OLD BEHAVIOR: Select top_percent from valid patches only
        cand = np.where(valid)[0]
        if len(cand) == 0:
            return np.array([], dtype=np.int32), 0.0, 0, 0
        
        cand_vals = patch_vals[cand]
        K = max(1, int(np.ceil(len(cand) * (top_percent / 100.0))))
        K = min(K, len(cand))
        K = min(K, max_query_points)
        
        if K >= len(cand):
            thr = float(np.min(cand_vals)) if len(cand_vals) > 0 else 0.0
            chosen = cand
        else:
            part_idx = np.argpartition(cand_vals, -K)[-K:]
            thr = float(np.min(cand_vals[part_idx]))
            mask = cand_vals >= thr
            chosen = cand[mask]
            if len(chosen) > K:
                chosen = rng.choice(chosen, size=K, replace=False)
        
        return chosen, thr, len(cand), len(chosen)
    
    else:
        # NEW BEHAVIOR: Select top_percent from ALL patches, then filter invalid
        # This ensures we actually get top_percent% of total patches (if they're valid)
        K_target = max(1, int(np.ceil(N * (top_percent / 100.0))))
        K_target = min(K_target, max_query_points)
        
        # Select top K patches by attention (regardless of validity)
        if K_target >= N:
            thr = float(np.min(patch_vals))
            top_k_idx = np.arange(N, dtype=np.int32)
        else:
            part_idx = np.argpartition(patch_vals, -K_target)[-K_target:]
            thr = float(np.min(patch_vals[part_idx]))
            mask = patch_vals >= thr
            top_k_idx = np.where(mask)[0]
            if len(top_k_idx) > K_target:
                top_k_idx = rng.choice(top_k_idx, size=K_target, replace=False)
        
        # Now filter by validity
        chosen = top_k_idx[valid[top_k_idx]]
        
        return chosen, thr, int(K_target), len(chosen)


def patchmap_to_fullres(patch_map: np.ndarray, H: int, W: int, patch: int) -> np.ndarray:
    """
    Upsample patch-level map (Hp, Wp) to full resolution (H, W).
    """
    p = int(patch)
    full = np.repeat(np.repeat(patch_map, p, axis=0), p, axis=1)
    return full[:H, :W]


def save_sampling_mask(sampling_mask: np.ndarray, mask_path: str) -> None:
    """
    Save binary sampling mask to .npz file.
    
    Args:
        sampling_mask: (S, Hp, Wp) bool array, True = selected patch
        mask_path: Path to save .npz file
    """
    os.makedirs(os.path.dirname(os.path.abspath(mask_path)), exist_ok=True)
    np.savez_compressed(mask_path, sampling_mask=sampling_mask)
    logging.info(f"Saved sampling mask to: {mask_path}")


class ReprojectionEvaluator(BaseEvaluator):
    """
    VGGT-based reprojection error metric.

    Main score: mean reprojection error in pixels (lower is better).
    """

    def __init__(
        self,
        sampling_rate: int = 1,             # not used (we override evaluate_video), kept for config compatibility
        max_frames: int = 10,               # S
        short_side: int = 448,
        patch_size: int = 4,
        max_query_points: int = 2048,       # K per reference
        top_percent: float = 20.0,          # attention top percentile
        track_conf: float = 0.05,
        attn_layer: int = 0,
        attn_reduce: str = "max",           # "max" or "topk"
        attn_topk: int = 8,
        enable_sky_onnx: bool = True,
        sky_onnx_path: Optional[str] = "skyseg.onnx",
        sky_threshold: int = 32,
        seed: int = 0,
        save_sampling_mask: bool = False,
        sampling_mask_dir: Optional[str] = None,
        filter_before_select: bool = False,
    ):  
        print("\n" + "=" * 60)
        print("[INFO] Initializing ReprojectionEvaluator...")
        super().__init__(sampling_rate=int(sampling_rate))

        self.max_frames = int(max_frames)
        self.short_side = int(short_side)
        self.patch_size = int(patch_size)
        self.max_query_points = int(max_query_points)
        self.top_percent = float(top_percent)
        self.track_conf = float(track_conf)
        self.filter_before_select = bool(filter_before_select)

        self.attn_layer = int(attn_layer)
        self.attn_reduce = str(attn_reduce)
        self.attn_topk = int(attn_topk)

        self.enable_sky_onnx = bool(enable_sky_onnx)
        self.sky_threshold = int(sky_threshold)
        self.seed = int(seed)

        self.save_sampling_mask = save_sampling_mask
        self.sampling_mask_dir = sampling_mask_dir

        self._rng = np.random.RandomState(self.seed)

        self._skyseg = None
        if self.enable_sky_onnx:
            if sky_onnx_path is None:
                # default: look for skyseg.onnx next to cwd
                sky_onnx_path = os.path.join(os.getcwd(), "skyseg.onnx")
            self._skyseg = SkySegONNX(sky_onnx_path)

        print(f"[INFO] Sky filtering enabled: {self.enable_sky_onnx}")
        print(f"[INFO] Save sampling mask: {self.save_sampling_mask}")
        if self.save_sampling_mask:
            print(f"[INFO] Sampling mask directory: {self.sampling_mask_dir}")
        print("=" * 60 + "\n")

    @property
    def name(self) -> str:
        return "reprojection_error"

    def compute_metrics(self, frame):
        """
        Not used (multi-frame metric). Present only to satisfy BaseEvaluator abstract API.
        """
        raise NotImplementedError("ReprojectionEvaluator overrides evaluate_video().")

    def aggregate_metrics(self, frame_metrics) -> Tuple[float, Dict[str, Any]]:
        """
        Not used. Present only to satisfy BaseEvaluator abstract API.
        """
        raise NotImplementedError("ReprojectionEvaluator overrides evaluate_video().")

    def evaluate_video(self, video_path: str) -> Tuple[float, Dict[str, Any]]:
        frames_bgr = extract_frames_uniform(
            video_path=video_path,
            max_frames=self.max_frames,
            target_short_side=self.short_side,
        )

        pack = run_vggt_once(frames_bgr)
        imgs = pack["images_rgb_0_1"]           # (S,H,W,3)
        world_points = pack["world_points"]     # (S,H,W,3)
        extr = pack["extrinsic"]                # (S,3,4)
        intr = pack["intrinsic"]                # (S,3,3)
        depth = pack["depth"]                   # (S,H,W)
        images_tensor = pack["images_tensor"]   # (S,3,H,W)

        S, H, W, _ = imgs.shape
        centers_uv, Hp, Wp = patch_centers(H, W, self.patch_size)

        # Sky masks (IMPORTANT: match VGGT resolution H,W)
        sky_masks: List[np.ndarray] = []
        if self.enable_sky_onnx:
            for i in range(S):
                frame_hw = cv2.resize(frames_bgr[i], (W, H), interpolation=cv2.INTER_AREA)
                is_sky = self._skyseg.segment_sky(frame_hw, sky_threshold=self.sky_threshold)
                if is_sky.shape[0] != H or is_sky.shape[1] != W:
                    # Hard safety: force to (H,W)
                    is_sky = cv2.resize(is_sky.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                sky_masks.append(is_sky.astype(bool))
        else:
            for _ in range(S):
                sky_masks.append(np.zeros((H, W), dtype=bool))

        # Attention maps
        attn_hw_01 = compute_reference_attention_maps(
            images_tensor_SCHW=images_tensor,
            attn_layer=self.attn_layer,
            reduce_mode=self.attn_reduce,
            topk=self.attn_topk,
        )  # (S,H,W)

        eps = 1e-8
        total_sum_d = 0.0
        total_used = 0

        total_selected = 0
        total_sky_skipped = 0
        total_proj_oob = 0
        total_proj_behind = 0

        thr_list: List[float] = []
        cand_cnt_list: List[int] = []
        keep_cnt_list: List[int] = []

        # NEW: Store binary sampling mask per frame (S, Hp, Wp)
        # This mask indicates which patches were selected based on VGGT attention
        sampling_mask = np.zeros((S, Hp, Wp), dtype=bool)

        for i in range(S):
            patch_vals, valid_patch, Hp2, Wp2, _ = attention_to_patch_values(
                attn_hw_01_i=attn_hw_01[i],
                patch_size=self.patch_size,
                sky_mask_i=sky_masks[i],
                depth_i=depth[i],
            )
            if Hp2 != Hp or Wp2 != Wp:
                raise RuntimeError("Patch grid mismatch unexpectedly.")

            sel_idx, thr, cand_cnt, kept_k = select_patches_top_percent(
                patch_vals=patch_vals,
                valid=valid_patch,
                top_percent=self.top_percent,
                max_query_points=self.max_query_points,
                rng=self._rng,
                filter_before_select=self.filter_before_select,
            )
            thr_list.append(float(thr))
            cand_cnt_list.append(int(cand_cnt))
            keep_cnt_list.append(int(kept_k))

            M = int(sel_idx.shape[0])
            total_selected += M
            
            # NEW: Mark selected patches in the binary mask
            if M > 0:
                # Convert flat indices to 2D patch coordinates
                patch_y = sel_idx // Wp
                patch_x = sel_idx % Wp
                sampling_mask[i, patch_y, patch_x] = True
            
            if M == 0:
                continue

            query_uv = centers_uv[sel_idx]  # (M,2)
            track_uv, conf = run_track_for_reference_i(images_tensor, query_uv, ref_i=i)  # (S,M,2), (S,M)

            ui = np.rint(query_uv[:, 0]).astype(np.int32)
            vi = np.rint(query_uv[:, 1]).astype(np.int32)
            ui = np.clip(ui, 0, W - 1)
            vi = np.clip(vi, 0, H - 1)

            Xw = world_points[i, vi, ui, :].astype(np.float32)   # (M,3)
            conf_i = conf[i].astype(np.float32)                  # (M,)

            sky_center = sky_masks[i][vi, ui]
            total_sky_skipped += int(np.sum(sky_center))
            valid_ref_center = ~sky_center

            ref_finite = np.isfinite(Xw).all(axis=1)

            for j in range(S):
                if j == i:
                    continue

                uv_track = track_uv[j].astype(np.float32)
                conf_j = conf[j].astype(np.float32)

                ok_conf = (conf_i >= self.track_conf) & (conf_j >= self.track_conf)

                uv_hat, z = project_world_to_pixel(Xw, extr[j], intr[j])
                infront = z > 1e-6

                uhat = uv_hat[:, 0]
                vhat = uv_hat[:, 1]
                inb_hat = (uhat >= 0.0) & (uhat <= (W - 1)) & (vhat >= 0.0) & (vhat <= (H - 1))

                utr = uv_track[:, 0]
                vtr = uv_track[:, 1]
                inb_tr = (utr >= 0.0) & (utr <= (W - 1)) & (vtr >= 0.0) & (vtr <= (H - 1))

                ok_base = valid_ref_center & ref_finite & ok_conf
                total_proj_behind += int(np.sum(ok_base & (~infront)))
                total_proj_oob += int(np.sum(ok_base & infront & (~inb_hat)))

                ok = ok_base & infront & inb_hat & inb_tr
                if not np.any(ok):
                    continue

                du = uv_hat[:, 0] - uv_track[:, 0]
                dv = uv_hat[:, 1] - uv_track[:, 1]
                d = np.sqrt(du * du + dv * dv).astype(np.float32)  # px

                total_sum_d += float(np.sum(d[ok]))
                total_used += int(np.sum(ok))

        mean_err = (total_sum_d / (float(total_used) + eps)) if total_used > 0 else -1.0

        result: Dict[str, Any] = {
            "mean_reprojection_error_px": float(mean_err),
            "used_pairs": int(total_used),
            "frames_used": int(S),
            "resolution": [int(W), int(H)],
            "patch_size": int(self.patch_size),
            "grid_hw": [int(Hp), int(Wp)],
            "max_query_points": int(self.max_query_points),
            "top_percent": float(self.top_percent),
            "track_conf": float(self.track_conf),
            "attn_layer": int(self.attn_layer),
            "attn_reduce": str(self.attn_reduce),
            "attn_topk": int(self.attn_topk),
            "enable_sky_onnx": bool(self.enable_sky_onnx),
            "sky_threshold": int(self.sky_threshold) if self.enable_sky_onnx else None,
            "selected_points_total": int(total_selected),
            "sky_skipped_selected_centers": int(total_sky_skipped),
            "proj_behind_camera": int(total_proj_behind),
            "proj_out_of_bounds": int(total_proj_oob),
            "attn_thr_mean": float(np.mean(thr_list)) if len(thr_list) else 0.0,
            "attn_cand_mean": float(np.mean(cand_cnt_list)) if len(cand_cnt_list) else 0.0,
            "attn_kept_mean": float(np.mean(keep_cnt_list)) if len(keep_cnt_list) else 0.0,
        }

        # NEW: Save sampling mask if enabled
        if self.save_sampling_mask and self.sampling_mask_dir is not None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            mask_path = os.path.join(self.sampling_mask_dir, f"{video_name}_sampling_mask.npz")
            save_sampling_mask(sampling_mask, mask_path)
            result["sampling_mask_path"] = mask_path

        # main score: lower is better
        return float(mean_err), result

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ReprojectionEvaluator":
        return cls(
            sampling_rate=config.get("sampling_rate", 1),
            max_frames=config.get("max_frames", 10),
            short_side=config.get("short_side", 448),
            patch_size=config.get("patch_size", 4),
            max_query_points=config.get("max_query_points", 2048),
            top_percent=config.get("top_percent", 20.0),
            track_conf=config.get("track_conf", 0.05),
            attn_layer=config.get("attn_layer", 0),
            attn_reduce=config.get("attn_reduce", "max"),
            attn_topk=config.get("attn_topk", 8),
            enable_sky_onnx=config.get("enable_sky_onnx", True),
            filter_before_select=config.get("filter_before_select", False),
            sky_onnx_path=config.get("sky_onnx_path", None),
            sky_threshold=config.get("sky_threshold", 32),
            seed=config.get("seed", 0),
            save_sampling_mask=config.get("save_sampling_mask", False),
            sampling_mask_dir=config.get("sampling_mask_dir", None),
        )