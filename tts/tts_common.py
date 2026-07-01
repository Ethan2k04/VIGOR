"""
Shared utilities for TTS algorithms (SoS / SoP / BS):
  - third_party path setup (Causal-Forcing + vggt)
  - VGGT singleton, frame conversion
  - Reprojection + epipolar scoring
  - KV / cross-attn cache clone & restore
"""

import os
import sys
import shutil
import logging
import tempfile
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Inject third_party paths so `pipeline`, `utils`, `demo_utils`, `vggt` resolve
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))          # VIGOR/tts
REPO_ROOT = os.path.dirname(ROOT_DIR)                           # VIGOR repo root
CAUSAL_FORCING_DIR = os.path.join(REPO_ROOT, "third_party", "Causal-Forcing")
VGGT_DIR = os.path.join(REPO_ROOT, "third_party", "vggt")
for _p in (CAUSAL_FORCING_DIR, VGGT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Default config path lives inside the Causal-Forcing submodule
DEFAULT_CONFIG_PATH = os.path.join(CAUSAL_FORCING_DIR, "configs", "default_config.yaml")

from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402
from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: E402

try:
    from kornia.geometry.epipolar import find_fundamental, sampson_epipolar_distance
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False

logger = logging.getLogger(__name__)

PENALTY = 1e6

# ============================================================
# VGGT singleton
# ============================================================
_VGGT_MODEL: Optional[VGGT] = None
VGGT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def amp_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def get_vggt_model() -> VGGT:
    global _VGGT_MODEL
    if _VGGT_MODEL is not None:
        return _VGGT_MODEL
    print("[TTS] Loading VGGT ...")
    m = VGGT()
    url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    m.load_state_dict(torch.hub.load_state_dict_from_url(url, map_location="cpu"))
    m.eval().to(VGGT_DEVICE)
    _VGGT_MODEL = m
    print("[TTS] VGGT ready.")
    return _VGGT_MODEL


# ============================================================
# Frame tensor -> BGR uint8 numpy list (optionally resized)
# ============================================================

def tensor_to_bgr(frames: torch.Tensor, short_side: int = 448) -> List[np.ndarray]:
    arr = frames.float().cpu().numpy()
    out: List[np.ndarray] = []
    for t in range(arr.shape[0]):
        rgb = np.clip(np.transpose(arr[t], (1, 2, 0)), 0, 1)
        bgr = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if short_side > 0:
            h, w = bgr.shape[:2]
            s = min(h, w)
            if s != short_side:
                sc = short_side / s
                bgr = cv2.resize(
                    bgr, (int(round(w * sc)), int(round(h * sc))),
                    interpolation=cv2.INTER_AREA,
                )
        out.append(bgr)
    return out


# ============================================================
# VGGT prediction (depth, pose, world points, tracked images)
# ============================================================

def _run_vggt(frames_bgr: List[np.ndarray]) -> Dict[str, Any]:
    tmp = tempfile.mkdtemp(prefix="tts_vggt_")
    paths = []
    for k, bgr in enumerate(frames_bgr):
        p = os.path.join(tmp, f"{k:06d}.png")
        cv2.imwrite(p, bgr)
        paths.append(p)

    images = load_and_preprocess_images(paths).to(VGGT_DEVICE)
    model = get_vggt_model()
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype()):
        preds = model(images)

    extr_raw, intr_raw = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])

    def _squeeze(t):
        a = t.detach().float().cpu().numpy()
        if a.ndim == 4 and a.shape[0] == 1:
            a = a[0]
        return a.astype(np.float32)

    d = preds["depth"].detach().float().cpu().numpy()
    if d.ndim == 5 and d.shape[0] == 1:
        d = d[0]
    if d.ndim == 4:
        d = d[..., 0]
    depth = d.astype(np.float32)

    extr = _squeeze(extr_raw)
    intr = _squeeze(intr_raw)
    wpts = unproject_depth_map_to_point_map(depth[..., None], extr, intr).astype(np.float32)

    shutil.rmtree(tmp, ignore_errors=True)
    return dict(world_points=wpts, extrinsic=extr, intrinsic=intr,
                depth=depth, images_tensor=images)


def _project(Xw: np.ndarray, extr: np.ndarray, K: np.ndarray):
    R, t = extr[:, :3], extr[:, 3]
    Xc = Xw @ R.T + t[None]
    z = Xc[:, 2].astype(np.float32)
    iz = 1.0 / np.maximum(z, 1e-8)
    u = K[0, 0] * Xc[:, 0] * iz + K[0, 2]
    v = K[1, 1] * Xc[:, 1] * iz + K[1, 2]
    return np.stack([u, v], -1).astype(np.float32), z


# ============================================================
# Reprojection score (mean per-pixel reprojection error)
# ============================================================

def compute_reprojection_score(
    frames_bgr: List[np.ndarray],
    max_pts: int = 512,
    conf_thr: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng(0)
    if len(frames_bgr) < 2:
        return PENALTY
    try:
        pk = _run_vggt(frames_bgr)
    except Exception as e:
        logger.warning(f"[TTS] VGGT failed: {e}")
        return PENALTY

    wpts, extr, intr, depth, imgs = (
        pk["world_points"], pk["extrinsic"], pk["intrinsic"],
        pk["depth"], pk["images_tensor"],
    )
    S, H, W, _ = wpts.shape
    model = get_vggt_model()
    dt = amp_dtype()
    total_d, total_n = 0.0, 0

    for i in range(S):
        ys = rng.integers(1, H - 1, size=max_pts)
        xs = rng.integers(1, W - 1, size=max_pts)
        valid = np.isfinite(depth[i, ys, xs]) & (depth[i, ys, xs] > 0)
        ys, xs = ys[valid], xs[valid]
        if len(ys) == 0:
            continue

        qp = torch.from_numpy(np.stack([xs, ys], 1).astype(np.float32)).to(VGGT_DEVICE)
        perm = [i] + [k for k in range(S) if k != i]
        inv = {orig: new for new, orig in enumerate(perm)}
        try:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dt):
                out = model(imgs[perm], query_points=qp)
        except Exception as e:
            logger.warning(f"[TTS] VGGT track failed frame {i}: {e}")
            continue

        track = out["track"].detach().float().cpu().numpy()[0]
        conf = out["conf"].detach().float().cpu().numpy()[0]
        tr_o = np.zeros_like(track); cf_o = np.zeros_like(conf)
        for orig in range(S):
            tr_o[orig] = track[inv[orig]]
            cf_o[orig] = conf[inv[orig]]

        Xw = wpts[i, ys, xs].astype(np.float32)
        ref_ok = np.isfinite(Xw).all(1)
        ci = cf_o[i]
        for j in range(S):
            if j == i:
                continue
            uv_hat, z = _project(Xw, extr[j], intr[j])
            uv_tr = tr_o[j].astype(np.float32)
            cj = cf_o[j]
            ok = (
                ref_ok
                & (ci >= conf_thr) & (cj >= conf_thr)
                & (z > 1e-6)
                & (uv_hat[:, 0] >= 0) & (uv_hat[:, 0] <= W - 1)
                & (uv_hat[:, 1] >= 0) & (uv_hat[:, 1] <= H - 1)
                & (uv_tr[:, 0] >= 0) & (uv_tr[:, 0] <= W - 1)
                & (uv_tr[:, 1] >= 0) & (uv_tr[:, 1] <= H - 1)
            )
            if not ok.any():
                continue
            du = uv_hat[ok, 0] - uv_tr[ok, 0]
            dv = uv_hat[ok, 1] - uv_tr[ok, 1]
            total_d += float(np.sum(np.sqrt(du * du + dv * dv)))
            total_n += int(ok.sum())

    return float(total_d / (total_n + 1e-8)) if total_n > 0 else PENALTY


# ============================================================
# Epipolar score (mean Sampson distance over consecutive pairs)
# ============================================================

class _SIFTMatcher:
    def __init__(self, ratio: float = 0.75, min_m: int = 2):
        self.ratio = ratio
        self.min_m = min_m
        self.sift = cv2.SIFT_create()

    def match(self, f1: np.ndarray, f2: np.ndarray):
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY) if f1.ndim == 3 else f1
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY) if f2.ndim == 3 else f2
        kp1, d1 = self.sift.detectAndCompute(g1, None)
        kp2, d2 = self.sift.detectAndCompute(g2, None)
        if len(kp1) < 8 or len(kp2) < 8 or d1 is None or d2 is None:
            return None, None
        good = [p[0] for p in cv2.BFMatcher().knnMatch(d1, d2, k=2)
                if len(p) == 2 and p[0].distance < self.ratio * p[1].distance]
        if len(good) < self.min_m:
            return None, None
        return (np.array([kp1[m.queryIdx].pt for m in good], np.float32),
                np.array([kp2[m.trainIdx].pt for m in good], np.float32))


class _LightGlueMatcher:
    def __init__(self, min_m: int = 2):
        self.min_m = min_m
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained("ETH-CVG/lightglue_superpoint")
        self.model = AutoModel.from_pretrained("ETH-CVG/lightglue_superpoint")

    def match(self, f1: np.ndarray, f2: np.ndarray):
        from PIL import Image
        try:
            i1 = Image.fromarray(cv2.cvtColor(f1, cv2.COLOR_BGR2RGB))
            i2 = Image.fromarray(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB))
            inputs = self.processor([i1, i2], return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            sizes = [[(i1.height, i1.width), (i2.height, i2.width)]]
            results = self.processor.post_process_keypoint_matching(outputs, sizes, threshold=0.2)
            if not results:
                return None, None
            r = results[0]
            if len(r["keypoints0"]) < self.min_m:
                return None, None
            return (r["keypoints0"].cpu().numpy().astype(np.float32),
                    r["keypoints1"].cpu().numpy().astype(np.float32))
        except Exception as e:
            logger.warning(f"[TTS] LightGlue failed: {e}")
            return None, None


_MATCHER_CACHE: Dict[str, Any] = {}


def _get_matcher(descriptor: str, ratio: float, min_m: int):
    if descriptor in _MATCHER_CACHE:
        return _MATCHER_CACHE[descriptor]
    if descriptor == "sift":
        _MATCHER_CACHE[descriptor] = _SIFTMatcher(ratio, min_m)
    elif descriptor == "lightglue":
        _MATCHER_CACHE[descriptor] = _LightGlueMatcher(min_m)
    else:
        raise ValueError(f"Unsupported descriptor: {descriptor}")
    return _MATCHER_CACHE[descriptor]


def compute_epipolar_score(
    frames_bgr: List[np.ndarray],
    descriptor: str = "sift",
    sampling_rate: int = 1,
    ratio: float = 0.75,
    min_matches: int = 2,
) -> float:
    if len(frames_bgr) < 2:
        return PENALTY
    if not KORNIA_AVAILABLE:
        raise RuntimeError("kornia is required for the epipolar metric (pip install kornia).")

    matcher = _get_matcher(descriptor, ratio, min_matches)
    scores: List[float] = []
    for idx in range(0, len(frames_bgr) - 1, max(1, sampling_rate)):
        pts1, pts2 = matcher.match(frames_bgr[idx], frames_bgr[idx + 1])
        if pts1 is None:
            continue
        try:
            t1 = torch.from_numpy(pts1).float().unsqueeze(0)
            t2 = torch.from_numpy(pts2).float().unsqueeze(0)
            F = find_fundamental(t1, t2)
            if F is None or torch.isnan(F).any():
                continue
            sd = torch.sqrt(sampson_epipolar_distance(t1, t2, F, squared=True) + 1e-8)
            scores.append(float(sd.mean()))
        except Exception:
            continue
    return float(np.mean(scores)) if scores else PENALTY


def score_frames(
    frames_bgr: List[np.ndarray],
    metric: str,
    cfg: Dict[str, Any],
) -> float:
    """Dispatch by metric. cfg keys depend on the metric (see callers)."""
    if metric == "reprojection":
        return compute_reprojection_score(
            frames_bgr,
            max_pts=cfg["max_pts"],
            conf_thr=cfg["conf_thr"],
            rng=cfg.get("rng"),
        )
    if metric == "epipolar":
        return compute_epipolar_score(
            frames_bgr,
            descriptor=cfg["ep_desc"],
            sampling_rate=cfg["ep_rate"],
            ratio=cfg["ep_ratio"],
            min_matches=cfg["ep_min_m"],
        )
    raise ValueError(f"Unknown metric: {metric}")


# ============================================================
# KV / cross-attn cache helpers (used by SoP and BS)
# ============================================================

def clone_kv(kv: List[dict]) -> List[dict]:
    out = []
    for e in kv:
        end = int(e["global_end_index"].item())
        out.append(dict(
            k_cpu=e["k"][:, :end].cpu(),
            v_cpu=e["v"][:, :end].cpu(),
            global_end_index=e["global_end_index"].clone().cpu(),
            local_end_index=e["local_end_index"].clone().cpu(),
        ))
    return out


def restore_kv(kv: List[dict], state: List[dict]) -> None:
    for dst, src in zip(kv, state):
        end = int(src["global_end_index"].item())
        if end > 0:
            dst["k"][:, :end].copy_(src["k_cpu"])
            dst["v"][:, :end].copy_(src["v_cpu"])
        if end < dst["k"].shape[1]:
            dst["k"][:, end:].zero_()
            dst["v"][:, end:].zero_()
        dst["global_end_index"].copy_(src["global_end_index"].to(dst["global_end_index"].device))
        dst["local_end_index"].copy_(src["local_end_index"].to(dst["local_end_index"].device))


def clone_ca(ca: List[dict]) -> List[dict]:
    return [dict(k_cpu=e["k"].cpu(), v_cpu=e["v"].cpu(), is_init=e["is_init"]) for e in ca]


def restore_ca(ca: List[dict], state: List[dict]) -> None:
    for dst, src in zip(ca, state):
        dst["k"].copy_(src["k_cpu"])
        dst["v"].copy_(src["v_cpu"])
        dst["is_init"] = src["is_init"]


def reset_pipeline_caches(pipeline, device: torch.device, dtype: torch.dtype) -> None:
    """Lazy-init caches on first call, otherwise zero the position counters."""
    if pipeline.kv_cache1 is None:
        pipeline._initialize_kv_cache(batch_size=1, dtype=dtype, device=device)
        pipeline._initialize_crossattn_cache(batch_size=1, dtype=dtype, device=device)
        return
    for b in range(pipeline.num_transformer_blocks):
        pipeline.crossattn_cache[b]["is_init"] = False
    for b in range(len(pipeline.kv_cache1)):
        pipeline.kv_cache1[b]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=device)
        pipeline.kv_cache1[b]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=device)


# ============================================================
# Shared denoising primitive (one block, returns latent + pixels)
# ============================================================

def denoise_block(
    pipeline,
    noise_slice: torch.Tensor,
    conditional_dict: dict,
    current_start_frame: int,
    device: torch.device,
    dtype: torch.dtype,
    decode_pixels: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the multi-step denoising for one block. Optionally decode pixels."""
    T = noise_slice.shape[1]
    x = noise_slice
    pred = None
    for idx, ts in enumerate(pipeline.denoising_step_list):
        timestep = torch.ones([1, T], device=device, dtype=torch.int64) * ts
        _, pred = pipeline.generator(
            noisy_image_or_video=x,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=pipeline.kv_cache1,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=current_start_frame * pipeline.frame_seq_length,
        )
        if idx < len(pipeline.denoising_step_list) - 1:
            nts = pipeline.denoising_step_list[idx + 1]
            x = pipeline.scheduler.add_noise(
                pred.flatten(0, 1),
                torch.randn_like(pred.flatten(0, 1)),
                nts * torch.ones([T], device=device, dtype=torch.long),
            ).unflatten(0, pred.shape[:2])

    pixels = None
    if decode_pixels:
        with torch.no_grad():
            pixels = pipeline.vae.decode_to_pixel(pred, use_cache=False)
            pixels = (pixels * 0.5 + 0.5).clamp(0, 1)
    return pred, pixels


def commit_to_context(
    pipeline,
    latent: torch.Tensor,
    conditional_dict: dict,
    current_start_frame: int,
    device: torch.device,
) -> None:
    """Inject the committed latent into KV cache at context_noise timestep."""
    T = latent.shape[1]
    ctx_ts = torch.ones([1, T], device=device, dtype=torch.int64) * pipeline.args.context_noise
    pipeline.generator(
        noisy_image_or_video=latent,
        conditional_dict=conditional_dict,
        timestep=ctx_ts,
        kv_cache=pipeline.kv_cache1,
        crossattn_cache=pipeline.crossattn_cache,
        current_start=current_start_frame * pipeline.frame_seq_length,
    )


def block_schedule(pipeline, num_frames: int, has_initial_latent: bool) -> List[int]:
    """Return per-block frame counts matching the pipeline's block layout."""
    nfb = pipeline.num_frame_per_block
    if not pipeline.independent_first_frame or (pipeline.independent_first_frame and has_initial_latent):
        assert num_frames % nfb == 0
        return [nfb] * (num_frames // nfb)
    assert (num_frames - 1) % nfb == 0
    return [1] + [nfb] * ((num_frames - 1) // nfb)
