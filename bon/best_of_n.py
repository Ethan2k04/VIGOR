"""Core Best-of-N orchestration for the VIGOR geometry reward.

The pipeline is deliberately simple:

1. Generate ``num_candidates`` videos per prompt with Wan2.1 (one seed each).
2. Score every candidate with a geometry reward evaluator.
3. Keep the best-scoring candidate (lower reprojection / epipolar error is better).

Heavy dependencies (``diffsynth``, the reward evaluators and their VGGT backend)
are imported lazily so that ``python run_bon.py --help`` stays cheap.
"""

import json
import logging
import os
import shutil
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

# Wan2.1 weight files expected inside ``model_path``.
WAN_FILES = (
    "diffusion_pytorch_model.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
)

# All supported reward metrics are "lower is better".
SUPPORTED_METRICS = ("reprojection", "reprojection_vanilla", "epipolar")


@dataclass
class BoNConfig:
    model_path: str
    prompts: List[str]
    output_dir: str = "outputs/bon"
    num_candidates: int = 4
    base_seed: int = 0
    metric: str = "reprojection"
    num_inference_steps: int = 40
    height: int = 480
    width: int = 832
    num_frames: int = 81
    fps: int = 15
    quality: int = 5
    negative_prompt: str = ""
    enable_sky_filter: bool = False
    save_all_candidates: bool = False


def load_wan_pipeline(model_path: str):
    """Build a Wan2.1 text-to-video pipeline from a local model directory."""
    import torch
    from diffsynth import ModelManager, WanVideoPipeline

    model_paths = [os.path.join(model_path, name) for name in WAN_FILES]
    missing = [p for p in model_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing Wan2.1 weight files:\n  " + "\n  ".join(missing)
        )

    model_manager = ModelManager(device="cpu", torch_dtype=torch.bfloat16)
    model_manager.load_models(model_paths)
    pipe = WanVideoPipeline.from_model_manager(
        model_manager, torch_dtype=torch.bfloat16, device="cuda"
    )
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    return pipe


def build_evaluator(metric: str, enable_sky_filter: bool = False):
    """Instantiate a reward evaluator by name."""
    if metric == "reprojection":
        from rewards.evaluator.reproj_pts import ReprojectionEvaluator

        return ReprojectionEvaluator(enable_sky_onnx=enable_sky_filter)
    if metric == "reprojection_vanilla":
        from rewards.evaluator.reproj_pix import ReprojectionVanillaEvaluator

        return ReprojectionVanillaEvaluator()
    if metric == "epipolar":
        from rewards.evaluator.epipolar import EpipolarEvaluator

        return EpipolarEvaluator()
    raise ValueError(
        f"Unknown metric '{metric}'. Choose from {SUPPORTED_METRICS}."
    )


def best_of_n(cfg: BoNConfig):
    """Run Best-of-N selection for every prompt in ``cfg.prompts``."""
    from diffsynth import save_video

    if cfg.metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unknown metric '{cfg.metric}'.")

    best_dir = os.path.join(cfg.output_dir, "best")
    cand_dir = os.path.join(cfg.output_dir, "candidates")
    os.makedirs(best_dir, exist_ok=True)
    if cfg.save_all_candidates:
        os.makedirs(cand_dir, exist_ok=True)

    pipe = load_wan_pipeline(cfg.model_path)
    evaluator = build_evaluator(cfg.metric, cfg.enable_sky_filter)

    manifest = []
    for p_idx, prompt in enumerate(cfg.prompts):
        candidates = []
        for c in range(cfg.num_candidates):
            seed = cfg.base_seed + c
            frames = pipe(
                prompt=prompt,
                negative_prompt=cfg.negative_prompt,
                num_inference_steps=cfg.num_inference_steps,
                seed=seed,
                tiled=True,
                width=cfg.width,
                height=cfg.height,
                num_frames=cfg.num_frames,
            )
            target_dir = cand_dir if cfg.save_all_candidates else best_dir
            cand_path = os.path.join(
                target_dir, f"prompt{p_idx:03d}_seed{seed:03d}.mp4"
            )
            save_video(frames, cand_path, fps=cfg.fps, quality=cfg.quality)

            try:
                score, _ = evaluator.evaluate_video(cand_path)
            except Exception as exc:  # scoring must not abort the whole run
                logger.warning("Scoring failed for %s: %s", cand_path, exc)
                score = float("inf")

            candidates.append({"seed": seed, "path": cand_path, "score": float(score)})
            logger.info(
                "prompt %d | seed %d | %s = %.4f", p_idx, seed, cfg.metric, score
            )

        # Lower error is better for every supported metric.
        candidates.sort(key=lambda d: d["score"])
        best = candidates[0]
        best_path = os.path.join(best_dir, f"prompt{p_idx:03d}_best.mp4")
        shutil.copyfile(best["path"], best_path)

        if not cfg.save_all_candidates:
            # Drop the temporary per-seed copies; keep only the winner.
            for cand in candidates:
                if cand["path"] != best_path and os.path.exists(cand["path"]):
                    os.remove(cand["path"])

        manifest.append(
            {
                "prompt_index": p_idx,
                "prompt": prompt,
                "metric": cfg.metric,
                "best_seed": best["seed"],
                "best_score": best["score"],
                "best_path": best_path,
                "candidates": candidates if cfg.save_all_candidates else None,
            }
        )
        logger.info(
            "prompt %d | best seed %d | %s = %.4f -> %s",
            p_idx,
            best["seed"],
            cfg.metric,
            best["score"],
            best_path,
        )

    manifest_path = os.path.join(cfg.output_dir, "bon_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest to %s", manifest_path)
    return manifest
