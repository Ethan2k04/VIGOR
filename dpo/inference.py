"""DPO inference: compare a VIGOR geometry-aligned LoRA against Wan2.1.

For every prompt the module generates a baseline video and a LoRA-adapted video
using the *same* seed, then optionally writes a side-by-side comparison so the
effect of the geometry reward is easy to inspect. Heavy dependencies
(``diffsynth``, ``torch``, ``cv2``) are imported lazily.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Wan2.1 weight files expected inside ``model_path``.
WAN_FILES = (
    "diffusion_pytorch_model.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
)


@dataclass
class DPOInferenceConfig:
    model_path: str
    lora_path: Optional[str] = None
    prompts: List[str] = field(default_factory=list)
    output_dir: str = "outputs/dpo"
    lora_alpha: float = 1.0
    num_inference_steps: int = 40
    height: int = 480
    width: int = 832
    num_frames: int = 81
    fps: int = 15
    quality: int = 5
    seed: int = 0
    negative_prompt: str = ""
    run_baseline: bool = True
    stack: bool = True


def load_wan_pipeline(model_path: str):
    """Build a Wan2.1 pipeline and return ``(pipe, model_manager)``.

    The ``model_manager`` is returned as well so a LoRA can be loaded onto the
    same weights after the baseline pass.
    """
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
    return pipe, model_manager


def _generate(pipe, prompt: str, cfg: DPOInferenceConfig):
    return pipe(
        prompt=prompt,
        negative_prompt=cfg.negative_prompt,
        num_inference_steps=cfg.num_inference_steps,
        seed=cfg.seed,
        tiled=True,
        width=cfg.width,
        height=cfg.height,
        num_frames=cfg.num_frames,
    )


def stack_side_by_side(baseline_path: str, lora_path: str, out_path: str, fps: int = 15):
    """Write a ``baseline | LoRA`` comparison video."""
    import cv2
    import numpy as np

    cap_b = cv2.VideoCapture(baseline_path)
    cap_l = cv2.VideoCapture(lora_path)
    width = int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width * 2, height))
    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        while True:
            ret_b, frame_b = cap_b.read()
            ret_l, frame_l = cap_l.read()
            if not ret_b or not ret_l:
                break
            cv2.putText(frame_b, "Baseline", (10, 30), font, 1, (0, 255, 0), 2)
            cv2.putText(frame_l, "VIGOR LoRA", (10, 30), font, 1, (0, 255, 0), 2)
            writer.write(np.hstack((frame_b, frame_l)))
    finally:
        cap_b.release()
        cap_l.release()
        writer.release()
    return out_path


def run_comparison(cfg: DPOInferenceConfig) -> List[Dict]:
    """Generate baseline / LoRA videos and optional side-by-side comparisons."""
    from diffsynth import save_video

    os.makedirs(cfg.output_dir, exist_ok=True)
    baseline_dir = os.path.join(cfg.output_dir, "baseline")
    lora_dir = os.path.join(cfg.output_dir, "lora")
    stacked_dir = os.path.join(cfg.output_dir, "stacked")

    pipe, model_manager = load_wan_pipeline(cfg.model_path)

    # ---- Baseline pass (frozen Wan2.1) -----------------------------------
    baseline_paths: Dict[int, str] = {}
    if cfg.run_baseline:
        os.makedirs(baseline_dir, exist_ok=True)
        for idx, prompt in enumerate(cfg.prompts):
            frames = _generate(pipe, prompt, cfg)
            path = os.path.join(baseline_dir, f"prompt_{idx:03d}.mp4")
            save_video(frames, path, fps=cfg.fps, quality=cfg.quality)
            baseline_paths[idx] = path
            logger.info("baseline | prompt %d -> %s", idx, path)

    # ---- LoRA pass (geometry-aligned) ------------------------------------
    lora_paths: Dict[int, str] = {}
    if cfg.lora_path:
        model_manager.load_lora(cfg.lora_path, lora_alpha=cfg.lora_alpha)
        os.makedirs(lora_dir, exist_ok=True)
        do_stack = cfg.stack and cfg.run_baseline
        if do_stack:
            os.makedirs(stacked_dir, exist_ok=True)
        for idx, prompt in enumerate(cfg.prompts):
            frames = _generate(pipe, prompt, cfg)
            path = os.path.join(lora_dir, f"prompt_{idx:03d}.mp4")
            save_video(frames, path, fps=cfg.fps, quality=cfg.quality)
            lora_paths[idx] = path
            logger.info("lora | prompt %d -> %s", idx, path)
            if do_stack and idx in baseline_paths:
                stacked = os.path.join(stacked_dir, f"prompt_{idx:03d}.mp4")
                stack_side_by_side(baseline_paths[idx], path, stacked, fps=cfg.fps)
    else:
        logger.warning("No --lora_path provided; only the baseline was generated.")

    manifest = [
        {
            "prompt_index": idx,
            "prompt": prompt,
            "seed": cfg.seed,
            "baseline_path": baseline_paths.get(idx),
            "lora_path": lora_paths.get(idx),
        }
        for idx, prompt in enumerate(cfg.prompts)
    ]
    manifest_path = os.path.join(cfg.output_dir, "dpo_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest to %s", manifest_path)
    return manifest
