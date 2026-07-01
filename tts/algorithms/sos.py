"""
Search on Start (SoS).

For each global seed s in [base_seed, base_seed+S-1], run a full AR rollout
sharing that seed across all frames, then pick the rollout with the lowest
reward score.

Complexity: O(S * N), where N = number of AR blocks.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch

from tts.tts_common import (
    block_schedule,
    commit_to_context,
    denoise_block,
    reset_pipeline_caches,
    score_frames,
    tensor_to_bgr,
)


def full_rollout(
    pipeline,
    noise: torch.Tensor,
    conditional_dict: dict,
    initial_latent: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run one complete AR pass. Returns (video_pixels, output_latents)."""
    _, num_frames, C, H, W = noise.shape
    n_in = initial_latent.shape[1] if initial_latent is not None else 0
    n_total = num_frames + n_in
    all_n_frames = block_schedule(pipeline, num_frames, initial_latent is not None)

    reset_pipeline_caches(pipeline, device, dtype)
    output_latents = torch.zeros([1, n_total, C, H, W], device=device, dtype=dtype)

    cur_frame = 0
    ts0 = torch.zeros([1, 1], device=device, dtype=torch.int64)

    if initial_latent is not None:
        if pipeline.independent_first_frame:
            n_in_blocks = (n_in - 1) // pipeline.num_frame_per_block
            output_latents[:, :1] = initial_latent[:, :1]
            pipeline.generator(
                noisy_image_or_video=initial_latent[:, :1],
                conditional_dict=conditional_dict,
                timestep=ts0, kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=0,
            )
            cur_frame += 1
        else:
            n_in_blocks = n_in // pipeline.num_frame_per_block
        nfb = pipeline.num_frame_per_block
        for _ in range(n_in_blocks):
            rl = initial_latent[:, cur_frame:cur_frame + nfb]
            output_latents[:, cur_frame:cur_frame + nfb] = rl
            pipeline.generator(
                noisy_image_or_video=rl,
                conditional_dict=conditional_dict,
                timestep=ts0, kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=cur_frame * pipeline.frame_seq_length,
            )
            cur_frame += nfb

    for n_block in all_n_frames:
        noise_slice = noise[:, cur_frame - n_in:cur_frame + n_block - n_in]
        pred, _ = denoise_block(
            pipeline, noise_slice, conditional_dict, cur_frame,
            device, dtype, decode_pixels=False,
        )
        output_latents[:, cur_frame:cur_frame + n_block] = pred
        commit_to_context(pipeline, pred, conditional_dict, cur_frame, device)
        cur_frame += n_block

    video = pipeline.vae.decode_to_pixel(output_latents, use_cache=False)
    video = (video * 0.5 + 0.5).clamp(0, 1)
    return video, output_latents


def make_noise_for_seed(
    seed: int, num_frames: int, frame_shape: Tuple[int, ...],
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """One-shot randn matching the baseline noise convention."""
    torch.manual_seed(seed)
    return torch.randn((1, num_frames, *frame_shape), device=device, dtype=dtype)


def search_on_start(
    pipeline,
    text_prompts: List[str],
    num_noise_frames: int,
    initial_latent: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    base_seed: int,
    num_candidates: int,
    metric: str,
    eval_frames: int,
    short_side: int,
    score_cfg: dict,
    frame_shape: Tuple[int, ...] = (16, 60, 104),
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Returns (best_video, best_latents, best_seed_index).
    Seed set: [base_seed, base_seed + 1, ..., base_seed + S - 1].
    """
    conditional_dict = pipeline.text_encoder(text_prompts=text_prompts)
    best_score = float("inf")
    best_video, best_latents, best_idx = None, None, 0

    for s in range(num_candidates):
        gseed = base_seed + s
        print(f"  [SoS] rollout {s + 1}/{num_candidates} (seed={gseed})")

        noise = make_noise_for_seed(gseed, num_noise_frames, frame_shape, device, dtype)
        video, latents = full_rollout(
            pipeline, noise, conditional_dict, initial_latent, device, dtype,
        )

        T = video.shape[1]
        idxs = np.unique(np.linspace(0, T - 1, eval_frames).round().astype(int)).tolist()
        eval_bgr = tensor_to_bgr(video[0, idxs], short_side=short_side)
        score = score_frames(eval_bgr, metric, score_cfg)
        print(f"  [SoS] seed={gseed} {metric}_score={score:.4f}")

        if score < best_score:
            best_score = score
            best_video = video
            best_latents = latents
            best_idx = s

        pipeline.vae.model.clear_cache()

    print(f"  [SoS] BEST seed_idx={best_idx} (seed={base_seed + best_idx}) "
          f"{metric}_score={best_score:.4f}")
    return best_video, best_latents, best_idx
