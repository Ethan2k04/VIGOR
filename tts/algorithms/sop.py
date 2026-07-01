"""
Search on Path (SoP).

For each new block, sample N candidate noises, generate N candidate blocks,
score each candidate together with a sliding window of recent committed
frames, and commit the best one.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from tts.tts_common import (
    block_schedule,
    clone_ca,
    clone_kv,
    commit_to_context,
    denoise_block,
    reset_pipeline_caches,
    restore_ca,
    restore_kv,
    score_frames,
    tensor_to_bgr,
)


class SoPSearch:
    """Per-block search wrapping a CausalInferencePipeline."""

    def __init__(
        self,
        pipeline,
        num_candidates: int,
        reward_context_frames: int,
        metric: str,
        short_side: int,
        seed: int,
        score_cfg: dict,
    ):
        self.pipeline = pipeline
        self.num_candidates = num_candidates
        self.reward_context = reward_context_frames
        self.metric = metric
        self.short_side = short_side
        self.base_seed = seed
        self.score_cfg = score_cfg

    def _best_block(
        self,
        committed_frames: torch.Tensor,
        block_shape: Tuple[int, ...],
        block_idx: int,
        conditional_dict: dict,
        cur_frame: int,
        device: torch.device,
        dtype: torch.dtype,
        baseline_noise: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pipe = self.pipeline
        ctx_start = max(0, committed_frames.shape[0] - self.reward_context)
        ctx = committed_frames[ctx_start:]
        n_ctx = ctx.shape[0]
        n_cand = self.num_candidates if n_ctx >= 1 else 1

        kv_state = clone_kv(pipe.kv_cache1)
        ca_state = clone_ca(pipe.crossattn_cache)

        best_score = float("inf")
        best_lat, best_pix = None, None

        for k in range(n_cand):
            if k > 0:
                restore_kv(pipe.kv_cache1, kv_state)
                restore_ca(pipe.crossattn_cache, ca_state)

            if k == 0 and baseline_noise is not None:
                noise = baseline_noise
            else:
                torch.manual_seed(self.base_seed + block_idx * 1000 + k)
                noise = torch.randn(block_shape, device=device, dtype=dtype)

            pred, pix = denoise_block(
                pipe, noise, conditional_dict, cur_frame, device, dtype,
            )

            if n_ctx >= 1:
                eval_t = torch.cat([ctx.to(pix.device), pix[0]], dim=0)
                eval_bgr = tensor_to_bgr(eval_t, short_side=self.short_side)
                score = score_frames(eval_bgr, self.metric, self.score_cfg)
            else:
                score = 0.0
            print(f"    [SoP] block={block_idx} cand={k + 1}/{n_cand} "
                  f"{self.metric}_score={score:.4f}")

            if score < best_score:
                best_score = score
                best_lat = pred
                best_pix = pix

        restore_kv(pipe.kv_cache1, kv_state)
        restore_ca(pipe.crossattn_cache, ca_state)
        print(f"  [SoP] block={block_idx} BEST {self.metric}_score={best_score:.4f}")
        return best_lat, best_pix

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
    ) -> torch.Tensor:
        pipe = self.pipeline
        device, dtype = noise.device, noise.dtype
        B, num_frames, C, H, W = noise.shape
        assert B == 1
        n_in = initial_latent.shape[1] if initial_latent is not None else 0
        n_total = num_frames + n_in
        all_n_frames = block_schedule(pipe, num_frames, initial_latent is not None)

        conditional_dict = pipe.text_encoder(text_prompts=text_prompts)
        reset_pipeline_caches(pipe, device, dtype)
        output_latents = torch.zeros([1, n_total, C, H, W], device=device, dtype=dtype)

        cur_frame = 0
        committed: List[torch.Tensor] = []
        ts0 = torch.zeros([1, 1], device=device, dtype=torch.int64)

        if initial_latent is not None:
            if pipe.independent_first_frame:
                n_in_blocks = (n_in - 1) // pipe.num_frame_per_block
                output_latents[:, :1] = initial_latent[:, :1]
                pipe.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict, timestep=ts0,
                    kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                    current_start=0,
                )
                cur_frame += 1
            else:
                n_in_blocks = n_in // pipe.num_frame_per_block
            nfb = pipe.num_frame_per_block
            for _ in range(n_in_blocks):
                rl = initial_latent[:, cur_frame:cur_frame + nfb]
                output_latents[:, cur_frame:cur_frame + nfb] = rl
                pipe.generator(
                    noisy_image_or_video=rl, conditional_dict=conditional_dict,
                    timestep=ts0, kv_cache=pipe.kv_cache1,
                    crossattn_cache=pipe.crossattn_cache,
                    current_start=cur_frame * pipe.frame_seq_length,
                )
                with torch.no_grad():
                    rp = pipe.vae.decode_to_pixel(rl, use_cache=False)
                    rp = (rp * 0.5 + 0.5).clamp(0, 1)
                committed.append(rp[0].cpu())
                cur_frame += nfb

        max_keep = self.reward_context // pipe.num_frame_per_block + 2

        for b_idx, n_block in enumerate(tqdm(all_n_frames, desc="[SoP] blocks")):
            noise_slice = noise[:, cur_frame - n_in:cur_frame + n_block - n_in]
            ctx = torch.cat(committed, dim=0) if committed else torch.zeros([0, 3, 1, 1])

            if self.num_candidates > 1:
                pred, pix = self._best_block(
                    ctx, (1, n_block, C, H, W), b_idx, conditional_dict,
                    cur_frame, device, dtype, noise_slice,
                )
                commit_to_context(pipe, pred, conditional_dict, cur_frame, device)
            else:
                pred, pix = denoise_block(
                    pipe, noise_slice, conditional_dict, cur_frame, device, dtype,
                )
                commit_to_context(pipe, pred, conditional_dict, cur_frame, device)

            output_latents[:, cur_frame:cur_frame + n_block] = pred
            committed.append(pix[0].cpu())
            if len(committed) > max_keep:
                committed = committed[-max_keep:]
            cur_frame += n_block

        video = pipe.vae.decode_to_pixel(output_latents, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        if return_latents:
            return video, output_latents
        return video
