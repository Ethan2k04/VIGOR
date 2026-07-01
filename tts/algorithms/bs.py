"""
Beam Search (BS).

Maintain K paths. At every block, each surviving path expands into N
children (different noise per child). After S consecutive expansion steps
(stride), evaluate every leaf over a sliding window and prune back to K.

Noise seed for child c at block b on a path with node_id n:
    seed = base_seed + b * 1000 + c   (unique within a stride window)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

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


@dataclass
class PathState:
    node_id: int
    kv_state: List[dict]
    crossattn_state: List[dict]
    committed_pixels: List[torch.Tensor]
    output_latents: torch.Tensor
    current_start_frame: int
    cumulative_score: float = 0.0
    seed_origin: int = 0


def _free(p: PathState) -> None:
    """Release the CPU KV snapshots so GC can reclaim memory."""
    p.kv_state = []
    p.crossattn_state = []


class BeamSearch:
    """Top-K beam search wrapping a CausalInferencePipeline."""

    def __init__(
        self,
        pipeline,
        top_k: int,
        n_candidates: int,
        stride: int,
        reward_window: int,
        metric: str,
        short_side: int,
        seed: int,
        score_cfg: dict,
    ):
        self.pipeline = pipeline
        self.top_k = top_k
        self.n_candidates = n_candidates
        self.stride = stride
        self.reward_window = reward_window
        self.metric = metric
        self.short_side = short_side
        self.base_seed = seed
        self.score_cfg = score_cfg
        self._node_counter = 0

    def _next_id(self) -> int:
        nid = self._node_counter
        self._node_counter += 1
        return nid

    def _make_noise(self, block_idx: int, c_idx: int, shape, device, dtype) -> torch.Tensor:
        torch.manual_seed(self.base_seed + block_idx * 1000 + c_idx)
        return torch.randn(shape, device=device, dtype=dtype)

    def _score_leaf(self, p: PathState) -> float:
        hist = p.committed_pixels
        if len(hist) < 2:
            return 0.0
        ctx_end = len(hist) - 1
        ctx_start = max(0, ctx_end - self.reward_window)
        ctx_frames = torch.cat(hist[ctx_start:ctx_end], dim=0)
        eval_tensor = torch.cat([ctx_frames, hist[-1]], dim=0)
        return score_frames(
            tensor_to_bgr(eval_tensor, short_side=self.short_side),
            self.metric, self.score_cfg,
        )

    def _expand(
        self,
        leaves: List[PathState],
        block_idx: int,
        n_block: int,
        C: int, H: int, W: int,
        cond: dict, device, dtype,
    ) -> List[PathState]:
        pipe = self.pipeline
        shape = (1, n_block, C, H, W)
        children: List[PathState] = []

        for parent in leaves:
            for c_idx in range(self.n_candidates):
                restore_kv(pipe.kv_cache1, parent.kv_state)
                restore_ca(pipe.crossattn_cache, parent.crossattn_state)

                noise = self._make_noise(block_idx, c_idx, shape, device, dtype)
                lat, pix = denoise_block(
                    pipe, noise, cond, parent.current_start_frame, device, dtype,
                )
                commit_to_context(pipe, lat, cond, parent.current_start_frame, device)

                new_lat = parent.output_latents.clone()
                new_lat[:, parent.current_start_frame:parent.current_start_frame + n_block] = lat.cpu()

                new_pix = list(parent.committed_pixels) + [pix[0].cpu()]
                max_keep = self.reward_window + 2
                if len(new_pix) > max_keep:
                    new_pix = new_pix[-max_keep:]

                children.append(PathState(
                    node_id=self._next_id(),
                    kv_state=clone_kv(pipe.kv_cache1),
                    crossattn_state=clone_ca(pipe.crossattn_cache),
                    committed_pixels=new_pix,
                    output_latents=new_lat,
                    current_start_frame=parent.current_start_frame + n_block,
                    cumulative_score=parent.cumulative_score,
                    seed_origin=parent.seed_origin,
                ))
        return children

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
        num_blocks = len(all_n_frames)

        cond = pipe.text_encoder(text_prompts=text_prompts)
        reset_pipeline_caches(pipe, device, dtype)

        # ---- I2V prefix ----
        cur_frame = 0
        init_pix: List[torch.Tensor] = []
        init_lat_prefix = (torch.zeros([1, n_in, C, H, W], device=device, dtype=dtype)
                          if n_in > 0 else None)

        if initial_latent is not None:
            ts0 = torch.zeros([1, 1], device=device, dtype=torch.int64)
            if pipe.independent_first_frame:
                n_in_blocks = (n_in - 1) // pipe.num_frame_per_block
                init_lat_prefix[:, :1] = initial_latent[:, :1]
                pipe.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=cond, timestep=ts0,
                    kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                    current_start=0,
                )
                cur_frame += 1
            else:
                n_in_blocks = n_in // pipe.num_frame_per_block
            nfb = pipe.num_frame_per_block
            for _ in range(n_in_blocks):
                rl = initial_latent[:, cur_frame:cur_frame + nfb]
                init_lat_prefix[:, cur_frame:cur_frame + nfb] = rl
                pipe.generator(
                    noisy_image_or_video=rl, conditional_dict=cond,
                    timestep=ts0, kv_cache=pipe.kv_cache1,
                    crossattn_cache=pipe.crossattn_cache,
                    current_start=cur_frame * pipe.frame_seq_length,
                )
                with torch.no_grad():
                    rp = pipe.vae.decode_to_pixel(rl, use_cache=False)
                    rp = (rp * 0.5 + 0.5).clamp(0, 1)
                init_pix.append(rp[0].cpu())
                cur_frame += nfb

        # ---- STEP 1: initialise K paths on block 0 ----
        T0 = all_n_frames[0]
        print(f"\n[BS] Initialising {self.top_k} paths (block 0) ...")
        prefix_kv = clone_kv(pipe.kv_cache1)
        prefix_ca = clone_ca(pipe.crossattn_cache)
        paths: List[PathState] = []

        for s in range(self.top_k):
            gseed = self.base_seed + s
            torch.manual_seed(gseed)
            fn = torch.randn([1, T0, C, H, W], device=device, dtype=dtype)
            restore_kv(pipe.kv_cache1, prefix_kv)
            restore_ca(pipe.crossattn_cache, prefix_ca)

            lat, pix = denoise_block(pipe, fn, cond, cur_frame, device, dtype)
            commit_to_context(pipe, lat, cond, cur_frame, device)

            out_lat = torch.zeros([1, n_total, C, H, W], device="cpu", dtype=dtype)
            if init_lat_prefix is not None:
                out_lat[:, :n_in] = init_lat_prefix.cpu()
            out_lat[:, cur_frame:cur_frame + T0] = lat.cpu()
            committed = list(init_pix) + [pix[0].cpu()]

            score = self._score_leaf(PathState(
                node_id=0, kv_state=[], crossattn_state=[],
                committed_pixels=committed, output_latents=out_lat,
                current_start_frame=cur_frame + T0, seed_origin=gseed,
            ))
            paths.append(PathState(
                node_id=self._next_id(),
                kv_state=clone_kv(pipe.kv_cache1),
                crossattn_state=clone_ca(pipe.crossattn_cache),
                committed_pixels=committed,
                output_latents=out_lat,
                current_start_frame=cur_frame + T0,
                cumulative_score=score,
                seed_origin=gseed,
            ))
            print(f"  path={s} seed={gseed} block0_{self.metric}={score:.4f}")

        # ---- STEP 2: expand x stride -> eval -> prune ----
        block_idx = 1
        while block_idx < num_blocks:
            end = min(block_idx + self.stride, num_blocks)
            window = list(range(block_idx, end))
            n_leaves = len(paths) * (self.n_candidates ** len(window))
            print(f"\n[BS] Stride blocks {window}: {len(paths)} paths -> {n_leaves} leaves")

            leaves = paths
            for b_idx in window:
                leaves = self._expand(
                    leaves, b_idx, all_n_frames[b_idx],
                    C, H, W, cond, device, dtype,
                )
                print(f"  expanded block {b_idx + 1}/{num_blocks}: {len(leaves)} leaves")

            scored: List[Tuple[float, PathState]] = []
            for leaf in leaves:
                s = self._score_leaf(leaf)
                leaf.cumulative_score += s
                scored.append((s, leaf))

            scored.sort(key=lambda x: x[0])
            survivors = [x[1] for x in scored[:self.top_k]]
            for _, d in scored[self.top_k:]:
                _free(d)
            print(f"  kept top-{self.top_k}: "
                  f"{[f'{x[0]:.4f}' for x in scored[:self.top_k]]}")

            paths = survivors
            block_idx = end

        # ---- STEP 3: final selection ----
        best = min(paths, key=lambda p: p.cumulative_score)
        print(f"\n[BS] BEST origin_seed={best.seed_origin} "
              f"cumulative_{self.metric}={best.cumulative_score:.4f}")

        best_lat = best.output_latents.to(device=device, dtype=dtype)
        with torch.no_grad():
            best_vid = pipe.vae.decode_to_pixel(best_lat, use_cache=False)
            best_vid = (best_vid * 0.5 + 0.5).clamp(0, 1)
        pipe.vae.model.clear_cache()

        if return_latents:
            return best_vid.cpu(), best_lat.cpu()
        return best_vid.cpu()
