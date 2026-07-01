"""
Test-Time Scaling inference for CausalForcing video generation.

Algorithms (selected via --algorithm):
  sos : Search on Start  -- pick the best of S full AR rollouts.
  sop : Search on Path   -- per-block best-of-N search with sliding-window reward.
  bs  : Beam Search      -- top-K beam search with stride S and N children per leaf.

Reward metrics (--metric):
  reprojection : VGGT reprojection error (default).
  epipolar     : Mean Sampson epipolar distance via SIFT or LightGlue.
"""

import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm

# torchvision.io.write_video requires the video backend (ffmpeg / av) to be
# built in; on many setups it is missing. Fall back to imageio if so.
try:
    from torchvision.io import write_video as _tv_write_video
except ImportError:
    _tv_write_video = None

# Import tts_common first so it injects third_party paths before downstream imports
from tts.tts_common import (
    CAUSAL_FORCING_DIR,
    DEFAULT_CONFIG_PATH,
    KORNIA_AVAILABLE,
    get_vggt_model,
)

from pipeline import CausalInferencePipeline, CausalDiffusionInferencePipeline  # noqa: E402
from utils.dataset import TextDataset, TextImagePairDataset  # noqa: E402
from utils.misc import set_seed  # noqa: E402
from demo_utils import memory as memory_mod  # noqa: E402
from demo_utils.memory import (  # noqa: E402
    get_cuda_free_memory_gb,
    DynamicSwapInstaller,
    move_model_to_device_with_memory_preservation,
)

from tts.algorithms.sos import search_on_start, make_noise_for_seed  # noqa: E402
from tts.algorithms.sop import SoPSearch  # noqa: E402
from tts.algorithms.bs import BeamSearch  # noqa: E402


FRAME_LATENT_SHAPE = (16, 60, 104)

# Default prompt suite / checkpoints (shared default across all entrypoints)
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(_REPO_ROOT, "prompts", "demos.txt")
DEFAULT_CF_CONFIG = os.path.join(
    CAUSAL_FORCING_DIR, "configs", "causal_forcing_dmd_framewise.yaml"
)
DEFAULT_CF_CHECKPOINT = os.path.join(
    CAUSAL_FORCING_DIR, "checkpoints", "framewise", "causal_forcing.pt"
)
DEFAULT_OUTPUT_FOLDER = os.path.join(_REPO_ROOT, "outputs", "tts")


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="CausalForcing TTS inference (SoS / SoP / BS)")

    # ---- pipeline ----
    p.add_argument("--algorithm", choices=["sos", "sop", "bs"], default="sos")
    p.add_argument("--config_path", default=DEFAULT_CF_CONFIG)
    p.add_argument("--checkpoint_path", default=DEFAULT_CF_CHECKPOINT)
    p.add_argument("--data_path", default=DEFAULT_DATA_PATH,
                   help="Prompt suite (defaults to prompts/demos.txt).")
    p.add_argument("--output_folder", default=DEFAULT_OUTPUT_FOLDER)
    p.add_argument("--default_config_path", default=DEFAULT_CONFIG_PATH,
                   help="Path to default_config.yaml (auto-resolved to the submodule by default).")
    p.add_argument("--num_output_frames", type=int, default=21)
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--i2v", action="store_true")
    p.add_argument("--device", default=None)

    # ---- metric ----
    p.add_argument("--metric", default="reprojection",
                   choices=["reprojection", "epipolar"])
    p.add_argument("--eval_short_side", type=int, default=448)
    p.add_argument("--eval_frames", type=int, default=8,
                   help="(SoS only) uniformly sampled frames for scoring.")
    p.add_argument("--vggt_max_query_points", type=int, default=512)
    p.add_argument("--vggt_track_conf", type=float, default=0.05)
    p.add_argument("--epipolar_descriptor", default="sift",
                   choices=["sift", "lightglue"])
    p.add_argument("--epipolar_sampling_rate", type=int, default=1)
    p.add_argument("--epipolar_ratio_thresh", type=float, default=0.75)
    p.add_argument("--epipolar_min_matches", type=int, default=2)

    # ---- SoS / SoP ----
    p.add_argument("--num_candidates", type=int, default=4,
                   help="SoS: number of seeds. SoP: candidates per block.")
    p.add_argument("--reward_context_frames", type=int, default=5,
                   help="SoP: sliding-window context size for scoring.")
    p.add_argument("--save_all_candidates", action="store_true",
                   help="SoS: also save every candidate for ablation.")

    # ---- BS ----
    p.add_argument("--top_k", type=int, default=4)
    p.add_argument("--n_candidates", type=int, default=4)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--reward_window", type=int, default=5)

    return p.parse_args()


# ============================================================
# Setup helpers
# ============================================================

def _chdir_to_causal_forcing(args) -> None:
    """Causal-Forcing's code references `wan_models/...` via relative paths.
    Absolutize all user-supplied paths, then chdir into the submodule so
    those lookups resolve to third_party/Causal-Forcing/wan_models/."""
    for attr in ("config_path", "checkpoint_path", "data_path",
                 "output_folder", "default_config_path"):
        v = getattr(args, attr, None)
        if v:
            setattr(args, attr, os.path.abspath(v))
    os.chdir(CAUSAL_FORCING_DIR)


def _setup_device(args) -> tuple:
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device or "cuda:0")
        torch.cuda.set_device(device)
        local_rank = 0
    memory_mod.gpu = device
    return device, local_rank


def _build_pipeline(args, device: torch.device):
    config = OmegaConf.merge(
        OmegaConf.load(args.default_config_path),
        OmegaConf.load(args.config_path),
    )
    if hasattr(config, "denoising_step_list"):
        pipeline = CausalInferencePipeline(config, device=device)
    else:
        pipeline = CausalDiffusionInferencePipeline(config, device=device)

    if args.checkpoint_path:
        sd = torch.load(args.checkpoint_path, map_location="cpu")
        key = "generator_ema" if args.use_ema else "generator"
        if key not in sd:
            avail = list(sd.keys()) if isinstance(sd, dict) else type(sd).__name__
            raise KeyError(
                f"Checkpoint '{args.checkpoint_path}' has no key '{key}'. "
                f"Available keys: {avail}. "
                f"Causal-Forcing's framewise/chunkwise checkpoints only ship "
                f"with 'generator_ema' -- pass --use_ema to use them."
            )
        try:
            pipeline.generator.load_state_dict(sd[key])
        except RuntimeError:
            fixed = {k.replace("model._fsdp_wrapped_module.", "model.", 1): v
                     for k, v in sd[key].items()}
            pipeline.generator.load_state_dict(fixed, strict=False)
    return pipeline.to(dtype=torch.bfloat16), config


def _place_modules(pipeline, device: torch.device, low_memory: bool) -> None:
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    else:
        pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)


def _preload_metric(args) -> None:
    if args.metric == "reprojection":
        get_vggt_model()
    elif args.metric == "epipolar" and not KORNIA_AVAILABLE:
        raise RuntimeError("kornia is required for the epipolar metric (pip install kornia).")


def _build_dataset(args):
    if args.i2v:
        assert not dist.is_initialized(), "I2V does not support distributed inference."
        tfm = transforms.Compose([
            transforms.Resize((480, 832)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        return TextImagePairDataset(args.data_path, transform=tfm)
    return TextDataset(
        prompt_path=args.data_path,
        extended_prompt_path=getattr(args, "extended_prompt_path", None),
    )


def _score_cfg(args, base_seed: int) -> dict:
    return dict(
        max_pts=args.vggt_max_query_points,
        conf_thr=args.vggt_track_conf,
        rng=np.random.default_rng(base_seed),
        ep_desc=args.epipolar_descriptor,
        ep_rate=args.epipolar_sampling_rate,
        ep_ratio=args.epipolar_ratio_thresh,
        ep_min_m=args.epipolar_min_matches,
    )


def _prepare_prompt_batch(batch_data, args, pipeline, device, num_noise_frames):
    """Returns (prompt, output_path, prompts, initial_latent, noise)."""
    batch = batch_data if isinstance(batch_data, dict) else batch_data[0]
    prompt = batch["prompts"][0]
    output_path = os.path.join(args.output_folder, f"{prompt[:100]}.mp4")
    if os.path.exists(output_path):
        return prompt, output_path, None, None, None

    if args.i2v:
        image = batch["image"].squeeze(0).unsqueeze(0).unsqueeze(2).to(
            device=device, dtype=torch.bfloat16)
        initial_latent = pipeline.vae.encode_to_latent(image).to(
            device=device, dtype=torch.bfloat16)
        prompts = [prompt]
    else:
        ep = batch.get("extended_prompts", [None])[0]
        prompts = [ep if ep is not None else prompt]
        initial_latent = None

    torch.manual_seed(args.seed)
    noise = torch.randn(
        [1, num_noise_frames, *FRAME_LATENT_SHAPE],
        device=device, dtype=torch.bfloat16,
    )
    return prompt, output_path, prompts, initial_latent, noise


def _save_video(video: torch.Tensor, path: str, fps: int = 16) -> None:
    """Write (B, T, C, H, W) float [0,1] video. Uses torchvision when
    available, otherwise imageio (mp4 via imageio-ffmpeg)."""
    frames = (255.0 * rearrange(video, "b t c h w -> b t h w c").cpu()
              ).clamp(0, 255).to(torch.uint8)[0]
    if _tv_write_video is not None:
        _tv_write_video(path, frames, fps=fps)
        return
    import imageio  # lazy import so missing imageio doesn't break startup
    imageio.mimsave(path, frames.numpy(), fps=fps, macro_block_size=1)


# ============================================================
# Algorithm dispatch
# ============================================================

def _run_sos(args, pipeline, prompts, initial_latent, num_noise_frames,
             device, dtype, score_cfg) -> torch.Tensor:
    best_video, _, best_idx = search_on_start(
        pipeline=pipeline,
        text_prompts=prompts,
        num_noise_frames=num_noise_frames,
        initial_latent=initial_latent,
        device=device, dtype=dtype,
        base_seed=args.seed,
        num_candidates=args.num_candidates,
        metric=args.metric,
        eval_frames=args.eval_frames,
        short_side=args.eval_short_side,
        score_cfg=score_cfg,
        frame_shape=FRAME_LATENT_SHAPE,
    )
    return best_video, best_idx


def _run_sop(args, pipeline, prompts, initial_latent, noise, score_cfg) -> torch.Tensor:
    sop = SoPSearch(
        pipeline=pipeline,
        num_candidates=args.num_candidates,
        reward_context_frames=args.reward_context_frames,
        metric=args.metric,
        short_side=args.eval_short_side,
        seed=args.seed,
        score_cfg=score_cfg,
    )
    return sop.inference(noise=noise, text_prompts=prompts,
                         initial_latent=initial_latent, return_latents=False)


def _run_bs(args, pipeline, prompts, initial_latent, noise, score_cfg) -> torch.Tensor:
    bs = BeamSearch(
        pipeline=pipeline,
        top_k=args.top_k,
        n_candidates=args.n_candidates,
        stride=args.stride,
        reward_window=args.reward_window,
        metric=args.metric,
        short_side=args.eval_short_side,
        seed=args.seed,
        score_cfg=score_cfg,
    )
    return bs.inference(noise=noise, text_prompts=prompts,
                        initial_latent=initial_latent, return_latents=False)


def _save_sos_candidates(args, pipeline, prompts, initial_latent,
                        num_noise_frames, device, dtype, best_idx, prompt) -> None:
    cand_dir = os.path.join(args.output_folder, "candidates", prompt[:60])
    os.makedirs(cand_dir, exist_ok=True)
    cond = pipeline.text_encoder(text_prompts=prompts)
    from algorithms.sos import full_rollout
    for s in range(args.num_candidates):
        gseed = args.seed + s
        noise = make_noise_for_seed(gseed, num_noise_frames, FRAME_LATENT_SHAPE, device, dtype)
        cand_video, _ = full_rollout(pipeline, noise, cond, initial_latent, device, dtype)
        tag = "BEST_" if s == best_idx else ""
        _save_video(cand_video, os.path.join(cand_dir, f"{tag}seed{gseed}.mp4"))
        pipeline.vae.model.clear_cache()
    print(f"[TTS] Saved {args.num_candidates} candidates -> {cand_dir}")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    _chdir_to_causal_forcing(args)
    device, local_rank = _setup_device(args)
    set_seed(args.seed)
    torch.set_grad_enabled(False)

    free_gb = get_cuda_free_memory_gb(device)
    low_memory = free_gb < 40
    print(f"[TTS] algorithm={args.algorithm} metric={args.metric} "
          f"free_vram={free_gb:.1f}GB low_memory={low_memory}")

    pipeline, config = _build_pipeline(args, device)
    _place_modules(pipeline, device, low_memory)
    _preload_metric(args)

    dataset = _build_dataset(args)
    print(f"[TTS] prompts: {len(dataset)}")

    sampler = (DistributedSampler(dataset, shuffle=False, drop_last=True)
               if dist.is_initialized() else SequentialSampler(dataset))
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0)

    if local_rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    dtype = torch.bfloat16
    for batch_data in tqdm(dataloader, disable=(local_rank != 0), desc="Prompts"):
        if args.i2v:
            assert config.num_frame_per_block == 1, "I2V requires frame-wise model."
            num_noise_frames = args.num_output_frames - 1
        else:
            num_noise_frames = args.num_output_frames

        prompt, output_path, prompts, initial_latent, noise = _prepare_prompt_batch(
            batch_data, args, pipeline, device, num_noise_frames,
        )
        if prompts is None:
            print(f"[TTS] skip (exists): {output_path}")
            continue

        print(f"\n[TTS] prompt: {prompt[:80]}")
        score_cfg = _score_cfg(args, args.seed)

        if low_memory and args.algorithm in ("sop", "bs"):
            move_model_to_device_with_memory_preservation(
                pipeline.text_encoder,
                target_device=device,
                preserved_memory_gb=get_cuda_free_memory_gb(device) + 5,
            )

        if args.algorithm == "sos":
            best_video, best_idx = _run_sos(
                args, pipeline, prompts, initial_latent,
                num_noise_frames, device, dtype, score_cfg,
            )
            _save_video(best_video, output_path)
            if args.save_all_candidates:
                _save_sos_candidates(
                    args, pipeline, prompts, initial_latent,
                    num_noise_frames, device, dtype, best_idx, prompt,
                )
        elif args.algorithm == "sop":
            video = _run_sop(args, pipeline, prompts, initial_latent, noise, score_cfg)
            _save_video(video, output_path)
        elif args.algorithm == "bs":
            video = _run_bs(args, pipeline, prompts, initial_latent, noise, score_cfg)
            _save_video(video, output_path)
        else:
            raise ValueError(args.algorithm)

        pipeline.vae.model.clear_cache()
        print(f"[TTS] saved: {output_path}")


if __name__ == "__main__":
    main()
