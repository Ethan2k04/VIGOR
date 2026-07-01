"""DPO inference entrypoint: VIGOR geometry-aligned LoRA vs. Wan2.1 baseline.

For each prompt this generates a baseline video and a LoRA-adapted video with the
same seed and writes a side-by-side comparison. Prompts default to
``prompts/demos.txt``.

Example
-------
    python infer_dpo.py \
        --model_path /path/to/Wan2.1-T2V-1.3B \
        --lora_path  /path/to/vigor_lora.safetensors
"""

import argparse
import logging

from common import DEFAULT_PROMPTS, load_prompts
from dpo.inference import DPOInferenceConfig, run_comparison


def parse_args():
    parser = argparse.ArgumentParser(
        description="VIGOR DPO inference (LoRA vs. baseline comparison)"
    )
    parser.add_argument(
        "--model_path", required=True, help="Local Wan2.1 base model directory."
    )
    parser.add_argument(
        "--lora_path",
        default=None,
        help="Geometry-aligned LoRA checkpoint. If omitted, only the baseline runs.",
    )
    parser.add_argument(
        "--data_path",
        default=DEFAULT_PROMPTS,
        help="Prompt file (one prompt per line). Defaults to prompts/demos.txt.",
    )
    parser.add_argument("--output_dir", default="outputs/dpo")
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument(
        "--no_baseline",
        action="store_true",
        help="Skip the baseline pass (generate LoRA videos only).",
    )
    parser.add_argument(
        "--no_stack",
        action="store_true",
        help="Do not write side-by-side comparison videos.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    prompts = load_prompts(args.data_path)
    logging.info("Loaded %d prompt(s) from %s", len(prompts), args.data_path)

    cfg = DPOInferenceConfig(
        model_path=args.model_path,
        lora_path=args.lora_path,
        prompts=prompts,
        output_dir=args.output_dir,
        lora_alpha=args.lora_alpha,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=args.fps,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        run_baseline=not args.no_baseline,
        stack=not args.no_stack,
    )
    run_comparison(cfg)


if __name__ == "__main__":
    main()
