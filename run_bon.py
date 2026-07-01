"""Best-of-N inference entrypoint for VIGOR.

Generate ``N`` candidate videos per prompt with Wan2.1, score each with the
geometry reward, and keep the best. Prompts default to ``prompts/demos.txt``.

Example
-------
    python run_bon.py \
        --model_path /path/to/Wan2.1-T2V-1.3B \
        --num_candidates 8 \
        --metric reprojection
"""

import argparse
import logging

from common import DEFAULT_PROMPTS, load_prompts
from bon.best_of_n import SUPPORTED_METRICS, BoNConfig, best_of_n


def parse_args():
    parser = argparse.ArgumentParser(description="VIGOR Best-of-N sampling")
    parser.add_argument(
        "--model_path", required=True, help="Local Wan2.1 model directory."
    )
    parser.add_argument(
        "--data_path",
        default=DEFAULT_PROMPTS,
        help="Prompt file (one prompt per line). Defaults to prompts/demos.txt.",
    )
    parser.add_argument("--output_dir", default="outputs/bon")
    parser.add_argument(
        "--num_candidates", type=int, default=4, help="Number of samples per prompt (N)."
    )
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument(
        "--metric", default="reprojection", choices=SUPPORTED_METRICS
    )
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument(
        "--enable_sky_filter",
        action="store_true",
        help="Use skyseg.onnx sky masking inside the geometry reward.",
    )
    parser.add_argument(
        "--save_all_candidates",
        action="store_true",
        help="Keep every candidate video instead of only the winner.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    prompts = load_prompts(args.data_path)
    logging.info("Loaded %d prompt(s) from %s", len(prompts), args.data_path)

    cfg = BoNConfig(
        model_path=args.model_path,
        prompts=prompts,
        output_dir=args.output_dir,
        num_candidates=args.num_candidates,
        base_seed=args.base_seed,
        metric=args.metric,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=args.fps,
        negative_prompt=args.negative_prompt,
        enable_sky_filter=args.enable_sky_filter,
        save_all_candidates=args.save_all_candidates,
    )
    best_of_n(cfg)


if __name__ == "__main__":
    main()
