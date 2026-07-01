"""
Video evaluation with the VIGOR geometry rewards.

Point the evaluator at a folder of videos::

    input_root/
        prompt_000.mp4
        prompt_001.mp4
        ...

Every video is scored with the metrics selected in the JSON config
(see ``config/config.json``); the results are written to a single JSON file
with per-video scores and a summary.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any
from datetime import datetime

import numpy as np

# Make the repository root importable so the `rewards` package resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from rewards.evaluator.epipolar import EpipolarEvaluator
from rewards.evaluator.reproj_pts import ReprojectionEvaluator
from rewards.evaluator.reproj_pix import ReprojectionVanillaEvaluator


# Metrics for which a lower score is better.
LOWER_IS_BETTER = {"epipolar_consistency", "reprojection_error", "reprojection_vanilla"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def set_device(device: str):
    """Select the CUDA device (or fall back to CPU)."""
    import torch

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU")
            device = "cpu"
        else:
            device_id = int(device.split(":")[-1]) if ":" in device else 0
            torch.cuda.set_device(device_id)
            print(f"Using device: {torch.cuda.get_device_name(device_id)} ({device})")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    else:
        print("Using device: CPU")
    return device


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a JSON configuration file."""
    with open(config_path, "r") as f:
        return json.load(f)


def get_evaluator(metric_name: str, config: Dict[str, Any],
                  save_sampling_mask: bool = False, sampling_mask_dir: str = None):
    """Instantiate an evaluator from its config block."""
    if metric_name == "epipolar_consistency":
        return EpipolarEvaluator.from_config(config.get("epipolar_consistency", {}))
    elif metric_name == "reprojection_error":
        cfg = config.get("reprojection_error", {})
        cfg["save_sampling_mask"] = save_sampling_mask
        if sampling_mask_dir:
            cfg["sampling_mask_dir"] = sampling_mask_dir
        return ReprojectionEvaluator.from_config(cfg)
    elif metric_name == "reprojection_vanilla":
        return ReprojectionVanillaEvaluator.from_config(config.get("reprojection_vanilla", {}))
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


def find_videos(input_root: str) -> List[Dict[str, Any]]:
    """Find all video files directly inside ``input_root``."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    all_videos = []

    if not os.path.exists(input_root):
        print(f"ERROR: Input root not found: {input_root}")
        return all_videos

    for filename in os.listdir(input_root):
        file_path = os.path.join(input_root, filename)
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in video_extensions):
            name = os.path.splitext(filename)[0]
            all_videos.append({'path': file_path, 'name': name, 'filename': filename})

    all_videos.sort(key=lambda x: x['name'])
    return all_videos


def _compute_summary(video_results: Dict[str, Any], metrics: List[str]) -> Dict[str, Any]:
    """Compute summary statistics across all videos for each metric."""
    summary = {}
    for metric in metrics:
        scores = [
            float(result[metric]["score"])
            for result in video_results.values()
            if metric in result and result[metric].get("score") is not None
        ]
        if scores:
            summary[metric] = {
                "count":  len(scores),
                "mean":   float(np.mean(scores)),
                "median": float(np.median(scores)),
                "std":    float(np.std(scores)),
                "min":    float(np.min(scores)),
                "max":    float(np.max(scores)),
                "note":   "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
            }
        else:
            summary[metric] = {"count": 0}
    return summary


def _save_output(video_results: Dict[str, Any], metrics: List[str], output_path: str):
    """Save results JSON: per-video details + compact score table + summary."""
    per_video_scores = {
        name: {m: result[m].get("score") if m in result else None for m in metrics}
        for name, result in video_results.items()
    }
    output_data = {
        "videos": video_results,
        "per_video_scores": per_video_scores,
        "summary": _compute_summary(video_results, metrics)
    }
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)


def evaluate_videos(
    input_root: str,
    output_path: str,
    metrics: List[str],
    config: Dict[str, Any],
    resume: bool = True,
    save_sampling_mask: bool = False,
):
    print(f"\n{'='*80}")
    print("Scanning for videos...")
    print(f"{'='*80}")

    all_videos = find_videos(input_root)
    if not all_videos:
        print("No videos found!")
        return {}

    print(f"Found {len(all_videos)} videos:")
    for v in all_videos:
        print(f"  {v['filename']}")

    # Load existing results if resuming
    existing_results: Dict[str, Any] = {}
    if resume and os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                existing_results = json.load(f).get("videos", {})
            print(f"Loaded {len(existing_results)} existing results from {output_path}")
        except Exception as e:
            print(f"Could not load existing results: {e}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    video_results: Dict[str, Any] = dict(existing_results)
    processed_count = skipped_count = 0

    pbar = tqdm(total=len(all_videos), desc="Overall Progress", unit="video",
                position=0, leave=True,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
                           '[{elapsed}<{remaining}, {rate_fmt}] {postfix}')

    for video_info in all_videos:
        name = video_info['name']
        video_path = video_info['path']

        pbar.set_description(f"[{datetime.now().strftime('%H:%M:%S')}] {name}")

        # Resume: skip if all metrics already have a score
        if resume and name in video_results:
            if all(m in video_results[name] and video_results[name][m].get("score") is not None
                   for m in metrics):
                skipped_count += 1
                pbar.set_postfix_str(f"Skipped | P:{processed_count} S:{skipped_count}")
                pbar.update(1)
                continue

        single_result: Dict[str, Any] = {
            'filename': video_info['filename'],
            'path': video_path,
        }

        success = True
        for metric_name in metrics:
            try:
                sampling_mask_dir = None
                if save_sampling_mask and metric_name == "reprojection_error":
                    sampling_mask_dir = os.path.join(
                        os.path.dirname(os.path.abspath(output_path)),
                        "sampling_masks", name)
                    os.makedirs(sampling_mask_dir, exist_ok=True)

                evaluator = get_evaluator(metric_name, config, save_sampling_mask, sampling_mask_dir)
                score, detailed_metrics = evaluator.evaluate_video(video_path)
                single_result[metric_name] = {"score": float(score), "details": detailed_metrics}
            except Exception as e:
                success = False
                single_result[metric_name] = {"score": None, "error": str(e)}

        video_results[name] = single_result
        _save_output(video_results, metrics, output_path)  # incremental save

        processed_count += 1
        pbar.set_postfix_str(f"{'✓' if success else '⚠'} | P:{processed_count} S:{skipped_count}")
        pbar.update(1)

    pbar.close()

    print(f"\n{'='*80}")
    print(f"Processed: {processed_count}  Skipped: {skipped_count}  Total: {len(all_videos)}")
    print(f"{'='*80}")
    _save_output(video_results, metrics, output_path)
    print(f"✓ Saved results to: {output_path}")
    return video_results


def print_summary(video_results: Dict[str, Any], metrics: List[str]):
    summary = _compute_summary(video_results, metrics)
    print(f"\n{'='*80}")
    print("Summary Statistics")
    print(f"{'='*80}")
    for metric, stats in summary.items():
        if stats.get("count", 0) == 0:
            print(f"\n{metric}: No results")
            continue
        print(f"\n{metric} ({stats.get('note', '')}):")
        print(f"  Count: {stats['count']}  Mean: {stats['mean']:.4f}  "
              f"Median: {stats['median']:.4f}  Std: {stats['std']:.4f}  "
              f"Min: {stats['min']:.4f}  Max: {stats['max']:.4f}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a folder of videos with the VIGOR geometry rewards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to the JSON configuration file.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (e.g. 'cuda:0', 'cpu').")
    args = parser.parse_args()

    device = set_device(args.device)
    config = load_config(args.config)

    input_root         = config.get("input_root", "input")
    output_path        = config.get("output_path", "output/results.json")
    metrics            = config.get("metrics", [])
    resume             = config.get("resume", True)
    save_sampling_mask = config.get("save_sampling_mask", False)

    print(f"\n{'='*80}")
    print("Video Evaluation Pipeline")
    print(f"  Input:   {input_root}")
    print(f"  Output:  {output_path}")
    print(f"  Metrics: {metrics}  Device: {device}  Resume: {resume}")
    print(f"{'='*80}")

    video_results = evaluate_videos(
        input_root=input_root,
        output_path=output_path,
        metrics=metrics,
        config=config,
        resume=resume,
        save_sampling_mask=save_sampling_mask,
    )

    if not video_results:
        print("No videos were evaluated.")
        return

    print_summary(video_results, metrics)
    print(f"\nDone! Results: {output_path}\n")


if __name__ == "__main__":
    main()
