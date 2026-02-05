"""
Video Evaluation Script for Multiple Metrics
Evaluates videos in subfolders and generates ranked results for each metric.
"""
import os
import re
import json
import logging
import argparse
import sys
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

import pathlib
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
THIRD_PARTY_DIR = SCRIPT_DIR / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

from metrics.evaluator.epipolar import EpipolarEvaluator
from metrics.evaluator.reprojection import ReprojectionEvaluator


# --------------------------
# Metric Configuration
# --------------------------
METRIC_DIRECTIONS: Dict[str, bool] = {
    "epipolar_consistency": False,   # Sampson distance: lower is better
    "reprojection_error": False,     # pixels: lower is better
}


def setup_logging(log_file: str = 'evaluation.log'):
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )


def parse_seed_from_filename(path: str) -> int:
    """
    Best-effort parse seed from filename.
    Examples:
      seed_0.mp4 -> 0
      00012.mp4  -> 12
      sample-seed=37_xxx.mp4 -> 37
    If none found, returns -1.
    """
    name = os.path.basename(path)
    # Try explicit seed patterns first
    m = re.search(r"(?:seed[_=\-]?)\s*(\d+)", name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Fallback to any number in filename
    m2 = re.search(r"(\d+)", name)
    if m2:
        return int(m2.group(1))
    return -1


def list_videos(folder: str) -> List[str]:
    """List all video files in a folder."""
    exts = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
    videos: List[str] = []
    
    if not os.path.isdir(folder):
        logging.warning(f"Folder not found: {folder}")
        return videos
    
    for fn in os.listdir(folder):
        p = os.path.join(folder, fn)
        if not os.path.isfile(p):
            continue
        if os.path.splitext(fn.lower())[1] in exts:
            videos.append(p)
    
    videos.sort()
    return videos


def list_subfolders(root: str) -> List[str]:
    """List direct child subfolders under root, sorted by name."""
    if not os.path.isdir(root):
        return []
    subs: List[str] = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            subs.append(p)
    subs.sort()
    return subs


def parse_metrics_arg(metrics_arg: str) -> Optional[List[str]]:
    """Parse --metrics argument. Return None if user didn't specify."""
    if not metrics_arg or not metrics_arg.strip():
        return None
    items = [x.strip() for x in metrics_arg.split(",") if x.strip()]
    return items if items else None


def make_evaluators_from_config(
    cfg: Dict[str, Any],
    only_metrics: Optional[List[str]] = None,
    save_error_mask: bool = False,
    error_mask_dir: Optional[str] = None,
) -> Tuple[List[Any], Dict[str, str]]:
    """
    Build evaluator instances for epipolar + reprojection.

    Args:
        cfg: Configuration dictionary
        only_metrics: List of metric names to enable (None = enable all)
        save_error_mask: Whether to save error masks for reprojection metric
        error_mask_dir: Directory to save error masks

    Returns:
        evaluators: List of instantiated evaluator objects
        skipped: Dict of metric_name -> reason for skipping
    """
    skipped: Dict[str, str] = {}
    evaluators: List[Any] = []

    def want(metric_name: str) -> bool:
        if only_metrics is None:
            return True
        return metric_name in set(only_metrics)

    # Epipolar consistency
    if want("epipolar_consistency"):
        try:
            if "epipolar_consistency" in cfg:
                evaluators.append(EpipolarEvaluator.from_config(cfg["epipolar_consistency"]))
                logging.info("Initialized EpipolarEvaluator from config")
            else:
                evaluators.append(EpipolarEvaluator())
                logging.info("Initialized EpipolarEvaluator with defaults")
        except Exception as e:
            skipped["epipolar_consistency"] = f"initialization failed: {e}"
            logging.error(f"Failed to initialize EpipolarEvaluator: {e}")

    # Reprojection error
    if want("reprojection_error"):
        try:
            if "reprojection_error" in cfg:
                # Add error mask saving parameters to config
                reproj_cfg = cfg["reprojection_error"].copy()
                reproj_cfg["save_error_mask"] = save_error_mask
                reproj_cfg["error_mask_dir"] = error_mask_dir
                evaluators.append(ReprojectionEvaluator.from_config(reproj_cfg))
                logging.info("Initialized ReprojectionEvaluator from config")
            else:
                evaluators.append(ReprojectionEvaluator(
                    save_error_mask=save_error_mask,
                    error_mask_dir=error_mask_dir
                ))
                logging.info("Initialized ReprojectionEvaluator with defaults")
        except Exception as e:
            skipped["reprojection_error"] = f"initialization failed: {e}"
            logging.error(f"Failed to initialize ReprojectionEvaluator: {e}")

    # Check for unsupported metrics
    if only_metrics is not None:
        supported = {"epipolar_consistency", "reprojection_error"}
        for m in only_metrics:
            if m not in supported:
                skipped[m] = "unsupported metric"
                logging.warning(f"Metric '{m}' is not supported")

    return evaluators, skipped


def sort_records(metric_name: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort records from best to worst based on metric direction.
    
    Args:
        metric_name: Name of the metric
        records: List of evaluation records
    
    Returns:
        Sorted list of records (best to worst)
    """
    higher_is_better = METRIC_DIRECTIONS.get(metric_name, True)

    def is_valid(score):
        return (
            score is not None 
            and isinstance(score, (int, float)) 
            and np.isfinite(score) 
            and score != -1
        )

    def key_fn(r):
        score = r.get("score", None)
        valid = is_valid(score)
        if not valid:
            # Invalid scores go to the end
            return (1, 0.0)
        s = float(score)
        # Sort: valid first (0), then by score
        return (0, -s if higher_is_better else s)

    return sorted(records, key=key_fn)


def load_previous_results(output_json: str) -> Dict[str, Dict[str, Any]]:
    """
    Load previously computed results from output JSON.
    
    Returns:
        Dictionary mapping video_name -> {metric_name -> record}
    """
    if not os.path.exists(output_json):
        return {}
    
    try:
        with open(output_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        previous = {}
        ranked = data.get("ranked", {})
        
        for metric_name, records in ranked.items():
            for rec in records:
                video = rec.get("video", "")
                if video:
                    if video not in previous:
                        previous[video] = {}
                    previous[video][metric_name] = rec
        
        logging.info(f"Loaded {len(previous)} previously processed videos from {output_json}")
        return previous
        
    except Exception as e:
        logging.warning(f"Could not load previous results from {output_json}: {e}")
        return {}


def evaluate_one_folder(
    folder: str,
    output_json: str,
    config_json: str = "",
    metrics: str = "",
    resume: bool = True,
    save_error_mask: bool = False,
) -> None:
    """
    Evaluate all videos in one folder and write results to output_json.
    
    Args:
        folder: Input folder containing videos
        output_json: Output JSON file path
        config_json: Optional config file for evaluator parameters
        metrics: Comma-separated list of metrics to evaluate
        resume: Whether to resume from previous results
        save_error_mask: Whether to save error masks for reprojection metric
    """
    logging.info(f"Processing folder: {folder}")
    
    # List videos
    videos = list_videos(folder)
    if len(videos) == 0:
        raise RuntimeError(f"No videos found in: {folder}")
    
    logging.info(f"Found {len(videos)} videos")
    
    # Load config
    cfg: Dict[str, Any] = {}
    if config_json and os.path.exists(config_json):
        with open(config_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        logging.info(f"Loaded config from {config_json}")
    
    # Parse metrics to evaluate
    only_metrics = parse_metrics_arg(metrics)
    
    # Setup error mask directory (same parent as output JSON)
    error_mask_dir = None
    if save_error_mask:
        output_dir = os.path.dirname(os.path.abspath(output_json))
        error_mask_dir = os.path.join(output_dir, "error_masks")
        os.makedirs(error_mask_dir, exist_ok=True)
        logging.info(f"Error masks will be saved to: {error_mask_dir}")
    
    # Initialize evaluators
    evaluators, skipped = make_evaluators_from_config(
        cfg, 
        only_metrics=only_metrics,
        save_error_mask=save_error_mask,
        error_mask_dir=error_mask_dir,
    )
    
    if len(evaluators) == 0:
        raise RuntimeError(
            f"No evaluators were instantiated.\n"
            f"Requested metrics: {only_metrics}\n"
            f"Skipped: {skipped}"
        )
    
    enabled_names = [ev.name for ev in evaluators]
    logging.info(f"Enabled metrics: {enabled_names}")
    if skipped:
        logging.info("Skipped metrics:")
        for k, v in skipped.items():
            logging.info(f"  - {k}: {v}")
    
    # Load previous results if resuming
    previous_results = {}
    if resume:
        previous_results = load_previous_results(output_json)
    
    # Track results by metric
    results: Dict[str, List[Dict[str, Any]]] = {name: [] for name in enabled_names}
    
    # Process each video
    processed_count = 0
    skipped_count = 0
    
    pbar = tqdm(videos, desc="Evaluating videos")
    for video_path in pbar:
        vname = os.path.basename(video_path)
        seed = parse_seed_from_filename(video_path)
        
        pbar.set_description(f"Processing: {vname[:30]}")
        
        for ev in evaluators:
            metric_name = ev.name
            
            # Check if already processed
            if vname in previous_results and metric_name in previous_results[vname]:
                rec = previous_results[vname][metric_name]
                results[metric_name].append(rec)
                skipped_count += 1
                logging.debug(f"Skipped {vname} for {metric_name} (already processed)")
                continue
            
            # Evaluate video
            try:
                score, details = ev.evaluate_video(video_path)
                rec_score = float(score) if isinstance(score, (int, float)) else -1.0
                logging.info(f"{vname}: {metric_name} = {rec_score:.4f}")
                
            except Exception as e:
                rec_score = -1.0
                details = {"error": str(e)}
                logging.error(f"Error evaluating {vname} with {metric_name}: {e}")
            
            # Create record
            rec = {
                "seed": int(seed),
                "video": vname,
                "score": rec_score,
                "details": details,
            }
            results[metric_name].append(rec)
            processed_count += 1
        
        # Save intermediate results periodically
        if processed_count > 0 and processed_count % 10 == 0:
            save_results(folder, videos, enabled_names, skipped, results, output_json, save_error_mask, error_mask_dir)
            logging.info(f"Saved intermediate results ({processed_count} evaluations)")
    
    pbar.close()
    
    # Final save
    save_results(folder, videos, enabled_names, skipped, results, output_json, save_error_mask, error_mask_dir)
    
    logging.info(f"Completed: {processed_count} evaluations, {skipped_count} skipped")
    logging.info(f"Results saved to: {output_json}")
    if save_error_mask and error_mask_dir:
        logging.info(f"Error masks saved to: {error_mask_dir}")


def save_results(
    folder: str,
    videos: List[str],
    enabled_names: List[str],
    skipped: Dict[str, str],
    results: Dict[str, List[Dict[str, Any]]],
    output_json: str,
    save_error_mask: bool = False,
    error_mask_dir: Optional[str] = None,
) -> None:
    """Save evaluation results to JSON file."""
    # Sort results by metric
    ranked: Dict[str, Any] = {}
    for metric_name, recs in results.items():
        ranked[metric_name] = sort_records(metric_name, recs)
    
    # Prepare output
    output_data = {
        "video_folder": os.path.abspath(folder),
        "num_videos": len(videos),
        "enabled_metrics": enabled_names,
        "skipped_metrics": skipped,
        "metric_directions_higher_is_better": METRIC_DIRECTIONS,
        "ranked": ranked,
    }
    
    # Add error mask info if enabled
    if save_error_mask and error_mask_dir:
        output_data["error_mask_dir"] = os.path.abspath(error_mask_dir)
        output_data["error_mask_format"] = "npz"
        output_data["error_mask_info"] = {
            "description": "Reprojection error masks stored as .npz files",
            "array_name": "error_mask",
            "shape": "(S, Hp, Wp) - frames x patch_height x patch_width",
            "dtype": "float32",
            "unit": "pixels",
        }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    
    # Write results
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate videos in subfolders for multiple metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default="input/static",
        help="Root folder containing subfolders to evaluate",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="output/static",
        help="Root folder to write JSON results",
    )
    parser.add_argument(
        "--config_json",
        type=str,
        default="",
        help="Optional path to a JSON config controlling evaluator params",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="",
        help="Comma-separated metric names: epipolar_consistency,reprojection_error (empty = all)",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="metrics.json",
        help="Output JSON filename under each output folder",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="evaluation.log",
        help="Log file path",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable resuming from previous results",
    )
    parser.add_argument(
        "--save_error_mask",
        action="store_true",
        help="Save error masks for reprojection metric (stored as .npz in error_masks/ subfolder)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file)
    logging.info("=" * 80)
    logging.info("Starting video evaluation")
    logging.info(f"Input root: {args.input_root}")
    logging.info(f"Output root: {args.output_root}")
    logging.info(f"Config: {args.config_json or 'None'}")
    logging.info(f"Metrics: {args.metrics or 'All'}")
    logging.info(f"Resume: {not args.no_resume}")
    logging.info(f"Save error masks: {args.save_error_mask}")
    logging.info("=" * 80)
    
    # List subfolders
    subfolders = list_subfolders(args.input_root)
    if len(subfolders) == 0:
        raise RuntimeError(f"No subfolders found under: {args.input_root}")
    
    logging.info(f"Found {len(subfolders)} subfolders to process")
    
    # Process each subfolder
    for idx, folder in enumerate(subfolders, 1):
        name = os.path.basename(folder.rstrip("/"))
        out_dir = os.path.join(args.output_root, name)
        out_json = os.path.join(out_dir, args.output_suffix)
        
        logging.info("")
        logging.info("=" * 80)
        logging.info(f"[{idx}/{len(subfolders)}] Evaluating: {name}")
        logging.info(f"Input folder : {folder}")
        logging.info(f"Output JSON  : {out_json}")
        logging.info("=" * 80)
        
        try:
            evaluate_one_folder(
                folder=folder,
                output_json=out_json,
                config_json=args.config_json,
                metrics=args.metrics,
                resume=not args.no_resume,
                save_error_mask=args.save_error_mask,
            )
        except Exception as e:
            logging.error(f"Failed to process {name}: {e}", exc_info=True)
            continue
    
    logging.info("")
    logging.info("=" * 80)
    logging.info("All evaluations finished!")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()