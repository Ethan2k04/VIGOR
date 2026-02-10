"""
Video Evaluation Script for Multiple Metrics
Evaluates videos in subfolders and generates ranked results for each metric.
Supports hierarchical folder structure: root/category/promptXXXXX/
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
    # Create directory for log file if it doesn't exist
    log_dir = os.path.dirname(os.path.abspath(log_file))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )


def set_device(device: str):
    """Set the device for computation."""
    import torch
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            logging.warning(f"CUDA not available, falling back to CPU")
            device = "cpu"
        else:
            # Set CUDA device
            device_id = int(device.split(":")[-1]) if ":" in device else 0
            torch.cuda.set_device(device_id)
            logging.info(f"Using device: {torch.cuda.get_device_name(device_id)} ({device})")
    else:
        logging.info(f"Using device: CPU")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1] if ":" in device else "0"
    return device


def parse_seed_from_filename(path: str) -> int:
    """
    Best-effort parse seed from filename.
    Examples:
      seed_0.mp4 -> 0
      seed00.mp4  -> 0
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


def extract_prompt_index_from_folder(folder_name: str) -> Optional[int]:
    """
    Extract prompt index from folder name like 'prompt02048' -> 2048
    """
    m = re.search(r"prompt(\d+)", folder_name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


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


def list_prompt_folders_in_category(category_path: str) -> List[Tuple[str, int]]:
    """
    List prompt folders in a category directory.
    Returns list of (folder_path, prompt_index) tuples.
    """
    if not os.path.isdir(category_path):
        return []
    
    folders_with_indices = []
    for name in os.listdir(category_path):
        p = os.path.join(category_path, name)
        if os.path.isdir(p):
            idx = extract_prompt_index_from_folder(name)
            if idx is not None:
                folders_with_indices.append((p, idx))
    
    # Sort by prompt index
    folders_with_indices.sort(key=lambda x: x[1])
    return folders_with_indices


def list_all_prompt_folders(input_root: str, categories: Optional[List[str]] = None) -> List[Tuple[str, str, int]]:
    """
    List all prompt folders across categories.
    
    Args:
        input_root: Root directory containing category folders
        categories: List of category names to process (None = auto-detect)
    
    Returns:
        List of (category_name, folder_path, prompt_index) tuples, sorted by category then index
    """
    if not os.path.isdir(input_root):
        raise RuntimeError(f"Input root not found: {input_root}")
    
    # Auto-detect categories if not specified
    if categories is None:
        categories = []
        for name in os.listdir(input_root):
            p = os.path.join(input_root, name)
            if os.path.isdir(p):
                categories.append(name)
        categories.sort()
    
    all_folders = []
    for category in categories:
        category_path = os.path.join(input_root, category)
        if not os.path.isdir(category_path):
            logging.warning(f"Category folder not found: {category_path}")
            continue
        
        prompt_folders = list_prompt_folders_in_category(category_path)
        for folder_path, prompt_idx in prompt_folders:
            all_folders.append((category, folder_path, prompt_idx))
    
    return all_folders


def parse_metrics_list(metrics_cfg: Any) -> Optional[List[str]]:
    """Parse metrics configuration. Return None if empty."""
    if metrics_cfg is None or metrics_cfg == "":
        return None
    
    if isinstance(metrics_cfg, list):
        items = [str(x).strip() for x in metrics_cfg if str(x).strip()]
        return items if items else None
    
    if isinstance(metrics_cfg, str):
        items = [x.strip() for x in metrics_cfg.split(",") if x.strip()]
        return items if items else None
    
    return None


def make_evaluators_from_config(
    cfg: Dict[str, Any],
    only_metrics: Optional[List[str]] = None,
    save_sampling_mask: bool = False,
    sampling_mask_dir: Optional[str] = None,
) -> Tuple[List[Any], Dict[str, str]]:
    """
    Build evaluator instances for epipolar + reprojection.

    Args:
        cfg: Configuration dictionary
        only_metrics: List of metric names to enable (None = enable all)
        save_sampling_mask: Whether to save sampling masks for reprojection metric
        sampling_mask_dir: Directory to save sampling masks

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
                # Add sampling mask saving parameters to config
                reproj_cfg = cfg["reprojection_error"].copy()
                reproj_cfg["save_sampling_mask"] = save_sampling_mask
                reproj_cfg["sampling_mask_dir"] = sampling_mask_dir
                evaluators.append(ReprojectionEvaluator.from_config(reproj_cfg))
                logging.info("Initialized ReprojectionEvaluator from config")
            else:
                evaluators.append(ReprojectionEvaluator(
                    save_sampling_mask=save_sampling_mask,
                    sampling_mask_dir=sampling_mask_dir
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
        logging.warning(f"Failed to load previous results from {output_json}: {e}")
        return {}


def evaluate_one_folder(
    folder: str,
    output_json: str,
    cfg: Dict[str, Any],
    only_metrics: Optional[List[str]],
    resume: bool,
    save_sampling_mask: bool,
) -> None:
    """
    Evaluate all videos in one folder for selected metrics.
    
    Args:
        folder: Input folder containing videos
        output_json: Output JSON file path
        cfg: Configuration dictionary
        only_metrics: List of metric names to enable
        resume: Whether to resume from previous results
        save_sampling_mask: Whether to save sampling masks
    """
    # List videos
    videos = list_videos(folder)
    if len(videos) == 0:
        logging.warning(f"No videos found in: {folder}")
        return
    
    logging.info(f"Found {len(videos)} videos")
    
    # Setup sampling mask directory if enabled
    sampling_mask_dir = None
    if save_sampling_mask:
        # Create output directory
        output_dir = os.path.dirname(os.path.abspath(output_json))
        os.makedirs(output_dir, exist_ok=True)
        sampling_mask_dir = output_dir  # Save masks in the same prompt folder
    
    # Initialize evaluators
    evaluators, skipped = make_evaluators_from_config(
        cfg, 
        only_metrics=only_metrics,
        save_sampling_mask=save_sampling_mask,
        sampling_mask_dir=sampling_mask_dir,
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
            save_results(folder, videos, enabled_names, skipped, results, output_json, save_sampling_mask, sampling_mask_dir)
            logging.info(f"Saved intermediate results ({processed_count} evaluations)")
    
    pbar.close()
    
    # Final save
    save_results(folder, videos, enabled_names, skipped, results, output_json, save_sampling_mask, sampling_mask_dir)
    
    logging.info(f"Completed: {processed_count} evaluations, {skipped_count} skipped")
    logging.info(f"Results saved to: {output_json}")
    if save_sampling_mask and sampling_mask_dir:
        logging.info(f"Sampling masks saved to: {sampling_mask_dir}")


def save_results(
    folder: str,
    videos: List[str],
    enabled_names: List[str],
    skipped: Dict[str, str],
    results: Dict[str, List[Dict[str, Any]]],
    output_json: str,
    save_sampling_mask: bool = False,
    sampling_mask_dir: Optional[str] = None,
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
    
    # Add sampling mask info if enabled
    if save_sampling_mask and sampling_mask_dir:
        output_data["sampling_mask_dir"] = os.path.abspath(sampling_mask_dir)
        output_data["sampling_mask_format"] = "npz"
        output_data["sampling_mask_info"] = {
            "description": "Binary sampling masks indicating selected patches based on VGGT attention",
            "array_name": "sampling_mask",
            "shape": "(S, Hp, Wp) - frames x patch_height x patch_width",
            "dtype": "bool",
            "values": "True = selected patch, False = not selected",
        }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    
    # Write results
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate videos in hierarchical prompt folders (root/category/promptXXXXX/) for multiple metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to configuration JSON file",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="Start prompt index (inclusive). Process prompts from this index onwards.",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="End prompt index (inclusive). Process prompts up to and including this index.",
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='GPU device to use (e.g., cuda:0, cuda:1, or cpu). Default: cuda:0'
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated list of categories to process (overrides config). E.g., 'static_indoor,dynamic_indoor'",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Log file path (default: evaluation.log in output_root)",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Extract settings from config
    input_root = cfg.get("input_root", "output")
    output_root = cfg.get("output_root", "output_metrics")
    
    # Categories: command line > config > auto-detect
    if args.categories:
        categories = [x.strip() for x in args.categories.split(",") if x.strip()]
    else:
        categories = cfg.get("categories", None)
    
    metrics_cfg = cfg.get("metrics", None)
    save_sampling_mask = cfg.get("save_sampling_mask", False)
    output_suffix = cfg.get("output_suffix", "metrics.json")
    resume = cfg.get("resume", True)
    
    # Parse metrics
    only_metrics = parse_metrics_list(metrics_cfg)
    
    # Setup device
    device = args.device
    set_device(device)
    
    # Setup logging
    if args.log_file is None:
        log_file = os.path.join(output_root, "evaluation.log")
    else:
        log_file = args.log_file
    
    setup_logging(log_file)
    
    logging.info("=" * 80)
    logging.info("Starting video evaluation")
    logging.info(f"Config file: {args.config}")
    logging.info(f"Input root: {input_root}")
    logging.info(f"Output root: {output_root}")
    logging.info(f"Categories: {categories or 'Auto-detect'}")
    logging.info(f"Device: {device}")
    logging.info(f"Metrics: {only_metrics or 'All'}")
    logging.info(f"Save sampling masks: {save_sampling_mask}")
    logging.info(f"Resume: {resume}")
    
    if args.start_index is not None or args.end_index is not None:
        logging.info(f"Prompt index range: [{args.start_index or 'start'}, {args.end_index or 'end'}]")
    
    logging.info("=" * 80)
    
    # List all prompt folders across categories
    all_folders = list_all_prompt_folders(input_root, categories)
    if len(all_folders) == 0:
        raise RuntimeError(f"No prompt folders found under: {input_root}")
    
    # Filter by index range if specified
    filtered_folders = []
    for category, folder_path, prompt_idx in all_folders:
        # Check index range
        if args.start_index is not None and prompt_idx < args.start_index:
            continue
        if args.end_index is not None and prompt_idx > args.end_index:
            continue
        
        filtered_folders.append((category, folder_path, prompt_idx))
    
    if len(filtered_folders) == 0:
        if args.start_index is not None or args.end_index is not None:
            raise RuntimeError(
                f"No prompt folders found in index range "
                f"[{args.start_index or 'start'}, {args.end_index or 'end'}]"
            )
        else:
            raise RuntimeError(f"No valid prompt folders found under: {input_root}")
    
    logging.info(f"Found {len(all_folders)} total prompt folders across all categories")
    logging.info(f"Processing {len(filtered_folders)} folders after filtering")
    
    # Group by category for summary
    category_counts = {}
    for category, _, _ in filtered_folders:
        category_counts[category] = category_counts.get(category, 0) + 1
    
    logging.info("Folders per category:")
    for category, count in sorted(category_counts.items()):
        logging.info(f"  - {category}: {count} folders")
    
    # Process each filtered folder
    for idx, (category, folder, prompt_idx) in enumerate(filtered_folders, 1):
        folder_name = os.path.basename(folder.rstrip("/"))
        
        # Create output directory: output_root/category/promptXXXXX/
        out_dir = os.path.join(output_root, category, f"prompt{prompt_idx:05d}")
        out_json = os.path.join(out_dir, output_suffix)
        
        logging.info("")
        logging.info("=" * 80)
        logging.info(f"[{idx}/{len(filtered_folders)}] Evaluating: {category}/{folder_name} (index: {prompt_idx})")
        logging.info(f"Input folder : {folder}")
        logging.info(f"Output folder: {out_dir}")
        logging.info(f"Output JSON  : {out_json}")
        logging.info("=" * 80)
        
        try:
            evaluate_one_folder(
                folder=folder,
                output_json=out_json,
                cfg=cfg,
                only_metrics=only_metrics,
                resume=resume,
                save_sampling_mask=save_sampling_mask,
            )
        except Exception as e:
            logging.error(f"Failed to process {category}/{folder_name}: {e}", exc_info=True)
            continue
    
    logging.info("")
    logging.info("=" * 80)
    logging.info("All evaluations finished!")
    logging.info(f"Results saved to: {output_root}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()