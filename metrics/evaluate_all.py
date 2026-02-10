import os
import json
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
import re
from datetime import datetime

import sys
import os
# Add third_party to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'third_party'))

from tqdm import tqdm

from metrics.evaluator.epipolar import EpipolarEvaluator
from metrics.evaluator.reprojection import ReprojectionEvaluator
from metrics.evaluator.reprojection_vanilla import ReprojectionVanillaEvaluator


def set_device(device: str):
    """Set the device for computation."""
    import torch
    
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"WARNING: CUDA not available, falling back to CPU")
            device = "cpu"
        else:
            # Extract device ID
            device_id = int(device.split(":")[-1]) if ":" in device else 0
            torch.cuda.set_device(device_id)
            print(f"Using device: {torch.cuda.get_device_name(device_id)} ({device})")
            # Set environment variable
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    else:
        print(f"Using device: CPU")
    
    return device


def extract_prompt_index_from_folder(folder_name: str) -> Optional[int]:
    """
    Extract prompt index from folder name like 'prompt02048' -> 2048
    Returns None if no valid index found.
    """
    # Match patterns like: prompt02048, prompt_02048, prompt-02048, etc.
    match = re.search(r"prompt[_-]?(\d+)", folder_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def get_evaluator(metric_name: str, config: Dict[str, Any], save_sampling_mask: bool = False, sampling_mask_dir: str = None):
    """Get evaluator instance based on metric name."""
    if metric_name == "epipolar_consistency":
        return EpipolarEvaluator.from_config(config.get("epipolar_consistency", {}))
    elif metric_name == "reprojection_error":
        # Pass save_sampling_mask and sampling_mask_dir to ReprojectionEvaluator
        cfg = config.get("reprojection_error", {})
        cfg['save_sampling_mask'] = save_sampling_mask
        if sampling_mask_dir:
            cfg['sampling_mask_dir'] = sampling_mask_dir
        return ReprojectionEvaluator.from_config(cfg)
    elif metric_name == "reprojection_vanilla":
        return ReprojectionVanillaEvaluator.from_config(config.get("reprojection_vanilla", {}))
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


def find_videos_recursive(
    input_root: str, 
    categories: List[str],
    start_index: Optional[int] = None,
    end_index: Optional[int] = None
) -> Dict[str, List[Dict[str, str]]]:
    """
    Find all video files in nested directory structure.
    
    Expected structure:
        input_root/
            category1/
                prompt00001/
                    video1.mp4
                    video2.mp4
                prompt00002/
                    video3.mp4
            category2/
                ...
    
    Args:
        input_root: Root directory containing category subdirectories
        categories: List of category names to search
        start_index: Start prompt index (inclusive, None = no filter)
        end_index: End prompt index (inclusive, None = no filter)
        
    Returns:
        Dictionary mapping category to list of video info dicts:
        {
            'category1': [
                {'path': 'full/path/to/video.mp4', 'prompt': 'prompt1', 'name': 'video1', 'prompt_index': 1},
                ...
            ]
        }
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    category_videos = {}
    
    for category in categories:
        category_path = os.path.join(input_root, category)
        
        if not os.path.exists(category_path):
            print(f"Warning: Category directory not found: {category_path}")
            continue
        
        videos = []
        
        # Iterate through prompt subdirectories
        for prompt_name in os.listdir(category_path):
            prompt_path = os.path.join(category_path, prompt_name)
            
            # Skip if not a directory
            if not os.path.isdir(prompt_path):
                continue
            
            # Extract prompt index
            prompt_index = extract_prompt_index_from_folder(prompt_name)
            
            # Skip if no valid prompt index found
            if prompt_index is None:
                print(f"Warning: Could not extract prompt index from folder: {prompt_name}")
                continue
            
            # Filter by index range
            if start_index is not None and prompt_index < start_index:
                continue
            if end_index is not None and prompt_index > end_index:
                continue
            
            # Find all video files in this prompt directory
            for filename in os.listdir(prompt_path):
                file_path = os.path.join(prompt_path, filename)
                
                # Check if it's a video file
                if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in video_extensions):
                    video_name = os.path.splitext(filename)[0]
                    videos.append({
                        'path': file_path,
                        'prompt': prompt_name,
                        'prompt_index': prompt_index,
                        'name': video_name,
                        'filename': filename
                    })
        
        category_videos[category] = sorted(videos, key=lambda x: (x['prompt_index'], x['name']))
    
    return category_videos


def evaluate_videos(
    input_root: str,
    output_root: str,
    categories: List[str],
    metrics: List[str],
    config: Dict[str, Any],
    output_suffix: str = "metrics.json",
    resume: bool = True,
    save_sampling_mask: bool = False,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
):
    """
    Evaluate all videos in specified categories using specified metrics.
    
    Args:
        input_root: Root directory containing category subdirectories with prompt subdirectories
        output_root: Root directory for output results
        categories: List of category names to evaluate
        metrics: List of metric names to compute
        config: Full configuration dictionary
        output_suffix: Suffix for output JSON files
        resume: If True, skip already evaluated videos
        save_sampling_mask: If True, save sampling masks for reprojection_error metric
        start_index: Start prompt index (inclusive, None = no filter)
        end_index: End prompt index (inclusive, None = no filter)
    """
    results = {}
    
    # Find all videos in nested structure
    print(f"\n{'='*80}")
    print("Scanning for videos...")
    print(f"{'='*80}")
    category_videos = find_videos_recursive(input_root, categories, start_index, end_index)
    
    # Print summary
    total_videos = sum(len(videos) for videos in category_videos.values())
    if total_videos == 0:
        print(f"No videos found matching criteria!")
        if start_index is not None or end_index is not None:
            print(f"  Prompt index range: [{start_index or 'start'}, {end_index or 'end'}]")
        return results
    
    print(f"Found {total_videos} total videos across {len(category_videos)} categories:")
    for category, videos in category_videos.items():
        prompts = set(v['prompt'] for v in videos)
        prompt_indices = sorted(set(v['prompt_index'] for v in videos))
        print(f"  {category}: {len(videos)} videos in {len(prompts)} prompts (indices: {prompt_indices[0]}-{prompt_indices[-1]})")
    
    if start_index is not None or end_index is not None:
        print(f"Filtering: prompt index range [{start_index or 'start'}, {end_index or 'end'}]")
    
    # Flatten all videos for progress bar
    all_videos_list = []
    for category in categories:
        videos = category_videos.get(category, [])
        for video_info in videos:
            all_videos_list.append((category, video_info))
    
    # Create master progress bar
    pbar = tqdm(
        total=len(all_videos_list),
        desc="Overall Progress",
        unit="video",
        position=0,
        leave=True,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
    )
    
    processed_count = 0
    skipped_count = 0
    
    for category in categories:
        videos = category_videos.get(category, [])
        
        if len(videos) == 0:
            continue
        
        category_output = os.path.join(output_root, category)
        os.makedirs(category_output, exist_ok=True)
        
        category_results = {}
        
        for video_info in videos:
            video_path = video_info['path']
            prompt_name = video_info['prompt']
            prompt_index = video_info['prompt_index']
            video_name = video_info['name']
            
            # Create unique identifier: prompt/videoname
            video_id = f"{prompt_name}/{video_name}"
            
            # Update progress bar description
            current_time = datetime.now().strftime("%H:%M:%S")
            pbar.set_description(f"[{current_time}] {category}/{video_id}")
            
            # Create output subdirectory for this prompt
            prompt_output_dir = os.path.join(category_output, prompt_name)
            os.makedirs(prompt_output_dir, exist_ok=True)
            
            output_file = os.path.join(prompt_output_dir, f"{video_name}_{output_suffix}")
            
            # Check if already evaluated
            if resume and os.path.exists(output_file):
                try:
                    with open(output_file, 'r') as f:
                        category_results[video_id] = json.load(f)
                    skipped_count += 1
                    pbar.set_postfix_str(f"Skipped (exist) | P:{processed_count} S:{skipped_count}")
                    pbar.update(1)
                    continue
                except Exception as e:
                    # If failed to load, re-evaluate
                    pass
            
            video_results = {
                'video_info': {
                    'category': category,
                    'prompt': prompt_name,
                    'prompt_index': prompt_index,
                    'video_name': video_name,
                    'filename': video_info['filename'],
                    'path': video_path
                }
            }
            
            # Evaluate all metrics
            success = True
            for metric_name in metrics:
                try:
                    # Set up sampling mask directory if needed
                    sampling_mask_dir = None
                    if save_sampling_mask and metric_name == "reprojection_error":
                        sampling_mask_dir = os.path.join(prompt_output_dir, "sampling_masks")
                        os.makedirs(sampling_mask_dir, exist_ok=True)
                    
                    evaluator = get_evaluator(metric_name, config, save_sampling_mask, sampling_mask_dir)
                    score, detailed_metrics = evaluator.evaluate_video(video_path)
                    
                    video_results[metric_name] = {
                        "score": float(score),
                        "details": detailed_metrics
                    }
                    
                except Exception as e:
                    success = False
                    video_results[metric_name] = {
                        "score": None,
                        "error": str(e)
                    }
            
            # Save results for this video
            category_results[video_id] = video_results
            
            with open(output_file, 'w') as f:
                json.dump(video_results, f, indent=2)
            
            processed_count += 1
            status = "✓" if success else "⚠"
            pbar.set_postfix_str(f"{status} | Processed:{processed_count} Skipped:{skipped_count}")
            pbar.update(1)
        
        results[category] = category_results
    
    pbar.close()
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"Evaluation Summary:")
    print(f"  Total videos:     {len(all_videos_list)}")
    print(f"  Processed:        {processed_count}")
    print(f"  Skipped:          {skipped_count}")
    print(f"{'='*80}")
    
    return results


def compute_rankings(results: Dict[str, Dict[str, Dict[str, Any]]], metrics: List[str]) -> Dict[str, Dict[str, List[tuple]]]:
    """
    Compute rankings for each metric within each category.
    
    Args:
        results: Nested dict {category: {video_id: {metric: {...}}}}
        metrics: List of metric names
        
    Returns:
        Rankings dict {category: {metric: [(video_id, score), ...]}}
    """
    rankings = {}
    
    for category, category_results in results.items():
        rankings[category] = {}
        
        for metric in metrics:
            # Collect (video_id, score) pairs
            scores = []
            for video_id, video_results in category_results.items():
                if metric in video_results and video_results[metric].get("score") is not None:
                    scores.append((video_id, video_results[metric]["score"]))
            
            # Sort by score
            # For epipolar_consistency, reprojection_error, and reprojection_vanilla: lower is better
            if metric in ["epipolar_consistency", "reprojection_error", "reprojection_vanilla"]:
                scores.sort(key=lambda x: x[1])  # ascending (lower is better)
            else:
                scores.sort(key=lambda x: x[1], reverse=True)  # descending (higher is better)
            
            rankings[category][metric] = scores
    
    return rankings


def compute_prompt_rankings(
    results: Dict[str, Dict[str, Dict[str, Any]]], 
    metrics: List[str],
    output_root: str
) -> Dict[str, Dict[str, Dict[str, List[tuple]]]]:
    """
    Compute rankings for each prompt within each category and save to prompt folders.
    
    Args:
        results: Nested dict {category: {video_id: {metric: {...}}}}
        metrics: List of metric names
        output_root: Root directory for output
        
    Returns:
        Rankings dict {category: {prompt: {metric: [(video_name, score), ...]}}}
    """
    prompt_rankings = {}
    
    for category, category_results in results.items():
        prompt_rankings[category] = {}
        
        # Group videos by prompt
        prompt_videos = {}
        for video_id, video_results in category_results.items():
            # video_id format: "prompt/video_name"
            if '/' in video_id:
                prompt, video_name = video_id.split('/', 1)
            else:
                prompt = "unknown"
                video_name = video_id
            
            if prompt not in prompt_videos:
                prompt_videos[prompt] = {}
            prompt_videos[prompt][video_name] = video_results
        
        # Compute rankings for each prompt
        for prompt, videos in prompt_videos.items():
            prompt_rankings[category][prompt] = {}
            
            for metric in metrics:
                # Collect (video_name, score) pairs
                scores = []
                for video_name, video_results in videos.items():
                    if metric in video_results and video_results[metric].get("score") is not None:
                        scores.append((video_name, video_results[metric]["score"]))
                
                # Sort by score
                if metric in ["epipolar_consistency", "reprojection_error", "reprojection_vanilla"]:
                    scores.sort(key=lambda x: x[1])  # ascending (lower is better)
                else:
                    scores.sort(key=lambda x: x[1], reverse=True)  # descending (higher is better)
                
                prompt_rankings[category][prompt][metric] = scores
            
            # Save prompt-level rankings to prompt folder
            prompt_output_dir = os.path.join(output_root, category, prompt)
            os.makedirs(prompt_output_dir, exist_ok=True)
            
            prompt_ranking_file = os.path.join(prompt_output_dir, "rankings.json")
            save_prompt_rankings(prompt_rankings[category][prompt], prompt_ranking_file, prompt)
    
    return prompt_rankings


def save_prompt_rankings(rankings: Dict[str, List[tuple]], output_path: str, prompt_name: str):
    """Save prompt-level rankings to JSON file."""
    rankings_serializable = {}
    
    for metric, scores in rankings.items():
        rankings_serializable[metric] = [
            {"rank": i+1, "video_name": name, "score": float(score)}
            for i, (name, score) in enumerate(scores)
        ]
    
    output_data = {
        "prompt": prompt_name,
        "num_videos": len(scores) if scores else 0,
        "rankings": rankings_serializable
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)


def print_rankings(rankings: Dict[str, Dict[str, List[tuple]]]):
    """Print rankings in a formatted way."""
    for category, metric_rankings in rankings.items():
        print(f"\n{'='*80}")
        print(f"Rankings for: {category}")
        print(f"{'='*80}")
        
        for metric, scores in metric_rankings.items():
            print(f"\n{metric}:")
            print(f"{'-'*80}")
            
            if len(scores) == 0:
                print("  No results available")
                continue
            
            # Add better/worse indicator based on metric type
            if metric in ["epipolar_consistency", "reprojection_error", "reprojection_vanilla"]:
                indicator = "(lower is better)"
            else:
                indicator = "(higher is better)"
            
            print(f"  {indicator}")
            
            # Show top 10
            for rank, (video_id, score) in enumerate(scores[:10], 1):
                print(f"  {rank:2d}. {video_id:60s} {score:10.4f}")
            
            if len(scores) > 10:
                print(f"  ... and {len(scores)-10} more videos")


def print_prompt_rankings(prompt_rankings: Dict[str, Dict[str, Dict[str, List[tuple]]]]):
    """Print prompt-level rankings."""
    for category, prompts in prompt_rankings.items():
        print(f"\n{'='*80}")
        print(f"Prompt-level Rankings for: {category}")
        print(f"{'='*80}")
        
        for prompt, metric_rankings in prompts.items():
            print(f"\n  Prompt: {prompt}")
            print(f"  {'-'*76}")
            
            for metric, scores in metric_rankings.items():
                if len(scores) == 0:
                    continue
                
                # Add indicator
                if metric in ["epipolar_consistency", "reprojection_error", "reprojection_vanilla"]:
                    indicator = "(lower is better)"
                else:
                    indicator = "(higher is better)"
                
                print(f"    {metric}: {indicator}")
                for rank, (video_name, score) in enumerate(scores[:5], 1):  # Show top 5
                    print(f"      {rank}. {video_name:40s} {score:10.4f}")
                
                if len(scores) > 5:
                    print(f"      ... and {len(scores)-5} more")


def save_rankings(rankings: Dict[str, Dict[str, List[tuple]]], output_path: str):
    """Save category-level rankings to JSON file."""
    # Convert to serializable format
    rankings_serializable = {}
    for category, metric_rankings in rankings.items():
        rankings_serializable[category] = {}
        for metric, scores in metric_rankings.items():
            rankings_serializable[category][metric] = [
                {"rank": i+1, "video_id": vid, "score": float(score)}
                for i, (vid, score) in enumerate(scores)
            ]
    
    with open(output_path, 'w') as f:
        json.dump(rankings_serializable, f, indent=2)
    
    print(f"✓ Saved category-level rankings to: {output_path}")


def print_summary_statistics(results: Dict[str, Dict[str, Dict[str, Any]]], metrics: List[str]):
    """Print summary statistics for each metric across all categories."""
    print(f"\n{'='*80}")
    print("Summary Statistics")
    print(f"{'='*80}")
    
    for metric in metrics:
        print(f"\n{metric}:")
        print(f"{'-'*80}")
        
        # Collect all scores across all categories
        all_scores = []
        category_stats = {}
        
        for category, category_results in results.items():
            category_scores = []
            for video_id, video_results in category_results.items():
                if metric in video_results and video_results[metric].get("score") is not None:
                    score = video_results[metric]["score"]
                    all_scores.append(score)
                    category_scores.append(score)
            
            if len(category_scores) > 0:
                category_stats[category] = {
                    'count': len(category_scores),
                    'mean': np.mean(category_scores),
                    'std': np.std(category_scores),
                    'min': np.min(category_scores),
                    'max': np.max(category_scores),
                }
        
        if len(all_scores) == 0:
            print("  No results available")
            continue
        
        # Overall statistics
        print(f"  Overall:")
        print(f"    Videos evaluated: {len(all_scores)}")
        print(f"    Mean:   {np.mean(all_scores):.4f}")
        print(f"    Median: {np.median(all_scores):.4f}")
        print(f"    Std:    {np.std(all_scores):.4f}")
        print(f"    Min:    {np.min(all_scores):.4f}")
        print(f"    Max:    {np.max(all_scores):.4f}")
        
        # Per-category statistics
        if len(category_stats) > 1:
            print(f"\n  Per-category:")
            for category, stats in category_stats.items():
                print(f"    {category}:")
                print(f"      Count: {stats['count']}, Mean: {stats['mean']:.4f}, "
                      f"Std: {stats['std']:.4f}, Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")


def save_summary_statistics(results: Dict[str, Dict[str, Dict[str, Any]]], metrics: List[str], output_path: str):
    """Save summary statistics to JSON file."""
    summary = {}
    
    for metric in metrics:
        # Collect all scores across all categories
        all_scores = []
        category_stats = {}
        
        for category, category_results in results.items():
            category_scores = []
            for video_id, video_results in category_results.items():
                if metric in video_results and video_results[metric].get("score") is not None:
                    score = video_results[metric]["score"]
                    all_scores.append(score)
                    category_scores.append(score)
            
            if len(category_scores) > 0:
                category_stats[category] = {
                    'count': int(len(category_scores)),
                    'mean': float(np.mean(category_scores)),
                    'median': float(np.median(category_scores)),
                    'std': float(np.std(category_scores)),
                    'min': float(np.min(category_scores)),
                    'max': float(np.max(category_scores)),
                }
        
        if len(all_scores) > 0:
            summary[metric] = {
                'overall': {
                    'count': int(len(all_scores)),
                    'mean': float(np.mean(all_scores)),
                    'median': float(np.median(all_scores)),
                    'std': float(np.std(all_scores)),
                    'min': float(np.min(all_scores)),
                    'max': float(np.max(all_scores)),
                },
                'per_category': category_stats
            }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved summary statistics to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate videos using multiple metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="Start prompt index (inclusive). Only process prompts with index >= this value."
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="End prompt index (inclusive). Only process prompts with index <= this value."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for computation (e.g., 'cuda:0', 'cuda:1', 'cpu')"
    )
    
    args = parser.parse_args()
    
    # Set device
    device = set_device(args.device)
    
    # Load configuration
    config = load_config(args.config)
    
    input_root = config.get("input_root", "metrics/input")
    output_root = config.get("output_root", "metrics/output")
    categories = config.get("categories", [])
    metrics = config.get("metrics", [])
    output_suffix = config.get("output_suffix", "metrics.json")
    resume = config.get("resume", True)
    save_sampling_mask = config.get("save_sampling_mask", False)
    
    print(f"\n{'='*80}")
    print("Video Evaluation Pipeline")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Config file:         {args.config}")
    print(f"  Input root:          {input_root}")
    print(f"  Output root:         {output_root}")
    print(f"  Categories:          {categories}")
    print(f"  Metrics:             {metrics}")
    print(f"  Device:              {device}")
    print(f"  Resume:              {resume}")
    print(f"  Save sampling mask:  {save_sampling_mask}")
    print(f"  Output suffix:       {output_suffix}")
    
    if args.start_index is not None or args.end_index is not None:
        print(f"  Prompt index range:  [{args.start_index or 'start'}, {args.end_index or 'end'}]")
    
    # Evaluate all videos
    results = evaluate_videos(
        input_root=input_root,
        output_root=output_root,
        categories=categories,
        metrics=metrics,
        config=config,
        output_suffix=output_suffix,
        resume=resume,
        save_sampling_mask=save_sampling_mask,
        start_index=args.start_index,
        end_index=args.end_index,
    )
    
    # Skip ranking and statistics if no results
    if not results or all(len(v) == 0 for v in results.values()):
        print(f"\n{'='*80}")
        print("No videos were evaluated. Exiting.")
        print(f"{'='*80}\n")
        return
    
    # Compute category-level rankings
    print(f"\n{'='*80}")
    print("Computing Rankings...")
    print(f"{'='*80}")
    
    rankings = compute_rankings(results, metrics)
    
    # Compute and save prompt-level rankings
    print(f"\nComputing prompt-level rankings...")
    prompt_rankings = compute_prompt_rankings(results, metrics, output_root)
    
    # Print category-level rankings
    print_rankings(rankings)
    
    # Print prompt-level rankings (summary)
    print_prompt_rankings(prompt_rankings)
    
    # Save category-level rankings
    rankings_path = os.path.join(output_root, "rankings.json")
    save_rankings(rankings, rankings_path)
    
    # Print summary statistics
    print_summary_statistics(results, metrics)
    
    # Save summary statistics
    summary_path = os.path.join(output_root, "summary_statistics.json")
    save_summary_statistics(results, metrics, summary_path)
    
    print(f"\n{'='*80}")
    print("Evaluation Complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_root}")
    print(f"  - Individual results: {output_root}/<category>/<prompt>/<video>_{output_suffix}")
    print(f"  - Prompt rankings:    {output_root}/<category>/<prompt>/rankings.json")
    if save_sampling_mask:
        print(f"  - Sampling masks:     {output_root}/<category>/<prompt>/sampling_masks/")
    print(f"  - Category rankings:  {rankings_path}")
    print(f"  - Summary statistics: {summary_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()