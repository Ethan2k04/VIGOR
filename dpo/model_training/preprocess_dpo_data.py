import os
import gc
import json
import argparse
import logging
import traceback
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(processName)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


DATASET_SOURCE_MAP = {
    "realestate10k":   "gb3dv25k",
    "real-estate-10k": "gb3dv25k",
    "dl3dv":           "gb3dv25k",
    "dl3dv-10k":       "gb3dv25k",
}

# Original metric names in rankings.json -> canonical names stored in metadata
METRIC_RENAME_MAP = {
    "epipolar_consistency":  "epipolar_consistency",   # unchanged
    "reprojection_vanilla":  "reprojection_pixel",
    "reprojection_error":    "reprojection_euclidean",
}

VIDEO_HEIGHT = 480
VIDEO_WIDTH  = 832
NUM_FRAMES   = 81


# ---------------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------------

def load_video_as_tensor(
    video_path: str,
    height: int = VIDEO_HEIGHT,
    width: int  = VIDEO_WIDTH,
    num_frames: int = NUM_FRAMES,
) -> torch.Tensor:
    """Return [C, T, H, W] float32 in the range [-1, 1]."""

    def sample_indices(total: int, n: int) -> List[int]:
        if total >= n:
            return np.linspace(0, total - 1, n, dtype=int).tolist()
        return list(range(total)) + [total - 1] * (n - total)

    try:
        import decord
        decord.bridge.set_bridge('torch')
        vr = decord.VideoReader(video_path, width=width, height=height)
        indices = sample_indices(len(vr), num_frames)
        frames = vr.get_batch(indices)               # [T, H, W, C] uint8
        frames = frames.permute(3, 0, 1, 2).float()  # [C, T, H, W]
        return frames / 127.5 - 1.0
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"decord failed ({e}), falling back to torchvision")

    import torchvision.io as tvio
    video, _, _ = tvio.read_video(video_path, pts_unit='sec')  # [T, H, W, C]
    T = video.shape[0]
    video = video.permute(0, 3, 1, 2).float()                  # [T, C, H, W]
    video = F.interpolate(video, size=(height, width), mode='bilinear', align_corners=False)
    video = video.permute(1, 0, 2, 3)                          # [C, T, H, W]
    indices = sample_indices(T, num_frames)
    return video[:, indices, :, :] / 127.5 - 1.0


# ---------------------------------------------------------------------------
# Model loading (each process loads independently, without interfering)
# ---------------------------------------------------------------------------

def load_wan_pipeline(wan_model_path: str, device: str):
    from diffsynth import WanVideoPipeline, ModelManager

    logger.info(f"[{device}] Loading model: {wan_model_path}")
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")

    preferred = ["Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth"]
    found = [
        os.path.join(wan_model_path, f)
        for f in preferred
        if os.path.exists(os.path.join(wan_model_path, f))
    ]
    if found:
        model_manager.load_models(found)
    elif os.path.isfile(wan_model_path):
        model_manager.load_models([wan_model_path])
    else:
        all_files = [
            os.path.join(wan_model_path, f)
            for f in os.listdir(wan_model_path)
            if f.endswith((".pth", ".safetensors", ".bin"))
        ]
        model_manager.load_models(all_files)

    pipe = WanVideoPipeline.from_model_manager(model_manager)
    pipe.device = "cpu"
    return pipe


# ---------------------------------------------------------------------------
# GPU memory management
# ---------------------------------------------------------------------------

def flush_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Encoding functions
# ---------------------------------------------------------------------------

def encode_text_condition(pipe, caption: str, device: str) -> dict:
    if hasattr(pipe, 'prompter') and hasattr(pipe.prompter, 'text_encoder'):
        pipe.prompter.text_encoder.to(device)
    pipe.device = device
    try:
        with torch.no_grad():
            prompt_emb = pipe.encode_prompt(prompt=caption, positive=True)
    finally:
        if hasattr(pipe, 'prompter') and hasattr(pipe.prompter, 'text_encoder'):
            pipe.prompter.text_encoder.to("cpu")
        pipe.device = "cpu"
        flush_cache()
    return {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in prompt_emb.items()}


def encode_video_to_latent(pipe, video_path: str, device: str) -> Optional[torch.Tensor]:
    try:
        video_tensor = load_video_as_tensor(video_path).to(torch.bfloat16)
        pipe.vae.to(device)
        try:
            with torch.no_grad():
                latent = pipe.vae.encode(
                    videos=[video_tensor],
                    device=device,
                    tiled=True,
                )
        finally:
            pipe.vae.to("cpu")
            flush_cache()

        if latent.dim() == 5 and latent.size(0) == 1:
            latent = latent.squeeze(0)
        return latent.cpu()

    except Exception as e:
        logger.error(f"Failed to encode video {video_path}: {e}")
        traceback.print_exc()
        return None


def encode_image_condition(pipe, video_path: str, device: str) -> dict:
    if not (hasattr(pipe, 'image_encoder') and pipe.image_encoder is not None):
        return {}
    try:
        from PIL import Image
        video_tensor = load_video_as_tensor(video_path)

        def to_pil(t):
            t = ((t.float() + 1.0) * 127.5).clamp(0, 255).byte()
            return Image.fromarray(t.permute(1, 2, 0).numpy())

        first_frame = to_pil(video_tensor[:, 0,  :, :])
        last_frame  = to_pil(video_tensor[:, -1, :, :])

        pipe.image_encoder.to(device)
        pipe.vae.to(device)
        pipe.device = device
        try:
            with torch.no_grad():
                image_emb = pipe.encode_image(
                    image=first_frame, end_image=last_frame,
                    num_frames=NUM_FRAMES, height=VIDEO_HEIGHT, width=VIDEO_WIDTH,
                    tiled=False,
                )
        finally:
            pipe.image_encoder.to("cpu")
            pipe.vae.to("cpu")
            pipe.device = "cpu"
            flush_cache()

        if image_emb is None:
            return {}
        return {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in image_emb.items()}

    except Exception as e:
        logger.warning(f"Failed to encode image condition {video_path}: {e}")
        traceback.print_exc()
        return {}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def normalize_dataset_source(raw: str, fallback: str = "gb3dv25k") -> str:
    return DATASET_SOURCE_MAP.get(raw.lower().strip(), fallback)


def parse_all_metric_scores(rankings: dict) -> Dict[str, Dict[str, float]]:
    """
    Parse the scores of every metric from the rankings dict and rename them
    according to METRIC_RENAME_MAP.

    Input:
        rankings = {
            "epipolar_consistency": [{"video_name": "seed00", "score": 2.37, "rank": 1}, ...],
            "reprojection_vanilla": [...],   # -> renamed to reprojection_pixel
            "reprojection_error":   [...],   # -> renamed to reprojection_euclidean
        }

    Returns (using the remapped names):
        {
            "seed00": {"epipolar_consistency": 2.37, "reprojection_pixel": 0.93, "reprojection_euclidean": 1.67},
            "seed01": {...},
            ...
        }
    """
    all_scores: Dict[str, Dict[str, float]] = defaultdict(dict)

    for raw_metric_name, items in rankings.items():
        # Apply the remapping; keep the original name if it is not in the map
        stored_name = METRIC_RENAME_MAP.get(raw_metric_name, raw_metric_name)
        for item in items:
            seed_name = item["video_name"]
            all_scores[seed_name][stored_name] = item["score"]

    return dict(all_scores)


def parse_flat_scores(per_video_scores: dict) -> Dict[str, Dict[str, float]]:
    """
    Parse the ``per_video_scores`` block written by the flat ``rewards/evaluate.py``
    and rename the metrics according to METRIC_RENAME_MAP.

    Input:
        per_video_scores = {
            "seed00": {"epipolar_consistency": 2.37, "reprojection_vanilla": 0.93, "reprojection_error": 1.67},
            "seed01": {...},
        }

    Returns (using the remapped names):
        {"seed00": {"epipolar_consistency": 2.37, "reprojection_pixel": 0.93, "reprojection_euclidean": 1.67}, ...}
    """
    all_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
    for seed_name, metric_scores in per_video_scores.items():
        for raw_metric_name, score in metric_scores.items():
            if score is None:
                continue
            stored_name = METRIC_RENAME_MAP.get(raw_metric_name, raw_metric_name)
            all_scores[seed_name][stored_name] = score
    return dict(all_scores)


def load_metric_scores(metric_prompt_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Load per-seed metric scores for one prompt, supporting two layouts:

    * the flat ``rewards/evaluate.py`` output — a JSON with a ``per_video_scores``
      block (``scores.json`` / ``results.json`` / ``metrics.json``), and
    * the legacy ``rankings.json`` — a JSON with a ``rankings`` block.

    Metric names are remapped via METRIC_RENAME_MAP. Returns
    ``{seed_name: {stored_metric: score, ...}}`` (empty if nothing was found).
    """
    # 1) flat evaluate.py output
    for fname in ("scores.json", "results.json", "metrics.json"):
        fpath = os.path.join(metric_prompt_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            data = json.load(f)
        if "per_video_scores" in data:
            return parse_flat_scores(data["per_video_scores"])
        if "rankings" in data:
            return parse_all_metric_scores(data["rankings"])

    # 2) legacy rankings.json
    rankings_path = os.path.join(metric_prompt_dir, "rankings.json")
    if os.path.exists(rankings_path):
        with open(rankings_path) as f:
            data = json.load(f)
        rankings = data.get("rankings", {})
        if rankings:
            return parse_all_metric_scores(rankings)

    return {}


# ---------------------------------------------------------------------------
# Process a single prompt folder
# ---------------------------------------------------------------------------

def process_prompt_folder(
    pipe,
    video_prompt_dir: str,
    metric_prompt_dir: str,
    output_prompt_dir: str,
    primary_metric: str,         # primary metric (validates the data; set by train.yaml at training time)
    default_dataset_source: str,
    device: str,
    overwrite: bool,
) -> list:
    os.makedirs(output_prompt_dir, exist_ok=True)

    # ---- Read the video metadata ----
    meta_path = os.path.join(video_prompt_dir, "metadata.json")
    if not os.path.exists(meta_path):
        logger.warning(f"metadata.json not found, skipping: {meta_path}")
        return []
    with open(meta_path) as f:
        meta = json.load(f)

    caption   = meta.get("caption", "")
    raw_ds    = meta.get("dataset", "")
    ds_source = normalize_dataset_source(raw_ds, default_dataset_source) if raw_ds else default_dataset_source

    # ---- Load per-seed metric scores (flat evaluate.py output or legacy rankings.json) ----
    all_scores = load_metric_scores(metric_prompt_dir)
    if not all_scores:
        logger.warning(
            f"No metric scores found (expected a flat evaluate.py 'per_video_scores' "
            f"JSON or a legacy rankings.json) in: {metric_prompt_dir}"
        )
        return []
    # available_metrics uses the remapped names
    available_metrics = list(next(iter(all_scores.values())).keys()) if all_scores else []

    # Verify the primary metric exists (primary_metric should use the remapped name)
    if primary_metric not in available_metrics:
        raw_names = list(rankings.keys())
        stored_names = [METRIC_RENAME_MAP.get(n, n) for n in raw_names]
        logger.warning(
            f"Primary metric '{primary_metric}' is not among the stored metrics {stored_names}; "
            f"original names in rankings.json: {raw_names}"
        )
        return []
    logger.debug(f"  Found metrics: {available_metrics}")

    # ---- Encode the prompt (once per prompt) ----
    shared_prompt_path = os.path.join(output_prompt_dir, "prompt_condition.pt")
    if overwrite or not os.path.exists(shared_prompt_path):
        prompt_emb = encode_text_condition(pipe, caption, device)
        torch.save({"prompt_embedding": prompt_emb}, shared_prompt_path)

    # ---- Process each seed ----
    entries     = []
    video_files = meta.get("video_files", [])

    for video_file in tqdm(
        video_files,
        desc=f"  [{device}] {os.path.basename(video_prompt_dir)}",
        leave=False,
    ):
        seed_name  = video_file.replace(".mp4", "")
        video_path = os.path.join(video_prompt_dir, video_file)

        if seed_name not in all_scores:
            logger.warning(f"  '{seed_name}' has no metric scores, skipping")
            continue
        if primary_metric not in all_scores[seed_name]:
            logger.warning(f"  '{seed_name}' is missing the primary metric '{primary_metric}', skipping")
            continue
        if not os.path.exists(video_path):
            logger.warning(f"  Video does not exist: {video_path}")
            continue

        latent_path    = os.path.join(output_prompt_dir, f"{seed_name}_latent.pt")
        condition_path = os.path.join(output_prompt_dir, f"{seed_name}_condition.pt")

        # -- latent --
        if overwrite or not os.path.exists(latent_path):
            latent = encode_video_to_latent(pipe, video_path, device)
            if latent is None:
                logger.error(f"  Latent encoding failed, skipping: {video_path}")
                continue
            torch.save(latent, latent_path)

        # -- condition --
        if overwrite or not os.path.exists(condition_path):
            shared     = torch.load(shared_prompt_path, map_location="cpu")
            prompt_emb = shared["prompt_embedding"]
            image_emb  = encode_image_condition(pipe, video_path, device)
            condition  = {"prompt_embedding": prompt_emb}
            if image_emb:
                condition["image_embedding"] = image_emb
            torch.save(condition, condition_path)

        # -- Build the entry: base fields + all metric scores --
        entry = {
            "original_video_path": video_prompt_dir,
            "latent_path":         os.path.abspath(latent_path),
            "condition_path":      os.path.abspath(condition_path),
            "dataset_source":      ds_source,
            "motion_dynamics":     0.0,
            "video_path":          os.path.abspath(video_path),
            "seed":                seed_name,
            "caption":             caption,
        }

        # Write the scores of all available metrics
        for metric_name in available_metrics:
            entry[metric_name] = all_scores[seed_name].get(metric_name, float('nan'))

        entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# Single-process worker
# ---------------------------------------------------------------------------

def worker(
    rank: int,
    device: str,
    prompt_tasks: list,
    primary_metric: str,
    default_dataset_source: str,
    wan_model_path: str,
    overwrite: bool,
    result_queue: mp.Queue,
    progress_queue: mp.Queue,  # new: used to report progress
):
    try:
        import setproctitle
        setproctitle.setproctitle(f"preprocess_{device}")
    except ImportError:
        pass

    logger.info(f"[{device}] Worker started, handling {len(prompt_tasks)} prompts")
    pipe        = load_wan_pipeline(wan_model_path, device)
    all_entries = []

    for i, (video_prompt_dir, metric_prompt_dir, output_prompt_dir) in enumerate(prompt_tasks):
        entries = process_prompt_folder(
            pipe=pipe,
            video_prompt_dir=video_prompt_dir,
            metric_prompt_dir=metric_prompt_dir,
            output_prompt_dir=output_prompt_dir,
            primary_metric=primary_metric,
            default_dataset_source=default_dataset_source,
            device=device,
            overwrite=overwrite,
        )
        all_entries.extend(entries)
        
        # Notify the main process to update progress after each prompt
        progress_queue.put(1)

    logger.info(f"[{device}] Worker finished, {len(all_entries)} entries in total")
    result_queue.put((rank, all_entries))


# ---------------------------------------------------------------------------
# Collect all prompt tasks
# ---------------------------------------------------------------------------

def collect_all_tasks(
    video_root: Path,
    metric_root: Path,
    output_root: Path,
    categories: List[str],
) -> list:
    tasks = []
    for category in categories:
        video_cat  = video_root  / category
        metric_cat = metric_root / category
        output_cat = output_root / category

        if not video_cat.exists():
            logger.warning(f"video category does not exist, skipping: {video_cat}")
            continue
        if not metric_cat.exists():
            logger.warning(f"metric category does not exist, skipping: {metric_cat}")
            continue

        prompt_dirs = sorted(d for d in video_cat.iterdir() if d.is_dir())
        logger.info(f"Category '{category}': {len(prompt_dirs)} prompts")

        for prompt_dir in prompt_dirs:
            prompt_name       = prompt_dir.name
            metric_prompt_dir = metric_cat / prompt_name
            output_prompt_dir = output_cat / prompt_name

            if not metric_prompt_dir.exists():
                logger.warning(f"  metric prompt does not exist, skipping: {metric_prompt_dir}")
                continue

            tasks.append((str(prompt_dir), str(metric_prompt_dir), str(output_prompt_dir)))

    return tasks


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess video data into the DPO training format (multi-GPU, stores all metrics)")
    parser.add_argument("--video_root",      required=True)
    parser.add_argument("--metric_root",     required=True)
    parser.add_argument("--output_root",     required=True)
    parser.add_argument("--wan_model_path",  required=True)
    parser.add_argument("--output_metadata", default="annotated_metadata.json")
    parser.add_argument("--metric_name",     default="epipolar_consistency",
                        help="Primary metric name (used for data validation; set in train.yaml at training time)")
    parser.add_argument("--dataset_source",  default="gb3dv25k")
    parser.add_argument("--devices",         nargs="+", default=["cuda:0"],
                        help="List of GPUs to use, e.g. --devices cuda:0 cuda:1")
    parser.add_argument("--overwrite",       action="store_true")
    parser.add_argument("--categories",      nargs="+", default=None)
    args = parser.parse_args()

    video_root  = Path(args.video_root)
    metric_root = Path(args.metric_root)
    output_root = Path(args.output_root)

    categories = args.categories or sorted(
        d.name for d in video_root.iterdir() if d.is_dir()
    )
    logger.info(f"Categories: {categories}")
    logger.info(f"Using devices: {args.devices}")
    logger.info(f"Primary metric: {args.metric_name} (all metrics in rankings.json are stored in metadata)")

    all_tasks = collect_all_tasks(video_root, metric_root, output_root, categories)
    logger.info(f"{len(all_tasks)} prompt tasks in total, distributed across {len(args.devices)} GPUs")

    num_gpus    = len(args.devices)
    task_shards = [[] for _ in range(num_gpus)]
    for i, task in enumerate(all_tasks):
        task_shards[i % num_gpus].append(task)
    for device, shard in zip(args.devices, task_shards):
        logger.info(f"  {device}: {len(shard)} prompts")

    # ---- Single GPU: the main process handles everything directly ----
    if num_gpus == 1:
        pipe        = load_wan_pipeline(args.wan_model_path, args.devices[0])
        all_entries = []
        
        # Single-GPU mode: show the progress bar directly in the main loop
        with tqdm(total=len(all_tasks), desc="Preprocessing", unit="prompt") as pbar:
            for video_prompt_dir, metric_prompt_dir, output_prompt_dir in all_tasks:
                entries = process_prompt_folder(
                    pipe=pipe,
                    video_prompt_dir=video_prompt_dir,
                    metric_prompt_dir=metric_prompt_dir,
                    output_prompt_dir=output_prompt_dir,
                    primary_metric=args.metric_name,
                    default_dataset_source=args.dataset_source,
                    device=args.devices[0],
                    overwrite=args.overwrite,
                )
                all_entries.extend(entries)
                pbar.update(1)

    # ---- Multi-GPU: spawn subprocesses ----
    else:
        mp.set_start_method("spawn", force=True)
        result_queue   = mp.Queue()
        progress_queue = mp.Queue()  # used by workers to report progress
        processes      = []

        for rank, (device, shard) in enumerate(zip(args.devices, task_shards)):
            p = mp.Process(
                target=worker,
                name=f"worker-{device}",
                kwargs=dict(
                    rank=rank,
                    device=device,
                    prompt_tasks=shard,
                    primary_metric=args.metric_name,
                    default_dataset_source=args.dataset_source,
                    wan_model_path=args.wan_model_path,
                    overwrite=args.overwrite,
                    result_queue=result_queue,
                    progress_queue=progress_queue,
                ),
            )
            p.start()
            processes.append(p)
            logger.info(f"Started worker PID={p.pid} -> {device}")

        # The main process monitors progress and updates the progress bar
        all_entries = []
        with tqdm(total=len(all_tasks), desc="Preprocessing", unit="prompt") as pbar:
            completed = 0
            workers_done = 0
            
            while workers_done < len(processes):
                # Non-blocking check for progress updates
                try:
                    progress_queue.get(timeout=0.1)
                    completed += 1
                    pbar.update(1)
                except:
                    pass
                
                # Check whether any worker has finished (non-blocking)
                try:
                    rank, entries = result_queue.get(timeout=0.1)
                    all_entries.append((rank, entries))
                    workers_done += 1
                except:
                    pass
            
            # Make sure the progress bar reaches 100%
            if completed < len(all_tasks):
                pbar.update(len(all_tasks) - completed)

        # Wait for all processes to exit
        for p in processes:
            p.join()
            if p.exitcode != 0:
                logger.error(f"Worker {p.name} exited abnormally, exitcode={p.exitcode}")

        # Merge results in rank order
        results_dict = dict(all_entries)
        all_entries = []
        for rank in sorted(results_dict.keys()):
            all_entries.extend(results_dict[rank])

    # ---- Save metadata ----
    with open(args.output_metadata, 'w') as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"All done! {len(all_entries)} videos → {args.output_metadata}")

    if all_entries:
        sources = Counter(e["dataset_source"] for e in all_entries)
        logger.info(f"Dataset distribution: {dict(sources)}")

        # Print statistics for every stored metric
        sample = all_entries[0]
        stored_metrics = [
            k for k in sample
            if k not in {"original_video_path", "latent_path", "condition_path",
                         "dataset_source", "motion_dynamics", "video_path", "seed", "caption"}
        ]
        logger.info(f"Stored metrics: {stored_metrics}")
        for m in stored_metrics:
            scores = [e[m] for e in all_entries if not np.isnan(e.get(m, float('nan')))]
            if scores:
                logger.info(
                    f"  {m}: min={min(scores):.3f}  max={max(scores):.3f}  "
                    f"mean={np.mean(scores):.3f}  n={len(scores)}"
                )

        groups      = defaultdict(int)
        for e in all_entries:
            groups[e["original_video_path"]] += 1
        valid_pairs = sum(1 for cnt in groups.values() if cnt >= 2)
        logger.info(f"Prompts that can form a DPO pair: {valid_pairs} / {len(groups)}")


if __name__ == "__main__":
    main()