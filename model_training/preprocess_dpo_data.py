import os
import gc
import json
import argparse
import logging
import traceback
from pathlib import Path
from typing import Optional, List
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

VIDEO_HEIGHT = 480
VIDEO_WIDTH  = 832
NUM_FRAMES   = 81


# ---------------------------------------------------------------------------
# 视频加载
# ---------------------------------------------------------------------------

def load_video_as_tensor(
    video_path: str,
    height: int = VIDEO_HEIGHT,
    width: int  = VIDEO_WIDTH,
    num_frames: int = NUM_FRAMES,
) -> torch.Tensor:
    """返回 [C, T, H, W] float32，值域 [-1, 1]"""

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
        logger.warning(f"decord 失败 ({e})，回退到 torchvision")

    import torchvision.io as tvio
    video, _, _ = tvio.read_video(video_path, pts_unit='sec')  # [T, H, W, C]
    T = video.shape[0]
    video = video.permute(0, 3, 1, 2).float()                  # [T, C, H, W]
    video = F.interpolate(video, size=(height, width), mode='bilinear', align_corners=False)
    video = video.permute(1, 0, 2, 3)                          # [C, T, H, W]
    indices = sample_indices(T, num_frames)
    return video[:, indices, :, :] / 127.5 - 1.0


# ---------------------------------------------------------------------------
# 模型加载（每个进程独立加载，互不干扰）
# ---------------------------------------------------------------------------

def load_wan_pipeline(wan_model_path: str, device: str):
    """在指定 device 上加载 pipeline，权重先放 CPU，按需移到 device"""
    from diffsynth import WanVideoPipeline, ModelManager

    logger.info(f"[{device}] 加载模型: {wan_model_path}")
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
# 显存管理
# ---------------------------------------------------------------------------

def flush_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 编码函数
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
        logger.error(f"编码视频失败 {video_path}: {e}")
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
        logger.warning(f"图像条件编码失败 {video_path}: {e}")
        traceback.print_exc()
        return {}


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def normalize_dataset_source(raw: str, fallback: str = "gb3dv25k") -> str:
    return DATASET_SOURCE_MAP.get(raw.lower().strip(), fallback)


# ---------------------------------------------------------------------------
# 处理单个 prompt 文件夹
# ---------------------------------------------------------------------------

def process_prompt_folder(
    pipe,
    video_prompt_dir: str,
    metric_prompt_dir: str,
    output_prompt_dir: str,
    metric_name: str,
    default_dataset_source: str,
    device: str,
    overwrite: bool,
) -> list:
    os.makedirs(output_prompt_dir, exist_ok=True)

    meta_path = os.path.join(video_prompt_dir, "metadata.json")
    if not os.path.exists(meta_path):
        logger.warning(f"找不到 metadata.json，跳过: {meta_path}")
        return []
    with open(meta_path) as f:
        meta = json.load(f)

    caption   = meta.get("caption", "")
    raw_ds    = meta.get("dataset", "")
    ds_source = normalize_dataset_source(raw_ds, default_dataset_source) if raw_ds else default_dataset_source

    rankings_path = os.path.join(metric_prompt_dir, "rankings.json")
    if not os.path.exists(rankings_path):
        logger.warning(f"找不到 rankings.json，跳过: {rankings_path}")
        return []
    with open(rankings_path) as f:
        rankings_data = json.load(f)

    rankings = rankings_data.get("rankings", {})
    if metric_name not in rankings:
        logger.warning(f"metric '{metric_name}' 不在 {rankings_path}，可用: {list(rankings.keys())}")
        return []

    metric_scores = {item["video_name"]: item["score"] for item in rankings[metric_name]}

    # 编码 prompt（每个 prompt 只做一次）
    shared_prompt_path = os.path.join(output_prompt_dir, "prompt_condition.pt")
    if overwrite or not os.path.exists(shared_prompt_path):
        prompt_emb = encode_text_condition(pipe, caption, device)
        torch.save({"prompt_embedding": prompt_emb}, shared_prompt_path)

    entries     = []
    video_files = meta.get("video_files", [])

    for video_file in tqdm(video_files, desc=f"  [{device}] {os.path.basename(video_prompt_dir)}", leave=False):
        seed_name  = video_file.replace(".mp4", "")
        video_path = os.path.join(video_prompt_dir, video_file)

        if seed_name not in metric_scores:
            logger.warning(f"  '{seed_name}' 无 metric 分数，跳过")
            continue
        if not os.path.exists(video_path):
            logger.warning(f"  视频不存在: {video_path}")
            continue

        latent_path    = os.path.join(output_prompt_dir, f"{seed_name}_latent.pt")
        condition_path = os.path.join(output_prompt_dir, f"{seed_name}_condition.pt")

        if overwrite or not os.path.exists(latent_path):
            latent = encode_video_to_latent(pipe, video_path, device)
            if latent is None:
                logger.error(f"  latent 编码失败，跳过: {video_path}")
                continue
            torch.save(latent, latent_path)

        if overwrite or not os.path.exists(condition_path):
            shared     = torch.load(shared_prompt_path, map_location="cpu")
            prompt_emb = shared["prompt_embedding"]
            image_emb  = encode_image_condition(pipe, video_path, device)
            condition  = {"prompt_embedding": prompt_emb}
            if image_emb:
                condition["image_embedding"] = image_emb
            torch.save(condition, condition_path)

        entries.append({
            "original_video_path": video_prompt_dir,
            "latent_path":         os.path.abspath(latent_path),
            "condition_path":      os.path.abspath(condition_path),
            metric_name:           metric_scores[seed_name],
            "dataset_source":      ds_source,
            "motion_dynamics":     0.0,
            "video_path":          os.path.abspath(video_path),
            "seed":                seed_name,
            "caption":             caption,
        })

    return entries


# ---------------------------------------------------------------------------
# 单进程worker：处理分配给自己的 prompt 列表
# ---------------------------------------------------------------------------

def worker(
    rank: int,
    device: str,
    prompt_tasks: list,          # [(video_prompt_dir, metric_prompt_dir, output_prompt_dir), ...]
    metric_name: str,
    default_dataset_source: str,
    wan_model_path: str,
    overwrite: bool,
    result_queue: mp.Queue,
):
    """
    每个 GPU 对应一个 worker 进程。
    处理完成后把 entries list 放入 result_queue。
    """
    # 设置进程名，方便日志区分
    import setproctitle
    try:
        setproctitle.setproctitle(f"preprocess_{device}")
    except ImportError:
        pass

    logger.info(f"[{device}] Worker 启动，负责 {len(prompt_tasks)} 个 prompt")

    pipe = load_wan_pipeline(wan_model_path, device)
    all_entries = []

    for video_prompt_dir, metric_prompt_dir, output_prompt_dir in tqdm(
        prompt_tasks, desc=f"[{device}]", position=rank
    ):
        entries = process_prompt_folder(
            pipe=pipe,
            video_prompt_dir=video_prompt_dir,
            metric_prompt_dir=metric_prompt_dir,
            output_prompt_dir=output_prompt_dir,
            metric_name=metric_name,
            default_dataset_source=default_dataset_source,
            device=device,
            overwrite=overwrite,
        )
        all_entries.extend(entries)

    logger.info(f"[{device}] Worker 完成，共 {len(all_entries)} 个 entries")
    result_queue.put((rank, all_entries))


# ---------------------------------------------------------------------------
# 收集所有 prompt 任务
# ---------------------------------------------------------------------------

def collect_all_tasks(
    video_root: Path,
    metric_root: Path,
    output_root: Path,
    categories: List[str],
) -> list:
    """收集所有需要处理的 (video_prompt_dir, metric_prompt_dir, output_prompt_dir) 三元组"""
    tasks = []
    for category in categories:
        video_cat  = video_root  / category
        metric_cat = metric_root / category
        output_cat = output_root / category

        if not video_cat.exists():
            logger.warning(f"video category 不存在，跳过: {video_cat}")
            continue
        if not metric_cat.exists():
            logger.warning(f"metric category 不存在，跳过: {metric_cat}")
            continue

        prompt_dirs = sorted(d for d in video_cat.iterdir() if d.is_dir())
        logger.info(f"Category '{category}': {len(prompt_dirs)} 个 prompt")

        for prompt_dir in prompt_dirs:
            prompt_name       = prompt_dir.name
            metric_prompt_dir = metric_cat / prompt_name
            output_prompt_dir = output_cat / prompt_name

            if not metric_prompt_dir.exists():
                logger.warning(f"  metric prompt 不存在，跳过: {metric_prompt_dir}")
                continue

            tasks.append((str(prompt_dir), str(metric_prompt_dir), str(output_prompt_dir)))

    return tasks


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="预处理视频数据为 DPO 训练格式（支持多卡）")
    parser.add_argument("--video_root",      required=True)
    parser.add_argument("--metric_root",     required=True)
    parser.add_argument("--output_root",     required=True)
    parser.add_argument("--wan_model_path",  required=True)
    parser.add_argument("--output_metadata", default="annotated_metadata.json")
    parser.add_argument("--metric_name",     default="epipolar_consistency")
    parser.add_argument("--dataset_source",  default="gb3dv25k")
    parser.add_argument("--devices",         nargs="+", default=["cuda:0"],
                        help="使用的 GPU 列表，如 --devices cuda:0 cuda:1")
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
    logger.info(f"使用设备: {args.devices}")

    # 收集全部任务
    all_tasks = collect_all_tasks(video_root, metric_root, output_root, categories)
    logger.info(f"共 {len(all_tasks)} 个 prompt 任务，分配到 {len(args.devices)} 张卡")

    # 按 GPU 数量切分任务（round-robin 均匀分配）
    num_gpus   = len(args.devices)
    task_shards = [[] for _ in range(num_gpus)]
    for i, task in enumerate(all_tasks):
        task_shards[i % num_gpus].append(task)

    for i, (device, shard) in enumerate(zip(args.devices, task_shards)):
        logger.info(f"  {device}: {len(shard)} 个 prompt")

    # 单卡直接在主进程跑，避免多进程 overhead
    if num_gpus == 1:
        logger.info("单卡模式，主进程直接处理")
        pipe = load_wan_pipeline(args.wan_model_path, args.devices[0])
        all_entries = []
        for video_prompt_dir, metric_prompt_dir, output_prompt_dir in tqdm(
            all_tasks, desc=f"[{args.devices[0]}]"
        ):
            entries = process_prompt_folder(
                pipe=pipe,
                video_prompt_dir=video_prompt_dir,
                metric_prompt_dir=metric_prompt_dir,
                output_prompt_dir=output_prompt_dir,
                metric_name=args.metric_name,
                default_dataset_source=args.dataset_source,
                device=args.devices[0],
                overwrite=args.overwrite,
            )
            all_entries.extend(entries)

    else:
        # 多卡：每张卡启动一个独立进程
        # 必须用 spawn，避免 CUDA 多进程 fork 问题
        mp.set_start_method("spawn", force=True)
        result_queue = mp.Queue()
        processes    = []

        for rank, (device, shard) in enumerate(zip(args.devices, task_shards)):
            p = mp.Process(
                target=worker,
                name=f"worker-{device}",
                kwargs=dict(
                    rank=rank,
                    device=device,
                    prompt_tasks=shard,
                    metric_name=args.metric_name,
                    default_dataset_source=args.dataset_source,
                    wan_model_path=args.wan_model_path,
                    overwrite=args.overwrite,
                    result_queue=result_queue,
                ),
            )
            p.start()
            processes.append(p)
            logger.info(f"已启动 worker PID={p.pid} -> {device}")

        # 等待所有进程完成并收集结果
        results = {}
        for _ in processes:
            rank, entries = result_queue.get()   # 阻塞直到有结果
            results[rank] = entries

        for p in processes:
            p.join()
            if p.exitcode != 0:
                logger.error(f"Worker {p.name} 异常退出，exitcode={p.exitcode}")

        # 按 rank 顺序合并，保证结果顺序确定
        all_entries = []
        for rank in sorted(results.keys()):
            all_entries.extend(results[rank])

    # ---- 保存 metadata ----
    with open(args.output_metadata, 'w') as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"全部完成！共 {len(all_entries)} 个视频 → {args.output_metadata}")

    if all_entries:
        sources = Counter(e["dataset_source"] for e in all_entries)
        logger.info(f"数据集分布: {dict(sources)}")

        scores = [e[args.metric_name] for e in all_entries]
        logger.info(
            f"Metric '{args.metric_name}': "
            f"min={min(scores):.3f}  max={max(scores):.3f}  "
            f"mean={np.mean(scores):.3f}  n={len(scores)}"
        )

        groups      = defaultdict(int)
        for e in all_entries:
            groups[e["original_video_path"]] += 1
        valid_pairs = sum(1 for cnt in groups.values() if cnt >= 2)
        logger.info(f"可组 DPO pair 的 prompt 数: {valid_pairs} / {len(groups)}")


if __name__ == "__main__":
    main()