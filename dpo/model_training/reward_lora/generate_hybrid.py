import os
import json
import shutil
import random
import logging

import cv2
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from diffsynth import ModelManager, WanVideoPipeline, save_video


def set_all_seeds(seed):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def load_captions_from_improved_json(json_path, num_samples=None, sources=None):
    with open(json_path, 'r') as f:
        metadata = json.load(f)

    logging.info(f"Loaded {len(metadata)} samples from improved captions")

    if sources is not None:
        metadata = [item for item in metadata if item.get('dataset_source', '') in sources]
        logging.info(f"Filtered to {len(metadata)} samples from sources: {sources}")

    if sources is not None and num_samples is not None:
        samples_by_source = {}
        for item in metadata:
            source = item.get('dataset_source', '')
            if source not in samples_by_source:
                samples_by_source[source] = []
            samples_by_source[source].append(item)

        result = []
        for source in sources:
            if source in samples_by_source:
                source_samples = samples_by_source[source]
                if num_samples < len(source_samples):
                    sampled_indices = random.sample(range(len(source_samples)), num_samples)
                    result.extend([source_samples[i] for i in sampled_indices])
                else:
                    result.extend(source_samples)
                logging.info(f"Selected {min(num_samples, len(source_samples))} samples from {source}")
        return result
    else:
        if num_samples is not None and num_samples < len(metadata):
            sampled_indices = random.sample(range(len(metadata)), num_samples)
            metadata = [metadata[i] for i in sampled_indices]
        return metadata


def initialize_pipeline(model_path, image_to_video: bool = False):
    """Initialize the pipeline without LoRA."""
    model_manager = ModelManager(device="cpu", torch_dtype=torch.bfloat16)

    model_paths = [
        f"{model_path}/diffusion_pytorch_model.safetensors",
        f"{model_path}/models_t5_umt5-xxl-enc-bf16.pth",
        f"{model_path}/Wan2.1_VAE.pth"
    ]

    logging.info(f"Loading models from: {model_paths}")
    model_manager.load_models(model_paths)

    if image_to_video:
        model_manager.load_models(
            [f"{model_path}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
            torch_dtype=torch.float32,
        )

    pipe = WanVideoPipeline.from_model_manager(
        model_manager,
        torch_dtype=torch.bfloat16,
        device="cuda"
    )
    pipe.enable_vram_management(num_persistent_param_in_dit=None)

    return pipe, model_manager


def add_lora_to_pipeline(pipe, model_manager, lora_path=None, lora_alpha=1.0,
                          sft_lora_path=None, sft_lora_alpha=1.0):
    """
    Add LoRA(s) to an existing pipeline.

    Correct order at inference time:
      1. Add the SFT LoRA first (fused into base, as the starting point for the DPO LoRA).
      2. Then add the new LoRA obtained from DPO training.

    This matches the training-time structure exactly:
      training base   = base DiT + SFT LoRA fused
      training policy = fused base + new DPO LoRA
    """
    if sft_lora_path:
        logging.info(f"Loading SFT LoRA from {sft_lora_path} with alpha={sft_lora_alpha}")
        model_manager.load_lora(sft_lora_path, lora_alpha=sft_lora_alpha)
        logging.info("SFT LoRA loaded and fused into base.")

    if lora_path:
        logging.info(f"Loading DPO LoRA from {lora_path} with alpha={lora_alpha}")
        model_manager.load_lora(lora_path, lora_alpha=lora_alpha)
        logging.info("DPO LoRA loaded.")

    return pipe


def extract_first_frame(video_path):
    """Extract the first frame from a video file using OpenCV."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    except Exception as e:
        logging.error(f"Error extracting frame from {video_path}: {e}")
        return None


def create_stacked_video(lora_video_path, output_dir, fps=15):
    """Create a side-by-side comparison video of baseline and LoRA outputs."""
    try:
        baseline_video_path = lora_video_path.replace('/lora/', '/baseline/')

        if not os.path.exists(baseline_video_path):
            logging.warning(f"Baseline video not found for {lora_video_path}")
            return None

        base_name = os.path.basename(lora_video_path)
        stacked_video_path = os.path.join(output_dir, f"stacked_{base_name}")

        cap_baseline = cv2.VideoCapture(baseline_video_path)
        cap_lora = cv2.VideoCapture(lora_video_path)

        width = int(cap_baseline.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_baseline.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(stacked_video_path, fourcc, fps, (width * 2, height))

        while True:
            ret_baseline, frame_baseline = cap_baseline.read()
            ret_lora, frame_lora = cap_lora.read()

            if not ret_baseline or not ret_lora:
                break

            cv2.putText(frame_baseline, "Baseline", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_lora, "LoRA", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            stacked_frame = np.hstack((frame_baseline, frame_lora))
            out.write(stacked_frame)

        cap_baseline.release()
        cap_lora.release()
        out.release()

        logging.info(f"Created stacked video: {stacked_video_path}")
        return stacked_video_path

    except Exception as e:
        logging.error(f"Error creating stacked video: {e}")
        return None


def generate_videos(cfg, pipe, samples, output_dir, prefix=""):
    """Generate videos using the provided pipeline."""
    videos_dir = os.path.join(output_dir, f"{prefix}videos")
    metadata_dir = os.path.join(output_dir, f"{prefix}metadata")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    stacked_dir = None
    if 'lora' in prefix:
        stacked_dir = os.path.join(output_dir, "stacked_videos")
        os.makedirs(stacked_dir, exist_ok=True)

    negative_prompt = cfg.get(
        'negative_prompt',
        "static view, frozen image, still frame, jump cuts, rapid scene changes"
    )
    generated_metadata = []

    for idx, sample in tqdm(enumerate(samples), total=len(samples),
                             desc=f"Generating {prefix}videos"):
        caption = sample.get('caption', '')
        if caption.startswith('"') and caption.endswith('"'):
            caption = caption[1:-1]

        original_video_path = sample.get('full_video_path', sample.get('original_video_path', ''))
        original_filename = os.path.basename(
            sample.get('original_video_path', f"unknown_{idx}")
        )

        if not caption:
            logging.warning(f"Empty caption for sample {idx}, skipping.")
            continue

        base_name = os.path.splitext(original_filename)[0]
        video_path = os.path.join(videos_dir, f"{base_name}.mp4")

        if os.path.exists(video_path) and not cfg.get('overwrite', False):
            logging.info(f"Skipping existing video: {video_path}")
            metadata_entry = {
                "original_video_path": sample.get('original_video_path', ''),
                "dataset_source": sample.get('dataset_source', ''),
                "short_caption": sample.get('short_caption', ''),
                "long_caption": sample.get('long_caption', ''),
                "caption": caption,
                "video_path": video_path,
                "num_frames": cfg.get('num_frames', 81),
                "seed": cfg.seed,
                "fps": cfg.get('fps', 15),
                "timestamp": cfg.get('timestamp', ""),
            }
            generated_metadata.append(metadata_entry)

            if stacked_dir and 'lora' in prefix:
                create_stacked_video(video_path, stacked_dir, fps=cfg.get('fps', 15))
            continue

        try:
            if cfg.get('image_to_video', False):
                first_frame = extract_first_frame(original_video_path)
                if first_frame is None:
                    logging.error(f"Could not extract first frame from {original_video_path}, skipping.")
                    continue
                max_area = 480 * 832
                aspect_ratio = first_frame.height / first_frame.width
                mod_value = 16
                height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
                width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
                first_frame = first_frame.resize((width, height))
                video_frames = pipe(
                    input_image=first_frame,
                    prompt=caption,
                    negative_prompt=negative_prompt,
                    num_inference_steps=cfg.get('num_inference_steps', 40),
                    seed=cfg.seed,
                    tiled=cfg.get('tiled', True),
                    width=width,
                    height=height,
                    num_frames=cfg.get('num_frames', 81),
                )
            else:
                video_frames = pipe(
                    prompt=caption,
                    negative_prompt=negative_prompt,
                    num_inference_steps=cfg.get('num_inference_steps', 40),
                    seed=cfg.seed,
                    tiled=cfg.get('tiled', True),
                    width=cfg.get('width', 832),
                    height=cfg.get('height', 480),
                    num_frames=cfg.get('num_frames', 81),
                )

            save_video(video_frames, video_path,
                       fps=cfg.get('fps', 15), quality=cfg.get('quality', 5))

            metadata_entry = {
                "original_video_path": sample.get('original_video_path', ''),
                "dataset_source": sample.get('dataset_source', ''),
                "short_caption": sample.get('short_caption', ''),
                "long_caption": sample.get('long_caption', ''),
                "caption": caption,
                "video_path": video_path,
                "num_frames": cfg.get('num_frames', 81),
                "seed": cfg.seed,
                "fps": cfg.get('fps', 15),
                "timestamp": cfg.get('timestamp', ""),
            }
            generated_metadata.append(metadata_entry)

            if stacked_dir and 'lora' in prefix:
                create_stacked_video(video_path, stacked_dir, fps=cfg.get('fps', 15))

        except Exception as e:
            logging.error(f"Error generating video {video_path}: {e}")

    metadata_path = os.path.join(metadata_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(generated_metadata, f, indent=2)

    logging.info(f"Saved {len(generated_metadata)} metadata entries to {metadata_path}")
    return metadata_path, generated_metadata


@hydra.main(config_path="config", config_name="test_restart", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logging.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    set_all_seeds(cfg.seed)

    logging.info(f"Loading captions from: {cfg.improved_captions_path}")
    samples = load_captions_from_improved_json(
        cfg.improved_captions_path,
        num_samples=cfg.get('num_samples_per_source', 100),
        sources=cfg.get('sources', None),
    )

    model_path = str(cfg.model_path)
    logging.info(f"Using model path: {model_path}")
    pipe, model_manager = initialize_pipeline(model_path, cfg.get('image_to_video', False))

    # ------------------------------------------------------------------ #
    #  Baseline generation
    # ------------------------------------------------------------------ #
    if cfg.get('run_baseline', True):
        baseline_output_dir = os.path.join(output_dir, "baseline")

        if cfg.get('baseline_video_path', None) is not None:
            logging.info(f"Using existing baseline videos from {cfg.baseline_video_path}")
            source_dir = os.path.join(cfg.baseline_video_path, "baseline")

            if os.path.exists(source_dir):
                os.makedirs(baseline_output_dir, exist_ok=True)
                for item in os.listdir(source_dir):
                    source_item = os.path.join(source_dir, item)
                    dest_item = os.path.join(baseline_output_dir, item)
                    if os.path.isdir(source_item):
                        shutil.copytree(source_item, dest_item, dirs_exist_ok=True)
                    else:
                        shutil.copy2(source_item, dest_item)
                baseline_metadata_path = os.path.join(
                    baseline_output_dir, "metadata", "metadata.json"
                )
                logging.info(f"Baseline content copied to {baseline_output_dir}")
            else:
                logging.warning(
                    f"Baseline directory {source_dir} not found, generating new videos."
                )
                baseline_metadata_path, _ = generate_videos(
                    cfg, pipe, samples, output_dir, prefix="baseline/"
                )
        else:
            logging.info("Generating baseline videos...")
            baseline_metadata_path, _ = generate_videos(
                cfg, pipe, samples, output_dir, prefix="baseline/"
            )
        logging.info(f"Baseline metadata at: {baseline_metadata_path}")

    # ------------------------------------------------------------------ #
    #  LoRA generation
    #  Correct order: fuse the SFT LoRA first, then add the DPO LoRA
    # ------------------------------------------------------------------ #
    if cfg.get('lora_path', None) is not None and cfg.get('run_lora', True):
        logging.info("Loading SFT + DPO LoRA and generating videos...")

        sft_lora_path = cfg.get('sft_lora_path', None)
        sft_lora_alpha = cfg.get('sft_lora_alpha', 1.0)
        lora_alpha = cfg.get('lora_alpha', 1.0)

        add_lora_to_pipeline(
            pipe, model_manager,
            lora_path=cfg.lora_path,
            lora_alpha=lora_alpha,
            sft_lora_path=sft_lora_path,
            sft_lora_alpha=sft_lora_alpha,
        )

        # If a baseline already exists, only generate LoRA videos for samples that have one
        lora_samples = samples
        if cfg.get('baseline_video_path', None) is not None:
            baseline_videos_dir = os.path.join(output_dir, "baseline", "videos")
            if os.path.exists(baseline_videos_dir):
                filtered_samples = []
                for sample in samples:
                    original_filename = os.path.basename(
                        sample.get('original_video_path', f"unknown_{samples.index(sample)}")
                    )
                    base_name = os.path.splitext(original_filename)[0]
                    baseline_video_path = os.path.join(baseline_videos_dir, f"{base_name}.mp4")
                    if os.path.exists(baseline_video_path):
                        filtered_samples.append(sample)
                    else:
                        logging.debug(
                            f"Skipping LoRA generation for {base_name}: no baseline video."
                        )
                lora_samples = filtered_samples
                logging.info(
                    f"Filtered to {len(lora_samples)} samples with existing baseline "
                    f"(from {len(samples)} total)"
                )
            else:
                logging.warning(
                    f"Baseline videos dir {baseline_videos_dir} not found, using all samples."
                )

        lora_metadata_path, _ = generate_videos(
            cfg, pipe, lora_samples, output_dir, prefix="lora/"
        )
        logging.info(f"LoRA metadata saved to: {lora_metadata_path}")

    logging.info(f"Video generation completed. Output saved to {output_dir}")


if __name__ == "__main__":
    main()