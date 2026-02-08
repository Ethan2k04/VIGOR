# Aligning Video Diffusion Model with Visual Geometry Grounded Reward (ECCV 2026)

## News

## Overview

## Installation

## Usage

Generate `static` and `static_dynamic` captions for prompting:

```bash
python -m caption.generate_caption --config caption/config/config.json
```

Your input files should be organized in below format:

```
caption/input/
          ├── dataset_A/
          │     ├── images/
          │     │     ├── 000.jpg
          │     │     └── ...
          │     └── videos
          │           ├── 000.mp4
          │           └── ...
          └── ...
```

Generate video samples using `Wan2.1` text to video model:

```bash
python -m videogen.generate_video --config videogen/config/config.json
```

Input .json caption files should be placed in `videogen/input` folder.

Use `epipolar` or `reprojection` metric to evaluate generated videos:

```bash
python -m metrics.evaluate_all \
--config metrics/config/config.json \
--input_root your/input/video/path \
--output_root your/output/json/path \
--metrics epipolar_consistency,reprojection_error 
```

Your input files should be organized in below format:

```
metrics/input/
          ├── static/
          │     ├── prompt_000_xxx/
          │     │     ├── seed_000.mp4
          │     │     └── ...
          │     └── ...
          └── ...

```

Train a LoRA adpater with DPO, Flow-DPO or Masked Flow-DPO target funtion:

```bash
python -m model_training.train --config model_training/config/config.json
```

## Citation

## License

## Acknowledgment