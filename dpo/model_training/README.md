# VIGOR — DPO Model Training

This folder contains the **post-hoc alignment** pathway of VIGOR: a LoRA adapter
is trained on the bidirectional **Wan2.1** text-to-video model with **Flow-DPO**,
using geometry-ranked preference pairs from the **GB3DV-25k** dataset. The reward
signal is the VIGOR geometry reward (see [`../../rewards`](../../rewards)).

The training code is adapted from
[KupynOrest/epipolar-dpo](https://github.com/KupynOrest/epipolar-dpo); the main
differences are the geometry reward (pointwise reprojection instead of epipolar
Sampson distance) and the GB3DV-25k data pipeline.

## Installation

Install this folder's pinned training dependencies (they cover `diffsynth`,
`peft`, and `lightning`):

```bash
pip install -r requirements.txt
```

Training needs a CUDA GPU (≥ 24 GB VRAM recommended).

## Directory layout

```text
model_training/
├── download.sh                 # download GB3DV-25k + Wan2.1-T2V-1.3B
├── preprocess_dpo_data.py      # build DPO preference latents (multi-GPU)
├── requirements.txt
└── reward_lora/
    ├── train.py                # DPO LoRA training (Flow-DPO)
    ├── train_sft.py            # SFT warm-up (optional SFT LoRA)
    ├── train_hybrid.py         # SFT-fused base + new DPO LoRA (hybrid)
    ├── generate.py             # generate baseline vs. LoRA videos
    ├── generate_hybrid.py      # generate with SFT + DPO LoRAs fused
    ├── evaluate.py             # score generated videos
    ├── dataset.py              # DPOLatentDataset (preference pairs)
    ├── loss.py                 # Flow-DPO loss strategies
    └── config/                 # train.yaml · test.yaml · *_hybrid.yaml
```

## Workflow

### Step 1 — Download the dataset and base model

`download.sh` pulls the geometry-ranked **GB3DV-25k** dataset (16 latent shards +
`annotated_metadata.json`) from ModelScope and the **Wan2.1-T2V-1.3B** weights.
Run it from inside this folder:

```bash
cd dpo/model_training
bash download.sh
```

This produces:

```text
model_training/
├── input_latent/               # pre-encoded latents (static/dynamic × indoor/outdoor)
├── annotated_metadata.json     # preference metadata with all metric scores
└── Wan2.1-T2V-1.3B/            # base model weights
```

Because GB3DV-25k ships **pre-encoded latents** and `annotated_metadata.json`, you
can go straight to Step 3.

### Step 2 — (Optional) Build preference latents from your own videos

Only needed if you want to regenerate latents from raw videos and reward scores
(e.g. a custom dataset). `preprocess_dpo_data.py` encodes each video to a Wan2.1
latent, encodes the text/image conditions once per prompt, and stores **every**
metric score into the metadata (multi-GPU capable). Per-prompt scores are read
from the flat [`rewards/evaluate.py`](../../rewards/evaluate.py) output
(`per_video_scores`), with a legacy `rankings.json` accepted as a fallback:

```bash
python preprocess_dpo_data.py \
    --video_root      /path/to/videos \
    --metric_root     /path/to/rankings \
    --output_root     ./input_latent \
    --wan_model_path  ./Wan2.1-T2V-1.3B \
    --output_metadata annotated_metadata.json \
    --metric_name     reprojection_euclidean \
    --devices         cuda:0 cuda:1
```

Metric names are normalized to `epipolar_consistency`, `reprojection_pixel`
(pixel-space warp), and `reprojection_euclidean` (pointwise reprojection, **ours**).

### Step 3 — Train the reward LoRA

Videos from the same prompt are grouped and paired (best vs. worst by the chosen
metric) and optimized with Flow-DPO. Training is configured with Hydra
([`reward_lora/config/train.yaml`](reward_lora/config/train.yaml)); override any
field on the command line:

```bash
cd reward_lora
python train.py \
    data.metadata_path=../annotated_metadata.json \
    data.metric_name=reprojection_euclidean \
    data.metric_mode=min \
    model.dit_path=../Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors \
    logging.output_path=./checkpoints
```

Variants:

- **SFT warm-up** — `python train_sft.py ...` trains an SFT LoRA first (its path
  can then be fused as the DPO starting point).
- **Hybrid (SFT + DPO)** — `python train_hybrid.py ...` fuses an SFT LoRA into the
  frozen base, then trains a new DPO LoRA on top (config: `config/train_hybrid.yaml`).

### Step 4 — Generate and evaluate

Generate baseline vs. LoRA videos and score them:

```bash
python generate.py  lora_path=./checkpoints/xxx.ckpt model_path=../Wan2.1-T2V-1.3B output_dir=./results
python evaluate.py  output_dir=./results
```

For a quick side-by-side comparison you can also use the repository-root entry
point [`../../infer_dpo.py`](../../infer_dpo.py).

## Configuration

Key fields in [`reward_lora/config/train.yaml`](reward_lora/config/train.yaml):

```yaml
training:
  learning_rate: 5e-6
  beta: 500              # DPO temperature
  train_strategy: dpo    # dpo | sft
  max_steps: 10000

lora:
  rank: 64
  alpha: 128.0
  target_modules: ["q", "k", "v", "o"]

data:
  metadata_path: "/path/to/annotated_metadata.json"
  metric_name: "reprojection_euclidean"   # the VIGOR pointwise reprojection reward
  metric_mode: "min"                        # lower reprojection error is preferred
  metric_threshold: 8.0                     # minimum best/worst gap to form a pair
```

Saved checkpoints contain **only the policy LoRA** weights and can be loaded at
inference time (see the root [`infer_dpo.py`](../../infer_dpo.py)).

## Acknowledgment

The training pipeline is forked from
[epipolar-dpo](https://github.com/KupynOrest/epipolar-dpo) and builds on
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and
[Wan2.1](https://github.com/Wan-Video/Wan2.1).
