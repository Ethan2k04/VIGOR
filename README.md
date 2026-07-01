<div align="center">

# VIGOR: VIdeo Geometry-Oriented Reward for Temporal Generative Alignment

<b>Tengjiao Yin<sup>1</sup>, Jinglei Shi<sup>1,†</sup>, Heng Guo<sup>2</sup>, Xi Wang<sup>3</sup></b>

<sup>1</sup>VCIP &amp; TMCC &amp; DISSec, College of Computer Science, Nankai University&nbsp;&nbsp;
<sup>2</sup>Beijing University of Posts and Telecommunications&nbsp;&nbsp;
<sup>3</sup>LIX, École Polytechnique, IP Paris

<sup>†</sup>Corresponding author

**European Conference on Computer Vision (ECCV) 2026**

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://vigor-geometry-reward.com/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.16271-b31b1b.svg)](https://arxiv.org/abs/2603.16271)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-GB3DV--25k-yellow)](https://huggingface.co/datasets/Ethan2k04/GB3DV-25k)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## News

- **[2026-07]** VIGOR is accepted to **ECCV 2026**! 🎉
- **[2026-07]** Code, the **GB3DV-25k** dataset, and pretrained LoRA adapters are released.

## Overview

![overview](figure/overview.png)

Modern video generators produce photorealistic frames but frequently violate the
underlying **3D geometry** of a scene — objects drift, camera trajectories bend,
and multi-view consistency breaks down. **VIGOR** is a geometry-oriented reward
that measures this temporal-geometric fidelity directly, without ground-truth
depth or camera poses.

Given a generated clip, VIGOR runs a pretrained **VGGT** to recover per-frame
geometry and camera parameters, then computes a **pointwise cross-frame
reprojection error**: salient 3D points are reprojected into neighbouring views
and their displacement is measured in pixel space. To focus the signal on
structurally meaningful regions, **Geometry-Aware Sampling (GAS)** uses VGGT's
shallow global-attention maps to select salient patches and filters out
ill-conditioned sky/ground regions.

VIGOR plugs into three complementary alignment pathways:

| Pathway | Model type | Where the reward is used | Entrypoint |
| --- | --- | --- | --- |
| **Best-of-N (BoN)** | any T2V model (Wan2.1) | rank *N* i.i.d. samples, keep the best | [`run_bon.py`](run_bon.py) |
| **Test-Time Scaling (TTS)** | causal autoregressive | reward as a path verifier during decoding (SoS / SoP / Beam Search) | [`inference_tts.py`](inference_tts.py) |
| **Post-hoc DPO** | bidirectional (Wan2.1) | preference optimization with a geometry-ranked LoRA | [`infer_dpo.py`](infer_dpo.py) |

Three reward variants are provided under [`rewards/evaluator`](rewards/evaluator):
`reproj_pts` (pointwise reprojection, **ours**), `reproj_pix` (pixel-space warp),
and `epipolar` (Sampson distance).

## Repository Structure

```text
VIGOR/
├── inference_tts.py          # Test-time scaling entrypoint (causal AR model)
├── run_bon.py                # Best-of-N sampling entrypoint
├── infer_dpo.py              # DPO: LoRA vs. baseline comparison entrypoint
├── common.py                 # Shared prompt-loading helper
├── prompts/
│   └── demos.txt             # Default prompt suite for all entrypoints
├── tts/                      # Test-time scaling core
│   ├── tts_common.py
│   └── algorithms/           # sos.py · sop.py · bs.py
├── bon/                      # Best-of-N core
│   └── best_of_n.py
├── dpo/                      # Post-hoc DPO alignment
│   ├── inference.py          # LoRA vs. baseline generation + comparison
│   └── model_training/       # DPO / SFT LoRA training (diffsynth + peft)
│       └── reward_lora/
├── rewards/                  # VIGOR geometry reward
│   ├── evaluator/            # reproj_pts · reproj_pix · epipolar
│   └── evaluate.py           # video evaluation (single folder)
├── third_party/              # Git submodules (external references)
│   ├── vggt/                 # Visual Geometry Grounded Transformer
│   └── Causal-Forcing/       # Causal autoregressive video backbone
├── figure/
└── requirements.txt
```

## Installation

VIGOR pins its external backbones (VGGT, Causal-Forcing) as **git submodules**,
so clone recursively:

```bash
git clone --recurse-submodules https://github.com/Ethan2k04/VIGOR.git
cd VIGOR
# already cloned without submodules? pull them now:
git submodule update --init --recursive
```

Create the environment (Python **3.10**), then install the third_party backends
first (they own `torch` / `torchvision` / `numpy`) and the VIGOR dependencies on
top:

```bash
conda create -n vigor python=3.10 -y
conda activate vigor

# 1) third_party backends
pip install -r third_party/vggt/requirements.txt            # required — geometry reward + TTS scoring
pip install -r third_party/Causal-Forcing/requirements.txt  # TTS only — causal AR backbone
# also download the Causal-Forcing checkpoints as per their README

# 2) VIGOR dependencies
pip install -r requirements.txt
```

`Causal-Forcing` is only needed for Test-Time Scaling; skip it for reward
evaluation, BoN, or DPO. COLMAP and the Gradio 3D demo are optional — see the
notes at the bottom of [`requirements.txt`](requirements.txt).

### Checkpoints & data

| Asset | Purpose | Notes |
| --- | --- | --- |
| **Wan2.1-T2V-1.3B** | base T2V model for BoN & DPO | expects `diffusion_pytorch_model.safetensors`, `models_t5_umt5-xxl-enc-bf16.pth`, `Wan2.1_VAE.pth` in one directory (`--model_path`) |
| **Causal-Forcing** checkpoint | causal AR backbone for TTS | see [`third_party/Causal-Forcing`](third_party/Causal-Forcing); default config/ckpt paths in [`inference_tts.py`](inference_tts.py) |
| **VGGT** weights | geometry reward backend | fetched automatically on first use |
| **skyseg.onnx** | sky masking for the geometry reward | download to the repo root — see below (skip if `enable_sky_onnx` is `false`) |
| **GB3DV-25k** | 25,600 geometry-ranked video pairs for DPO | `bash dpo/model_training/download.sh` (pulls `Ethan2k04/GB3DV-25k`) |

The geometry reward filters out ill-conditioned sky regions with an ONNX
segmentation model. The reward looks for `skyseg.onnx` in the working directory
you launch from (the repo root, per the commands below), so download it once into
the repository root:

```bash
wget https://huggingface.co/spaces/facebook/vggt/resolve/main/skyseg.onnx -O skyseg.onnx
```

Override the location with `sky_onnx_path` in
[`rewards/config/config.json`](rewards/config/config.json), or set
`enable_sky_onnx` to `false` there to disable sky filtering entirely.

## Usage

All entrypoints read prompts from [`prompts/demos.txt`](prompts/demos.txt) by
default (one prompt per line); override with `--data_path`.

### Best-of-N (BoN)

Generate *N* candidates per prompt with Wan2.1, score each with the geometry
reward, and keep the best:

```bash
python run_bon.py \
    --model_path /path/to/Wan2.1-T2V-1.3B \
    --num_candidates 8 \
    --metric reprojection \
    --output_dir outputs/bon
```

Winners are written to `outputs/bon/best/` alongside a `bon_manifest.json`.
Use `--save_all_candidates` to keep every sample and `--metric {reprojection,reprojection_vanilla,epipolar}` to switch reward.

### Test-Time Scaling (TTS)

Use the reward as a path verifier while decoding the causal autoregressive model.
Choose a search strategy with `--algorithm {sos,sop,bs}`:

```bash
python inference_tts.py \
    --algorithm sos \
    --data_path prompts/demos.txt \
    --output_folder outputs/tts
```

- **SoS** — *Search on Start*: branch at the first block, then decode greedily.
- **SoP** — *Search on Path*: branch and prune at every block.
- **BS**  — *Beam Search*: maintain a beam of geometry-verified paths.

### Post-hoc DPO

Compare a geometry-aligned LoRA against the frozen Wan2.1 baseline (same seed,
side-by-side output):

```bash
python infer_dpo.py \
    --model_path /path/to/Wan2.1-T2V-1.3B \
    --lora_path  /path/to/vigor_lora.safetensors \
    --output_dir outputs/dpo
```

Baseline / LoRA / stacked comparison videos are written under `outputs/dpo/`.

To **train** the LoRA yourself, download GB3DV-25k, build the preference latents,
and launch DPO training (Hydra config in `reward_lora/config/train.yaml`):

```bash
cd dpo/model_training
bash download.sh                       # GB3DV-25k + weights
python preprocess_dpo_data.py          # build preference latents
cd reward_lora
python train.py                        # DPO LoRA (override cfg via Hydra, e.g. beta=500)
```

### Evaluation

Score every video in a folder with the geometry reward (input/output paths and
metrics are set in the config):

```bash
python -m rewards.evaluate --config rewards/config/config.json
```

## Citation

If you find VIGOR useful in your research, please cite:

```bibtex
@inproceedings{yin2026vigor,
  title     = {VIGOR: VIdeo Geometry-Oriented Reward for Temporal Generative Alignment},
  author    = {Yin, Tengjiao and Shi, Jinglei and Guo, Heng and Wang, Xi},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Acknowledgment

VIGOR builds on excellent open-source work, including
[VGGT](https://github.com/facebookresearch/vggt),
[Causal-Forcing](https://github.com/thu-ml/Causal-Forcing),
[Wan2.1](https://github.com/Wan-Video/Wan2.1), and
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio).
We thank the authors for releasing their code and models.

## License

This project is licensed under the terms of the **MIT License**.
