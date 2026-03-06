# VIGOR: VIdeo Geometry-Oriented Reward for Temporal Generative Alignment (ECCV 2026)

## News

## Overview

## Installation

## How to train

**Prerequisite: Python 3.10 and a torch version that support cuda 12.8**

* Step 1.  `cd model_training/` and run `pip install -r requirements.txt` to install dependencies.
* Step 2.  run `bash download.bash` to download preprocessed *gb3dv-25k* dataset along with *Wan2.1-T2V-1.3B* model.
* Step 3.  `cd reward_lora/config/` and configure your training settings in `train.yaml`
* Step 4.  `cd reward_lora/` and run `python train.py` to start the training process

**P.S. Under the new setup, to run train_restart.py with sft fintuned model as base model, you should create a folder under reward_lora like reward_lora/lora_ckpt/ and put the sft_stepxxx.ckpt in it. Also be aware to modify the config files.**

## Citation

If you use this code in your research, please cite:

## License

This project is licensed under the MIT License.

## Acknowledgment
