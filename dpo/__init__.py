"""Direct Preference Optimization (DPO) with the VIGOR geometry reward.

This package contains the post-hoc alignment pathway: a LoRA adapter is trained
on Wan2.1 with geometry-ranked preference pairs (``dpo/model_training``) and can
be compared against the frozen baseline at inference time (``dpo/inference.py``).
See ``infer_dpo.py`` at the repository root for the CLI.
"""
