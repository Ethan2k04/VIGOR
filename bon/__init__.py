"""Best-of-N (BoN) sampling with the VIGOR geometry reward.

Generate ``N`` candidate videos per prompt (different seeds) with a Wan2.1
text-to-video pipeline, score each candidate with the geometry-based reward,
and keep the best one. See ``run_bon.py`` at the repository root for the CLI.
"""
