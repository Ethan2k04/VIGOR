"""Shared helpers for the VIGOR command-line entrypoints.

Kept intentionally light so that ``inference_tts.py``, ``run_bon.py`` and
``infer_dpo.py`` share a single prompt-loading convention.
"""

import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROMPTS = os.path.join(REPO_ROOT, "prompts", "demos.txt")


def load_prompts(path: str = DEFAULT_PROMPTS):
    """Read a prompt-per-line file.

    Each non-empty line is one prompt. Lines may optionally be prefixed with an
    index and a tab (``<idx>\\t<prompt>``); the index is stripped. Blank lines
    and lines starting with ``#`` are ignored.
    """
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2 and parts[0].strip().isdigit():
                line = parts[1].strip()
            prompts.append(line)
    return prompts
