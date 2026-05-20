"""Shared imports and constants for ``zoonomia_projection_dataset``.

URLs come from ``CONSERVATION_TRACKS`` — single source of truth in
``src/marin_dna/evals/conservation.py``. Only ``phyloP_447m`` is downloaded
at pipeline runtime; ``phyloP_241m`` is fetched only by the one-off
calibration script.

Pipeline is hg38-only by design — the phyloP bigWigs are human-anchored
and the downstream cross-mammal projection is human-anchored too. No
``{species}`` wildcard.
"""

import gzip
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from marin_dna.pipelines.evals.conservation import CONSERVATION_TRACKS


def _git_commit_sha() -> str:
    """Return the current git commit SHA, or ``"main"`` if git is unavailable.

    Resolution order:
      1. ``--config commit_sha=...`` override (or env var ``COMMIT_SHA``),
         used by ``sky/upload.yaml`` to forward the launcher's SHA to the
         cluster — needed because a synced git *worktree* has a ``.git``
         file pointing at the host's main repo path, which doesn't exist
         on the cluster, so ``git rev-parse HEAD`` would fail there.
      2. ``git rev-parse HEAD`` from the workflow dir (works on the host
         and on any cluster with a real ``.git`` directory synced in).
      3. The literal string ``"main"`` (fallback so permalinks still
         resolve, just to the head of main instead of a pinned commit).

    Used by per-repo HF dataset cards (validation, region_labels) to
    embed a commit-pinned permalink. Resolved once at Snakefile-load time
    so all cards in a single run reference the same SHA.
    """
    override = config.get("commit_sha") or os.environ.get("COMMIT_SHA")
    if override and len(override) == 40:
        return override
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workflow.basedir,
            capture_output=True,
            text=True,
            check=False,
        )
        sha = result.stdout.strip()
        if result.returncode == 0 and len(sha) == 40:
            return sha
    except FileNotFoundError:
        pass
    return "main"


GIT_COMMIT_SHA = _git_commit_sha()


WINDOW_SIZE = int(config["window_size"])
STEP_SIZE = int(config["step_size"])
PHYLOP_447M_THRESHOLD = float(config["phyloP_447m_threshold"])
STANDARD_CHROMS = list(config["standard_chroms"])

# Sanity-check: track must be in the registry.
assert "phyloP_447m" in CONSERVATION_TRACKS, (
    "phyloP_447m must be present in marin_dna.pipelines.evals.conservation.CONSERVATION_TRACKS"
)


# ===== Cross-mammal projection knobs (rule all_projected) =====

HAL_PATH = config["hal_path"]
SPECIES_TSV = config["species_tsv"]
PROJECT_MIN_P = str(config["project_min_p"])
TARGET_LEN = int(config["target_len"])
PRE_RESIZE_MIN = int(config["pre_resize_min_len"])
PRE_RESIZE_MAX = int(config["pre_resize_max_len"])
TIER = config.get("tier", "full")
assert TIER in {"full", "smoke"}, f"tier must be full or smoke, got {TIER!r}"
assert TARGET_LEN == WINDOW_SIZE, (
    f"projection target_len ({TARGET_LEN}) must equal source WINDOW_SIZE "
    f"({WINDOW_SIZE}); the 255+BOS=256 invariant decouples from the input "
    f"window length only intentionally."
)
assert 0 < PRE_RESIZE_MIN <= TARGET_LEN <= PRE_RESIZE_MAX

SPECIES = pl.read_csv(SPECIES_TSV, separator="\t")["species"].to_list()
assert {"Homo_sapiens", "Mus_musculus", "Bos_taurus"}.issubset(SPECIES), (
    "species TSV must include Homo_sapiens, Mus_musculus, Bos_taurus "
    "(force-included by the dedup policy)"
)
if TIER == "smoke":
    SPECIES = ["Homo_sapiens", "Mus_musculus", "Bos_taurus", "Loxodonta_africana"]
