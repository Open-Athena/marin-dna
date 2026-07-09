"""Resolve scaling-ladder checkpoints to a local HF dir for fine-tuning (#369).

Checkpoints are pulled from the evals_v2 S3 checkpoint cache (clean model names, no
run-dir hash), which is populated for every rung the evals pipeline scored. GCS is the
source of truth but needs the per-run hash + gcloud auth; the S3 cache is the easy path.
"""

from __future__ import annotations

from pathlib import Path

# display size -> scaling-v0.5 run stem (all step-215573). From issue #341.
MODEL_STEMS: dict[str, str] = {
    "46M": "h640-p46M",
    "76M": "h768-p76M",
    "128M": "h896-p128M",
    "255M": "h1152-p255M",
    "476M": "h1408-p476M",
    "1B": "h1920-p1B",
    "2B": "h2432-p2B",
    "4B": "h2944-p4B",
}
STEP = "step-215573"
S3_CKPT_CACHE = (
    "s3://oa-bolinas/snakemake/analysis/evals_v2/results/checkpoints"
)


def model_name(size: str) -> str:
    assert size in MODEL_STEMS, f"unknown size {size!r}; pick from {list(MODEL_STEMS)}"
    return f"scaling-v0.5-{MODEL_STEMS[size]}-{STEP}"


def s3_checkpoint_uri(size: str) -> str:
    return f"{S3_CKPT_CACHE}/{model_name(size)}"


def download_checkpoint(size: str, dest_root: str | Path) -> Path:
    """Download the HF checkpoint dir for ``size`` from the S3 cache to ``dest_root``.

    Uses ``s3fs`` (the ``genome-s3`` dep group, already needed for ``Genome``) rather
    than the ``aws`` CLI — the ``aws-cli`` group conflicts with ``genome-s3``. Returns the
    local checkpoint directory (``config.json`` + ``model.safetensors`` + tokenizer).
    Idempotent — a populated dest is reused.
    """
    import s3fs

    dest = Path(dest_root) / model_name(size)
    if (dest / "config.json").exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    src = s3_checkpoint_uri(size).removeprefix("s3://")
    fs = s3fs.S3FileSystem()
    assert fs.exists(src), (
        f"{s3_checkpoint_uri(size)} not in the S3 checkpoint cache — fall back to GCS"
    )
    fs.get(src + "/", str(dest) + "/", recursive=True)
    assert (dest / "config.json").exists(), (
        f"no config.json under {dest} after sync from {s3_checkpoint_uri(size)}"
    )
    return dest
