"""Build evals_sge v3 (AUPRC-only) from the existing v2 unsplit parquet (#301).

v3 = v2 + a boolean ``label`` (True = impactful / calibrated abnormal, False =
normal) + a row filter to label-non-null variants (drops intermediate +
uncalibrated/null incl. BRCA2). This is a *targeted* build: it applies the
committed ``attach_label`` and the same chrom-parity split the
``split_dataset_by_chrom`` rule uses, directly to the v2 unsplit parquet — so it
re-derives nothing (no genome re-stage, no cdot re-recode) and is guaranteed to
equal "v2 labeled-subset", keeping the AUPRC numbers identical. The pipeline's
``sge.smk`` carries the same label+filter, so a future full run reproduces this.

Outputs (gitignored scratch): v3 train/test parquets + the rendered dataset card,
for review before the HF upload (a separate, gated step).

Run: ``uv run python scripts/issue301_build_sge_v3.py``
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import polars as pl

from marin_dna.pipelines.evals import hf_readme
from marin_dna.pipelines.evals.sge import attach_label

S3 = "s3://oa-bolinas/snakemake/evals/results"
SO = {"aws_region": "us-east-2"}
OUT = Path(__file__).resolve().parents[1] / "scratch" / "sge_v3"

# Mirror snakemake/evals common.smk: CHROMS[::2] → train (odd + X), [1::2] → test.
CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]
SPLIT_CHROMS = {"train": CHROMS[::2], "test": CHROMS[1::2]}
# PRIMARY_COLS = COORDS + [label, subset, match_group]; SGE has no match_group.
PRIMARY_COLS = ["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]


def _reorder(df: pl.DataFrame) -> pl.DataFrame:
    primary = [c for c in PRIMARY_COLS if c in df.columns]
    return df.select(primary + [c for c in df.columns if c not in primary])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    unsplit = pl.read_parquet(f"{S3}/dataset_unsplit/sge.parquet", storage_options=SO)
    print(f"v2 unsplit: {unsplit.height} rows")

    labeled = attach_label(unsplit).filter(pl.col("label").is_not_null())
    n_abn = labeled.filter(pl.col("label")).height
    n_norm = labeled.filter(~pl.col("label")).height
    print(f"v3 labeled: {labeled.height} rows ({n_abn} abnormal / {n_norm} normal)")
    assert labeled.height == n_abn + n_norm
    # Every shipped row carries a clean bool (the filter guarantees it).
    assert labeled["label"].null_count() == 0

    labeled = _reorder(labeled)
    paths = {}
    for split, chroms in SPLIT_CHROMS.items():
        part = labeled.filter(pl.col("chrom").is_in(chroms))
        path = OUT / f"{split}.parquet"
        part.write_parquet(path)
        paths[split] = path
        print(
            f"  {split}: {part.height} rows "
            f"({part.filter(pl.col('label')).height} abnormal / "
            f"{part.filter(~pl.col('label')).height} normal) "
            f"chroms={sorted(part['chrom'].unique().to_list())}"
        )
    assert sum(pl.read_parquet(p).height for p in paths.values()) == labeled.height

    # Calibration companion (for the card's Score-calibration section).
    calib = OUT / "calibrations.parquet"
    pl.read_parquet(f"{S3}/sge/calibrations.parquet", storage_options=SO).write_parquet(
        calib
    )

    card = hf_readme.render(
        "sge", sha, paths["train"], paths["test"], calibration_path=calib
    )
    (OUT / "README.md").write_text(card)
    print(f"\nwrote {OUT}/ (train/test/calibrations parquet + README.md); sha={sha[:7]}")


if __name__ == "__main__":
    main()
