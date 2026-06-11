"""End-to-end validation for the #310 caQTL/dsQTL datasets.

Reproduces the #262 *all-chroms* ChromBPNet / Enformer baseline numbers directly
from the built dataset's **carried** score columns. This is the sign guard for the
orientation fix: ``|score|`` (causality auPRC) is swap-invariant, and signed-Pearson
(direction) is invariant only if each signed score column was flipped together with
``effect`` on a ref/alt swap (``orient_variants``). A sign bug would surface here as a
collapsed or inverted direction Pearson.

Reads the all-chroms ``dataset_unsplit`` parquet (default: the pipeline's S3 outputs;
override with ``--caqtl`` / ``--dsqtl`` paths) and prints the protocol table per
dataset, asserting the ChromBPNet-ATAC headline reproduces #262 within tolerance.

    uv run python scripts/issue310_validate.py
    uv run python scripts/issue310_validate.py --caqtl results/dataset_unsplit/caqtl.parquet ...
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import polars as pl

from marin_dna.pipelines.chrombpnet_eval.metrics import compute_supervised_qtl_metrics

S3_BASE = "s3://oa-bolinas/snakemake/evals/results/dataset_unsplit"

# #262 reported all-chroms numbers (causality auPRC, direction Pearson) + the
# positive-rate random-auPRC baseline. The carried columns must reproduce these.
REFERENCE = {
    "caqtl": {
        "random_auprc": 0.0852,
        "ChromBPNet ATAC": (0.428, 0.689),
        "ChromBPNet DNase": (0.375, 0.674),
        "Enformer DNase": (0.412, 0.637),
    },
    "dsqtl": {
        "random_auprc": 0.0200,
        "ChromBPNet ATAC": (0.538, 0.759),
        "ChromBPNet DNase": (0.429, 0.739),
        "Enformer DNase": (0.526, 0.727),
    },
}

# (model label, causality column [|·|], direction column [signed]).
BASELINES = [
    ("ChromBPNet ATAC", "chrombpnet_atac_ips", "chrombpnet_atac_logfc"),
    ("ChromBPNet DNase", "chrombpnet_dnase_ips", "chrombpnet_dnase_logfc"),
    ("Enformer DNase", "enformer_dnase_local_logfc", "enformer_dnase_local_logfc"),
]

# Tolerance vs #262: caQTL is native hg38 (orientation-invariant -> exact); dsQTL drops
# ~27 variants to liftover, a negligible shift. 0.01 covers both + rounding.
TOL = 0.012


def _metric(df: pd.DataFrame, col: str, metric: str) -> float:
    """One metric for one score column (dropping non-finite scores first)."""
    sub = df[["label", "effect", col]].rename(columns={col: "score"})
    sub = sub[np.isfinite(sub["score"].to_numpy())]
    m = compute_supervised_qtl_metrics(sub, score_col="score", n_bootstrap=0)
    return float(m.set_index("metric").loc[metric, "value"])


def validate_dataset(name: str, path: str) -> bool:
    # polars reads S3 natively (object_store) — no s3fs; convert for the metric fn.
    df = pl.read_parquet(path).to_pandas()
    n = len(df)
    n_pos = int(df["label"].sum())
    print(
        f"\n=== {name}  (n={n:,}, pos={n_pos:,}, rate={n_pos / n:.4f}, "
        f"random_auPRC {REFERENCE[name]['random_auprc']}) ==="
    )
    print(f"{'model':16} {'causality auPRC':>16} {'direction Pearson':>18}   vs #262")
    ok = True
    for model, caus_col, dir_col in BASELINES:
        if caus_col not in df.columns or dir_col not in df.columns:
            print(f"{model:16}  (columns absent — skipped)")
            continue
        auprc = _metric(df, caus_col, "AUPRC")
        pear = _metric(df, dir_col, "pearson")
        ref_a, ref_p = REFERENCE[name][model]
        da, dp = auprc - ref_a, pear - ref_p
        print(
            f"{model:16} {auprc:16.4f} {pear:18.4f}   "
            f"(#262 {ref_a:.3f}/{ref_p:.3f}; Δ {da:+.4f}/{dp:+.4f})"
        )
        if model == "ChromBPNet ATAC":
            if abs(da) > TOL or abs(dp) > TOL:
                ok = False
                print(
                    f"  !! {name} ChromBPNet-ATAC off by >{TOL}: "
                    f"Δcausality={da:+.4f} Δdirection={dp:+.4f}"
                )
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caqtl", default=f"{S3_BASE}/caqtl.parquet")
    ap.add_argument("--dsqtl", default=f"{S3_BASE}/dsqtl.parquet")
    args = ap.parse_args()
    ok = True
    ok &= validate_dataset("caqtl", args.caqtl)
    ok &= validate_dataset("dsqtl", args.dsqtl)
    assert ok, "baseline reproduction FAILED — suspect a score-column sign bug"
    print("\nOK — carried baselines reproduce #262 (orientation sign-flip correct).")


if __name__ == "__main__":
    main()
