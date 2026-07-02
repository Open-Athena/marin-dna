"""SGE-AUPRC regression check for the #354 benchmark scores (offline, CPU).

The benchmark's ``sge_scores.parquet`` is row-aligned with the HF SGE dataset
(the benchmark kept ``label``/``subset`` to prove it). Re-attach ``mavedb_urn``/
``gene`` from HF, derive the ``minus_llr_*``/``jsd_*`` score columns exactly as
``metrics.smk`` does, rerun ``compute_sge_metrics``, and compare the macro-avg
AUPRC to the existing official cell — confirming the compiled GH200/A10G scoring
didn't regress.

    uv run python scripts/issue354/check_sge_auprc.py --scores issue354_out/sge_scores.parquet
"""

from __future__ import annotations

import argparse

import pandas as pd
import polars as pl
from datasets import load_dataset

from marin_dna.pipelines.evals.metrics import compute_sge_metrics

HF_DATASET = "bolinas-dna/evals_sge"
HF_REVISION = "225d3d1ea32a4af547891b13c33b5e92a5aae849"
OFFICIAL = (
    "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics/"
    "mix-v0.9-p1B-i24-exp135-m5.1-step-59158/sge.parquet"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="benchmark sge_scores.parquet")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    args = ap.parse_args()

    ds = load_dataset(HF_DATASET, split="train", revision=HF_REVISION).to_pandas()
    sc = pd.read_parquet(args.scores)

    # Row-alignment proof (the benchmark scored the dataset in order).
    assert len(ds) == len(sc), f"len mismatch {len(ds)} vs {len(sc)}"
    assert (ds["label"].to_numpy() == sc["label"].to_numpy()).all(), "label misaligned"
    assert (
        ds["subset"].astype(str).to_numpy() == sc["subset"].astype(str).to_numpy()
    ).all(), "subset misaligned"

    # Derive score columns exactly as metrics.smk (average raw LLR, then negate).
    sc["llr_avg"] = (sc["llr_fwd"] + sc["llr_rc"]) / 2
    sc["jsd_avg"] = (sc["jsd_fwd"] + sc["jsd_rc"]) / 2
    for s in ("fwd", "rc", "avg"):
        sc[f"minus_llr_{s}"] = -sc[f"llr_{s}"]
    score_cols = [
        "minus_llr_fwd",
        "jsd_fwd",
        "minus_llr_rc",
        "jsd_rc",
        "minus_llr_avg",
        "jsd_avg",
    ]

    mine = compute_sge_metrics(
        dataset=ds[["mavedb_urn", "gene", "subset", "label"]],
        scores=sc[score_cols],
        score_columns=score_cols,
        n_bootstrap=args.n_bootstrap,
        rng=0,
    )
    mine_macro = mine[
        (mine["accession"] == "_macro_avg_") & (mine["subset"] == "_macro_avg_")
    ].set_index("score_type")["value"]

    off = pl.read_parquet(OFFICIAL).to_pandas()
    off_macro = off[
        (off["accession"] == "_macro_avg_") & (off["subset"] == "_macro_avg_")
    ].set_index("score_type")["value"]

    print(f"{'score_type':<16}{'benchmark':>12}{'official':>12}{'abs_diff':>12}")
    ok = True
    for st in score_cols:
        b, o = (
            float(mine_macro.get(st, float("nan"))),
            float(off_macro.get(st, float("nan"))),
        )
        d = abs(b - o)
        ok = ok and (d < 0.01)
        print(f"{st:<16}{b:>12.4f}{o:>12.4f}{d:>12.4f}")
    print(f"\nregression check: {'PASS' if ok else 'REVIEW'} (all |Δ| < 0.01)")


if __name__ == "__main__":
    main()
