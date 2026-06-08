"""Cross-scale analysis for issue #296 — stratified LL gap vs missense VEP AUPRC.

Collects the per-model Stage-2 stratum LL gaps
(``scratch/issue296/stage2/<model>/val_cds_stratum_ll_gap.parquet``) and the
per-model **missense** VEP AUPRC (``minus_llr_avg`` on ``missense_variant`` from
the evals_v2 ``mendelian_traits`` metrics, the #279 y-axis), aligns them across
the scaling-v0.5 ladder, and asks: does any stratum's LL gap — especially
``LLgap | {codon 1,2}`` — predict missense AUPRC better (more monotonically) than
the vanilla all-token gap?

Correlations use only the models found on disk (run the smaller ones first), so
the table grows as more checkpoints are cached. CPU-only. One-off.

    uv run python scripts/issue296/scaling_analysis.py
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import polars as pl

METRICS_ROOT = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics"


def _params_millions(model: str) -> float:
    m = re.search(r"p(\d+(?:\.\d+)?)([MB])", model)
    assert m, f"no param token in {model!r}"
    return float(m.group(1)) * (1000.0 if m.group(2) == "B" else 1.0)


def _vep_auprc(model: str, subset: str) -> float:
    df = pl.read_parquet(f"{METRICS_ROOT}/{model}/mendelian_traits.parquet")
    r = df.filter(
        (pl.col("subset") == subset) & (pl.col("score_type") == "minus_llr_avg")
    )
    assert len(r) == 1, f"{model}/{subset}: expected 1 row, got {len(r)}"
    return float(r["value"][0])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ = Pearson on ranks (no scipy; fine for the small ladder)."""
    if len(x) < 3 or np.isnan(x).any():
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage2-dir", default="scratch/issue296/stage2")
    ap.add_argument("--out-dir", default="scratch/issue296")
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(f"{args.stage2_dir}/*/val_cds_stratum_ll_gap.parquet")):
        model = Path(p).parent.name
        gaps = {
            r["stratum"]: r["gap"] for r in pl.read_parquet(p).iter_rows(named=True)
        }
        rows.append(
            {
                "model": model,
                "params_M": _params_millions(model),
                "missense_auprc": _vep_auprc(model, "missense_variant"),
                "synonymous_auprc": _vep_auprc(model, "synonymous_variant"),
                "splicing_auprc": _vep_auprc(model, "splicing"),
                **gaps,
            }
        )
    assert rows, f"no stratum parquets under {args.stage2_dir}"
    tab = pl.DataFrame(rows).sort("params_M")
    meta = ["model", "params_M", "missense_auprc", "synonymous_auprc", "splicing_auprc"]
    strata = [c for c in tab.columns if c not in meta]

    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(f"[scaling] {len(tab)} models:")
        print(tab.select(["model", "params_M", "missense_auprc", *strata]))

    # Per-stratum correlation of the stratum's LL gap vs missense AUPRC.
    y = tab["missense_auprc"].to_numpy()
    corr = pl.DataFrame(
        [
            {
                "stratum": s,
                "spearman_vs_missense": _spearman(tab[s].to_numpy(), y),
                "pearson_vs_missense": (
                    float(np.corrcoef(tab[s].to_numpy(), y)[0, 1])
                    if not np.isnan(tab[s].to_numpy()).any() and len(tab) >= 2
                    else float("nan")
                ),
            }
            for s in strata
        ]
    ).sort("spearman_vs_missense", descending=True)
    print(
        f"\n[scaling] per-stratum LL-gap correlation with missense AUPRC "
        f"(n={len(tab)} models; vanilla baseline = `all_token`):"
    )
    with pl.Config(tbl_rows=-1):
        print(corr)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tab.write_parquet(out / "scaling_table.parquet")
    corr.write_parquet(out / "scaling_corr.parquet")
    print(f"\n[scaling] wrote scaling_table.parquet + scaling_corr.parquet → {out}")


if __name__ == "__main__":
    main()
