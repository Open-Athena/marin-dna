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

    meta_cols = [
        "model",
        "params_M",
        "missense_auprc",
        "synonymous_auprc",
        "splicing_auprc",
    ]
    rows_gap, rows_loss = [], []
    for p in sorted(glob.glob(f"{args.stage2_dir}/*/val_cds_stratum_ll_gap.parquet")):
        model = Path(p).parent.name
        df = pl.read_parquet(p)
        meta = {
            "model": model,
            "params_M": _params_millions(model),
            "missense_auprc": _vep_auprc(model, "missense_variant"),
            "synonymous_auprc": _vep_auprc(model, "synonymous_variant"),
            "splicing_auprc": _vep_auprc(model, "splicing"),
        }
        rows = {r["stratum"]: r for r in df.iter_rows(named=True)}
        rows_gap.append({**meta, **{s: r["gap"] for s, r in rows.items()}})
        rows_loss.append({**meta, **{s: r["mean_loss"] for s, r in rows.items()}})

    assert rows_gap, f"no stratum parquets under {args.stage2_dir}"
    tab_gap = pl.DataFrame(rows_gap).sort("params_M")
    tab_loss = pl.DataFrame(rows_loss).sort("params_M")
    strata = [c for c in tab_gap.columns if c not in meta_cols]

    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=50):
        print(f"[scaling] {len(tab_gap)} models — per-stratum **LL gap**:")
        print(tab_gap.select(["params_M", "missense_auprc", *strata]))
        print("\n[scaling] per-stratum **mean loss** (nats; lower = better predicted):")
        print(tab_loss.select(["params_M", "missense_auprc", *strata]))

    # Per-stratum correlation of BOTH the gap and the mean loss vs missense AUPRC.
    y = tab_gap["missense_auprc"].to_numpy()
    corr = pl.DataFrame(
        [
            {
                "stratum": s,
                "gap_spearman": _spearman(tab_gap[s].to_numpy(), y),
                "meanloss_spearman": _spearman(tab_loss[s].to_numpy(), y),
            }
            for s in strata
        ]
    ).sort("gap_spearman", descending=True)
    print(
        f"\n[scaling] Spearman ρ of each stratum's gap / mean-loss vs missense "
        f"AUPRC (n={len(tab_gap)} models; vanilla baseline = `all_token`):"
    )
    with pl.Config(tbl_rows=-1):
        print(corr)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tab_gap.write_parquet(out / "scaling_gap.parquet")
    tab_loss.write_parquet(out / "scaling_meanloss.parquet")
    corr.write_parquet(out / "scaling_corr.parquet")
    print(f"\n[scaling] wrote scaling_gap / scaling_meanloss / scaling_corr → {out}")


if __name__ == "__main__":
    main()
