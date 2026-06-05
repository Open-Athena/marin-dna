"""Cross-check: offline evals_v2 metrics (just computed) vs the online
in-training wandb metric, for the two finished exp232 arms with CLEAN final
steps (utr3, bg). If offline ~ online here, it confirms the two paths agree at
the final step when there's no glitch — isolating ccre/distal/FWD as the anomaly.
"""

from __future__ import annotations

import polars as pl
import wandb

ARMS = ["v4_utr3", "v4_bg"]
METRICS = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics/exp232-{arm}-step-4999/mendelian_traits.parquet"
api = wandb.Api(timeout=40)
proj = "gonzalobenegas/marin"


def online_vals(arm: str) -> dict:
    rs = list(
        api.runs(
            proj,
            filters={"display_name": {"$regex": f"exp232.*{arm}-v0.1$"}},
            per_page=5,
        )
    )
    r = rs[0]
    out = {}
    for k, v in r.summary.items():
        # lm_eval/mendelian_traits_255/<subset>/<strand>/auprc
        parts = k.split("/")
        # lm_eval / mendelian_traits_255 / <subset> / <strand> / auprc  (5 parts)
        if (
            len(parts) == 5
            and parts[0] == "lm_eval"
            and parts[1] == "mendelian_traits_255"
            and parts[4] == "auprc"
        ):
            subset, strand = parts[2], parts[3]
            out[(subset, strand)] = float(v)
    return out


for arm in ARMS:
    off = pl.read_parquet(METRICS.format(arm=arm))
    on = online_vals(arm)
    # offline minus_llr_{fwd,rc,avg} → map to online fwd/rc/avg
    off = off.filter(pl.col("score_type").str.starts_with("minus_llr_"))
    off = off.with_columns(
        pl.col("score_type").str.replace("minus_llr_", "").alias("strand")
    )
    print(
        f"\n{'=' * 78}\n### {arm}-step-4999   offline minus_llr  vs  online (in-training final)\n{'=' * 78}"
    )
    print(f"{'subset':<34}{'strand':<6}{'offline':>9}{'online':>9}{'Δ(off-on)':>11}")
    rows = (
        off.select(["subset", "strand", "value", "n_rows"])
        .sort(["subset", "strand"])
        .to_dicts()
    )
    big = []
    for row in rows:
        sub, strand, val = row["subset"], row["strand"], row["value"]
        onv = on.get((sub, strand))
        if onv is None:
            print(f"{sub:<34}{strand:<6}{val:>9.4f}{'—':>9}{'(no online)':>11}")
            continue
        d = val - onv
        flag = "  <<" if abs(d) > 0.05 else ""
        print(f"{sub:<34}{strand:<6}{val:>9.4f}{onv:>9.4f}{d:>+11.4f}{flag}")
        if abs(d) > 0.05:
            big.append((sub, strand, val, onv, d))
    if big:
        print(
            "  >>> |Δ|>0.05:",
            [(s, st, f"off={o:.3f}", f"on={n:.3f}") for s, st, o, n, _ in big],
        )
    else:
        print(
            "  >>> all |Δ| <= 0.05 — offline reproduces online (no glitch at this final step)"
        )
