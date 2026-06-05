"""Issue #274: does the functional/non-functional LL gap predict VEP better than
loss across the 8-model parameter-scaling ladder?

Computes Spearman ρ of VEP AUPRC vs two predictors — validation loss and the LL
gap — per Mendelian variant type, across the 8 `dna-bolinas-scaling-v0.5` sizes
(46M→4B, step-215573). Everything is reproducible from W&B + the evals_v2
pipeline outputs (no external/blog data):

  - loss + in-training Mendelian AUPRC: W&B (eric-czech/marin scaling runs).
  - LL gap per (model, region): the evals_v2 `ll_gap` summary parquet (S3/local).
  - [primary VEP] evals_v2 Mendelian AUPRC per (model, variant): the
    `results/metrics/<model>/mendelian_traits.parquet` parquets (S3/local).

Usage:
    uv run python scripts/issue274_scaling_correlation.py \
        --gap-summary s3://oa-bolinas/snakemake/analysis/evals_v2/results/ll_gap/summary.parquet \
        --metrics-prefix s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics \
        --out scratch/issue274/correlation.parquet
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr

WANDB_ENTITY = "eric-czech/marin"
SIZES = [
    "h640-p46M",
    "h768-p76M",
    "h896-p128M",
    "h1152-p255M",
    "h1408-p476M",
    "h1920-p1B",
    "h2432-p2B",
    "h2944-p4B",
]
# Mendelian TraitGym variant types (in-training lm_eval naming; evals_v2 uses the
# same consequence names as subsets).
VARIANTS = [
    "missense_variant",
    "5_prime_UTR_variant",
    "3_prime_UTR_variant",
    "splicing",
    "synonymous_variant",
    "tss_proximal",
    "distal",
    "non_coding_transcript_exon_variant",
    "all",
]
# Which gap region best matches each variant type's locus (for the region-matched
# comparison). cds = v5, upstream/promoter = v1, downstream/3'UTR = v15.
VARIANT_REGION = {
    "missense_variant": "cds",
    "synonymous_variant": "cds",
    "splicing": "cds",
    "5_prime_UTR_variant": "upstream",
    "tss_proximal": "upstream",
    "3_prime_UTR_variant": "downstream",
    "non_coding_transcript_exon_variant": "downstream",
    "distal": "macro",
    "all": "macro",
}

_SIZE_RE = re.compile(r"(h\d+-p[\dA-Za-z]+)")


def size_token(name: str) -> str:
    """Extract the `h<width>-p<size>` token from a run/model/dir name."""
    m = _SIZE_RE.search(name)
    assert m, f"no size token in {name!r}"
    return m.group(1)


def pull_wandb_baseline(entity: str = WANDB_ENTITY) -> pd.DataFrame:
    """Per-size loss (overall + per region) and in-training Mendelian AUPRC from W&B."""
    import wandb

    api = wandb.Api()
    rows = []
    for s in SIZES:
        dn = f"dna-bolinas-scaling-v0.5-{s}"
        cands = [
            r
            for r in api.runs(entity, filters={"display_name": dn})
            if r.summary.get("global_step") == 215573
        ]
        assert len(cands) == 1, f"{dn}: expected 1 final run, got {len(cands)}"
        r = cands[0]
        d: dict[str, object] = {
            "size": s,
            "params": r.summary.get("parameter_count"),
            "eval_loss": r.summary.get("eval/loss"),
            "loss_cds": r.summary.get("eval/val_cds/loss"),
            "loss_upstream": r.summary.get("eval/val_upstream/loss"),
            "loss_downstream": r.summary.get("eval/val_downstream/loss"),
        }
        for v in VARIANTS:
            d[f"intrain_auprc_{v}"] = r.summary.get(
                f"lm_eval/traitgym_mendelian_v2_255/{v}/auprc"
            )
        rows.append(d)
    df = pd.DataFrame(rows).sort_values("params").reset_index(drop=True)
    assert df["eval_loss"].notna().all(), "missing eval_loss for some size"
    return df


def load_gap_by_size(gap_summary_path: str) -> pd.DataFrame:
    """Pivot the ll_gap summary to one row per size: gap per region + macro gap."""
    g = pl.read_parquet(gap_summary_path).to_pandas()
    # Keep only the scaling ladder; the summary also holds the m5.1 gate row,
    # which has no h<width>-p<size> token.
    g = g[g["model"].str.contains(r"scaling-v0\.5-h\d+-p", regex=True)].copy()
    g["size"] = g["model"].map(size_token)
    wide = g.pivot(index="size", columns="region", values="gap")
    wide.columns = [f"gap_{c}" for c in wide.columns]
    wide["gap_macro"] = wide.mean(axis=1)  # mean of the per-region gaps
    return wide.reset_index()


def load_evals_v2_vep(metrics_prefix: str, score_type: str) -> pd.DataFrame | None:
    """Per-size evals_v2 Mendelian AUPRC per variant subset, for one score_type.

    Returns one row per size with `v2_auprc_<variant>` columns, or None if the
    metrics parquets aren't present yet.
    """
    rows = []
    for s in SIZES:
        model = f"scaling-v0.5-{s}-step-215573"
        path = f"{metrics_prefix}/{model}/mendelian_traits.parquet"
        try:
            m = pl.read_parquet(path).to_pandas()
        except Exception:
            return None
        m = m[m["score_type"] == score_type]
        d: dict[str, object] = {"size": s}
        for v in VARIANTS:
            subset = "_global_" if v == "all" else v
            hit = m[m["subset"] == subset]["value"]
            d[f"v2_auprc_{v}"] = float(hit.iloc[0]) if len(hit) else np.nan
        rows.append(d)
    return pd.DataFrame(rows)


def rho_comparison(df: pd.DataFrame, auprc_prefix: str, label: str) -> pd.DataFrame:
    """For each variant: ρ(loss, AUPRC), ρ(macro gap, AUPRC), ρ(matched-region gap,
    AUPRC), across the 8 sizes. `auprc_prefix` selects the VEP column family."""
    out = []
    for v in VARIANTS:
        y = df[f"{auprc_prefix}{v}"].to_numpy(dtype=float)
        if np.isnan(y).any():
            continue
        loss = df["eval_loss"].to_numpy(dtype=float)
        gap_macro = df["gap_macro"].to_numpy(dtype=float)
        region = VARIANT_REGION[v]
        gap_m = df["gap_macro" if region == "macro" else f"gap_{region}"].to_numpy(
            dtype=float
        )
        r_loss, p_loss = spearmanr(loss, y)
        r_macro, p_macro = spearmanr(gap_macro, y)
        r_match, p_match = spearmanr(gap_m, y)
        out.append(
            {
                "vep_source": label,
                "variant": v,
                "rho_loss": r_loss,
                "p_loss": p_loss,
                "rho_gap_macro": r_macro,
                "p_gap_macro": p_macro,
                "matched_region": region,
                "rho_gap_matched": r_match,
                "p_gap_matched": p_match,
                "gap_wins": abs(r_match) > abs(r_loss),
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gap-summary", required=True, help="ll_gap summary.parquet (S3 or local)"
    )
    ap.add_argument(
        "--metrics-prefix",
        default=None,
        help="evals_v2 results/metrics prefix (S3 or local) for the primary VEP",
    )
    ap.add_argument(
        "--score-type",
        default="minus_llr_avg",
        help="evals_v2 score_type column to read AUPRC from",
    )
    ap.add_argument("--out", default="scratch/issue274/correlation.parquet")
    args = ap.parse_args()

    base = pull_wandb_baseline()
    gap = load_gap_by_size(args.gap_summary)
    df = base.merge(gap, on="size", how="left", validate="1:1")
    assert df["gap_macro"].notna().all(), (
        "missing gap for some size — run `snakemake ll_gap`"
    )

    tables = [rho_comparison(df, "intrain_auprc_", "in_training")]
    if args.metrics_prefix:
        v2 = load_evals_v2_vep(args.metrics_prefix, args.score_type)
        if v2 is not None:
            df = df.merge(v2, on="size", how="left", validate="1:1")
            tables.append(
                rho_comparison(df, "v2_auprc_", f"evals_v2:{args.score_type}")
            )
        else:
            print("[warn] evals_v2 metrics not found yet — in-training VEP only")

    result = pd.concat(tables, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.out, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    for label, sub in result.groupby("vep_source"):
        print(f"\n=== VEP = {label}  (n=8 sizes; ρ = Spearman) ===")
        show = sub[
            [
                "variant",
                "rho_loss",
                "rho_gap_macro",
                "matched_region",
                "rho_gap_matched",
                "gap_wins",
            ]
        ].round(3)
        print(show.to_string(index=False))
        wins = int(sub["gap_wins"].sum())
        print(
            f"  matched-region gap beats loss (|ρ|) on {wins}/{len(sub)} variant types"
        )
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
