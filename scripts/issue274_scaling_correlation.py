"""Issue #274: which likelihood signal best predicts VEP across the scaling ladder?

For the 8-model `dna-bolinas-scaling-v0.5` ladder (46M→4B, step-215573) this
correlates Mendelian VEP AUPRC against **4 likelihood predictors**, each derived
from the evals_v2 `ll_gap` eval (per validation-interval region):

  - ``LL_all``   — equal-weight mean log-prob over all target tokens (clean −perplexity)
  - ``LL_upper`` — mean log-prob on functional (uppercase / phyloP-conserved) tokens
  - ``LL_lower`` — mean log-prob on non-functional (lowercase) tokens
  - ``gap``      — LL_upper − LL_lower

We deliberately **exclude the W&B `eval/loss`**: training's base val spec uses
``lowercase_weight=0.01``, so that "loss" is functional-position-dominated (not a
clean equal-weight loss) and would be a confounded baseline. ``LL_all`` is the
clean overall-LL baseline instead.

Both Pearson r and Spearman ρ are reported (n=8): Spearman ties the
monotonic-in-scale predictors, Pearson separates them by value shape.

Inputs (reproducible, no external data):
  - in-training Mendelian AUPRC: W&B (eric-czech/marin scaling runs).
  - LL quantities per (model, region): the evals_v2 `ll_gap` summary parquet.
  - [primary VEP] official evals_v2 Mendelian AUPRC: `results/metrics/<model>/mendelian_traits.parquet`.

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
from scipy.stats import pearsonr, spearmanr

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
REGIONS = ["cds", "upstream", "downstream", "macro"]
# The 4 likelihood predictors (all from the ll_gap eval; W&B loss excluded).
PREDICTORS = ["LL_all", "LL_upper", "LL_lower", "gap"]
PRED_LABEL = {
    "LL_all": "LL (overall)",
    "LL_upper": "LL functional",
    "LL_lower": "LL non-functional",
    "gap": "LL gap",
}
# Region-matched variant sets: correlate each region's LLs only against the
# variant types that live in that region (issue #274 follow-up). The full grid
# (all regions × variants) is still saved; this drives the focused print.
REGION_VARIANTS = {
    "cds": ["missense_variant", "synonymous_variant", "splicing"],
    "upstream": ["tss_proximal", "5_prime_UTR_variant"],  # tss_proximal ≈ promoter
    "downstream": ["3_prime_UTR_variant"],
}

_SIZE_RE = re.compile(r"(h\d+-p[\dA-Za-z]+)")


def size_token(name: str) -> str:
    m = _SIZE_RE.search(name)
    assert m, f"no size token in {name!r}"
    return m.group(1)


def pull_wandb_auprc(entity: str = WANDB_ENTITY) -> pd.DataFrame:
    """Per-size in-training Mendelian AUPRC (+ params) from W&B. The W&B
    `eval/loss` is intentionally NOT pulled — it is the `lowercase_weight=0.01`
    (functional-dominated) loss, a confounded baseline; the clean overall LL
    (`LL_all`) is the baseline instead."""
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
            "size": size_token(s),
            "params": r.summary.get("parameter_count"),
        }
        for v in VARIANTS:
            d[f"intrain_auprc_{v}"] = r.summary.get(
                f"lm_eval/traitgym_mendelian_v2_255/{v}/auprc"
            )
        rows.append(d)
    return pd.DataFrame(rows).sort_values("params").reset_index(drop=True)


def load_gap_quantities(gap_summary_path: str) -> pd.DataFrame:
    """Per-size LL quantities per region: e.g. `gap_cds`, `LL_upper_cds`, …, plus
    a `<q>_macro` (mean over the 3 regions) for each quantity."""
    g = pl.read_parquet(gap_summary_path).to_pandas()
    g = g[g["model"].str.contains(r"scaling-v0\.5-h\d+-p", regex=True)].copy()
    g["size"] = g["model"].map(size_token)
    out: pd.DataFrame | None = None
    for q in PREDICTORS:
        w = g.pivot(index="size", columns="region", values=q)
        w.columns = [f"{q}_{r}" for r in w.columns]
        out = w if out is None else out.join(w)
    assert out is not None
    for q in PREDICTORS:
        out[f"{q}_macro"] = out[[f"{q}_cds", f"{q}_upstream", f"{q}_downstream"]].mean(
            axis=1
        )
    return out.reset_index()


def load_evals_v2_vep(metrics_prefix: str, score_type: str) -> pd.DataFrame | None:
    """Per-size official evals_v2 Mendelian AUPRC per variant subset, for one
    score_type. None if the metrics parquets aren't all present yet."""
    rows = []
    for s in SIZES:
        path = f"{metrics_prefix}/scaling-v0.5-{s}-step-215573/mendelian_traits.parquet"
        try:
            m = pl.read_parquet(path).to_pandas()
        except Exception as e:
            # Genuinely-absent metrics (not built yet) → fall back to in-training.
            # Any other error (corrupt parquet, S3 auth, schema drift) must
            # surface rather than be silently misread as "not present yet".
            if any(
                s in str(e).lower()
                for s in ("not found", "no such", "does not exist", "404", "nosuchkey")
            ):
                return None
            raise
        m = m[m["score_type"] == score_type]
        d: dict[str, object] = {"size": size_token(s)}
        for v in VARIANTS:
            subset = "_global_" if v == "all" else v
            hit = m[m["subset"] == subset]["value"]
            d[f"v2_auprc_{v}"] = float(hit.iloc[0]) if len(hit) else np.nan
        rows.append(d)
    return pd.DataFrame(rows)


def decomposition(df: pd.DataFrame, auprc_prefix: str, vep_label: str) -> pd.DataFrame:
    """Pearson r + Spearman ρ of every (region × predictor) vs every variant's
    AUPRC, across the 8 sizes."""
    rows = []
    for region in REGIONS:
        for q in PREDICTORS:
            x = df[f"{q}_{region}"].to_numpy(dtype=float)
            for v in VARIANTS:
                col = f"{auprc_prefix}{v}"
                if col not in df.columns or df[col].isna().any():
                    continue
                y = df[col].to_numpy(dtype=float)
                pear = pearsonr(x, y)
                spear = spearmanr(x, y)
                rows.append(
                    {
                        "vep_source": vep_label,
                        "region": region,
                        "predictor": PRED_LABEL[q],
                        "variant": v,
                        "pearson": pear.statistic,
                        "pearson_p": pear.pvalue,
                        "spearman": spear.statistic,
                        "spearman_p": spear.pvalue,
                    }
                )
    return pd.DataFrame(rows)


def _print_matched(result: pd.DataFrame) -> None:
    """Print each region-matched (region → variant) pair's 4 predictor
    correlations as Pearson r / Spearman ρ."""
    hdr = ["LL_all", "LL_func", "LL_nonf", "LL_gap"]
    for label, sub in result.groupby("vep_source"):
        print(f"\n=== VEP={label} | region-matched | Pearson r / Spearman ρ (n=8) ===")
        print(f"{'region → variant':24}" + "".join(f"{h:>16}" for h in hdr))
        for region, variants in REGION_VARIANTS.items():
            for v in variants:
                cells = []
                for q in PREDICTORS:
                    row = sub[
                        (sub["region"] == region)
                        & (sub["predictor"] == PRED_LABEL[q])
                        & (sub["variant"] == v)
                    ]
                    cells.append(
                        f"{row.pearson.iloc[0]:+.2f}/{row.spearman.iloc[0]:+.2f}"
                        if len(row)
                        else "—"
                    )
                name = f"{region}→{v.replace('_variant', '')}"
                print(f"{name:24}" + "".join(f"{c:>16}" for c in cells))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gap-summary", required=True)
    ap.add_argument("--metrics-prefix", default=None)
    ap.add_argument("--score-type", default="minus_llr_avg")
    ap.add_argument("--out", default="scratch/issue274/correlation.parquet")
    args = ap.parse_args()

    df = pull_wandb_auprc().merge(
        load_gap_quantities(args.gap_summary), on="size", validate="1:1"
    )
    # Require every (predictor × region) cell. A partial summary (e.g. one sky
    # cell failed) would otherwise yield silent-NaN correlations and a skipna
    # macro mean over the wrong number of regions.
    _need = [f"{q}_{r}" for q in PREDICTORS for r in ("cds", "upstream", "downstream")]
    _missing = [c for c in _need if df[c].isna().any()]
    assert not _missing, f"missing LL cells (run `snakemake ll_gap`): {_missing}"

    tables = [decomposition(df, "intrain_auprc_", "in_training")]
    if args.metrics_prefix:
        v2 = load_evals_v2_vep(args.metrics_prefix, args.score_type)
        if v2 is not None:
            df = df.merge(v2, on="size", validate="1:1")
            tables.append(decomposition(df, "v2_auprc_", f"evals_v2:{args.score_type}"))
        else:
            print(
                "[warn] official evals_v2 metrics not all present yet — in-training only"
            )

    result = pd.concat(tables, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.out, index=False)

    # Focused print: region-matched (region → variant) pairs.
    _print_matched(result)
    print(f"\nsaved full grid (all regions × predictors × variants) → {args.out}")


if __name__ == "__main__":
    main()
