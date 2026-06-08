"""Paired cluster-bootstrap of the order−family AUPRC delta per matched subset (#255).

The endpoint bars / per-step AUPRC show a family edge on ncRNA & 5'UTR using *per-model*
(independence) SEs, which over-state the delta's uncertainty because the two arms are
scored on the SAME matched-pair variants. This runs the rigorous PAIRED test: for each
matched subset, load the order and family region-specialist's step-4999 evals_v2 scores
parquets (per-variant llr_fwd/llr_rc + label/subset/match_group), derive the BOS-faithful
minus_llr_avg score (= −(llr_fwd + llr_rc)/2), align the two arms by variant, and run
``paired_metric_delta_bootstrap`` (resample match_groups once, recompute AUPRC for both
arms on that resample, delta). Prints Δ(order−family) ± paired SE, 95% CI and two-sided p.

Run:  uv run python scripts/issue255_paired_bootstrap.py
"""

from __future__ import annotations

import polars as pl

from marin_dna.pipelines.evals.metrics import paired_metric_delta_bootstrap

S3 = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
KEY = ["chrom", "pos", "ref", "alt"]
# arm -> (order model, family model, matched subsets)
ARMS: dict[str, tuple[str, str, list[str]]] = {
    "cds": (
        "exp255-v4_cds_order-step-4999",
        "exp232-v4_cds-step-4999",
        ["missense_variant", "synonymous_variant", "splicing"],
    ),
    "utr3": (
        "exp255-v4_utr3_order-step-4999",
        "exp232-v4_utr3-step-4999",
        ["3_prime_UTR_variant"],
    ),
    "ncrna": (
        "exp255-v4_ncrna_exon_order-step-4999",
        "exp232-v4_ncrna_exon-step-4999",
        ["non_coding_transcript_exon_variant"],
    ),
    "tss": (
        "exp255-v4_tss_region_and_utr5_order-step-4999",
        "exp232-v4_tss_region_and_utr5-step-4999",
        ["5_prime_UTR_variant", "tss_proximal"],
    ),
    "ccre": (
        "exp255-v4_ccre_non_promoter_order-step-4999",
        "exp232-v4_ccre_non_promoter-step-4999",
        ["distal"],
    ),
}


def _load(model: str) -> pl.DataFrame:
    df = pl.read_parquet(f"{S3}/{model}/mendelian_traits.parquet")
    return df.with_columns((-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("score"))


def main() -> None:
    out = []
    for arm, (om, fm, subs) in ARMS.items():
        o = _load(om).select([*KEY, "label", "subset", "match_group", "score"])
        f = _load(fm).select([*KEY, "score"]).rename({"score": "score_fam"})
        m = o.join(f, on=KEY, how="inner")
        assert len(m) == len(o) == len(f), (
            f"{arm}: join {len(o)}/{len(f)}->{len(m)} not 1:1"
        )
        for ss in subs:
            sub = m.filter(pl.col("subset") == ss)
            r = paired_metric_delta_bootstrap(
                label=sub["label"].to_pandas(),
                score_a=sub["score"].to_pandas(),
                score_b=sub["score_fam"].to_pandas(),
                match_group=sub["match_group"].to_pandas(),
            )
            out.append((arm, ss, r))

    print(
        f"\n{'arm':6}{'subset':30}{'Δ(o−f)':>9}{'pairedSE':>10}{'95% CI':>20}{'p':>8}  sig"
    )
    print("-" * 87)
    for arm, ss, r in out:
        ci = f"[{r['ci_low']:+.3f},{r['ci_high']:+.3f}]"
        sig = "***" if (r["ci_low"] > 0) == (r["ci_high"] > 0) else ""  # CI excludes 0
        print(
            f"{arm:6}{ss.replace('_variant', ''):30}{r['delta']:>+9.3f}{r['se']:>10.3f}"
            f"{ci:>20}{r['p_two_sided']:>8.3f}  {sig}"
        )


if __name__ == "__main__":
    main()
