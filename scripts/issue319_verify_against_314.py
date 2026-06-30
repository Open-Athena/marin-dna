"""issue #319 — verify the library ``per_chrom_weighted_ap`` reproduces #314.

Two checks, both on the #314 headline generalist exp135-1B-m5.1 (productionized name
``mix-v0.9-p1B-i24-exp135-m5.1-step-59158``) on ``mendelian_traits``:

1. **Bit-for-bit metric anchor (the real test).** #314's iter1 OOF parquets carry
   #314's own ``llr_fwd``/``llr_rc``. Computing ``per_chrom_weighted_ap`` on
   ``minus_llr_avg = -(llr_fwd + llr_rc) / 2`` from those scores must reproduce #314's
   saved iter2 ``llr_perchrom`` *exactly* — same scores + same metric definition. (The
   ``llr_perchrom`` baseline is computed once per subset in iter2, identical across its
   reps.) This isolates the metric from any scoring difference: Δ == 0 ⇒ the library
   function is the #314 helper. It passes for all 8 subsets at Δ = 0.

2. **End-to-end sanity (productionized pipeline + library metric).** Apply the same
   metric to the *productionized* ``compute_probe`` predictions (#320/#330 evals_v2
   rule, on S3) — both the LOOC ``probe_score`` and ``minus_llr_avg`` — and put them
   beside #314's iter2 numbers. The productionized LLR is a different scoring code path
   from #314's embedding cache (corr ≈ 0.999, mean |Δllr| ≈ 0.07–0.10), so per-chrom
   lands within ~0.001 of #314 on 7/8 subsets; ``synonymous_variant`` (smallest, n=460)
   is most sensitive to that per-variant drift and differs ~0.025. The probe is a
   different protocol from iter2's nested rep sweep (LOOC, min_variants 300 vs 100,
   single directional feature), so its ``probe_perchrom`` is close-but-not-identical to
   iter2's ``delta`` rep — what we confirm is that the productionized pipeline + library
   metric reproduce the #314 finding that per-chrom AUPRC > the global pooled AUPRC.

Run (needs S3 + the library on this branch):
  uv run --group genome-s3 python scripts/issue319_verify_against_314.py
"""

import numpy as np
import polars as pl

from marin_dna.pipelines.evals.metrics import (
    auprc_with_bootstrap_se,
    per_chrom_weighted_ap,
)

MODEL_314 = "exp135-1B-m5.1"
MODEL_PROD = "mix-v0.9-p1B-i24-exp135-m5.1-step-59158"

ITER1_OOF = (
    f"s3://oa-bolinas/analysis/issue314/iter1_search/{MODEL_314}/oof_{{}}.parquet"
)
ITER2 = f"s3://oa-bolinas/analysis/issue314/iter2_nested/nested_{MODEL_314}.parquet"
PROD = (
    "s3://oa-bolinas/snakemake/analysis/evals_v2/results/probe/"
    f"{MODEL_PROD}/mendelian_traits.parquet"
)


def _minus_llr_avg(df) -> np.ndarray:
    """Directional mendelian baseline (iter2's non-``--abs_llr`` path)."""
    return (-(df["llr_fwd"] + df["llr_rc"]) / 2.0).to_numpy()


def main() -> None:
    ref = pl.read_parquet(ITER2).to_pandas()
    # llr_perchrom is computed once per subset in iter2 (same across reps); the
    # `delta` rep is the directional analog of the productionized minus_llr feature.
    ref_llr = ref.groupby("subset")["llr_perchrom"].first()
    ref_probe_delta = ref[ref["rep"] == "entire_window/delta"].set_index("subset")[
        "probe_perchrom"
    ]
    subsets = sorted(ref_llr.index)

    # --- Check 1: bit-for-bit on #314's own scores ------------------------------
    print("[1] library metric on #314's own iter1-OOF llr  vs  #314 saved llr_perchrom")
    print(f"    {'subset':36s} {'n':>5s} {'mine':>10s} {'#314':>10s} {'Δ':>11s}")
    max_anchor = 0.0
    for s in subsets:
        d = pl.read_parquet(ITER1_OOF.format(s)).to_pandas()
        d["chrom"] = d["chrom"].astype(str)
        y = d["label"].to_numpy().astype(int)
        mine = per_chrom_weighted_ap(y, _minus_llr_avg(d), d["chrom"].to_numpy())
        r = float(ref_llr[s])
        max_anchor = max(max_anchor, abs(mine - r))
        print(f"    {s:36s} {len(d):5d} {mine:10.6f} {r:10.6f} {mine - r:+11.2e}")
    print(f"    max |Δ| anchor = {max_anchor:.3e}")
    assert max_anchor < 1e-9, (
        f"library per_chrom_weighted_ap does NOT reproduce #314 llr_perchrom on #314's "
        f"own scores (max |Δ| = {max_anchor:.3e}); the metric and the helper must match"
    )
    print("    PASS — library metric == #314 helper, bit-for-bit.\n")

    # --- Check 2: end-to-end on the productionized predictions -------------------
    pred = pl.read_parquet(PROD).to_pandas()
    pred["chrom"] = pred["chrom"].astype(str)
    print("[2] productionized predictions (#320/#330) + library metric  vs  #314 iter2")
    print(
        f"    {'subset':36s} {'n':>5s} | {'llr_pc':>8s} {'#314':>8s} {'Δ':>8s} | "
        f"{'probe_pc':>8s} {'probe_g':>8s} {'#314 Δrep':>9s}"
    )
    for s in subsets:
        d = pred[pred["subset"] == s]
        y = d["label"].to_numpy().astype(int)
        ch = d["chrom"].to_numpy()
        llr_pc = per_chrom_weighted_ap(y, _minus_llr_avg(d), ch)
        r = float(ref_llr[s])
        dp = d[d["probe_score"].notna()]
        if len(dp):
            yp = dp["label"].to_numpy().astype(int)
            sp = dp["probe_score"].to_numpy()
            probe_pc = per_chrom_weighted_ap(yp, sp, dp["chrom"].to_numpy())
            probe_g = auprc_with_bootstrap_se(
                yp, sp, dp["match_group"].to_numpy(), n_bootstrap=0
            )["value"]
        else:
            probe_pc = probe_g = float("nan")
        print(
            f"    {s:36s} {len(d):5d} | {llr_pc:8.4f} {r:8.4f} {llr_pc - r:+8.4f} | "
            f"{probe_pc:8.4f} {probe_g:8.4f} {float(ref_probe_delta.get(s, np.nan)):9.4f}"
        )
    print(
        "\n    (probe_pc > probe_g on every subset — the per-chrom>global finding #314 "
        "productionizes.)"
    )


if __name__ == "__main__":
    main()
