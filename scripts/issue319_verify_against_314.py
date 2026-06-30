"""issue #319 — verify the metric AND the parent #320 probe reproduce #314.

All on the #314 headline generalist exp135-1B-m5.1 (productionized name
``mix-v0.9-p1B-i24-exp135-m5.1-step-59158``) on ``mendelian_traits``; every subset's
row count matches #314 exactly, so it's the same variant set.

[1] **Metric, bit-for-bit (validates #319).** #314's iter1 OOF parquets carry #314's
    own ``llr_fwd``/``llr_rc``. ``per_chrom_weighted_ap`` on ``minus_llr_avg`` from
    those scores reproduces #314's saved iter2 ``llr_perchrom`` *exactly* (Δ = 0, all 8
    subsets) — same scores + same metric ⇒ the library function is the #314 helper.

[2] **Productionized probe vs #314's *matching* arm (validates the parent end-to-end).**
    The mendelian probe feature is ``concat_ref_delta`` (the ``minus_llr`` protocol →
    ``PROBE_FEATURE_BY_PROTOCOL``), i.e. #314's ref-context ablation arm — so the right
    reference is ``iter2_nested_refdelta``'s ``concat_ref_delta`` rep, NOT the plain
    ``delta`` rep (plain ``delta`` is ~0.1 lower; ref-context is the whole point of that
    #314 ablation). Per-chrom AUPRC from the productionized predictions lands within
    ~0.01 of that arm (mean), the residual being a different scoring/pooling code path
    (evals_v2 #318 vs #314's cache; LLR corr ≈ 0.999, mean |Δllr| ≈ 0.1). ``probe`` >
    its global pooled AUPRC on every subset — the calibration artifact #314 found.

[3] **Parent #320 reproducibility (validates the training rule).** Re-run #320's own
    ``run_subset_probes`` on the pipeline's input embeddings (``results/scores/...``,
    the 125 MB emb-bearing parquet) for the two smallest probed subsets; the saved
    ``probe_score`` reproduces to fp/threading tolerance (synonymous ~2e-10, distal
    ~3e-3 — parallel-``GridSearchCV`` nondeterminism), per-chrom AUPRC within ~1e-3.

Run (needs S3 + the library on this branch; [3] downloads the 125 MB scores parquet):
  uv run --group genome-s3 python scripts/issue319_verify_against_314.py
"""

import numpy as np
import pandas as pd
import polars as pl

from marin_dna.pipelines.evals.metrics import (
    auprc_with_bootstrap_se,
    per_chrom_weighted_ap,
)
from marin_dna.pipelines.evals.variant_probe import run_subset_probes

MODEL_314 = "exp135-1B-m5.1"
MODEL_PROD = "mix-v0.9-p1B-i24-exp135-m5.1-step-59158"

ITER1_OOF = (
    f"s3://oa-bolinas/analysis/issue314/iter1_search/{MODEL_314}/oof_{{}}.parquet"
)
ITER2 = f"s3://oa-bolinas/analysis/issue314/iter2_nested/nested_{MODEL_314}.parquet"
ITER2_REFDELTA = f"s3://oa-bolinas/analysis/issue314/iter2_nested_refdelta/nested_{MODEL_314}.parquet"
_PROD = "s3://oa-bolinas/snakemake/analysis/evals_v2/results"
PRED = f"{_PROD}/probe/{MODEL_PROD}/mendelian_traits.parquet"
SCORES = f"{_PROD}/scores/{MODEL_PROD}/mendelian_traits.parquet"  # emb-bearing input

# The productionized mendelian probe feature (config: score_protocol minus_llr →
# PROBE_FEATURE_BY_PROTOCOL). Pinned here so [2]/[3] reference the right #314 arm.
PROBE_FEATURE = "concat_ref_delta"
PROBE_C_GRID = np.logspace(-12, 4, 17)  # config probe.c_grid
KEY = ["chrom", "pos", "ref", "alt"]


def _minus_llr_avg(df: pd.DataFrame) -> np.ndarray:
    """Directional mendelian baseline (iter2's non-``--abs_llr`` path)."""
    return (-(df["llr_fwd"] + df["llr_rc"]) / 2.0).to_numpy()


def check1_metric_anchor(ref_llr: pd.Series, subsets: list[str]) -> None:
    print("[1] metric on #314's own iter1-OOF llr  vs  #314 saved llr_perchrom")
    print(f"    {'subset':36s} {'n':>5s} {'mine':>10s} {'#314':>10s} {'Δ':>11s}")
    worst = 0.0
    for s in subsets:
        d = pl.read_parquet(ITER1_OOF.format(s)).to_pandas()
        d["chrom"] = d["chrom"].astype(str)
        y = d["label"].to_numpy().astype(int)
        mine = per_chrom_weighted_ap(y, _minus_llr_avg(d), d["chrom"].to_numpy())
        r = float(ref_llr[s])
        worst = max(worst, abs(mine - r))
        print(f"    {s:36s} {len(d):5d} {mine:10.6f} {r:10.6f} {mine - r:+11.2e}")
    print(f"    max |Δ| = {worst:.3e}")
    assert worst < 1e-9, (
        f"metric does NOT reproduce #314 llr_perchrom (max|Δ|={worst:.2e})"
    )
    print("    PASS — library metric == #314 helper, bit-for-bit.\n")


def check2_probe_vs_matching_arm(ref_concat: pd.Series, subsets: list[str]) -> None:
    pred = pl.read_parquet(PRED).to_pandas()
    pred["chrom"] = pred["chrom"].astype(str)
    print(
        f"[2] productionized probe ({PROBE_FEATURE}) per-chrom  vs  #314 concat_ref_delta arm"
    )
    print(
        f"    {'subset':36s} {'n':>5s} {'probe_pc':>9s} {'probe_g':>8s} "
        f"{'#314':>8s} {'Δ':>8s}"
    )
    deltas = []
    for s in subsets:
        dp = pred[(pred["subset"] == s) & pred["probe_score"].notna()]
        y = dp["label"].to_numpy().astype(int)
        sp = dp["probe_score"].to_numpy()
        pc = per_chrom_weighted_ap(y, sp, dp["chrom"].to_numpy())
        g = auprc_with_bootstrap_se(y, sp, dp["match_group"].to_numpy(), n_bootstrap=0)[
            "value"
        ]
        r = float(ref_concat[s])
        deltas.append(abs(pc - r))
        assert pc > g, f"{s}: per-chrom {pc:.3f} !> global {g:.3f} (the #314 finding)"
        print(f"    {s:36s} {len(dp):5d} {pc:9.4f} {g:8.4f} {r:8.4f} {pc - r:+8.4f}")
    print(f"    mean |Δ| = {np.mean(deltas):.4f}, max |Δ| = {np.max(deltas):.4f}")
    print(
        "    (probe per-chrom > global on every subset; ~0.01 from #314's matching arm.)\n"
    )


def check3_parent_reproducibility(ref_concat: pd.Series) -> None:
    print(
        "[3] re-run #320 run_subset_probes on its input embeddings  vs  saved probe_score"
    )
    saved = pd.read_parquet(PRED, columns=KEY + ["subset", "label", "probe_score"])
    print(
        f"    {'subset':22s} {'n':>5s} {'max|Δscore|':>12s} {'pc(rerun)':>10s} {'pc(saved)':>10s}"
    )
    for s in ("synonymous_variant", "distal"):  # two smallest probed subsets (fast)
        inp = pd.read_parquet(SCORES, filters=[("subset", "==", s)])
        preds, _ = run_subset_probes(
            inp,
            feature_combo=PROBE_FEATURE,
            c_grid=PROBE_C_GRID,
            min_variants=300,
            min_chroms=3,
            inner_splits=5,
            n_jobs=4,
        )
        rerun = preds[KEY + ["probe_score"]].rename(columns={"probe_score": "rerun"})
        sv = saved[saved["subset"] == s].reset_index(drop=True).merge(rerun, on=KEY)
        sv = sv[sv["probe_score"].notna() & sv["rerun"].notna()]
        mad = float((sv["probe_score"] - sv["rerun"]).abs().max())
        y = sv["label"].to_numpy().astype(int)
        ch = sv["chrom"].astype(str).to_numpy()
        pc_re = per_chrom_weighted_ap(y, sv["rerun"].to_numpy(), ch)
        pc_sv = per_chrom_weighted_ap(y, sv["probe_score"].to_numpy(), ch)
        assert mad < 0.05 and abs(pc_re - pc_sv) < 0.02, (
            f"{s}: re-run diverges from saved (max|Δscore|={mad:.2e}, "
            f"Δpc={pc_re - pc_sv:+.2e}) beyond fp/threading tolerance"
        )
        print(f"    {s:22s} {len(sv):5d} {mad:12.2e} {pc_re:10.4f} {pc_sv:10.4f}")
    print("    PASS — saved S3 probe predictions reproduce #320's run_subset_probes.\n")


def main() -> None:
    ref = pl.read_parquet(ITER2).to_pandas()
    ref_llr = ref.groupby("subset")["llr_perchrom"].first()
    rd = pl.read_parquet(ITER2_REFDELTA).to_pandas()
    ref_concat = rd[rd["rep"] == "entire_window/concat_ref_delta"].set_index("subset")[
        "probe_perchrom"
    ]
    subsets = sorted(ref_llr.index)

    check1_metric_anchor(ref_llr, subsets)
    check2_probe_vs_matching_arm(ref_concat, subsets)
    check3_parent_reproducibility(ref_concat)


if __name__ == "__main__":
    main()
