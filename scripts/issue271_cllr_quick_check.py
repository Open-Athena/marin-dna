"""Quick CPU-only sanity check for cLLR stage 4 (#271): does mutation-rate
calibration improve Mendelian AUPRC?

This is a one-off validation of the stage-4 success criterion *before* wiring
calibration into the evals_v2 metrics rule. It reuses two already-computed
artifacts — no GPU, no model forward passes:

  1. The per-variant raw-LLR scores from evals_v2 `compute_scores`
     (`results/scores/<model>/mendelian_traits.parquet`: per-strand `llr_fwd` /
     `llr_rc` + the matched-pair columns `label/subset/match_group`).
  2. The per-checkpoint `llr_neutral_mean` calibration table from stage 3 (#270)
     (`results/calibration/<model>/llr_neutral_mean_n{n}.parquet`: per
     `pentanuc_mut` cell, FWD/RC/avg neutral-mean LLR).

The only new computation is the per-variant pentanucleotide context (a 5 bp read
per variant against the local reference) — light CPU work. We then derive the
calibrated `minus_cllr` columns and compare AUPRC to raw `minus_llr`.

Calibration (mendelian protocol, per the issue):

    calibrated LLR = llr − llr_neutral_mean(pentanuc_mut)
    minus_cllr     = −(calibrated LLR)                       # higher = more pathogenic

Strand convention matches the evals_v2 metrics rule and the stage-3 aggregator:
average the *raw* LLR first (`llr_avg = (llr_fwd + llr_rc)/2`), then apply the
protocol — so `minus_cllr_avg = −(llr_avg − llr_neutral_mean_avg)` with
`llr_neutral_mean_avg = (mean_fwd + mean_rc)/2`.

Coordinates: eval `pos` is 1-based VCF (the center base at 0-based `pos-1` equals
`ref`, asserted), so the central 5-mer is `genome[pos-3:pos+2]` (0-based
half-open) — identical to `annotate_pentanucleotide` in the neutral-sites
pipeline, so eval and neutral 5-mers are binned the same way.

Run (defaults point at the exp135-1B-m5.1 artifacts staged under scratch/):

    uv run python scripts/issue271_cllr_quick_check.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from marin_dna.data.dna import NUCLEOTIDES
from marin_dna.data.genome import Genome
from marin_dna.pipelines.evals.metrics import (
    GLOBAL_SUBSET,
    MACRO_AVG_SUBSET,
    compute_auprc_metrics,
)

DEFAULT_MODEL = "mix-v0.9-p1B-i24-exp135-m5.1-step-59158"
SCRATCH = Path("scratch/issue271")


def extract_pentanuc(variants: pd.DataFrame, genome: Genome) -> np.ndarray:
    """Central 5-mer (uppercase) per variant via a sparse per-variant read.

    Mirrors `annotate_pentanucleotide`'s convention exactly but with one small
    read per variant (eval variants are sparse — a per-chrom span read would be
    a pathological whole-chromosome decompress). For 1-based `pos`, the central
    base is 0-based `pos-1` and the 5-mer spans `[pos-3, pos+2)`. The center
    base is asserted to equal `ref`, validating the coordinate convention
    end-to-end; a non-ACGT flank (N near an assembly gap) yields a 5-mer that
    won't index a calibration cell and is dropped by the caller.
    """
    pent = np.empty(len(variants), dtype=object)
    for i, (chrom, pos, ref) in enumerate(
        zip(variants["chrom"], variants["pos"], variants["ref"])
    ):
        # 0-based half-open [pos-3, pos+2): 2 bp flank | center (pos-1) | 2 bp flank.
        mer = genome(str(chrom), int(pos) - 3, int(pos) + 2, strand="+").upper()
        assert len(mer) == 5, f"{chrom}:{pos} returned {len(mer)} bp, expected 5"
        assert mer[2] == ref, (
            f"pentanuc center {mer[2]!r} != ref {ref!r} at {chrom}:{pos} — "
            f"coordinate convention mismatch (expected 1-based VCF pos)"
        )
        pent[i] = mer
    return pent


def paired_cluster_bootstrap_delta(
    df: pd.DataFrame,
    raw_col: str,
    cal_col: str,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Cluster-bootstrap the *paired* global-AUPRC delta (cal − raw).

    Both score columns are scored on the *same* resampled match_groups each
    iteration, so the shared-variant correlation cancels — the right way to
    compare two scorers on one dataset (see `compute_qtl_metrics`' caveat about
    not reading a delta off overlapping marginal ±SE bars). Returns the point
    delta, bootstrap SE, a 95% percentile CI, and the one-sided fraction of
    resamples with cal > raw.
    """
    label = df["label"].to_numpy().astype(int)
    raw = df[raw_col].to_numpy(dtype=float)
    cal = df[cal_col].to_numpy(dtype=float)
    groups = list(
        pd.Series(df["match_group"]).groupby(df["match_group"]).indices.values()
    )
    n_groups = len(groups)
    point = float(average_precision_score(label, cal)) - float(
        average_precision_score(label, raw)
    )
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        sampled = rng.integers(0, n_groups, size=n_groups)
        idx = np.concatenate([groups[i] for i in sampled])
        y = label[idx]
        if y.sum() == 0 or y.sum() == len(y):
            deltas[b] = np.nan
            continue
        deltas[b] = average_precision_score(y, cal[idx]) - average_precision_score(
            y, raw[idx]
        )
    finite = deltas[~np.isnan(deltas)]
    return {
        "delta": point,
        "se": float(np.std(finite, ddof=1)),
        "ci_lo": float(np.percentile(finite, 2.5)),
        "ci_hi": float(np.percentile(finite, 97.5)),
        "frac_cal_gt_raw": float((finite > 0).mean()),
        "n_groups": n_groups,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--scores", default=str(SCRATCH / "mendelian_scores.parquet"))
    ap.add_argument(
        "--calibration", default=str(SCRATCH / "llr_neutral_mean_n100.parquet")
    )
    ap.add_argument(
        "--genome",
        default=str(
            SCRATCH / "genome/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"
        ),
    )
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[cLLR check] model={args.model}")
    scores = pd.read_parquet(args.scores)
    cal = pd.read_parquet(args.calibration)
    print(f"  scores: {len(scores)} rows | calibration: {len(cal)} cells")

    # --- SNV sanity: the LLR scoring path (and 5-mer binning) is SNV-only. ---
    assert scores["ref"].isin(NUCLEOTIDES).all(), "non-ACGT ref in eval scores"
    assert scores["alt"].isin(NUCLEOTIDES).all(), "non-ACGT alt in eval scores"
    assert (scores["ref"] != scores["alt"]).all(), "ref == alt row in eval scores"
    for col in ("llr_fwd", "llr_rc", "label", "subset", "match_group"):
        assert col in scores.columns, f"scores missing {col!r}"
    assert scores[["llr_fwd", "llr_rc"]].notna().all().all(), "NaN in raw LLR"

    # --- Pentanucleotide context per eval variant (the only new computation). ---
    genome = Genome(args.genome)
    scores = scores.copy()
    scores["pentanuc"] = extract_pentanuc(scores, genome)
    acgt = scores["pentanuc"].str.fullmatch("[ACGT]{5}")
    n_drop = int((~acgt).sum())
    if n_drop:
        print(f"  dropped {n_drop} variant(s) with a non-ACGT 5-mer flank (N near gap)")
    scores = scores.loc[acgt].reset_index(drop=True)
    scores["pentanuc_mut"] = scores["pentanuc"] + "_" + scores["alt"]

    # --- Join the stage-3 neutral-mean table on the calibration cell. ---
    cal_cols = [
        "pentanuc_mut",
        "llr_neutral_mean_fwd",
        "llr_neutral_mean_rc",
        "llr_neutral_mean_avg",
    ]
    merged = scores.merge(
        cal[cal_cols], on="pentanuc_mut", how="left", validate="many_to_one"
    )
    miss = int(merged["llr_neutral_mean_avg"].isna().sum())
    assert miss == 0, (
        f"{miss} variant(s) have a pentanuc_mut absent from the calibration "
        f"table — the table should cover all 1024×3 ACGT cells"
    )

    # --- Derive raw (minus_llr) and calibrated (minus_cllr) per strand. ---
    merged["llr_avg"] = (merged["llr_fwd"] + merged["llr_rc"]) / 2.0
    score_cols: list[str] = []
    for strand in ("fwd", "rc", "avg"):
        raw = merged[f"llr_{strand}"]
        neutral = merged[f"llr_neutral_mean_{strand}"]
        merged[f"minus_llr_{strand}"] = -raw  # raw protocol
        merged[f"minus_cllr_{strand}"] = -(raw - neutral)  # calibrated
        score_cols += [f"minus_llr_{strand}", f"minus_cllr_{strand}"]

    # --- AUPRC (per-subset + _global_ + _macro_avg_) for raw vs calibrated. ---
    metrics = compute_auprc_metrics(
        dataset=merged[["label", "subset", "match_group"]],
        scores=merged[score_cols],
        score_columns=score_cols,
        n_bootstrap=args.n_bootstrap,
        rng=args.seed,
    )

    def row(score_type: str, subset: str) -> pd.Series:
        return metrics[
            (metrics.score_type == score_type) & (metrics.subset == subset)
        ].iloc[0]

    print("\n=== AUPRC: raw (minus_llr) vs calibrated (minus_cllr) ===")
    print(
        f"{'aggregate':<14}{'strand':<7}{'raw':>10}{'calibrated':>13}{'Δ (cal−raw)':>14}"
    )
    for agg_label, agg in [
        ("_global_", GLOBAL_SUBSET),
        ("_macro_avg_", MACRO_AVG_SUBSET),
    ]:
        for strand in ("fwd", "rc", "avg"):
            r = row(f"minus_llr_{strand}", agg)
            c = row(f"minus_cllr_{strand}", agg)
            print(
                f"{agg_label:<14}{strand:<7}{r['value']:>10.4f}{c['value']:>13.4f}"
                f"{c['value'] - r['value']:>+14.4f}"
            )

    # --- Per-subset breakdown for the headline avg strand. ---
    print("\n=== Per-subset AUPRC (avg strand) ===")
    print(f"{'subset':<26}{'n_grp':>6}{'raw':>10}{'calibrated':>13}{'Δ':>10}")
    subs = metrics[(metrics.score_type == "minus_llr_avg")]
    subs = subs[~subs.subset.isin([GLOBAL_SUBSET, MACRO_AVG_SUBSET])]
    for _, r in subs.sort_values("subset").iterrows():
        c = row("minus_cllr_avg", r["subset"])
        print(
            f"{r['subset']:<26}{int(r['n_groups']):>6}{r['value']:>10.4f}"
            f"{c['value']:>13.4f}{c['value'] - r['value']:>+10.4f}"
        )

    # --- Paired cluster-bootstrap of the global-AUPRC delta (avg strand). ---
    pb = paired_cluster_bootstrap_delta(
        merged,
        "minus_llr_avg",
        "minus_cllr_avg",
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    print(
        "\n=== Paired cluster-bootstrap: global AUPRC delta (cal − raw), avg strand ==="
    )
    print(
        f"  Δ = {pb['delta']:+.4f}  (SE {pb['se']:.4f}, 95% CI "
        f"[{pb['ci_lo']:+.4f}, {pb['ci_hi']:+.4f}])  P(cal>raw) = {pb['frac_cal_gt_raw']:.3f}"
    )
    verdict = "IMPROVES" if pb["delta"] > 0 else "does NOT improve"
    print(f"\n[verdict] calibration {verdict} global Mendelian AUPRC (avg strand).")


if __name__ == "__main__":
    main()
