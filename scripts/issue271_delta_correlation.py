"""4x4 correlation of the per-variant calibration shift (cLLR − LLR) across the
3 GPN-Star models + our exp135-1B-m5.1, on the SAME 16,140 Mendelian variants
(revision 4aed58e), plus a provenance check of the dashboard's GPN-Star numbers.

The per-variant shift is `delta = calibrated_LLR − LLR = −llr_neutral_mean(context)`
— a per-pentanuc_mut quantity. Correlating these deltas asks: do the models agree
on *which contexts get adjusted*, even though calibration helps GPN-Star and
mildly hurts us?

GPN-Star predictions come from the **current-revision** gist (issue #145
comment 2026-05-19, gist `db282f89` @ `02484d5`), which is row-aligned to
`evals_mendelian_traits@4aed58e` — NOT the stale gist pinned in
`marin_dna.pipelines.evals.gpn_star.GPN_STAR_GIST_BASE` (old revision, 9,820
train). Ours reuses the stage-3 n=100 `llr_neutral_mean` table.

    uv run python scripts/issue271_delta_correlation.py
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score

from marin_dna.data.genome import Genome

SCRATCH = Path("scratch/issue271")
KEYS = ["chrom", "pos", "ref", "alt"]
# Dashboard's GPN-Star AUPRC (metrics gist cba23a7), global, for the provenance check.
GIST_AUPRC_GLOBAL = {
    ("V", "minus_llr"): 0.477695,
    ("V", "minus_llr_calibrated"): 0.483443,
    ("M", "minus_llr"): 0.470490,
    ("M", "minus_llr_calibrated"): 0.478777,
    ("P", "minus_llr"): 0.385800,
    ("P", "minus_llr_calibrated"): 0.391931,
}


def _extract_pentanuc(variants: pd.DataFrame, genome: Genome) -> np.ndarray:
    """Reuse the quick-check's pentanuc extractor (same convention)."""
    spec = importlib.util.spec_from_file_location(
        "qc", "scripts/issue271_cllr_quick_check.py"
    )
    qc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qc)
    return qc.extract_pentanuc(variants, genome)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
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
    ap.add_argument(
        "--gpn-glob",
        default=str(SCRATCH / "gpn_new_{model}.parquet"),
        help="path template for the current-revision GPN-Star prediction parquets",
    )
    args = ap.parse_args()

    # --- Ours: per-variant calibration shift = -llr_neutral_mean_avg. ---
    o = pd.read_parquet(args.scores)
    o["chrom"] = o["chrom"].astype(str)
    o["pentanuc"] = _extract_pentanuc(o, Genome(args.genome))
    o["pentanuc_mut"] = o["pentanuc"] + "_" + o["alt"]
    cal = pd.read_parquet(args.calibration)
    o = o.merge(
        cal[["pentanuc_mut", "llr_neutral_mean_avg"]], on="pentanuc_mut", how="left"
    )
    assert o["llr_neutral_mean_avg"].notna().all(), "unmatched pentanuc_mut in our join"
    o["delta_ours"] = -o["llr_neutral_mean_avg"]  # cLLR_llr - LLR (avg strand)

    merged = o[[*KEYS, "label", "subset", "match_group", "delta_ours"]].copy()

    # --- GPN-Star V/M/P: per-variant shift = llr_calibrated - llr (current gist). ---
    print(
        "=== Provenance check: do the new-gist predictions reproduce the dashboard? ==="
    )
    print(f"{'model':<8}{'score':<24}{'recomputed':>12}{'gist':>10}{'Δ':>9}")
    for m in ("V", "M", "P"):
        g = pd.read_parquet(args.gpn_glob.format(model=m))
        g["chrom"] = g["chrom"].astype(str)
        gt = g[g["split"] == "train"].copy()
        gt[f"delta_{m}"] = gt["llr_calibrated"] - gt["llr"]
        # provenance: recompute global AUPRC of -llr and -llr_calibrated on OUR labels
        chk = o[[*KEYS, "label"]].merge(
            gt[[*KEYS, "llr", "llr_calibrated"]],
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )
        assert len(chk) == len(o), f"{m}: only {len(chk)}/{len(o)} aligned"
        y = chk["label"].to_numpy().astype(int)
        for col, sign_col in (
            ("minus_llr", "llr"),
            ("minus_llr_calibrated", "llr_calibrated"),
        ):
            ap_val = average_precision_score(y, -chk[sign_col].to_numpy())
            ref = GIST_AUPRC_GLOBAL[(m, col)]
            print(
                f"{'GPN-' + m:<8}{col:<24}{ap_val:>12.4f}{ref:>10.4f}{ap_val - ref:>+9.4f}"
            )
        merged = merged.merge(
            gt[[*KEYS, f"delta_{m}"]], on=KEYS, how="inner", validate="one_to_one"
        )

    assert len(merged) == len(o), f"alignment lost: {len(merged)} of {len(o)}"

    # --- 4x4 correlation of per-variant deltas. ---
    cols = ["delta_V", "delta_M", "delta_P", "delta_ours"]
    labels = ["GPN-V", "GPN-M", "GPN-P", "ours(exp135)"]
    X = merged[cols].to_numpy()

    def corr_matrix(fn) -> np.ndarray:
        k = len(cols)
        out = np.eye(k)
        for i in range(k):
            for j in range(i + 1, k):
                r = fn(X[:, i], X[:, j])[0]
                out[i, j] = out[j, i] = r
        return out

    for name, fn in (("Pearson", pearsonr), ("Spearman", spearmanr)):
        M = corr_matrix(fn)
        print(
            f"\n=== {name} correlation of per-variant calibration shift (cLLR − LLR), n={len(merged)} ==="
        )
        print(f"{'':<14}" + "".join(f"{lbl:>14}" for lbl in labels))
        for i, lbl in enumerate(labels):
            print(f"{lbl:<14}" + "".join(f"{M[i, j]:>14.3f}" for j in range(len(cols))))

    # --- Cell-level (unweighted over the 3,072 pentanuc_mut contexts) robustness check. ---
    merged["pentanuc_mut"] = o["pentanuc_mut"].to_numpy()
    cell = merged.groupby("pentanuc_mut")[cols].mean()
    print(
        f"\n=== Pearson at the context level (one row per pentanuc_mut, n={len(cell)}) ==="
    )
    Xc = cell.to_numpy()
    print(f"{'':<14}" + "".join(f"{lbl:>14}" for lbl in labels))
    for i, lbl in enumerate(labels):
        row = [pearsonr(Xc[:, i], Xc[:, j])[0] for j in range(len(cols))]
        print(f"{lbl:<14}" + "".join(f"{v:>14.3f}" for v in row))


if __name__ == "__main__":
    main()
