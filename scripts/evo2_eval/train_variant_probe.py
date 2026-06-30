"""Train the frozen-embedding linear probe on an Evo2 bundle parquet (issue #131).

The Evo2 analog of the evals_v2 probe rule (#320) + per-chrom AUPRC metric (#331),
in script form — Evo2 isn't in the evals_v2 Snakemake pipeline. Reuses the exact
library code, no new logistic regression or metric:

  1. ``run_subset_probes`` (#320) — per-subset, nested leave-one-chromosome-out L2
     probe over the in-bundle pooled ``emb_ref``/``emb_alt`` (the #131 embedding
     extension), emitting a LOOC ``probe_score`` per variant.
  2. ``per_chrom_weighted_ap`` (#331) — the TraitGym per-chromosome-weighted AUPRC,
     the protocol's headline metric; scored on ``probe_score`` (probe) and on the
     zero-shot LLR baseline through the SAME metric, plus the global
     match-group-bootstrap AUPRC (secondary) via ``auprc_with_bootstrap_se``.

Input is the parquet from ``eval_matched_pair.py --return-embeddings``. CPU-only
(sklearn on cached embeddings — no GPU); run it on the dev box / a CPU instance
after rsyncing the bundle, like ``compute_auprc_metrics.py``.

    uv run python scripts/evo2_eval/train_variant_probe.py \
        --input results/evo2_mendelian_traits/evo2_7b_train.parquet \
        --output-predictions results/evo2_mendelian_traits/evo2_7b_train_probe.parquet \
        --output-metrics results/evo2_mendelian_traits/evo2_7b_train_probe_metrics.parquet \
        --model evo2_7b --dataset mendelian_traits
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from marin_dna.pipelines.evals.metrics import (
    auprc_with_bootstrap_se,
    per_chrom_weighted_ap,
)
from marin_dna.pipelines.evals.variant_probe import PAIR_COMBOS, run_subset_probes

# Dataset → (feature_combo, baseline LLR column). Directional datasets (a real
# ref→alt direction) take the asymmetric `concat_ref_delta = [ref, alt−ref]` and the
# signed `minus_llr` baseline; swap-invariant ones take the symmetric `sum_absdiff =
# [ref+alt, |alt−ref|]` and the `abs_llr` baseline. Mirrors the evals_v2
# common.smk `_PROTOCOL_FEATURE` rule (issue #314), keyed on dataset rather than
# score-protocol since the Evo2 bundle has both `minus_llr` and `abs_llr` columns.
_DATASET_PROTOCOL: dict[str, tuple[str, str]] = {
    "mendelian_traits": ("concat_ref_delta", "minus_llr"),
    "sge": ("concat_ref_delta", "minus_llr"),
    "complex_traits": ("sum_absdiff", "abs_llr"),
    "caqtl": ("sum_absdiff", "abs_llr"),
    "dsqtl": ("sum_absdiff", "abs_llr"),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        required=True,
        help="Evo2 bundle parquet with emb_ref/emb_alt "
        "(eval_matched_pair.py --return-embeddings).",
    )
    p.add_argument(
        "--output-predictions",
        required=True,
        help="Per-variant LOOC predictions parquet (adds probe_score; drops the "
        "bulky emb_ref/emb_alt).",
    )
    p.add_argument(
        "--output-metrics",
        required=True,
        help="Per-subset probe-vs-LLR AUPRC metrics parquet.",
    )
    p.add_argument(
        "--output-classifiers",
        default=None,
        help="Optional joblib of the fitted per-subset classifiers (for cross-"
        "dataset transfer). Defaults next to --output-predictions (.joblib).",
    )
    p.add_argument("--model", required=True, help="Model id — stamped into outputs.")
    p.add_argument(
        "--dataset",
        default="mendelian_traits",
        help="Dataset name — selects the feature/baseline protocol and is stamped "
        f"into outputs. Known: {sorted(_DATASET_PROTOCOL)}.",
    )
    p.add_argument("--split", default="train", help="Split name — stamped in.")
    p.add_argument(
        "--feature-combo",
        default=None,
        choices=PAIR_COMBOS,
        help="Override the dataset-derived ref/alt embedding feature.",
    )
    p.add_argument(
        "--baseline-col",
        default=None,
        help="Override the dataset-derived zero-shot LLR baseline column "
        "(default: minus_llr for directional, abs_llr for symmetric).",
    )
    p.add_argument(
        "--min-variants",
        type=int,
        default=300,
        help="Min variants per subset to train a probe (evals_v2 default 300).",
    )
    p.add_argument(
        "--min-chroms",
        type=int,
        default=3,
        help="Min distinct chromosomes per subset (nested LOOC needs >=3).",
    )
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Cluster-bootstrap iterations for the secondary global AUPRC SE.",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="GridSearchCV parallelism. Default 4; cap at nproc/2 on the shared "
        "dev box (local-node etiquette).",
    )
    args = p.parse_args()

    if args.dataset not in _DATASET_PROTOCOL and (
        args.feature_combo is None or args.baseline_col is None
    ):
        raise SystemExit(
            f"--dataset {args.dataset!r} is not in the known protocol map "
            f"{sorted(_DATASET_PROTOCOL)}; pass --feature-combo and --baseline-col "
            f"explicitly."
        )
    default_feature, default_baseline = _DATASET_PROTOCOL.get(
        args.dataset, (None, None)
    )
    feature_combo = args.feature_combo or default_feature
    baseline_col = args.baseline_col or default_baseline
    assert feature_combo is not None and baseline_col is not None

    df = pd.read_parquet(args.input)
    assert "emb_ref" in df.columns and "emb_alt" in df.columns, (
        f"{args.input} lacks emb_ref/emb_alt — re-score with "
        f"eval_matched_pair.py --return-embeddings"
    )
    for c in ("chrom", "label", baseline_col):
        assert c in df.columns, f"{args.input} lacks required column {c!r}"

    print(
        f"[evo2-probe] {args.input}"
        f"\n  rows: {len(df)}  subsets: "
        f"{sorted(df['subset'].unique()) if 'subset' in df.columns else ['all']}"
        f"\n  feature_combo: {feature_combo}  baseline: {baseline_col}"
        f"\n  min_variants={args.min_variants} min_chroms={args.min_chroms} "
        f"n_jobs={args.n_jobs}",
        flush=True,
    )

    predictions, classifiers = run_subset_probes(
        df,
        feature_combo=feature_combo,
        min_variants=args.min_variants,
        min_chroms=args.min_chroms,
        n_jobs=args.n_jobs,
    )

    # Per-subset probe-vs-LLR AUPRC. Primary = per-chromosome-weighted AUPRC (#331);
    # secondary = global match-group-bootstrap AUPRC (#175 / dashboard). The LLR
    # baseline is scored on the SAME finite-score rows through the SAME per-chrom
    # metric so the probe−LLR delta is apples-to-apples.
    subset = (
        predictions["subset"].astype(str)
        if "subset" in predictions.columns
        else pd.Series(np.full(len(predictions), "all"), index=predictions.index)
    )
    rows: list[dict[str, object]] = []
    for s in sorted(subset.unique()):
        sub = predictions[subset.to_numpy() == s]
        scored = sub[sub["probe_score"].notna()]
        if scored.empty:
            print(f"[evo2-probe] {s}: unprobed (below threshold) — skipped")
            continue
        y = scored["label"].to_numpy().astype(int)
        chrom = scored["chrom"].astype(str).to_numpy()
        mg = scored["match_group"].to_numpy()
        probe_score = scored["probe_score"].to_numpy()
        llr = scored[baseline_col].to_numpy()
        probe_pc = per_chrom_weighted_ap(y, probe_score, chrom)
        llr_pc = per_chrom_weighted_ap(y, llr, chrom)
        probe_g = auprc_with_bootstrap_se(
            y, probe_score, mg, n_bootstrap=args.n_bootstrap, rng=0
        )
        llr_g = auprc_with_bootstrap_se(y, llr, mg, n_bootstrap=0)
        rows.append(
            {
                "subset": s,
                "n": int(len(scored)),
                "n_pos": int(y.sum()),
                "probe_per_chrom_ap": probe_pc,
                "llr_per_chrom_ap": llr_pc,
                "delta_per_chrom_ap": probe_pc - llr_pc,
                "probe_global_ap": probe_g["value"],
                "probe_global_ap_se": probe_g["se"],
                "llr_global_ap": llr_g["value"],
            }
        )
        print(
            f"[evo2-probe] {s:32s} n={len(scored):5d} pos={int(y.sum()):4d}  "
            f"probe_pc={probe_pc:.4f}  llr_pc={llr_pc:.4f}  "
            f"Δ={probe_pc - llr_pc:+.4f}"
        )

    metrics = pd.DataFrame(rows)
    metrics["model"] = args.model
    metrics["dataset"] = args.dataset
    metrics["split"] = args.split
    metrics["feature_combo"] = feature_combo
    metrics["baseline_col"] = baseline_col

    pred_path = Path(args.output_predictions)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(pred_path, index=False)

    metrics_path = Path(args.output_metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(metrics_path, index=False)

    clf_path = (
        Path(args.output_classifiers)
        if args.output_classifiers
        else pred_path.with_suffix(".joblib")
    )
    joblib.dump(classifiers, clf_path)

    n_scored = int(predictions["probe_score"].notna().sum())
    print(
        f"[evo2-probe] wrote {pred_path} ({n_scored}/{len(predictions)} scored), "
        f"{metrics_path} ({len(metrics)} subsets), {clf_path} "
        f"({len(classifiers)} probes)"
    )


if __name__ == "__main__":
    main()
