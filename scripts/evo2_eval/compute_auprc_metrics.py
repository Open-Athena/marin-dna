"""Compute AUPRC + cluster-bootstrap SE on an existing Evo2 predictions
parquet from ``eval_matched_pair.py``.

Post-hoc replacement for the in-script PA path, which asserts on 1:9
matching after PR #194. Run ``eval_matched_pair.py --skip-metrics``, then
this. Mirrors ``snakemake/analysis/evals_v2/workflow/rules/metrics.smk``
(PR #195) in script form — evo2 isn't in the evals_v2 Snakemake pipeline.
"""

import argparse
from pathlib import Path

import pandas as pd

from marin_dna.pipelines.evals.metrics import compute_auprc_metrics


# evo2 predictions parquet → metrics parquet `score_type`. ``_rev`` →
# ``_rc`` matches the evals_v2 / dashboard reverse-complement convention;
# ``next_token_jsd_mean`` → ``jsd`` for parity with marin_dna's ``jsd_*``.
COL_RENAME: dict[str, str] = {
    "llr": "llr_avg",
    "llr_fwd": "llr_fwd",
    "llr_rev": "llr_rc",
    "minus_llr": "minus_llr_avg",
    "minus_llr_fwd": "minus_llr_fwd",
    "minus_llr_rev": "minus_llr_rc",
    "abs_llr": "abs_llr_avg",
    "abs_llr_fwd": "abs_llr_fwd",
    "abs_llr_rev": "abs_llr_rc",
    "next_token_jsd_mean": "jsd_avg",
    "next_token_jsd_mean_fwd": "jsd_fwd",
    "next_token_jsd_mean_rev": "jsd_rc",
}

# Subset of `COL_RENAME` values actually fed to AUPRC: ``leaderboard.PROTOCOLS["evo2"]``
# selects from these. ``abs_llr_*`` and raw ``llr_*`` are renamed (so the
# parquet schema is uniform across mendelian / complex) but not scored on
# mendelian. Keep in sync with `COL_RENAME` when adding a protocol.
SCORE_COLUMNS: list[str] = [
    "minus_llr_avg",
    "minus_llr_fwd",
    "minus_llr_rc",
    "jsd_avg",
    "jsd_fwd",
    "jsd_rc",
]

REQUIRED_DATASET_COLUMNS: tuple[str, ...] = (
    "chrom",
    "pos",
    "ref",
    "alt",
    "label",
    "subset",
    "match_group",
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        required=True,
        help="Predictions parquet (output of eval_matched_pair.py).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Metrics parquet path to write.",
    )
    p.add_argument(
        "--model",
        required=True,
        help="Model id (e.g. evo2_1b_base) — stamped into the output parquet.",
    )
    p.add_argument(
        "--dataset",
        default="mendelian_traits",
        help="Dataset name — stamped into the output parquet.",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Split name — stamped into the output parquet.",
    )
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Cluster-bootstrap iterations per (subset, score_column).",
    )
    p.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="RNG seed for the bootstrap. 0 (default) → reproducible across "
        "re-runs on identical inputs.",
    )
    args = p.parse_args()

    df = pd.read_parquet(args.input)
    missing_ds = [c for c in REQUIRED_DATASET_COLUMNS if c not in df.columns]
    assert not missing_ds, (
        f"input is missing dataset columns: {missing_ds}; "
        f"is this an Evo2 predictions parquet?"
    )

    # Rename per COL_RENAME. Missing source columns are skipped (e.g. an
    # ablation run with --no-rc-avg won't have *_rev atoms, so the *_rc
    # outputs are dropped from SCORE_COLUMNS below).
    present_renames = {src: dst for src, dst in COL_RENAME.items() if src in df.columns}
    df = df.rename(columns=present_renames)

    # If the input came from `eval_matched_pair.py --no-rc-avg`, only
    # the `*_fwd` atoms exist (no `_rev` → no `_rc`, no `_avg`). All six
    # `SCORE_COLUMNS` may be missing; the dashboard's LLR/JSD protocols
    # both point at `_avg`, so the resulting metrics parquet won't drive
    # any leaderboard cell — but the per-strand columns are still useful
    # for ablations, so we proceed if anything is left.
    score_cols = [c for c in SCORE_COLUMNS if c in df.columns]
    assert score_cols, (
        f"none of SCORE_COLUMNS={SCORE_COLUMNS} are present in input "
        f"(after rename); did the predictions parquet skip RC averaging?"
    )

    print(
        f"[evo2-auprc] {args.input} → {args.output}"
        f"\n  rows: {len(df)}  match_groups: {df['match_group'].nunique()}"
        f"\n  subsets: {sorted(df['subset'].unique())}"
        f"\n  score_columns: {score_cols}"
        f"\n  n_bootstrap={args.n_bootstrap} seed={args.bootstrap_seed}",
        flush=True,
    )

    metrics = compute_auprc_metrics(
        dataset=df[list(REQUIRED_DATASET_COLUMNS)],
        scores=df[score_cols],
        score_columns=score_cols,
        n_bootstrap=args.n_bootstrap,
        rng=args.bootstrap_seed,
    )
    metrics["model"] = args.model
    metrics["dataset"] = args.dataset
    metrics["split"] = args.split

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(out_path, index=False)
    print(
        f"[evo2-auprc] wrote {out_path}  ({len(metrics)} rows: "
        f"{len(score_cols)} score_types × "
        f"{int(len(metrics) / len(score_cols))} subset rows)"
    )


if __name__ == "__main__":
    main()
