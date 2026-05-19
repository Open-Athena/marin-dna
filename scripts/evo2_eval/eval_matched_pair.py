"""Score matched-pair variants (mendelian or complex traits) with Evo2.

Per-variant score bundle (LLR + next-token JSD, FWD/RC/avg) over an
8192-bp window, followed by PairwiseAccuracy ± SE per consequence
subset on every score column. One model per run.

Default dataset: ``bolinas-dna/evals_mendelian_traits``; override with
``--dataset-name complex_traits`` (auto-derives the HF path and the
output dir) or pass ``--dataset-hf-path`` explicitly.

Single-GPU execution only — Evo2's Vortex backend handles its own
multi-GPU sharding (e.g. 40B on GH200's 96 GB sits on one device). We do
not use ``torchrun`` here, matching the cluster yaml's single-process
convention.
"""

# --- header guard: must run before importing torch / evo2 / biofoundation ---
# From biofoundation/examples/evo2_llr.py. Kept for parity with the LL-gap
# script in case this entry is ever invoked under torchrun; under plain
# ``python`` it's a no-op (LOCAL_RANK unset).
import os

_local_rank = os.environ.get("LOCAL_RANK")
if _local_rank is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(_local_rank))
# ---------------------------------------------------------------------------

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from datasets import load_dataset  # noqa: E402

from bolinas.pipelines.evals.evo2 import EVO2_MODEL_CHOICES  # noqa: E402
from bolinas.pipelines.evals.metrics import (  # noqa: E402
    GLOBAL_SUBSET,
    MACRO_AVG_SUBSET,
    compute_pairwise_metrics,
)

# Script-local Evo2 scoring (no KV-cache; doesn't go through
# bolinas.model.runner / HF Trainer — see scripts/evo2_eval/_evo2_scoring.py
# docstring for why).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _evo2_scoring import compute_evo2_bundle  # noqa: E402, I001


# Schema of the matched-pair eval datasets. Same tuple as
# `bolinas.pipelines.evals.conservation.REQUIRED_VARIANT_COLUMNS`, hardcoded
# here to avoid importing conservation.py (which triggers a top-level
# `import pyBigWig` — needs libBigWig.so on the host).
REQUIRED_VARIANT_COLUMNS = (
    "chrom",
    "pos",
    "ref",
    "alt",
    "label",
    "subset",
    "match_group",
)
# Bundle output columns we score through compute_pairwise_metrics. Each
# applies to FWD+RC-averaged + per-strand variants when rc_avg is on.
# `minus_llr` is the mendelian leaderboard's canonical column;
# `abs_llr` is complex_traits' canonical column; `next_token_jsd_mean`
# is direction-agnostic by construction (always ≥ 0) — included on both
# datasets for cross-method comparison.
SCORE_COLUMN_BASES = ("minus_llr", "abs_llr", "next_token_jsd_mean")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=EVO2_MODEL_CHOICES)
    p.add_argument("--split", default="train")
    p.add_argument(
        "--dataset-name",
        default="mendelian_traits",
        help="Short dataset name (e.g. 'mendelian_traits' or 'complex_traits'). "
        "Drives the HF path default (bolinas-dna/evals_{name}) and the output "
        "directory (results/evo2_{name}/).",
    )
    p.add_argument(
        "--dataset-hf-path",
        default=None,
        help="HF dataset ID. Defaults to bolinas-dna/evals_{dataset_name}.",
    )
    p.add_argument(
        "--dataset-revision",
        default=None,
        help="Pin the HF dataset to a specific commit SHA / tag / branch. "
        "Defaults to HEAD of the default branch.",
    )
    p.add_argument(
        "--genome-path",
        default="results/genome.fa.gz",
        help="Path to GRCh38 reference (Ensembl release-113 primary assembly).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Scores parquet path. Defaults to "
        "results/evo2_{dataset_name}/{model}_{split}.parquet",
    )
    p.add_argument(
        "--output-metrics",
        default=None,
        help="Metrics parquet path. Defaults to "
        "results/evo2_{dataset_name}/{model}_{split}_metrics.parquet",
    )
    p.add_argument("--window-size", type=int, default=8192)
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-batch row count. Each batch feeds the model "
        "[2*batch_size, window_size] tokens (ref+alt concatenated). "
        "Default 16 worked for evo2_1b_base on GH200 at window=8192; "
        "halve for 7B, more for 40B-needs-bs=1.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only score the first N variants. Smoke-test flag.",
    )
    p.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Filter to a single consequence subset (e.g. 'splicing'). "
        "Useful for targeted #175-style sanity checks (e.g. RC averaging "
        "helps most on splicing).",
    )
    p.add_argument(
        "--no-rc-avg",
        action="store_true",
        help="Disable FWD+RC averaging (ablation only). Default is RC on, "
        "matching evals_v2 and the #161 leaderboard protocol.",
    )
    p.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Write the scores parquet and exit; skip PairwiseAccuracy. "
        "Use when the dataset isn't 1:1 paired (e.g. 1:9 in evals_mendelian_traits "
        "post-#194) and the pairwise metric no longer applies.",
    )
    args = p.parse_args()

    if args.dataset_hf_path is None:
        args.dataset_hf_path = f"bolinas-dna/evals_{args.dataset_name}"
    out_dir = f"results/evo2_{args.dataset_name}"
    if args.output is None:
        args.output = f"{out_dir}/{args.model}_{args.split}.parquet"
    if args.output_metrics is None:
        args.output_metrics = f"{out_dir}/{args.model}_{args.split}_metrics.parquet"

    ds = load_dataset(
        args.dataset_hf_path, split=args.split, revision=args.dataset_revision
    ).to_pandas()
    missing = [c for c in REQUIRED_VARIANT_COLUMNS if c not in ds.columns]
    assert not missing, f"dataset is missing required columns: {missing}"
    assert ds["label"].isna().sum() == 0, "label column contains NaN"
    assert set(ds["label"].astype(int).unique()) <= {0, 1}, "label is not binary 0/1"
    if args.subset is not None:
        assert args.subset in set(ds["subset"]), (
            f"--subset {args.subset!r} not in dataset; available: "
            f"{sorted(set(ds['subset']))}"
        )
        ds = ds[ds["subset"] == args.subset].reset_index(drop=True)
        print(f"[evo2] --subset {args.subset} applied, scoring {len(ds)} variants")
    if args.limit is not None:
        ds = ds.head(args.limit).reset_index(drop=True)
        print(f"[evo2] --limit {args.limit} applied, scoring {len(ds)} variants")

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        rc_state = "OFF" if args.no_rc_avg else "ON"
        print(
            f"[evo2] model={args.model} split={args.split} "
            f"n={len(ds)} rc_avg={rc_state} window={args.window_size}"
        )
        print(f"[evo2] subsets:\n{ds['subset'].value_counts().to_string()}")
        n_pairs_input = ds["match_group"].nunique()
        print(f"[evo2] match_groups (pairs): {n_pairs_input}")

    scores = compute_evo2_bundle(
        model_name=args.model,
        df=ds[["chrom", "pos", "ref", "alt"]],
        genome_path=args.genome_path,
        window_size=args.window_size,
        batch_size=args.batch_size,
        rc_avg=not args.no_rc_avg,
    )

    if rank != 0:
        return

    out = pd.concat(
        [
            ds[list(REQUIRED_VARIANT_COLUMNS)].reset_index(drop=True),
            scores.reset_index(drop=True),
        ],
        axis=1,
    )
    assert len(out) == len(ds)
    assert np.isfinite(out["llr"]).all()
    assert np.isfinite(out["next_token_jsd_mean"]).all()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(
        f"[evo2] wrote {out_path}  "
        f"llr min={out.llr.min():.3f} max={out.llr.max():.3f} "
        f"mean={out.llr.mean():.3f}  "
        f"jsd mean={out.next_token_jsd_mean.mean():.4f}"
    )

    if args.skip_metrics:
        print("[evo2] --skip-metrics set; skipping PairwiseAccuracy")
        return

    # Score each base (minus_llr, abs_llr, next_token_jsd_mean) on its
    # avg + per-strand variants when present. Lets us verify #175's
    # patterns (avg > single strand) and compare LLR-based vs JSD-based
    # signal on both mendelian (canonical: minus_llr) and complex_traits
    # (canonical: abs_llr).
    score_cols: list[str] = []
    for base in SCORE_COLUMN_BASES:
        for suffix in ("", "_fwd", "_rev"):
            col = f"{base}{suffix}"
            if col in out.columns:
                score_cols.append(col)
    try:
        metrics = compute_pairwise_metrics(
            dataset=out[list(REQUIRED_VARIANT_COLUMNS)],
            scores=out[score_cols],
            score_columns=score_cols,
        )
    except AssertionError as e:
        # At small --limit we may not have ≥1 subset with n_pairs ≥ n_min=30
        # (the macro-avg threshold). Smoke-test scenario; metrics aren't the
        # deliverable. Skip them and let the scores parquet stand.
        print(f"[evo2] WARNING: skipping metrics ({e})", flush=True)
        return
    metrics["model"] = args.model
    metrics["dataset"] = args.dataset_name
    metrics["split"] = args.split

    metrics_path = Path(args.output_metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(metrics_path, index=False)
    print(f"[evo2] wrote {metrics_path}")

    # Print PA per subset for each score_type side-by-side, so the FWD vs
    # RC vs avg pattern is easy to eyeball.
    print(f"[evo2] {args.model} PairwiseAccuracy by subset × score_type:")
    print(
        f"  {'subset':36s} {'n_pairs':>8s}  "
        + "  ".join(f"{sc:>22s}" for sc in score_cols)
    )
    # Sort subsets by n_pairs desc; put _global_ / _macro_avg_ first.
    pivot = metrics.pivot_table(
        index="subset",
        columns="score_type",
        values=["value", "se", "n_pairs"],
        aggfunc="first",
    )
    ordered_subsets = [GLOBAL_SUBSET, MACRO_AVG_SUBSET] + [
        s
        for s, _ in sorted(
            (
                (s, int(pivot.loc[s, ("n_pairs", score_cols[0])]))
                for s in pivot.index
                if s not in (GLOBAL_SUBSET, MACRO_AVG_SUBSET)
            ),
            key=lambda x: -x[1],
        )
    ]
    for subset_name in ordered_subsets:
        if subset_name not in pivot.index:
            continue
        n_pairs = int(pivot.loc[subset_name, ("n_pairs", score_cols[0])])
        cells = []
        for sc in score_cols:
            v = pivot.loc[subset_name, ("value", sc)]
            se = pivot.loc[subset_name, ("se", sc)]
            cells.append(f"{v:.3f} ± {se:.3f}")
        label = subset_name
        if subset_name == GLOBAL_SUBSET:
            label = "(Global)"
        elif subset_name == MACRO_AVG_SUBSET:
            label = "(Macro Avg)"
        print(f"  {label:36s} {n_pairs:>8d}  " + "  ".join(f"{c:>22s}" for c in cells))


if __name__ == "__main__":
    main()
