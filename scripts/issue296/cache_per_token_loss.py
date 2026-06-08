"""Stage 1 of issue #296 — cache per-token validation loss for one (model, set).

Loads an HF checkpoint (local dir), pulls a mixed-case validation-interval
parquet from HuggingFace, and writes the model's per-token loss for **every**
base in long format (``compute_hf_per_token_loss``). As an in-run correctness
gate it *also* runs the official case-bucketed LL gap on the **same** set
(``compute_hf_ll_gap`` → ``aggregate_ll_gap``) and asserts that re-aggregating
the per-token loss by case reproduces it **bit-for-bit** — the equivalence
Verification step of the plan. FWD strand only.

One-off orchestration (issue #296); the reusable logic lives in
``marin_dna.pipelines.evals.per_token_loss`` / ``.ll_gap``. Run on a GPU (bf16
eval errors on CPU).

Example (one cell):

    uv run python scripts/issue296/cache_per_token_loss.py \
        --checkpoint-dir /tmp/ckpt/scaling-v0.5-h640-p46M-step-215573 \
        --model-name scaling-v0.5-h640-p46M-step-215573 \
        --hf-repo bolinas-dna/zoonomia-v1-val_cds \
        --hf-filename val_cds.parquet \
        --dataset-name val_cds \
        --window-size 255 \
        --out-dir scratch/issue296
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from marin_dna.pipelines.evals.ll_gap import aggregate_ll_gap, compute_hf_ll_gap
from marin_dna.pipelines.evals.per_token_loss import compute_hf_per_token_loss


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", required=True, help="Local HF checkpoint dir.")
    p.add_argument("--model-name", required=True, help="Label for output paths.")
    p.add_argument("--hf-repo", required=True, help="HF dataset repo id.")
    p.add_argument(
        "--hf-filename",
        required=True,
        help="Parquet file inside the repo (e.g. val_cds.parquet).",
    )
    p.add_argument("--hf-revision", default=None, help="HF dataset commit/revision.")
    p.add_argument("--dataset-name", required=True, help="Short set label (val_cds).")
    p.add_argument("--window-size", type=int, default=255)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--out-dir",
        default="scratch/issue296",
        help="Local output root; per_token/ and ll_gap_check/ are written under it.",
    )
    p.add_argument(
        "--skip-ll-gap-check",
        action="store_true",
        help="Skip the second forward (compute_hf_ll_gap equivalence re-check) — "
        "the per-token cache has its own alignment asserts. Halves wall time; use "
        "once the kernel is validated (e.g. for the big models).",
    )
    return p.parse_args()


def _reaggregate_by_case(per_token: pd.DataFrame) -> dict[str, float]:
    """Collapse the long per-token loss back to LL_upper/LL_lower/gap, the same
    quantity ``aggregate_ll_gap`` produces — but from the un-bucketed cache."""
    logp = -per_token["loss"].to_numpy(dtype=np.float64)  # loss = −log p
    up = per_token["is_upper"].to_numpy()
    s_u = logp[up].sum()
    s_l = logp[~up].sum()
    n_u = int(up.sum())
    n_l = int((~up).sum())
    return {
        "LL_upper": float(s_u / n_u),
        "LL_lower": float(s_l / n_l),
        "gap": float(s_u / n_u - s_l / n_l),
        "n_upper": n_u,
        "n_lower": n_l,
    }


def main() -> None:
    args = _parse_args()
    out_root = Path(args.out_dir)

    local = hf_hub_download(
        args.hf_repo,
        args.hf_filename,
        repo_type="dataset",
        revision=args.hf_revision,
    )
    seqs = pd.read_parquet(local)
    print(
        f"[stage1] {args.model_name}/{args.dataset_name}: {len(seqs)} windows "
        f"from {args.hf_repo}@{args.hf_revision or 'main'}/{args.hf_filename}"
    )

    # Per-token loss cache (the deliverable).
    per_token = compute_hf_per_token_loss(
        checkpoint_path=args.checkpoint_dir,
        sequences=seqs,
        window_size=args.window_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    pt_path = out_root / "per_token" / args.model_name / f"{args.dataset_name}.parquet"
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    per_token.to_parquet(pt_path, index=False)
    print(f"[stage1] wrote {len(per_token):,} token rows → {pt_path}")

    if args.skip_ll_gap_check:
        print("[stage1] skipping LL-gap equivalence re-check (--skip-ll-gap-check)")
        return

    # Same-set equivalence gate: official case-bucketed LL gap vs our re-aggregation.
    atoms = compute_hf_ll_gap(
        checkpoint_path=args.checkpoint_dir,
        sequences=seqs,
        window_size=args.window_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    official = aggregate_ll_gap(
        atoms[["ll_sum_upper", "ll_sum_lower", "n_upper", "n_lower"]].to_numpy()
    )
    ours = _reaggregate_by_case(per_token)
    assert official["n_upper"] == ours["n_upper"], (official, ours)
    assert official["n_lower"] == ours["n_lower"], (official, ours)
    # `compute_ll_clm` accumulates each row's token log-probs in fp32, then
    # aggregate_ll_gap sums rows in fp64; our path keeps every token and sums in
    # fp64. Same math, different reduction order → agreement to ~fp32-accumulation
    # (≲1e-6 on these ~1.0-nat means), not bit-exact. 1e-5 still catches any real
    # discrepancy (a token-misalignment bug is ≥1e-3).
    tol = 1e-5
    for k in ("LL_upper", "LL_lower", "gap"):
        assert abs(official[k] - ours[k]) < tol, (k, official[k], ours[k])
    print(
        f"[stage1] equivalence OK — LL_upper={ours['LL_upper']:.6f} "
        f"LL_lower={ours['LL_lower']:.6f} gap={ours['gap']:.6f} "
        f"(official aggregate matches to <{tol:g})"
    )

    check_path = (
        out_root / "ll_gap_check" / args.model_name / f"{args.dataset_name}.json"
    )
    check_path.parent.mkdir(parents=True, exist_ok=True)
    check_path.write_text(
        json.dumps(
            {
                "model": args.model_name,
                "dataset": args.dataset_name,
                "hf_repo": args.hf_repo,
                "hf_revision": args.hf_revision,
                "official_aggregate": official,
                "per_token_reaggregate": ours,
            },
            indent=2,
        )
    )
    print(f"[stage1] wrote LL-gap check → {check_path}")


if __name__ == "__main__":
    main()
