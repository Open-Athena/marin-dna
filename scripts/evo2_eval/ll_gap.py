"""Compute LL gap for an Evo2 model on a mixed-case CDS validation set.

LL gap = mean log-likelihood on phyloP-functional (uppercase) target tokens
minus mean LL on non-functional (lowercase) target tokens. Positive gap
means the model finds functional bases easier to predict than non-functional
ones — a self-supervised proxy for "captures functional/non-functional
sequence structure" (biofoundation PR #18; marin-dna issue #131
follow-up).

Default dataset: ``bolinas-dna/genomes-v5-validation-intervals-v5_255_255``
(16,384 × 255-bp CDS sequences, columns ``id`` + ``seq``).

Single-process / single-GPU; the LOCAL_RANK header is kept to allow a
torchrun scale-out later without code changes.
"""

# --- header guard: must run before importing torch / evo2 / biofoundation ---
import os

_local_rank = os.environ.get("LOCAL_RANK")
if _local_rank is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(_local_rank))
# ---------------------------------------------------------------------------

import argparse  # noqa: E402
import itertools  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from datasets import Dataset, load_dataset  # noqa: E402

from marin_dna.pipelines.evals.evo2 import (  # noqa: E402
    EVO2_MODEL_CHOICES,
    aggregate_ll_gap,
    compute_evo2_ll,
)


DEFAULT_DATASET = "bolinas-dna/genomes-v5-validation-intervals-v5_255_255"
DEFAULT_SPLIT = "validation"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=EVO2_MODEL_CHOICES)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Score only the first N rows (fast-iteration default).",
    )
    p.add_argument(
        "--streaming",
        action="store_true",
        help="Load the dataset in streaming mode and take the first --limit "
        "rows. Use for sharded datasets where downloading the full set "
        "is expensive (e.g. genome-set-animals-* JSONL shards).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Per-device eval batch size. If omitted, OOM-descent tune.",
    )
    p.add_argument(
        "--tune-start",
        type=int,
        default=512,
        help="Starting batch size for OOM-descent tuning. Default 512 "
        "because context here is short (255 tokens); seeding lower "
        "would settle at a wastefully small batch.",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--output",
        default=None,
        help="Output parquet. Defaults to results/evo2_ll_gap/{model}_n{limit}.parquet",
    )
    args = p.parse_args()

    if args.output is None:
        # Sanitize dataset name for filename (slashes -> underscores).
        ds_tag = args.dataset.split("/")[-1]
        args.output = (
            f"results/evo2_ll_gap/{args.model}__{ds_tag}__n{args.limit}.parquet"
        )

    if args.streaming:
        assert args.limit is not None, "--streaming requires --limit"
        stream = load_dataset(args.dataset, split=args.split, streaming=True)
        rows = list(itertools.islice(stream, args.limit))
        assert rows, "empty stream"
        assert "seq" in rows[0], (
            f"dataset missing 'seq' column; got {list(rows[0].keys())}"
        )
        ds = Dataset.from_list(rows)
    else:
        ds = load_dataset(args.dataset, split=args.split)
        assert "seq" in ds.column_names, (
            f"dataset missing 'seq' column; got {ds.column_names}"
        )
        if args.limit is not None and args.limit < len(ds):
            ds = ds.select(range(args.limit))
    n = len(ds)
    assert n > 0, "empty dataset after --limit"

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(
            f"[evo2-ll] model={args.model} dataset={args.dataset} "
            f"split={args.split} n={n}"
        )
        sample_seq = ds[0]["seq"]
        n_up = sum(c.isupper() for c in sample_seq)
        n_lo = sum(c.islower() for c in sample_seq)
        print(
            f"[evo2-ll] sample row 0: len={len(sample_seq)} upper={n_up} lower={n_lo}"
        )

    pred = compute_evo2_ll(
        model_name=args.model,
        dataset=ds,
        batch_size=args.batch_size,
        tune_start=args.tune_start,
        num_workers=args.num_workers,
    )

    if rank != 0:
        return

    # Per-row parquet: keep sums + counts (not means) — means break for
    # all-upper or all-lower rows where one count is 0.
    out = pd.DataFrame(
        {
            "id": ds["id"] if "id" in ds.column_names else np.arange(n),
            "ll_sum_upper": pred[:, 0].astype(np.float64),
            "ll_sum_lower": pred[:, 1].astype(np.float64),
            "n_upper": pred[:, 2].astype(np.int64),
            "n_lower": pred[:, 3].astype(np.int64),
        }
    )

    summary = aggregate_ll_gap(pred)
    out.attrs["model"] = args.model
    out.attrs["dataset"] = args.dataset
    out.attrs["split"] = args.split
    out.attrs["n"] = n

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print(
        f"[evo2-ll] wrote {out_path}  "
        f"rows={n}  target_tokens={summary['n_upper'] + summary['n_lower']} "
        f"(upper={summary['n_upper']}, lower={summary['n_lower']})"
    )
    print(f"[evo2-ll] LL_all   = {summary['LL_all']:+.4f}")
    print(f"[evo2-ll] LL_upper = {summary['LL_upper']:+.4f}  (functional)")
    print(f"[evo2-ll] LL_lower = {summary['LL_lower']:+.4f}  (non-functional)")
    print(f"[evo2-ll] gap      = {summary['gap']:+.4f}  (LL_upper - LL_lower)")


if __name__ == "__main__":
    main()
