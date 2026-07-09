"""Recompute the frozen-probe + zero-shot-LLR baselines for issue #369, both the
full 12-chromosome per-chrom-weighted AUPRC (validation vs #341) and the
**leave-out-chr1 fold** value (the number LoRA fine-tuning must beat on the dev
fold).

For each scaling-ladder rung, on Mendelian **missense** (train split):

- Frozen probe: ``traitgym_nested_oof`` (nested LOCO, inner C-search) on the
  ``concat_ref_delta`` feature built from the in-bundle ``emb_ref``/``emb_alt``
  (the productionized #320 path) → OOF ``probe_score``. The chr1 fold value is
  ``per_chrom_weighted_ap`` over the chr1 rows of that OOF (= probe trained on the
  other 11 chroms, scored on chr1).
- Zero-shot LLR: ``minus_llr_avg = -((llr_fwd + llr_rc) / 2)`` (derived from the
  raw atoms in the scores parquet), scored on identical rows.

Run: ``uv run python scripts/issue369/recompute_chr1_baselines.py`` (optionally
pass a comma-separated model-size subset, e.g. ``46M,255M``).
"""

import sys

import numpy as np
import pandas as pd
import polars as pl

from marin_dna.pipelines.evals.metrics import per_chrom_weighted_ap
from marin_dna.pipelines.evals.variant_probe import (
    DEFAULT_C_GRID,
    _inner_c_search,
    pair_feature_from_bundle,
    traitgym_nested_oof,
)

SCORES_PREFIX = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
# display size -> scaling-v0.5 run-dir stem (all step-215573). From issue #341.
MODELS: dict[str, str] = {
    "46M": "h640-p46M",
    "76M": "h768-p76M",
    "128M": "h896-p128M",
    "255M": "h1152-p255M",
    "476M": "h1408-p476M",
    "1B": "h1920-p1B",
    "2B": "h2432-p2B",
    "4B": "h2944-p4B",
}
SUBSET = "missense_variant"
DEV_CHROM = "1"  # the leave-out fold
N_JOBS = 2  # shared 4-vCPU box — cap at nproc/2 (local etiquette)


def scores_path(stem: str) -> str:
    return f"{SCORES_PREFIX}/scaling-v0.5-{stem}-step-215573/mendelian_traits.parquet"


def one_model(size: str, stem: str, full12: bool) -> dict[str, float | int]:
    df = pl.read_parquet(scores_path(stem)).to_pandas()
    mis = df[df["subset"] == SUBSET].reset_index(drop=True)
    assert len(mis) > 0, f"no {SUBSET} rows for {size}"
    label = mis["label"].to_numpy().astype(int)
    chrom = mis["chrom"].astype(str).to_numpy()
    feat = pair_feature_from_bundle(mis, "concat_ref_delta")
    minus_llr = -((mis["llr_fwd"].to_numpy() + mis["llr_rc"].to_numpy()) / 2.0)
    chr1 = chrom == DEV_CHROM

    # chr1 dev-fold baseline: probe trained on the other 11 chroms, scored on chr1
    # (the chr1 slice of the nested-LOCO OOF — one outer fold, so ~12x cheaper).
    gs = _inner_c_search(DEFAULT_C_GRID, k=5, n_jobs=N_JOBS)
    gs.fit(feat[~chr1], label[~chr1], groups=chrom[~chr1])
    probe_chr1_oof = gs.predict_proba(feat[chr1])[:, 1]

    out: dict[str, float | int] = {
        "size": size,
        "n_mis": int(len(mis)),
        "n_pos": int(label.sum()),
        "n_chr1": int(chr1.sum()),
        "n_pos_chr1": int(label[chr1].sum()),
        "probe_chr1": per_chrom_weighted_ap(
            label[chr1], probe_chr1_oof, chrom[chr1]
        ),
        "llr_chr1": per_chrom_weighted_ap(label[chr1], minus_llr[chr1], chrom[chr1]),
        "probe_full12": float("nan"),
        "llr_full12": per_chrom_weighted_ap(label, minus_llr, chrom),
    }
    if full12:  # slow validation vs #341 (full 12-fold nested OOF)
        oof, _ = traitgym_nested_oof(feat, label, chrom, n_jobs=N_JOBS)
        out["probe_full12"] = per_chrom_weighted_ap(label, oof, chrom)
    return out


def main() -> None:
    argv = [a for a in sys.argv[1:] if a != "--full"]
    full12 = "--full" in sys.argv
    want = argv[0].split(",") if argv else list(MODELS)
    rows = []
    for size in want:
        assert size in MODELS, f"unknown size {size!r}; pick from {list(MODELS)}"
        print(f"[baseline] {size} ...", flush=True)
        r = one_model(size, MODELS[size], full12)
        rows.append(r)
        print(
            f"[baseline] {size}: probe chr1={r['probe_chr1']:.3f} | "
            f"LLR chr1={r['llr_chr1']:.3f} full12={r['llr_full12']:.3f} | "
            f"probe full12={r['probe_full12']:.3f} "
            f"(chr1 n={r['n_chr1']}/{r['n_pos_chr1']}+)",
            flush=True,
        )
    out = pd.DataFrame(rows)
    dest = "scratch/issue369/chr1_baselines.parquet"
    out.to_parquet(dest, index=False)

    print("\n| size | probe chr1 | LLR chr1 | LLR full-12 | probe full-12 |")
    print("|---|--:|--:|--:|--:|")
    for _, r in out.iterrows():
        print(
            f"| {r['size']} | {r['probe_chr1']:.3f} | {r['llr_chr1']:.3f} "
            f"| {r['llr_full12']:.3f} | {r['probe_full12']:.3f} |"
        )
    print(f"\nsaved -> {dest}", flush=True)


if __name__ == "__main__":
    main()
