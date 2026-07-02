"""Recompute the Evo 2 frozen-embedding probe metrics WITH chromosome-cluster bootstrap SE
and a ``_macro_avg_`` row (#349), so they carry the exact schema the dashboard Supervised
loader (``probe_normalized_rows``) requires.

The #335 Evo 2 probe metrics (`s3://oa-bolinas/analysis/evo2_embeddings/mendelian_traits/
probe/{model}_train_probe_metrics.parquet`) were produced by the one-off `train_variant_probe.py`
script in a *wide* format (`probe_per_chrom_ap`, `probe_global_ap_se`, ...) with the
pre-#349 `per_chrom_weighted_ap` — point estimates only, no per-chrom-weighted SE, no
`_macro_avg_` row. This script re-runs the **metrics step only** over the existing per-variant
probe **predictions** (`{model}_train_probe.parquet`, already on S3) through the same
`per_chrom_ap_table` call that evals_v2's `compute_probe_metrics` rule makes, emitting the
long `[score_type, subset, value, se, n, n_pos, n_chrom] + model/dataset/split` schema. CPU
only — no GPU re-inference, no probe re-training.

Only reconciliation vs the evals_v2 rule: the #335 bundle names its reverse-complement LLR
atom ``llr_rev``; `compute_probe_metrics` expects ``llr_rc``. The FWD/RC-averaged baseline is
otherwise identical (``minus_llr = -(llr_fwd + llr_rev)/2`` in the bundle).

Provenance: issues #131 / #352. Run from the repo root:

    uv run python scripts/evo2_eval/recompute_probe_metrics_with_se.py

Outputs land in ``scratch/evo2_probe_metrics_se/`` named exactly as the gist expects
(`mendelian_{model}_train_probe_metrics.parquet`), ready to push to EVO2_METRICS_GIST.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import polars as pl

from marin_dna.pipelines.evals.metrics import SCORE_PROTOCOLS, per_chrom_ap_table

DATASET = "mendelian_traits"
SHORT = "mendelian"  # EVO2_DATASET_SHORT[mendelian_traits]; the gist filename prefix
SPLIT = "train"
# Mendelian is directional (concat_ref_delta feature): the zero-shot baseline is -LLR on the
# FWD/RC-averaged raw LLR, exactly as compute_probe_metrics derives it (score_protocol=minus_llr).
SCORE_PROTOCOL = "minus_llr"
MODELS = ["evo2_1b_base", "evo2_7b", "evo2_40b"]

S3_PROBE = "s3://oa-bolinas/analysis/evo2_embeddings/mendelian_traits/probe"
OUT = pathlib.Path("scratch/evo2_probe_metrics_se")

# per_chrom_ap_table knobs — must match config/config.yaml `probe:` (n_bootstrap/n_min) and the
# rule's rng=0 pin, so the SE is bit-stable and comparable to the marin_dna probe rows.
N_BOOTSTRAP = 1000
N_MIN = 30
MACRO = "_macro_avg_"


def recompute(model: str) -> pd.DataFrame:
    """Load one model's probe predictions and return the SE-carrying metrics table."""
    preds = pl.read_parquet(f"{S3_PROBE}/{model}_train_probe.parquet").to_pandas()
    # #335 bundle uses the `_rev` suffix; evals_v2 compute_probe_metrics uses `_rc`.
    preds = preds.rename(columns={"llr_rev": "llr_rc"})
    for col in ("probe_score", "label", "subset", "chrom", "llr_fwd", "llr_rc"):
        assert col in preds.columns, f"{model}: predictions missing column {col!r}"

    # Zero-shot baseline on the identical rows: dataset score protocol on the FWD/RC-avg raw
    # LLR (== compute_metrics `_avg`), mirroring compute_probe_metrics exactly.
    transform = SCORE_PROTOCOLS[SCORE_PROTOCOL]
    baseline_col = f"{SCORE_PROTOCOL}_avg"
    preds[baseline_col] = transform((preds["llr_fwd"] + preds["llr_rc"]) / 2)

    metrics = per_chrom_ap_table(
        preds,
        ["probe_score", baseline_col],
        n_bootstrap=N_BOOTSTRAP,
        rng=0,
        n_min=N_MIN,
    )
    metrics["model"] = model
    metrics["dataset"] = DATASET
    metrics["split"] = SPLIT
    return metrics


def verify(model: str, metrics: pd.DataFrame) -> None:
    """Correctness gate: the recomputed per-subset point estimates must reproduce the #335
    wide metrics parquet (== the #131 results table) exactly — we only *add* SE."""
    old = pl.read_parquet(f"{S3_PROBE}/{model}_train_probe_metrics.parquet").to_pandas()
    baseline_col = f"{SCORE_PROTOCOL}_avg"

    new_probe = metrics[
        (metrics.score_type == "probe_score") & (metrics.subset != MACRO)
    ].set_index("subset")["value"]
    new_llr = metrics[
        (metrics.score_type == baseline_col) & (metrics.subset != MACRO)
    ].set_index("subset")["value"]
    old_probe = old.set_index("subset")["probe_per_chrom_ap"]
    old_llr = old.set_index("subset")["llr_per_chrom_ap"]

    # `per_chrom_ap_table` emits a row for EVERY subset in the predictions, incl. below-gate
    # ones the #335 wide table dropped (e.g. mature_miRNA_variant, n<300 → NaN probe row).
    # That's the evals_v2 `compute_probe_metrics` schema, so we require ⊆, not ==, and surface
    # the extras.
    assert set(old_probe.index) <= set(new_probe.index), (
        f"{model}: #335 subsets not all reproduced: "
        f"{set(old_probe.index) - set(new_probe.index)}"
    )
    extra = sorted(set(new_probe.index) - set(old_probe.index))
    if extra:
        print(
            f"    [{model}] extra (below-gate) subsets emitted by per_chrom_ap_table:"
        )
        for s in extra:
            print(
                f"        {s}: probe value={new_probe[s]!r} (expected NaN, below 300 gate)"
            )
    for subset in old_probe.index:
        assert abs(new_probe[subset] - old_probe[subset]) < 1e-9, (
            f"{model} {subset}: probe AUPRC drift "
            f"{new_probe[subset]:.6f} vs #335 {old_probe[subset]:.6f}"
        )
        assert abs(new_llr[subset] - old_llr[subset]) < 1e-9, (
            f"{model} {subset}: LLR AUPRC drift "
            f"{new_llr[subset]:.6f} vs #335 {old_llr[subset]:.6f}"
        )

    # Schema the dashboard loader requires: an `se` column + a `_macro_avg_` row.
    assert "se" in metrics.columns, f"{model}: no se column"
    macro = metrics[(metrics.score_type == "probe_score") & (metrics.subset == MACRO)]
    assert len(macro) == 1, f"{model}: expected exactly one probe _macro_avg_ row"
    assert pd.notna(macro["se"].iloc[0]), f"{model}: macro se is NaN"
    # Every subset that cleared the gate carries a finite SE.
    finite = metrics[(metrics.score_type == "probe_score") & metrics["value"].notna()]
    assert finite["se"].notna().all(), f"{model}: a finite probe value has NaN se"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        metrics = recompute(model)
        verify(model, metrics)
        out = OUT / f"{SHORT}_{model}_train_probe_metrics.parquet"
        metrics.to_parquet(out, index=False)

        macro = metrics[
            (metrics.score_type == "probe_score") & (metrics.subset == MACRO)
        ].iloc[0]
        # K = gated per-subset probe rows with a finite value (the set entering the macro);
        # excludes the below-gate NaN subset the pipeline still emits as a row.
        k = int(
            (
                (metrics.score_type == "probe_score")
                & (metrics.subset != MACRO)
                & metrics["value"].notna()
            ).sum()
        )
        print(f"[{model}] wrote {out}  ({len(metrics)} rows)")
        print(
            f"    probe macro AUPRC = {macro['value']:.4f} ± {macro['se']:.4f} "
            f"(K={k} gated subsets)"
        )
        with pd.option_context("display.width", 160, "display.max_columns", 20):
            show = metrics[metrics.score_type == "probe_score"][
                ["subset", "value", "se", "n", "n_pos", "n_chrom"]
            ]
            print(show.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
