"""Apply the production BF16 probe unchanged to aligned BF16 and FP8 embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from marin_dna.pipelines.evals.metrics import (
    paired_auprc_degradation_metrics,
    per_chrom_ap_table,
)
from marin_dna.pipelines.evals.variant_probe import pair_feature_from_bundle


KEY_COLUMNS = ["chrom", "pos", "ref", "alt"]


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["chrom"] = out["chrom"].astype(str)
    out["pos"] = out["pos"].astype(np.int64)
    out["ref"] = out["ref"].astype(str)
    out["alt"] = out["alt"].astype(str)
    return out


def align_embedding_bundles(
    bf16: pd.DataFrame, fp8: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one-to-one key-aligned bundles in canonical BF16 row order."""
    required_bf16 = {
        *KEY_COLUMNS,
        "label",
        "subset",
        "match_group",
        "emb_ref",
        "emb_alt",
    }
    required_fp8 = {
        *KEY_COLUMNS,
        "target",
        "subset",
        "match_group",
        "emb_ref",
        "emb_alt",
    }
    assert not (required_bf16 - set(bf16.columns)), required_bf16 - set(bf16.columns)
    assert not (required_fp8 - set(fp8.columns)), required_fp8 - set(fp8.columns)
    left = _normalize_keys(bf16)
    right = _normalize_keys(fp8)
    assert not left.duplicated(KEY_COLUMNS).any(), "duplicate BF16 variant keys"
    assert not right.duplicated(KEY_COLUMNS).any(), "duplicate FP8 variant keys"
    left_keys = pd.MultiIndex.from_frame(left[KEY_COLUMNS])
    right_keys = pd.MultiIndex.from_frame(right[KEY_COLUMNS])
    missing = left_keys.difference(right_keys)
    extra = right_keys.difference(left_keys)
    assert len(missing) == 0 and len(extra) == 0, (
        f"variant-key mismatch: missing FP8={len(missing)}, extra FP8={len(extra)}"
    )
    right = right.set_index(KEY_COLUMNS).loc[left_keys].reset_index()
    assert left[KEY_COLUMNS].equals(right[KEY_COLUMNS])
    assert np.array_equal(left["label"].astype(bool), right["target"].astype(bool))
    assert np.array_equal(left["subset"].astype(str), right["subset"].astype(str))
    assert np.array_equal(left["match_group"], right["match_group"])
    return left.reset_index(drop=True), right.reset_index(drop=True)


def _top_overlap(a: np.ndarray, b: np.ndarray, fraction: float) -> float:
    assert 0 < fraction <= 1
    k = max(1, int(np.ceil(len(a) * fraction)))
    top_a = set(np.argpartition(a, -k)[-k:].tolist())
    top_b = set(np.argpartition(b, -k)[-k:].tolist())
    return len(top_a & top_b) / k


def _drift_summary(bf16: np.ndarray, fp8: np.ndarray) -> dict[str, float]:
    delta = fp8 - bf16
    abs_delta = np.abs(delta)
    return {
        "mean_abs": float(abs_delta.mean()),
        "p50_abs": float(np.quantile(abs_delta, 0.50)),
        "p95_abs": float(np.quantile(abs_delta, 0.95)),
        "p99_abs": float(np.quantile(abs_delta, 0.99)),
        "max_abs": float(abs_delta.max()),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "pearson": float(np.corrcoef(bf16, fp8)[0, 1]),
        "spearman": float(pd.Series(bf16).corr(pd.Series(fp8), method="spearman")),
        "top_1pct_overlap": _top_overlap(bf16, fp8, 0.01),
        "top_5pct_overlap": _top_overlap(bf16, fp8, 0.05),
    }


def apply_frozen_classifiers(
    bf16: pd.DataFrame, fp8: pd.DataFrame, classifiers: dict[str, dict[str, Any]]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Score aligned embeddings through the identical fitted BF16 pipelines."""
    assert len(bf16) == len(fp8)
    predictions = bf16[[*KEY_COLUMNS, "label", "subset", "match_group"]].copy()
    predictions["probe_bf16"] = np.nan
    predictions["probe_fp8"] = np.nan
    subset_rows: list[dict[str, Any]] = []
    for subset, artifact in sorted(classifiers.items()):
        mask = predictions["subset"].astype(str).to_numpy() == str(subset)
        n = int(mask.sum())
        assert n == int(artifact["n"]), (subset, n, artifact["n"])
        combo = str(artifact["feature"])
        bf16_feature = pair_feature_from_bundle(bf16.loc[mask], combo)
        fp8_feature = pair_feature_from_bundle(fp8.loc[mask], combo)
        assert bf16_feature.shape == fp8_feature.shape
        pipeline = artifact["pipeline"]
        assert pipeline.n_features_in_ == bf16_feature.shape[1], (
            subset,
            pipeline.n_features_in_,
            bf16_feature.shape,
        )
        bf16_pred = pipeline.predict_proba(bf16_feature)[:, 1]
        fp8_pred = pipeline.predict_proba(fp8_feature)[:, 1]
        assert np.isfinite(bf16_pred).all() and np.isfinite(fp8_pred).all()
        predictions.loc[mask, "probe_bf16"] = bf16_pred
        predictions.loc[mask, "probe_fp8"] = fp8_pred
        feature_delta = fp8_feature - bf16_feature
        subset_rows.append(
            {
                "subset": str(subset),
                "n": n,
                "n_pos": int(predictions.loc[mask, "label"].astype(int).sum()),
                "feature": combo,
                "C": float(artifact["C"]),
                "feature_relative_l2": float(
                    np.linalg.norm(feature_delta) / np.linalg.norm(bf16_feature)
                ),
                **{
                    f"prediction_{key}": value
                    for key, value in _drift_summary(bf16_pred, fp8_pred).items()
                },
            }
        )
    scored = predictions[["probe_bf16", "probe_fp8"]].notna().all(axis=1)
    expected = predictions["subset"].astype(str).isin(classifiers)
    assert np.array_equal(scored, expected)
    return predictions.loc[scored].reset_index(drop=True), subset_rows


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-scores", required=True)
    parser.add_argument("--fp8-scores", required=True)
    parser.add_argument("--classifiers", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--metric-bootstrap", type=int, default=1000)
    parser.add_argument("--paired-bootstrap", type=int, default=10_000)
    args = parser.parse_args()
    assert args.metric_bootstrap >= 0
    assert args.paired_bootstrap >= 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bf16, fp8 = align_embedding_bundles(
        pd.read_parquet(args.bf16_scores), pd.read_parquet(args.fp8_scores)
    )
    classifiers = joblib.load(args.classifiers)
    assert isinstance(classifiers, dict) and classifiers
    predictions, subset_rows = apply_frozen_classifiers(bf16, fp8, classifiers)

    metrics = per_chrom_ap_table(
        predictions,
        ["probe_bf16", "probe_fp8"],
        n_bootstrap=args.metric_bootstrap,
        rng=0,
        n_min=30,
    )
    metric_wide = metrics.pivot(index="subset", columns="score_type", values="value")
    metric_wide["degradation"] = metric_wide["probe_bf16"] - metric_wide["probe_fp8"]
    metric_wide = metric_wide.reset_index()
    metric_wide.columns.name = None

    paired = paired_auprc_degradation_metrics(
        predictions,
        predictions["probe_bf16"],
        predictions["probe_fp8"],
        n_bootstrap=args.paired_bootstrap,
        rng=0,
        n_min=30,
        degradation_threshold=0.01,
    )
    global_drift = _drift_summary(
        predictions["probe_bf16"].to_numpy(),
        predictions["probe_fp8"].to_numpy(),
    )
    report = {
        "contract": (
            "same production BF16 StandardScaler+L2 classifiers applied unchanged "
            "to aligned production BF16 and candidate FP8 f16-stored embeddings"
        ),
        "interpretation_caveat": (
            "absolute frozen-classifier AUPRC is in-sample because the published "
            "joblib contains all-data classifiers; BF16-to-FP8 deltas isolate "
            "embedding compatibility and are the decision evidence"
        ),
        "n_variants_total": len(bf16),
        "n_variants_scored": len(predictions),
        "n_classifiers": len(classifiers),
        "excluded_subsets": sorted(set(bf16["subset"].astype(str)) - set(classifiers)),
        "global_prediction_drift": global_drift,
        "per_subset_prediction_drift": subset_rows,
        "per_chrom_weighted_auprc": _records(metric_wide),
        "paired_global_auprc_degradation": _records(paired),
    }
    predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    metrics.to_csv(out_dir / "per_chrom_metrics.csv", index=False)
    metric_wide.to_csv(out_dir / "per_chrom_comparison.csv", index=False)
    paired.to_csv(out_dir / "paired_global_degradation.csv", index=False)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
