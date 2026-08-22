"""Compare terminal VEP performance for random and chromosome-18 validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exp473_center_seeded_projection.analyze_evals import ROW_IDENTITY_COLUMNS
from exp473_center_seeded_projection.paired_metrics import paired_policy_bootstrap
from exp473_center_seeded_projection.random_validation_vep_config import (
    RELEVANT_SUBSETS,
    validate_experiment_commit,
)

DEVELOPMENT_SPLIT = "train"
MATURE_MIRNA_SUBSET = "mature_miRNA_variant"
CHR18_POLICY = "chr18_validation"
RANDOM_POLICY = "random_validation"
MATCHED_PROTOCOLS = {
    "mendelian_traits": "minus_llr",
    "complex_traits": "abs_llr",
}
BENCHMARK_LABELS = {
    "mendelian_traits": "Mendelian",
    "complex_traits": "Complex traits",
    "sge": "SGE",
}
SUBSET_LABELS = {
    "missense_variant": "Missense",
    "splicing": "Splicing",
    "synonymous_variant": "Synonymous",
}
REPORT_ORDER = tuple(
    (benchmark, subset)
    for benchmark in ("mendelian_traits", "complex_traits", "sge")
    for subset in RELEVANT_SUBSETS[benchmark]
)


def exclude_mature_mirna_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove every matched group whose canonical subset is mature miRNA."""
    assert "subset" in frame.columns, "matched VEP frame is missing subset"
    subset = frame["subset"].astype(str)
    if not (subset == MATURE_MIRNA_SUBSET).any():
        result = frame.copy()
    else:
        assert "match_group" in frame.columns, (
            "mature-miRNA exclusion requires match_group"
        )
        excluded_groups = set(
            frame.loc[subset == MATURE_MIRNA_SUBSET, "match_group"].tolist()
        )
        result = frame.loc[~frame["match_group"].isin(excluded_groups)].copy()
    assert MATURE_MIRNA_SUBSET not in set(result["subset"].astype(str))
    return result.reset_index(drop=True)


def assert_development_metric(frame: pd.DataFrame, *, label: str) -> None:
    """Require explicit development-split provenance on a metric table."""
    assert "split" in frame.columns, f"{label}: metric table is missing split"
    observed = set(frame["split"].astype(str))
    assert observed == {DEVELOPMENT_SPLIT}, (
        f"{label}: expected split={DEVELOPMENT_SPLIT!r}, got {sorted(observed)}"
    )


def validate_score_pair(chr18: pd.DataFrame, random: pd.DataFrame) -> None:
    """Require exact labeled-row identity before a paired calculation."""
    for label, frame in ((CHR18_POLICY, chr18), (RANDOM_POLICY, random)):
        missing = set(ROW_IDENTITY_COLUMNS) - set(frame.columns)
        assert not missing, f"{label}: missing identity columns {sorted(missing)}"
    assert len(chr18) == len(random), (
        f"score row count differs: chr18={len(chr18)} random={len(random)}"
    )
    chr18_identity = chr18[list(ROW_IDENTITY_COLUMNS)].reset_index(drop=True)
    random_identity = random[list(ROW_IDENTITY_COLUMNS)].reset_index(drop=True)
    assert chr18_identity.equals(random_identity), (
        "terminal score bundles are not row-identical; paired VEP is invalid"
    )


def llr_score(frame: pd.DataFrame, *, protocol: str) -> np.ndarray:
    """Return the official FWD+RC score for one matched benchmark."""
    missing = {"llr_fwd", "llr_rc"} - set(frame.columns)
    assert not missing, f"score bundle is missing {sorted(missing)}"
    average = (
        frame["llr_fwd"].to_numpy(dtype=float) + frame["llr_rc"].to_numpy(dtype=float)
    ) / 2.0
    if protocol == "minus_llr":
        score = -average
    elif protocol == "abs_llr":
        score = np.abs(average)
    else:
        raise ValueError(f"unsupported score protocol {protocol!r}")
    assert np.isfinite(score).all(), f"{protocol} score contains non-finite values"
    return score


def compare_matched_subset(
    chr18: pd.DataFrame,
    random: pd.DataFrame,
    *,
    benchmark: str,
    subset: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute paired AUPRC and Group SMD for one relevant matched subset."""
    protocol = MATCHED_PROTOCOLS[benchmark]
    chr18 = exclude_mature_mirna_groups(chr18)
    random = exclude_mature_mirna_groups(random)
    validate_score_pair(chr18, random)
    selected = chr18["subset"].astype(str) == subset
    chr18_subset = chr18.loc[selected].reset_index(drop=True)
    random_subset = random.loc[selected].reset_index(drop=True)
    assert len(chr18_subset) > 0, f"{benchmark}/{subset}: no rows"
    assert set(chr18_subset["subset"].astype(str)) == {subset}
    scores = pd.DataFrame(
        {
            CHR18_POLICY: llr_score(chr18_subset, protocol=protocol),
            RANDOM_POLICY: llr_score(random_subset, protocol=protocol),
        }
    )
    result = paired_policy_bootstrap(
        chr18_subset["label"],
        scores,
        chr18_subset["match_group"],
        center_column=RANDOM_POLICY,
        full_column=CHR18_POLICY,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    points = result.point.set_index(["policy", "metric"])
    rows: list[dict[str, Any]] = []
    for delta in result.deltas.itertuples(index=False):
        rows.append(
            {
                "benchmark": benchmark,
                "subset": subset,
                "metric": delta.metric,
                "chr18_validation": float(
                    points.loc[(CHR18_POLICY, delta.metric), "value"]
                ),
                "random_validation": float(
                    points.loc[(RANDOM_POLICY, delta.metric), "value"]
                ),
                "delta_random_minus_chr18": float(delta.delta_center_minus_full),
                "delta_ci_low": float(delta.ci_low),
                "delta_ci_high": float(delta.ci_high),
                "probability_random_better": float(delta.probability_center_better),
                "chr18_se": float(points.loc[(CHR18_POLICY, delta.metric), "se"]),
                "random_se": float(points.loc[(RANDOM_POLICY, delta.metric), "se"]),
                "n_groups": int(points.loc[(CHR18_POLICY, delta.metric), "n_groups"]),
                "n_rows": int(points.loc[(CHR18_POLICY, delta.metric), "n_rows"]),
                "uncertainty": "paired match-group bootstrap",
                "split": DEVELOPMENT_SPLIT,
            }
        )
    samples = result.samples.assign(
        benchmark=benchmark,
        subset=subset,
        split=DEVELOPMENT_SPLIT,
        bootstrap_seed=seed,
    )
    return pd.DataFrame(rows), samples


def select_sge_endpoint(frame: pd.DataFrame, *, subset: str) -> pd.Series:
    """Select one official assay-macro SGE AUPRC endpoint."""
    assert_development_metric(frame, label="sge")
    assert MATURE_MIRNA_SUBSET not in set(frame["subset"].astype(str))
    selected = frame[
        (frame["metric"].astype(str) == "AUPRC")
        & (frame["score_type"].astype(str) == "minus_llr_avg")
        & (frame["subset"].astype(str) == subset)
        & (frame["accession"].astype(str) == "_macro_avg_")
        & (frame["gene"].astype(str) == "_macro_avg_")
    ]
    assert len(selected) == 1, (
        f"SGE {subset}: expected one endpoint, got {len(selected)}"
    )
    return selected.iloc[0]


def compare_sge(
    chr18_metrics: pd.DataFrame, random_metrics: pd.DataFrame
) -> pd.DataFrame:
    """Compare the official assay-macro SGE AUPRC endpoints."""
    rows: list[dict[str, Any]] = []
    for subset in RELEVANT_SUBSETS["sge"]:
        chr18 = select_sge_endpoint(chr18_metrics, subset=subset)
        random = select_sge_endpoint(random_metrics, subset=subset)
        rows.append(
            {
                "benchmark": "sge",
                "subset": subset,
                "metric": "auprc",
                "chr18_validation": float(chr18["value"]),
                "random_validation": float(random["value"]),
                "delta_random_minus_chr18": float(random["value"] - chr18["value"]),
                "delta_ci_low": np.nan,
                "delta_ci_high": np.nan,
                "probability_random_better": np.nan,
                "chr18_se": float(chr18["se"]),
                "random_se": float(random["se"]),
                "n_groups": np.nan,
                "n_rows": int(random["n"]),
                "uncertainty": "official per-arm assay-macro bootstrap SE",
                "split": DEVELOPMENT_SPLIT,
            }
        )
    return pd.DataFrame(rows)


def _ordered_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    row_order = {key: index for index, key in enumerate(REPORT_ORDER)}
    metric_order = {"auprc": 0, "group_smd": 1}
    result = frame.copy()
    result["_row_order"] = [
        row_order[(benchmark, subset)]
        for benchmark, subset in zip(result["benchmark"], result["subset"], strict=True)
    ]
    result["_metric_order"] = result["metric"].map(metric_order)
    return (
        result.sort_values(["_metric_order", "_row_order"])
        .drop(columns=["_row_order", "_metric_order"])
        .reset_index(drop=True)
    )


def write_summary(comparison: pd.DataFrame, output: Path) -> None:
    """Write the relevant terminal comparison in reading order."""
    lines = [
        "# Issue #473 terminal random-validation VEP comparison",
        "",
        "All results use the labeled development split: odd autosomes and chromosome X.",
        "Complete mature-miRNA match groups are excluded before matched-data metrics.",
        "Positive deltas favor the random-validation training arm.",
        "",
        "## AUPRC",
        "",
        "| Benchmark | Subset | Chr18 validation | Random validation | Delta | Uncertainty |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in comparison[comparison["metric"] == "auprc"].itertuples():
        if np.isfinite(row.delta_ci_low):
            uncertainty = (
                f"paired 95% CI [{row.delta_ci_low:.6f}, {row.delta_ci_high:.6f}]"
            )
        else:
            uncertainty = f"arm SEs {row.chr18_se:.6f}, {row.random_se:.6f}"
        lines.append(
            f"| {BENCHMARK_LABELS[row.benchmark]} | {SUBSET_LABELS[row.subset]} | "
            f"{row.chr18_validation:.6f} | {row.random_validation:.6f} | "
            f"{row.delta_random_minus_chr18:+.6f} | {uncertainty} |"
        )
    lines.extend(
        [
            "",
            "## Group SMD",
            "",
            "| Benchmark | Subset | Chr18 validation | Random validation | Delta | Paired 95% CI |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in comparison[comparison["metric"] == "group_smd"].itertuples():
        lines.append(
            f"| {BENCHMARK_LABELS[row.benchmark]} | {SUBSET_LABELS[row.subset]} | "
            f"{row.chr18_validation:.6f} | {row.random_validation:.6f} | "
            f"{row.delta_random_minus_chr18:+.6f} | "
            f"[{row.delta_ci_low:.6f}, {row.delta_ci_high:.6f}] |"
        )
    lines.extend(
        [
            "",
            "Mendelian and Complex-trait deltas use aligned match-group bootstrap draws.",
            "SGE reports the official per-arm assay-macro AUPRC and bootstrap SE because its aggregation unit is accession rather than match group.",
            "No even-autosome or chromosome-Y labeled row, prediction, or aggregate is accessed.",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_terminal_comparison(
    *,
    chr18_score_paths: dict[str, Path],
    random_score_paths: dict[str, Path],
    chr18_metric_paths: dict[str, Path],
    random_metric_paths: dict[str, Path],
    input_uris: dict[str, dict[str, str]],
    comparison_output: Path,
    samples_output: Path,
    summary_output: Path,
    manifest_output: Path,
    snapshot_commit: str,
    n_bootstrap: int,
    seed: int,
) -> None:
    """Run the complete terminal comparison and write durable audit outputs."""
    snapshot_commit = validate_experiment_commit(snapshot_commit)
    expected = set(RELEVANT_SUBSETS)
    for label, paths in (
        ("chr18 scores", chr18_score_paths),
        ("random scores", random_score_paths),
        ("chr18 metrics", chr18_metric_paths),
        ("random metrics", random_metric_paths),
    ):
        assert set(paths) == expected, f"{label}: expected {sorted(expected)}"

    comparisons: list[pd.DataFrame] = []
    samples: list[pd.DataFrame] = []
    for benchmark_index, (benchmark, protocol) in enumerate(MATCHED_PROTOCOLS.items()):
        chr18_scores = pd.read_parquet(chr18_score_paths[benchmark])
        random_scores = pd.read_parquet(random_score_paths[benchmark])
        chr18_metrics = pd.read_parquet(chr18_metric_paths[benchmark])
        random_metrics = pd.read_parquet(random_metric_paths[benchmark])
        assert_development_metric(chr18_metrics, label=f"chr18 {benchmark}")
        assert_development_metric(random_metrics, label=f"random {benchmark}")
        assert MATURE_MIRNA_SUBSET not in set(random_metrics["subset"].astype(str))
        assert protocol == MATCHED_PROTOCOLS[benchmark]
        for subset_index, subset in enumerate(RELEVANT_SUBSETS[benchmark]):
            comparison, draws = compare_matched_subset(
                chr18_scores,
                random_scores,
                benchmark=benchmark,
                subset=subset,
                n_bootstrap=n_bootstrap,
                seed=seed + benchmark_index * 100 + subset_index,
            )
            comparisons.append(comparison)
            samples.append(draws)

    comparisons.append(
        compare_sge(
            pd.read_parquet(chr18_metric_paths["sge"]),
            pd.read_parquet(random_metric_paths["sge"]),
        )
    )
    comparison = _ordered_comparison(pd.concat(comparisons, ignore_index=True))
    paired_samples = pd.concat(samples, ignore_index=True)
    assert set(comparison["split"]) == {DEVELOPMENT_SPLIT}
    assert set(paired_samples["split"]) == {DEVELOPMENT_SPLIT}

    for path in (comparison_output, samples_output, summary_output, manifest_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_parquet(comparison_output, index=False)
    paired_samples.to_parquet(samples_output, index=False)
    write_summary(comparison, summary_output)
    manifest = {
        "snapshot_commit": snapshot_commit,
        "split": DEVELOPMENT_SPLIT,
        "held_out_access": False,
        "mature_mirna_exclusion": "complete match_group",
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "relevant_subsets": {
            name: list(subsets) for name, subsets in RELEVANT_SUBSETS.items()
        },
        "input_uris": input_uris,
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (comparison_output, samples_output, summary_output)
        },
    }
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
