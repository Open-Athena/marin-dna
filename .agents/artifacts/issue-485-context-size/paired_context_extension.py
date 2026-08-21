"""Paired context-extension uncertainty for issue #485.

The zero-shot comparison resamples matched groups within each consequence subset.
The probe comparison resamples chromosomes and preserves the shared chromosome draw
across both context arms; its macro comparison also shares that draw across subsets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


SUBSETS = (
    "missense_variant",
    "splicing",
    "synonymous_variant",
    "tss_proximal",
    "5_prime_UTR_variant",
    "3_prime_UTR_variant",
    "distal",
    "non_coding_transcript_exon_variant",
)
MACRO = "_macro_avg_"
KEY_COLUMNS = ["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]


def _summary(
    point: float,
    bootstrap: np.ndarray,
    *,
    n_clusters: int,
    n_rows: int,
) -> dict[str, float | int | bool]:
    bootstrap = bootstrap[np.isfinite(bootstrap)]
    if bootstrap.size == 0:
        raise ValueError("paired bootstrap produced no finite draws")
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    p_value = min(
        2.0
        * min(
            float(np.mean(bootstrap <= 0.0)),
            float(np.mean(bootstrap >= 0.0)),
        ),
        1.0,
    )
    p_value = max(p_value, 1.0 / bootstrap.size)
    return {
        "delta_extension_minus_baseline": point,
        "se": float(np.std(bootstrap, ddof=1)),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_two_sided": p_value,
        "significant_0_05": bool(low > 0.0 or high < 0.0),
        "n_bootstrap_valid": int(bootstrap.size),
        "n_clusters": n_clusters,
        "n_rows": n_rows,
    }


def _read_aligned(
    baseline_path: Path,
    extension_path: Path,
    score_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = KEY_COLUMNS + score_columns
    baseline = pd.read_parquet(baseline_path, columns=columns)
    extension = pd.read_parquet(extension_path, columns=columns)
    if not baseline[KEY_COLUMNS].equals(extension[KEY_COLUMNS]):
        raise ValueError("baseline and extension rows are not exactly aligned")
    if set(baseline["subset"]) != set(extension["subset"]):
        raise ValueError("baseline and extension subset sets differ")
    return baseline, extension


def _zero_shot_deltas(
    baseline_path: Path,
    extension_path: Path,
    *,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, object]]:
    baseline, extension = _read_aligned(
        baseline_path,
        extension_path,
        ["llr_fwd", "llr_rc"],
    )
    baseline_score = -(
        baseline["llr_fwd"].to_numpy() + baseline["llr_rc"].to_numpy()
    ) / 2
    extension_score = -(
        extension["llr_fwd"].to_numpy() + extension["llr_rc"].to_numpy()
    ) / 2
    if not np.isfinite(baseline_score).all() or not np.isfinite(extension_score).all():
        raise ValueError("zero-shot scores contain non-finite values")

    rows: list[dict[str, object]] = []
    subset_bootstraps: list[np.ndarray] = []
    total_groups = 0
    total_rows = 0
    child_seeds = np.random.SeedSequence(seed).spawn(len(SUBSETS))
    for subset, child_seed in zip(SUBSETS, child_seeds, strict=True):
        keep = baseline["subset"].to_numpy() == subset
        labels = baseline.loc[keep, "label"].to_numpy(dtype=int)
        groups = baseline.loc[keep, "match_group"].to_numpy()
        baseline_arm = baseline_score[keep]
        extension_arm = extension_score[keep]
        group_rows = list(pd.Series(groups).groupby(groups).indices.values())
        if len(group_rows) < 30:
            raise ValueError(f"{subset} does not satisfy the macro group gate")
        generator = np.random.default_rng(child_seed)
        bootstrap = np.empty(n_bootstrap, dtype=float)
        for iteration in range(n_bootstrap):
            sampled = generator.integers(0, len(group_rows), size=len(group_rows))
            indices = np.concatenate([group_rows[index] for index in sampled])
            bootstrap[iteration] = average_precision_score(
                labels[indices], extension_arm[indices]
            ) - average_precision_score(labels[indices], baseline_arm[indices])
        point = average_precision_score(labels, extension_arm) - average_precision_score(
            labels, baseline_arm
        )
        rows.append(
            {
                "protocol": "zero-shot",
                "subset": subset,
                **_summary(
                    point,
                    bootstrap,
                    n_clusters=len(group_rows),
                    n_rows=len(labels),
                ),
            }
        )
        subset_bootstraps.append(bootstrap)
        total_groups += len(group_rows)
        total_rows += len(labels)

    rows.insert(
        0,
        {
            "protocol": "zero-shot",
            "subset": MACRO,
            **_summary(
                float(np.mean([row["delta_extension_minus_baseline"] for row in rows])),
                np.mean(np.stack(subset_bootstraps), axis=0),
                n_clusters=total_groups,
                n_rows=total_rows,
            ),
        },
    )
    return rows


def _chrom_components(
    labels: np.ndarray,
    scores: np.ndarray,
    chroms: np.ndarray,
) -> dict[object, tuple[float, float]]:
    components: dict[object, tuple[float, float]] = {}
    for chrom in np.unique(chroms):
        keep = (chroms == chrom) & np.isfinite(scores)
        kept_labels = labels[keep]
        if 0 < int(kept_labels.sum()) < len(kept_labels):
            components[chrom] = (
                float(average_precision_score(kept_labels, scores[keep])),
                float(keep.sum()),
            )
    return components


def _weighted_component(
    components: dict[object, tuple[float, float]],
    multiplicities: dict[object, int] | None = None,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for chrom, (value, weight) in components.items():
        multiplicity = 1 if multiplicities is None else multiplicities.get(chrom, 0)
        numerator += value * weight * multiplicity
        denominator += weight * multiplicity
    return numerator / denominator if denominator else float("nan")


def _paired_chrom_bootstrap(
    baseline_arm: dict[object, tuple[float, float]],
    extension_arm: dict[object, tuple[float, float]],
    *,
    n_bootstrap: int,
    generator: np.random.Generator,
) -> tuple[float, np.ndarray, int]:
    union = sorted(set(baseline_arm) | set(extension_arm))
    point = _weighted_component(extension_arm) - _weighted_component(baseline_arm)
    bootstrap = np.empty(n_bootstrap, dtype=float)
    for iteration in range(n_bootstrap):
        sampled = generator.choice(union, size=len(union), replace=True)
        multiplicities = dict(zip(*np.unique(sampled, return_counts=True), strict=True))
        bootstrap[iteration] = _weighted_component(
            extension_arm, multiplicities
        ) - _weighted_component(baseline_arm, multiplicities)
    return point, bootstrap, len(union)


def _probe_deltas(
    baseline_path: Path,
    extension_path: Path,
    *,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, object]]:
    baseline, extension = _read_aligned(
        baseline_path, extension_path, ["probe_score"]
    )
    subset_components: dict[
        str,
        tuple[dict[object, tuple[float, float]], dict[object, tuple[float, float]]],
    ] = {}
    rows: list[dict[str, object]] = []
    child_seeds = np.random.SeedSequence(seed).spawn(len(SUBSETS) + 1)
    for subset, child_seed in zip(SUBSETS, child_seeds[:-1], strict=True):
        keep = baseline["subset"].to_numpy() == subset
        labels = baseline.loc[keep, "label"].to_numpy(dtype=int)
        chroms = baseline.loc[keep, "chrom"].to_numpy()
        baseline_arm = _chrom_components(
            labels,
            baseline.loc[keep, "probe_score"].to_numpy(dtype=float),
            chroms,
        )
        extension_arm = _chrom_components(
            labels,
            extension.loc[keep, "probe_score"].to_numpy(dtype=float),
            chroms,
        )
        point, bootstrap, n_chroms = _paired_chrom_bootstrap(
            baseline_arm,
            extension_arm,
            n_bootstrap=n_bootstrap,
            generator=np.random.default_rng(child_seed),
        )
        rows.append(
            {
                "protocol": "probe",
                "subset": subset,
                **_summary(
                    point,
                    bootstrap,
                    n_clusters=n_chroms,
                    n_rows=len(labels),
                ),
            }
        )
        subset_components[subset] = (baseline_arm, extension_arm)

    union = sorted(
        {
            chrom
            for components in subset_components.values()
            for arm in components
            for chrom in arm
        }
    )
    point = float(np.mean([row["delta_extension_minus_baseline"] for row in rows]))
    generator = np.random.default_rng(child_seeds[-1])
    bootstrap = np.empty(n_bootstrap, dtype=float)
    for iteration in range(n_bootstrap):
        sampled = generator.choice(union, size=len(union), replace=True)
        multiplicities = dict(zip(*np.unique(sampled, return_counts=True), strict=True))
        deltas = [
            _weighted_component(extension_arm, multiplicities)
            - _weighted_component(baseline_arm, multiplicities)
            for baseline_arm, extension_arm in subset_components.values()
        ]
        bootstrap[iteration] = (
            float(np.mean(deltas)) if np.isfinite(deltas).all() else np.nan
        )
    rows.insert(
        0,
        {
            "protocol": "probe",
            "subset": MACRO,
            **_summary(
                point,
                bootstrap,
                n_clusters=len(union),
                n_rows=int(baseline["probe_score"].notna().sum()),
            ),
        },
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-context", type=int, required=True)
    parser.add_argument("--extension-context", type=int, required=True)
    parser.add_argument("--zero-shot-baseline", type=Path, required=True)
    parser.add_argument("--zero-shot-extension", type=Path, required=True)
    parser.add_argument("--probe-baseline", type=Path, required=True)
    parser.add_argument("--probe-extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=485)
    args = parser.parse_args()
    if args.n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be positive")
    if args.extension_context <= args.baseline_context:
        raise ValueError("--extension-context must exceed --baseline-context")

    rows = _zero_shot_deltas(
        args.zero_shot_baseline,
        args.zero_shot_extension,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    rows.extend(
        _probe_deltas(
            args.probe_baseline,
            args.probe_extension,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + 1,
        )
    )
    output = pd.DataFrame(rows)
    output.insert(2, "baseline_context_bp", args.baseline_context)
    output.insert(3, "extension_context_bp", args.extension_context)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
