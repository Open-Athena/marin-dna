"""Registered AUPRC and paired-bootstrap summaries for exp479 VEP."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

GLOBAL = "_global_"
MACRO = "_macro_avg_"
SGE_POOLED = "both"


def _cluster_ap(
    labels: Iterable[bool | int],
    scores: Iterable[float],
    groups: Iterable[object],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | int]:
    y = np.asarray(list(labels), dtype=int)
    score = np.asarray(list(scores), dtype=float)
    group = np.asarray(list(groups))
    if not (len(y) == len(score) == len(group)):
        raise ValueError("labels, scores, and groups must have equal lengths")
    if not np.isfinite(score).all():
        raise ValueError("VEP scores must be finite")
    if not 0 < y.sum() < len(y):
        raise ValueError("AUPRC requires both classes")
    group_rows = list(pd.Series(group).groupby(group).indices.values())
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled = rng.integers(0, len(group_rows), size=len(group_rows))
        rows = np.concatenate([group_rows[value] for value in sampled])
        sampled_y = y[rows]
        bootstrap[index] = (
            average_precision_score(sampled_y, score[rows])
            if 0 < sampled_y.sum() < len(sampled_y)
            else np.nan
        )
    return {
        "value": float(average_precision_score(y, score)),
        "se": float(np.nanstd(bootstrap, ddof=1)),
        "n_groups": len(group_rows),
        "n_rows": len(y),
    }


def matched_metrics(
    variants: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    n_bootstrap: int = 1_000,
    seed: int = 0,
    n_min_groups: int = 30,
) -> pd.DataFrame:
    """Match evals_v2 per-subset, global, and macro AUPRC semantics."""

    required = {"label", "subset", "match_group"}
    if not required.issubset(variants.columns):
        raise ValueError(f"matched VEP frame lacks {sorted(required - set(variants.columns))}")
    if len(variants) != len(scores):
        raise ValueError("variant and score rows differ")
    frame = pd.concat([variants.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    if (frame.groupby("match_group")["subset"].nunique() > 1).any():
        raise ValueError("a match_group spans consequence subsets")

    rows: list[dict[str, object]] = []
    for score_name in scores.columns:
        subset_rows: list[dict[str, object]] = []
        for subset, cell in frame.groupby("subset", sort=False):
            result = _cluster_ap(
                cell["label"],
                cell[score_name],
                cell["match_group"],
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            row: dict[str, object] = {
                "score_type": score_name,
                "subset": str(subset),
                **result,
            }
            rows.append(row)
            subset_rows.append(row)

        global_result = _cluster_ap(
            frame["label"],
            frame[score_name],
            frame["match_group"],
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        rows.append({"score_type": score_name, "subset": GLOBAL, **global_result})
        qualifying = [row for row in subset_rows if int(row["n_groups"]) >= n_min_groups]
        if not qualifying:
            raise ValueError(f"no subsets qualify for {score_name} macro AUPRC")
        count = len(qualifying)
        rows.append(
            {
                "score_type": score_name,
                "subset": MACRO,
                "value": float(sum(float(row["value"]) for row in qualifying) / count),
                "se": float(math.sqrt(sum(float(row["se"]) ** 2 for row in qualifying)) / count),
                "n_groups": count,
                "n_rows": sum(int(row["n_rows"]) for row in qualifying),
            }
        )
    return pd.DataFrame(rows)


def _macro(children: list[dict[str, float | int]]) -> dict[str, float | int] | None:
    if not children:
        return None
    count = len(children)
    return {
        "value": float(sum(float(child["value"]) for child in children) / count),
        "se": float(math.sqrt(sum(float(child["se"]) ** 2 for child in children)) / count),
        "n": count,
        "n_pos": sum(int(child["n_pos"]) for child in children),
    }


def _sge_cell(
    cell: pd.DataFrame,
    score_name: str,
    *,
    n_bootstrap: int,
    seed: int,
    n_min_class: int,
) -> dict[str, float | int] | None:
    labels = cell["label"].astype(bool)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos < n_min_class or n_neg < n_min_class:
        return None
    result = _cluster_ap(
        labels,
        cell[score_name],
        np.arange(len(cell)),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return {
        "value": float(result["value"]),
        "se": float(result["se"]),
        "n": len(cell),
        "n_pos": n_pos,
    }


def sge_metrics(
    variants: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    n_bootstrap: int = 1_000,
    seed: int = 0,
    n_min_class: int = 30,
) -> pd.DataFrame:
    """Match evals_v2 accession- and consequence-aware SGE AUPRC."""

    required = {"mavedb_urn", "gene", "subset", "label"}
    if not required.issubset(variants.columns):
        raise ValueError(f"SGE frame lacks {sorted(required - set(variants.columns))}")
    if variants[list(required)].isna().any().any():
        raise ValueError("SGE grouping and label columns cannot be null")
    if len(variants) != len(scores):
        raise ValueError("variant and score rows differ")
    frame = pd.concat([variants.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    if (frame.groupby("mavedb_urn")["gene"].nunique() != 1).any():
        raise ValueError("an SGE accession maps to more than one gene")
    subsets = sorted(str(value) for value in frame["subset"].unique())
    if SGE_POOLED in subsets or MACRO in subsets:
        raise ValueError("SGE subset collides with a reserved aggregate name")
    genes = frame.groupby("mavedb_urn")["gene"].first().to_dict()

    rows: list[dict[str, object]] = []
    for score_name in scores.columns:
        cells: dict[tuple[str, str], dict[str, float | int]] = {}
        gated: list[tuple[str, str, int, int]] = []
        for accession, accession_frame in frame.groupby("mavedb_urn", sort=False):
            accession = str(accession)
            scopes = {
                subset: accession_frame[accession_frame["subset"] == subset] for subset in subsets
            }
            scopes[SGE_POOLED] = accession_frame
            for scope, cell in scopes.items():
                result = _sge_cell(
                    cell,
                    score_name,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                    n_min_class=n_min_class,
                )
                if result is None:
                    n_pos = int(cell["label"].astype(bool).sum())
                    if len(cell):
                        gated.append((accession, scope, len(cell), n_pos))
                else:
                    cells[(accession, scope)] = result
            macro = _macro(
                [cells[(accession, subset)] for subset in subsets if (accession, subset) in cells]
            )
            if macro is not None:
                cells[(accession, MACRO)] = macro

        for (accession, scope), result in cells.items():
            rows.append(
                {
                    "metric": "AUPRC",
                    "subset": scope,
                    "accession": accession,
                    "gene": genes[accession],
                    "score_type": score_name,
                    **result,
                }
            )
        for accession, scope, n_rows, n_pos in gated:
            rows.append(
                {
                    "metric": "AUPRC",
                    "subset": scope,
                    "accession": accession,
                    "gene": genes[accession],
                    "score_type": score_name,
                    "value": float("nan"),
                    "se": float("nan"),
                    "n": n_rows,
                    "n_pos": n_pos,
                }
            )
        accessions = [str(value) for value in frame["mavedb_urn"].unique()]
        for scope in [*subsets, SGE_POOLED, MACRO]:
            macro = _macro(
                [
                    cells[(accession, scope)]
                    for accession in accessions
                    if (accession, scope) in cells
                ]
            )
            if macro is not None:
                rows.append(
                    {
                        "metric": "AUPRC",
                        "subset": scope,
                        "accession": MACRO,
                        "gene": MACRO,
                        "score_type": score_name,
                        **macro,
                    }
                )
    return pd.DataFrame(rows)


def paired_ap_delta(
    labels: Iterable[bool | int],
    candidate: Iterable[float],
    baseline: Iterable[float],
    groups: Iterable[object],
    *,
    n_bootstrap: int = 1_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Paired AUPRC difference with a shared natural-unit bootstrap."""

    y = np.asarray(list(labels), dtype=int)
    first = np.asarray(list(candidate), dtype=float)
    second = np.asarray(list(baseline), dtype=float)
    group = np.asarray(list(groups))
    if not (len(y) == len(first) == len(second) == len(group)):
        raise ValueError("paired VEP inputs have different lengths")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("paired VEP scores must be finite")
    group_rows = list(pd.Series(group).groupby(group).indices.values())
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled = rng.integers(0, len(group_rows), size=len(group_rows))
        rows = np.concatenate([group_rows[value] for value in sampled])
        sampled_y = y[rows]
        bootstrap[index] = (
            average_precision_score(sampled_y, first[rows])
            - average_precision_score(sampled_y, second[rows])
            if 0 < sampled_y.sum() < len(sampled_y)
            else np.nan
        )
    point = float(average_precision_score(y, first) - average_precision_score(y, second))
    low, high = np.nanpercentile(bootstrap, [2.5, 97.5])
    return {
        "delta": point,
        "se": float(np.nanstd(bootstrap, ddof=1)),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_groups": len(group_rows),
        "n_rows": len(y),
    }
