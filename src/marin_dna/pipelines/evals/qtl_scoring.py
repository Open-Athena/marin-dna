"""Model-agnostic supervised accessibility-QTL benchmark (issue #311).

The **per-variant score parquet is the plug-in interface**: every scorer emits
``[chrom, pos, ref, alt, causality_score, direction_score]`` and the shared metric
treats them uniformly — **causality** = auPRC (+AUROC) on ``|causality_score|`` over
all variants in a split; **direction** = signed Pearson (+Spearman) of
``direction_score`` vs the study ``effect`` over the positives. Single-signal models
(AlphaGenome, Enformer, a fine-tuned gLM) write the same value into both columns;
ChromBPNet writes ATAC **IPS** for causality and **logFC** for direction (the
ChromBPNet-paper recommended scores).

Adding a model = drop one ``scores/{model}/{dataset}.parquet`` — no metric-code change
(:func:`compute_qtl_split_metrics` takes ``model`` as an opaque label).

Splits are a metrics-time slice (not a re-score): ``all`` / our ``train`` (odd chroms)
/ ``test`` (even chroms) — the production splits — plus ``ag_test`` (AlphaGenome's test
chromosomes), used only to **reproduce** AlphaGenome Suppl Table 4.

No per-dataset sign-flip lives in the metric. AlphaGenome's raw DNase-LFC sits in a
per-dataset accessibility *convention* that is uniformly flipped on dsQTL relative to
the study (a lift-related #262 artifact — uniform across orientation swaps, not a
per-variant orientation issue). It is aligned to the dataset's convention **once**
upstream (:func:`align_score_sign` anchored on the carried ChromBPNet logFC →
``scripts/correct_ag_predictions.py``), so by the time scores reach the metric every
model is already a positively-oriented accessibility score, exactly like the carried
ChromBPNet/Enformer columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import pearsonr

from marin_dna.pipelines.chrombpnet_eval.metrics import compute_supervised_qtl_metrics

# Per-model protocol columns: (causality_col, direction_col). The carried column names
# are the canonical names produced by ``chrombpnet_qtl.load_standardized_qtl``;
# ``alphagenome_dnase_lfc`` is the column written by
# ``alphagenome.score_dnase_lfc_resumable``. ChromBPNet uses IPS for causality and
# logFC for direction; every other model is single-signal (same column twice).
MODEL_SCORE_COLUMNS: dict[str, tuple[str, str]] = {
    "alphagenome": ("alphagenome_dnase_lfc", "alphagenome_dnase_lfc"),
    "chrombpnet": ("chrombpnet_atac_ips", "chrombpnet_atac_logfc"),
    "enformer": ("enformer_dnase_local_logfc", "enformer_dnase_local_logfc"),
}

# AlphaGenome's test chromosomes (Suppl Table 4 is reported on this slice). Used only
# to reproduce the published numbers — production reporting uses our train/test splits.
AG_TEST_CHROMS: frozenset[str] = frozenset(
    {"3", "6", "9", "12", "16", "18", "19", "21"}
)

# The named splits the benchmark reports, each a row-mask over the scored variants.
QTL_SPLITS: tuple[str, ...] = ("all", "train", "test", "ag_test")

# AlphaGenome Suppl Table 4 (published): (causality auPRC, direction Pearson) on the
# AG-test slice. Borzoi-ensemble is the one model we have no per-variant scores for, so
# it is a pure static reference; ChromBPNet/AlphaGenome are kept as published targets to
# show our reproduced numbers against. Random-baseline auPRC anchors are separate.
SUPPL_TABLE4_REFERENCE: dict[str, dict[str, tuple[float, float]]] = {
    "caqtl": {
        "ChromBPNet": (0.4148, 0.6737),
        "Borzoi-ensemble": (0.4624, 0.6485),
        "AlphaGenome": (0.5643, 0.7368),
    },
    "dsqtl": {
        "ChromBPNet": (0.5432, 0.7722),
        "Borzoi-ensemble": (0.6056, 0.7922),
        "AlphaGenome": (0.6308, 0.8323),
    },
}

# AlphaGenome's reported random-baseline causality auPRC (= positive rate); direction
# Pearson for a random scorer is 0. Anchors for the dashboard (#312).
RANDOM_BASELINE_AUPRC: dict[str, float] = {"caqtl": 0.0852, "dsqtl": 0.0200}

# Canonical caQTL/dsQTL HF dataset revisions (#313 build) — the single source for the
# benchmark scripts (correct_ag_predictions.py / qtl_benchmark.py). Keep in sync with
# snakemake/alphagenome_eval/config/config.yaml (snakemake reads YAML and can't import
# this).
QTL_HF_REVISION: dict[str, str] = {
    "caqtl": "27a24296f50ed55afdc412d1612df680d13138d6",
    "dsqtl": "4a3bf152cd7c28be290adde48a402ec40992cb62",
}

# Dashboard benchmark model registry (#312): maps each scored model's key (its
# ``metrics/{model}/…`` S3 prefix, also ``compute_qtl_split_metrics``'s ``model``
# label) to a display name + a coarse group. The group separates the external
# supervised baselines from our (future) fine-tuned gLMs (#243) — the comparison the
# page exists to make — and drives the row swatch. A model discovered on S3 without an
# entry here falls back to its raw key + ``"other"`` (opaque passthrough).
QTL_BENCHMARK_MODELS: dict[str, dict[str, str]] = {
    "alphagenome": {"display": "AlphaGenome", "group": "supervised"},
    "chrombpnet": {"display": "ChromBPNet", "group": "supervised"},
    "enformer": {"display": "Enformer", "group": "supervised"},
}

# The two accessibility-QTL assays the dashboard's "macro" scope averages over.
QTL_BENCHMARK_DATASETS: tuple[str, ...] = ("caqtl", "dsqtl")

_KEY: tuple[str, ...] = ("chrom", "pos", "ref", "alt")


def to_score_interface(
    df: pl.DataFrame, causality_col: str, direction_col: str
) -> pl.DataFrame:
    """Project a frame onto the uniform score interface.

    Returns ``[chrom, pos, ref, alt, causality_score, direction_score]``. For a
    single-signal model pass the same column name twice. ``df`` must carry the key
    columns and both score columns.
    """
    for col in (*_KEY, causality_col, direction_col):
        assert col in df.columns, f"frame missing column {col!r}"
    return df.select(
        [
            *_KEY,
            pl.col(causality_col).alias("causality_score"),
            pl.col(direction_col).alias("direction_score"),
        ]
    )


def align_score_sign(
    scores: np.ndarray, reference: np.ndarray, *, min_abs_corr: float = 0.5
) -> tuple[np.ndarray, float]:
    """Flip ``scores`` to agree in sign with ``reference``; return ``(aligned, sign)``.

    Used to put an externally-computed accessibility score that lives in a per-dataset
    flipped convention (the #262 AlphaGenome DNase-LFC: uniformly anti-correlated with
    the study on dsQTL) into the dataset's accessibility convention, anchored on a
    carried baseline that is already study-aligned (ChromBPNet logFC). Two accessibility
    predictors of the same allelic effect should correlate strongly; the **sign** of
    that correlation is the convention. Asserts ``|corr| >= min_abs_corr`` so a
    near-zero (ambiguous) correlation fails loud rather than silently guessing a sign.

    ``sign`` is ``+1.0`` (kept) or ``-1.0`` (flipped). Non-finite pairs are dropped for
    the correlation; ``aligned`` is ``scores * sign`` (NaNs preserved).
    """
    scores = np.asarray(scores, dtype=float)
    reference = np.asarray(reference, dtype=float)
    assert scores.shape == reference.shape, "scores/reference length mismatch"
    finite = np.isfinite(scores) & np.isfinite(reference)
    assert finite.sum() >= 2, "need >=2 finite pairs to align sign"
    r = float(pearsonr(scores[finite], reference[finite])[0])
    assert abs(r) >= min_abs_corr, (
        f"|corr|={abs(r):.3f} < {min_abs_corr} — sign alignment is ambiguous; the "
        "anchor (carried ChromBPNet logFC) should strongly track the accessibility score"
    )
    sign = 1.0 if r >= 0 else -1.0
    return scores * sign, sign


def qtl_split_masks(
    chrom: np.ndarray, split_source: np.ndarray
) -> dict[str, np.ndarray]:
    """Row-masks for the named splits.

    ``chrom`` = per-variant chromosome (unprefixed str); ``split_source`` = each
    variant's home split in the dataset (``"train"`` / ``"test"``, the odd/even chrom
    partition). ``all`` = every row; ``train``/``test`` = membership; ``ag_test`` =
    chrom ∈ :data:`AG_TEST_CHROMS` (an independent slice that spans both).
    """
    chrom = np.asarray(chrom).astype(str)
    split_source = np.asarray(split_source).astype(str)
    extra = set(np.unique(split_source).tolist()) - {"train", "test"}
    assert not extra, f"unexpected split_source values {sorted(extra)}"
    return {
        "all": np.ones(len(chrom), dtype=bool),
        "train": split_source == "train",
        "test": split_source == "test",
        "ag_test": np.isin(chrom, list(AG_TEST_CHROMS)),
    }


def _pick(metrics: pd.DataFrame, metric: str) -> tuple[float, float]:
    """``(value, se)`` for one metric row from ``compute_supervised_qtl_metrics``."""
    row = metrics.loc[metrics["metric"] == metric]
    assert len(row) == 1, f"expected exactly one {metric!r} row, got {len(row)}"
    return float(row["value"].iloc[0]), float(row["se"].iloc[0])


def compute_qtl_split_metrics(
    scores_df: pl.DataFrame,
    dataset_df: pl.DataFrame,
    *,
    dataset: str,
    model: str,
    splits: tuple[str, ...] = QTL_SPLITS,
    n_bootstrap: int = 1000,
    rng: int = 0,
) -> pd.DataFrame:
    """Official caQTL/dsQTL metrics for one model's scores, per split.

    Joins ``scores_df`` (``[chrom,pos,ref,alt,causality_score,direction_score]``) to
    ``dataset_df`` (must carry ``label``, ``effect``, ``split_source``) on the variant
    key, then for each split reuses :func:`compute_supervised_qtl_metrics` — causality
    metrics (AUROC/AUPRC on ``|causality_score|``) from the causality column, direction
    metrics (Pearson/Spearman on ``direction_score`` over positives) from the direction
    column (a single call when the two columns are identical, i.e. single-signal
    models). ``model`` is an opaque label — **any** model flows through unchanged.

    Returns a wide row per split: ``[dataset, model, split, n_rows, n_pos, coverage,
    causality_auPRC, causality_se, causality_AUROC, causality_AUROC_se,
    direction_pearson, direction_pearson_se, direction_spearman,
    direction_spearman_se]``.
    """
    key = list(_KEY)
    for col in ("causality_score", "direction_score"):
        assert col in scores_df.columns, f"scores_df missing {col!r}"
    for col in ("label", "effect", "split_source"):
        assert col in dataset_df.columns, f"dataset_df missing {col!r}"

    joined = scores_df.join(
        dataset_df.select([*key, "label", "effect", "split_source"]),
        on=key,
        how="inner",
    )
    assert joined.height == scores_df.height, (
        f"{joined.height}/{scores_df.height} scored variants matched the dataset — every "
        "scored variant must be a known dataset variant (genome-orientation mismatch, or "
        f"a (chrom,pos,ref,alt) key-dtype skew between the two frames: "
        f"{[scores_df.schema[c] for c in key]} vs {[dataset_df.schema[c] for c in key]})"
    )
    coverage = round(joined.height / dataset_df.height, 4)
    for col in ("causality_score", "direction_score"):
        assert joined.get_column(col).is_finite().all(), (
            f"non-finite {col!r} — fill/align upstream before scoring"
        )

    single_signal = bool(
        (
            joined.get_column("causality_score") == joined.get_column("direction_score")
        ).all()
    )
    masks = qtl_split_masks(
        joined.get_column("chrom").to_numpy(),
        joined.get_column("split_source").to_numpy(),
    )

    rows: list[dict] = []
    for split in splits:
        assert split in masks, f"unknown split {split!r}"
        sub = joined.filter(pl.Series(masks[split])).to_pandas()
        causality = compute_supervised_qtl_metrics(
            sub,
            score_col="causality_score",
            label_col="label",
            effect_col="effect",
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        direction = (
            causality
            if single_signal
            else compute_supervised_qtl_metrics(
                sub,
                score_col="direction_score",
                label_col="label",
                effect_col="effect",
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
        )
        auprc, auprc_se = _pick(causality, "AUPRC")
        auroc, auroc_se = _pick(causality, "AUROC")
        pearson, pearson_se = _pick(direction, "pearson")
        spearman, spearman_se = _pick(direction, "spearman")
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "split": split,
                "n_rows": int(len(sub)),
                "n_pos": int(sub["label"].sum()),
                "coverage": coverage,
                "causality_auPRC": auprc,
                "causality_se": auprc_se,
                "causality_AUROC": auroc,
                "causality_AUROC_se": auroc_se,
                "direction_pearson": pearson,
                "direction_pearson_se": pearson_se,
                "direction_spearman": spearman,
                "direction_spearman_se": spearman_se,
            }
        )
    return pd.DataFrame(rows)


def reference_metrics(dataset: str) -> pd.DataFrame:
    """Static published reference rows (AlphaGenome Suppl Table 4), ``split="ag_test"``.

    Borzoi-ensemble (no per-variant scores) is the essential static row; ChromBPNet and
    AlphaGenome are kept as published targets to compare our reproduced numbers against.
    A ``Random`` row carries the random-baseline auPRC anchor (direction Pearson 0).
    The ``source`` column distinguishes these from our computed metrics.
    """
    rows: list[dict] = []
    for model, (auprc, pearson) in SUPPL_TABLE4_REFERENCE[dataset].items():
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "split": "ag_test",
                "causality_auPRC": auprc,
                "direction_pearson": pearson,
                "source": "AlphaGenome Suppl Table 4 (published)",
            }
        )
    rows.append(
        {
            "dataset": dataset,
            "model": "Random",
            "split": "ag_test",
            "causality_auPRC": RANDOM_BASELINE_AUPRC[dataset],
            "direction_pearson": 0.0,
            "source": "random baseline",
        }
    )
    return pd.DataFrame(rows)


# The metric columns the dashboard renders (value + SE), a subset of
# compute_qtl_split_metrics' output (AUROC/Spearman stay in the per-model parquets but
# aren't surfaced on the page).
_DASH_METRIC_COLS: tuple[str, ...] = (
    "causality_auPRC",
    "causality_se",
    "direction_pearson",
    "direction_pearson_se",
)
_DASH_COUNT_COLS: tuple[str, ...] = ("n_rows", "n_pos")


def macro_average_metrics(
    metrics: pl.DataFrame, datasets: tuple[str, ...] = QTL_BENCHMARK_DATASETS
) -> pl.DataFrame:
    """Equal-weight macro-average of each model's per-assay metrics → one row per (model, split).

    The dashboard's "macro" scope (#312): for each ``(model, split)`` the metric value is
    the unweighted mean over ``datasets`` and the SE is ``sqrt(Σ se_i²) / k`` — the SE of
    a mean of ``k`` *independent* estimates, which these are (caQTL and dsQTL are disjoint
    variant sets; contrast the conservative independent-*sum* used for matched-pair model
    deltas). ``n_rows`` / ``n_pos`` are summed (pooled counts, for the tooltip). Asserts
    every ``(model, split)`` group carries exactly ``datasets`` so a missing assay fails
    loud rather than silently averaging one. Returns the standard per-(model, split) metric
    columns plus ``dataset="macro"``.
    """
    k = len(datasets)
    assert k >= 1, "need at least one dataset to average"
    sub = metrics.filter(pl.col("dataset").is_in(datasets))
    # One pass: aggregate the metrics and the per-group dataset count together, then
    # assert completeness before returning (an incomplete group's wrong mean never escapes).
    macro = sub.group_by("model", "split").agg(
        pl.col("dataset").n_unique().alias("_nd"),
        pl.col("causality_auPRC").mean().alias("causality_auPRC"),
        (pl.col("causality_se").pow(2).sum().sqrt() / k).alias("causality_se"),
        pl.col("direction_pearson").mean().alias("direction_pearson"),
        (pl.col("direction_pearson_se").pow(2).sum().sqrt() / k).alias(
            "direction_pearson_se"
        ),
        pl.col("n_rows").sum().alias("n_rows"),
        pl.col("n_pos").sum().alias("n_pos"),
    )
    incomplete = macro.filter(pl.col("_nd") != k)
    assert incomplete.height == 0, (
        f"macro-average needs all of {sorted(datasets)} per (model, split); incomplete "
        f"groups: {incomplete.select('model', 'split').rows()}"
    )
    return macro.drop("_nd").with_columns(pl.lit("macro").alias("dataset"))


def assemble_benchmark_rows(metrics: pl.DataFrame) -> pl.DataFrame:
    """Tidy long-form rows for the Accessibility QTL dashboard page (#312).

    ``metrics`` is the concatenation of the per-model ``metrics/{model}/{dataset}.parquet``
    frames (:func:`compute_qtl_split_metrics` output), already filtered to the split(s) the
    page shows (the dashboard passes ``split == "train"`` only). Appends a ``macro`` scope
    (:func:`macro_average_metrics`) to the per-assay rows, attaches each model's display
    name + group from :data:`QTL_BENCHMARK_MODELS` (unknown models fall back to their key +
    ``"other"`` — opaque passthrough, so a newly-dropped model still appears), and returns
    ``[model, display, group, scope, split, *metric cols, n_rows, n_pos]`` with
    ``scope ∈ {caqtl, dsqtl, macro}``.
    """
    needed = ("dataset", "model", "split", *_DASH_METRIC_COLS, *_DASH_COUNT_COLS)
    for col in needed:
        assert col in metrics.columns, f"metrics frame missing column {col!r}"
    per_assay = metrics.select(needed)
    macro = macro_average_metrics(metrics).select(needed)
    combined = pl.concat([per_assay, macro], how="vertical").rename(
        {"dataset": "scope"}
    )

    registry = pl.DataFrame(
        {
            "model": list(QTL_BENCHMARK_MODELS),
            "display": [m["display"] for m in QTL_BENCHMARK_MODELS.values()],
            "group": [m["group"] for m in QTL_BENCHMARK_MODELS.values()],
        }
    )
    combined = combined.join(registry, on="model", how="left").with_columns(
        pl.col("display").fill_null(pl.col("model")),
        pl.col("group").fill_null(pl.lit("other")),
    )
    return combined.select(
        "model",
        "display",
        "group",
        "scope",
        "split",
        *_DASH_METRIC_COLS,
        *_DASH_COUNT_COLS,
    )
