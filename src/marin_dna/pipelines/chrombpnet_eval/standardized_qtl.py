"""Evaluate precomputed ChromBPNet / Enformer baselines on the **standardized**
caQTL/dsQTL benchmarks (ChromBPNet paper, Synapse ``syn64126763``).

These are the canonical benchmark sets used by *both* the ChromBPNet and
AlphaGenome papers. Each TSV carries precomputed per-variant scores — ChromBPNet
GM12878 DNase (``encsr000emt``) + ATAC (``encsr637xsc``) ``logfc``/``ips``, and
Enformer's recomputed local-2kb ``local_logfc`` — so we reproduce the papers'
reported baseline numbers **without running any model**.

The standard protocol (issue #262): **causality** = auPRC on ``|score|`` over all
``var.isused`` variants; **direction** = signed Pearson of the score vs the study
``effect`` over positives only. Both are computed by
:func:`marin_dna.pipelines.chrombpnet_eval.metrics.compute_supervised_qtl_metrics`.

No liftover / genome is needed here: the precomputed scores already live on each
file's native build (caQTL hg38, dsQTL hg19) and the metrics only classify/correlate.
(Liftover to hg38 is only needed downstream, to score these variants with hg38-only
models like AlphaGenome.)

Schema quirks handled below: the per-score column **infix** differs between files
(``variantscore`` for caQTL, ``varscore`` for dsQTL); both use ``obs.label`` (bool
significance), ``var.isused`` (benchmark membership), and a per-file effect column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from marin_dna.pipelines.chrombpnet_eval.metrics import compute_supervised_qtl_metrics

# ENCODE experiment id -> human-readable GM12878 assay label.
ENCODE_ASSAY: dict[str, str] = {
    "encsr000emt": "GM12878 DNase",
    "encsr637xsc": "GM12878 ATAC",
}


@dataclass(frozen=True)
class StandardizedQTL:
    """One standardized benchmark file (Synapse ``syn64126763`` children)."""

    name: str
    synapse_id: str
    build: str  # "hg38" | "hg19"
    chrom_col: str
    pos_col: str
    effect_col: str
    score_infix: str  # "variantscore" (caqtl) | "varscore" (dsqtl)


STANDARDIZED_QTL: dict[str, StandardizedQTL] = {
    "caqtl": StandardizedQTL(
        name="caqtl",
        synapse_id="syn64126781",
        build="hg38",
        chrom_col="var.chr",
        pos_col="var.pos_hg38",
        effect_col="obs.beta",
        score_infix="variantscore",
    ),
    "dsqtl": StandardizedQTL(
        name="dsqtl",
        synapse_id="syn64126779",
        build="hg19",
        chrom_col="var.chr",
        pos_col="var.pos_hg19",
        effect_col="obs.estimate",
        score_infix="varscore",
    ),
}


@dataclass(frozen=True)
class Baseline:
    """A precomputed baseline = (model, assay) with its protocol score columns.

    ``causality_col`` feeds the auPRC (on ``|score|``); ``direction_col`` feeds the
    signed Pearson over positives. ChromBPNet uses **IPS** for causality and
    **logFC** for direction (the ChromBPNet paper's recommended scores); Enformer
    uses its local-2kb ``local_logfc`` for both.
    """

    model: str
    assay: str
    causality_col: str
    direction_col: str


def baselines_for(df: pd.DataFrame) -> list[Baseline]:
    """The baselines whose score columns are present in ``df`` (canonical names
    from :func:`load_standardized_qtl`)."""
    candidates = [
        Baseline(
            "ChromBPNet", "GM12878 ATAC", "chrombpnet_atac_ips", "chrombpnet_atac_logfc"
        ),
        Baseline(
            "ChromBPNet",
            "GM12878 DNase",
            "chrombpnet_dnase_ips",
            "chrombpnet_dnase_logfc",
        ),
        Baseline(
            "Enformer",
            "GM12878 DNase",
            "enformer_dnase_local_logfc",
            "enformer_dnase_local_logfc",
        ),
    ]
    return [
        b for b in candidates if {b.causality_col, b.direction_col} <= set(df.columns)
    ]


def _truthy(s: pd.Series) -> np.ndarray:
    """Boolean array from a flag column stored as bool, 0/1, or ``"True"``/``"1"``
    (used for ``var.isused``)."""
    if s.dtype == bool:
        return s.to_numpy()
    if np.issubdtype(s.dtype, np.number):
        return s.to_numpy() != 0
    return s.astype(str).str.lower().isin(["1", "true", "t", "yes"]).to_numpy()


def _positive_label(s: pd.Series) -> np.ndarray:
    """Positive (= significant QTL) mask from ``obs.label``.

    The two standardized files encode the label differently: caQTL uses a boolean
    (``True`` = significant), dsQTL uses an integer **``1`` = significant, ``-1`` =
    control**. So a plain truthiness test mislabels dsQTL's ``-1`` controls as
    positive — map numeric labels with ``== 1`` instead (``-1``/``0`` → negative).
    """
    if s.dtype == bool:
        return s.to_numpy()
    if np.issubdtype(s.dtype, np.number):
        vals = set(np.unique(s.to_numpy()).tolist())
        assert vals <= {-1, 0, 1}, f"unexpected obs.label values {sorted(vals)}"
        return s.to_numpy() == 1
    return s.astype(str).str.lower().isin(["1", "true", "t", "yes"]).to_numpy()


def load_standardized_qtl(path: str, spec: StandardizedQTL) -> pd.DataFrame:
    """Parse a standardized benchmark TSV into a metric-ready frame.

    Restricts to ``var.isused`` (the benchmark set: positives −log10p>5 ∪ controls
    −log10p<3, dropping the ambiguous middle), and projects onto
    ``[chrom, pos, allele1, allele2, label, effect, <score columns>]``. Score
    columns are renamed to canonical, file-independent names
    (``chrombpnet_{atac,dnase}_{ips,logfc}``, ``enformer_dnase_local_logfc``); any
    not present in this file are simply omitted.

    ``label`` is the boolean ``obs.label`` (True = significant QTL); ``effect`` is
    the signed study effect (caQTL ``obs.beta`` / dsQTL ``obs.estimate``). The
    benchmark logFC correlates *positively* with ``effect`` on both files, so no
    sign flip is applied here.
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    for col in (
        "var.isused",
        "obs.label",
        spec.chrom_col,
        spec.pos_col,
        spec.effect_col,
    ):
        assert col in df.columns, (
            f"{spec.name}: standardized TSV missing column {col!r}"
        )
    df = df[_truthy(df["var.isused"])].copy()
    assert len(df) > 0, f"{spec.name}: no var.isused variants"

    out = pd.DataFrame(
        {
            "chrom": df[spec.chrom_col]
            .astype(str)
            .str.replace(r"^chr", "", regex=True),
            "pos": df[spec.pos_col].astype("int64"),
            "allele1": df["var.allele1"].astype(str).str.upper(),
            "allele2": df["var.allele2"].astype(str).str.upper(),
            "label": _positive_label(df["obs.label"]),
            "effect": df[spec.effect_col].astype(float).to_numpy(),
        }
    )
    inf = spec.score_infix
    score_map = {
        "chrombpnet_atac_ips": f"pred.chrombpnet.encsr637xsc.{inf}.ips",
        "chrombpnet_atac_logfc": f"pred.chrombpnet.encsr637xsc.{inf}.logfc",
        "chrombpnet_dnase_ips": f"pred.chrombpnet.encsr000emt.{inf}.ips",
        "chrombpnet_dnase_logfc": f"pred.chrombpnet.encsr000emt.{inf}.logfc",
        "enformer_dnase_local_logfc": f"pred.enformer.encsr000emt.{inf}.local_logfc",
    }
    for canon, src in score_map.items():
        if src in df.columns:
            out[canon] = df[src].astype(float).to_numpy()
    # Every positive must carry a measured effect (controls may not) — the metric
    # asserts this too, but fail here with a clearer message.
    n_nan_pos = int(np.isnan(out.loc[out["label"], "effect"].to_numpy()).sum())
    assert n_nan_pos == 0, f"{spec.name}: {n_nan_pos} positives with NaN effect"
    return out


def evaluate_score_columns(
    df: pd.DataFrame,
    score_columns: list[str],
    *,
    n_bootstrap: int = 1000,
    rng: int = 0,
) -> pd.DataFrame:
    """Run the supervised QTL metric set for each score column on ``df``.

    For every column: drops rows with a non-finite score (reporting ``coverage``),
    then computes ``compute_supervised_qtl_metrics`` (AUROC/AUPRC on ``|score|``
    over all rows; signed Pearson/Spearman vs ``effect`` over positives). Returns a
    long frame ``[score_column, metric, value, se, n_rows, n_pos, coverage]``.
    """
    assert {"label", "effect"} <= set(df.columns), "df needs label + effect columns"
    parts: list[pd.DataFrame] = []
    for col in score_columns:
        assert col in df.columns, f"missing score column {col!r}"
        sub = df[["label", "effect", col]].rename(columns={col: "score"})
        n_before = len(sub)
        sub = sub[np.isfinite(sub["score"].to_numpy())].copy()
        m = compute_supervised_qtl_metrics(
            sub, score_col="score", n_bootstrap=n_bootstrap, rng=rng
        )
        m.insert(0, "score_column", col)
        m["coverage"] = round(len(sub) / n_before, 4)
        parts.append(m)
    return pd.concat(parts, ignore_index=True)
