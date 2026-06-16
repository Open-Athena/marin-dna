"""Saturation genome editing (SGE) variant-effect dataset construction (issue #289).

Builds the ``bolinas-dna/evals_sge`` HF artifact: per-SNV experimental function
scores from endogenous saturation-genome-editing assays, normalized to GRCh38 with
the standard consequence/distance annotation.

How it differs from the other eval datasets:

- vs the matched-pair datasets (``mendelian_traits`` / ``complex_traits``): **no
  matching, no subsetting** — every assayed SNV is kept.
- vs the QTL datasets (``caqtl`` / ``dsqtl``, which keep all consequences): the
  HIGH-impact ``exclude_consequences`` (canonical splice, nonsense, frameshift, …)
  are **dropped**. Those are trivially-deleterious and not the discriminative
  signal an SGE benchmark is about.

Every original author column is preserved verbatim under an ``author_`` prefix
(namespaced so nothing collides with the pipeline's annotation columns —
``consequence``, ``consequence_final``, ``distance_*``, …); no binary label is
imposed here (an eval-time decision).

Phase-1 source: BRCA1 (Findlay et al. 2018, *Nature* 562:217-222), via the
Evo2-bundled supplementary table ``41586_2018_461_MOESM3_ESM.xlsx``. It carries
hg19 genomic coordinates for **all** SNVs including intronic ones (unlike MaveDB,
whose cDNA->genome mapping drops intronic variants), so it is the cleanest BRCA1
source; coordinates are lifted hg19->GRCh38 here.
"""

import json
import re
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from marin_dna.data.genome import Genome
from marin_dna.pipelines.evals.trait_intervals import add_exon, add_tss
from marin_dna.pipelines.evals.variants import (
    COORDINATES,
    NUCLEOTIDES,
    attach_per_chrom_consequences,
    check_ref_alt,
    filter_chroms,
    filter_snp,
    lift_hg19_to_hg38,
)

# Findlay 2018 BRCA1 SGE xlsx columns that become the standard pipeline coords;
# every other column is preserved verbatim under an ``author_`` prefix.
_BRCA1_COORD_SOURCE: dict[str, str] = {
    "chromosome": "chrom",
    "position (hg19)": "pos",
    "reference": "ref",
    "alt": "alt",
    "gene": "gene",
}


def author_slug(name: str) -> str:
    """``author_``-prefixed, identifier-safe column name (lowercased; each run of
    non-alphanumerics -> ``_``). Namespaces source columns so they never collide
    with pipeline columns (``consequence``, ``consequence_final``, ``distance_*``)."""
    return "author_" + re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()


# --------------------------------------------------------------------------- #
# MaveDB study-level metadata: assay facts (experiment keywords) + score
# calibrations (investigator + ClinGen/ExCALIBR ACMG thresholds).
#
# These describe the *assay*, not individual variants, so they are study-level
# (one set per score-set). Two complementary shapes:
#   - assay facts: a flat dict of controlled-vocabulary keywords (assay readout,
#     molecular mechanism, model system, library mechanism, …) -> joined onto the
#     dataset as constant-per-gene `assay_*` columns (queryable inline).
#   - score calibrations: a variable-length list of threshold schemes, each with a
#     variable-length list of functional classes (GO call, score range, variant
#     count, and often an ACMG criterion PS3/BS3 + evidence strength) -> a tidy
#     long companion table (one row per gene x calibration x class), NOT joined
#     per-variant (wrong grain).
# --------------------------------------------------------------------------- #
_MAVEDB_API = "https://api.mavedb.org/api/v1"

# Tidy long-format schema for the score-calibration companion table.
_CALIBRATION_SCHEMA: dict[str, pl.DataType] = {
    "gene": pl.Utf8,
    "mavedb_urn": pl.Utf8,
    "calibration_title": pl.Utf8,
    "research_use_only": pl.Boolean,
    "baseline_score": pl.Float64,
    "prior_probability_pathogenicity": pl.Float64,
    "threshold_source_pmids": pl.Utf8,
    "class_label": pl.Utf8,
    "go_classification": pl.Utf8,  # normal / abnormal / not_specified
    "range_lower": pl.Float64,
    "range_upper": pl.Float64,
    "inclusive_lower": pl.Boolean,
    "inclusive_upper": pl.Boolean,
    "variant_count": pl.Int64,
    "acmg_criterion": pl.Utf8,  # PS3 (pathogenic) / BS3 (benign)
    "acmg_evidence_strength": pl.Utf8,  # SUPPORTING … VERY_STRONG
    "acmg_points": pl.Int64,  # ExCALIBR signed points (negative = benign)
}


def _http_get_json(url: str) -> dict:
    """GET ``url`` and parse the JSON body (stdlib only). Injected via ``get_fn``
    in :func:`build_mavedb_metadata` so tests need no network."""
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (https only)
        return json.load(resp)


def assay_fact_slug(key: str) -> str:
    """``assay_``-prefixed, identifier-safe column name for a MaveDB keyword key
    (e.g. ``"Phenotypic Assay Method"`` -> ``assay_phenotypic_assay_method``)."""
    return "assay_" + re.sub(r"[^0-9a-zA-Z]+", "_", str(key)).strip("_").lower()


def fetch_mavedb_score_set(
    urn: str, *, get_fn: Callable[[str], dict] | None = None
) -> dict:
    """Fetch a MaveDB score-set record (full JSON). ``get_fn`` is injectable for
    tests (default = live HTTPS GET of ``/score-sets/{urn}``)."""
    get = get_fn or _http_get_json
    return get(f"{_MAVEDB_API}/score-sets/{urn}")


def extract_assay_facts(score_set: dict) -> dict[str, str]:
    """Pull a score-set's MaveDB **assay facts** (the experiment's controlled
    keywords) into ``{assay_<slug>: label}``.

    MaveDB annotates each experiment with controlled-vocabulary keywords — assay
    readout (``Phenotypic Assay Method``), mechanism (``Phenotypic Assay
    Mechanism`` = loss/gain of function), model system, endogenous-locus library
    mechanism, … Returns ``{}`` for an unannotated experiment (e.g. BRCA2
    ``urn:mavedb:00001225-a-1`` carries no keywords).
    """
    kws = (score_set.get("experiment") or {}).get("keywords") or []
    facts: dict[str, str] = {}
    for entry in kws:
        kw = entry.get("keyword") or {}
        key, label = kw.get("key"), kw.get("label")
        if key and label:
            facts[assay_fact_slug(key)] = label
    # Sort by column name so the assay_ column order is deterministic regardless of the
    # order MaveDB returns the experiment's keywords in (byte-reproducible artifact).
    return dict(sorted(facts.items()))


def extract_score_calibrations(
    score_set: dict, *, gene: str, mavedb_urn: str
) -> list[dict]:
    """Flatten a score-set's ``scoreCalibrations`` into tidy long rows (one per
    calibration x functional class) matching :data:`_CALIBRATION_SCHEMA`.

    Each calibration is a threshold scheme (investigator-provided, or a
    ClinGen/ExCALIBR ACMG calibration). A functional class carries its GO call
    (normal/abnormal/not_specified), score range, variant count, and — for the
    ACMG schemes — the criterion (PS3/BS3), evidence strength, and signed points.
    Calibration-level ``prior_probability_pathogenicity`` (the OddsPath prior) and
    PubMed threshold sources are carried on every row of that calibration. A
    calibration with no classes still emits one (class-null) row so its existence
    is recorded. Returns ``[]`` if the score-set has no calibrations (e.g. BRCA2).
    """
    rows: list[dict] = []
    for cal in score_set.get("scoreCalibrations") or []:
        meta = cal.get("calibrationMetadata") or {}
        pmids = ",".join(
            s["identifier"]
            for s in (cal.get("thresholdSources") or [])
            if s.get("dbName") == "PubMed" and s.get("identifier")
        )
        base = {
            "gene": gene,
            "mavedb_urn": mavedb_urn,
            "calibration_title": cal.get("title"),
            "research_use_only": cal.get("researchUseOnly"),
            "baseline_score": cal.get("baselineScore"),
            "prior_probability_pathogenicity": meta.get(
                "prior_probability_pathogenicity"
            ),
            "threshold_source_pmids": pmids or None,
        }
        fcs = cal.get("functionalClassifications") or []
        if not fcs:
            rows.append(
                {k: base.get(k) for k in _CALIBRATION_SCHEMA}
            )  # class fields -> null
            continue
        for fc in fcs:
            rng = fc.get("range") or [None, None]
            assert len(rng) == 2, f"{gene}: unexpected calibration range {rng!r}"
            acmg = fc.get("acmgClassification") or {}
            rows.append(
                {
                    **base,
                    "class_label": fc.get("label"),
                    "go_classification": fc.get("functionalClassification"),
                    "range_lower": rng[0],
                    "range_upper": rng[1],
                    "inclusive_lower": fc.get("inclusiveLowerBound"),
                    "inclusive_upper": fc.get("inclusiveUpperBound"),
                    "variant_count": fc.get("variantCount"),
                    "acmg_criterion": acmg.get("criterion"),
                    "acmg_evidence_strength": acmg.get("evidenceStrength"),
                    "acmg_points": acmg.get("points"),
                }
            )
    return rows


def build_mavedb_metadata(
    gene_to_urn: list[tuple[str, str]],
    *,
    get_fn: Callable[[str], dict] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fetch every SGE study's MaveDB record once and return ``(assay_facts,
    calibrations)``.

    ``assay_facts``: one row per ``(gene, mavedb_urn)`` with the sparse union of
    ``assay_*`` keyword columns (genes with no keywords contribute nulls) — joined
    onto the dataset by ``mavedb_urn`` so the facts are queryable per-variant.
    ``calibrations``: the tidy long companion table (:func:`extract_score_calibrations`).
    ``get_fn`` is injectable for tests.
    """
    fact_rows: list[pl.DataFrame] = []
    calib_rows: list[dict] = []
    for gene, urn in gene_to_urn:
        ss = fetch_mavedb_score_set(urn, get_fn=get_fn)
        facts = extract_assay_facts(ss)
        fact_rows.append(pl.DataFrame({"gene": gene, "mavedb_urn": urn, **facts}))
        calib = extract_score_calibrations(ss, gene=gene, mavedb_urn=urn)
        calib_rows.extend(calib)
        print(
            f"[sge mavedb-meta {gene}] {len(facts)} assay facts, "
            f"{len(calib)} calibration class-rows"
        )
    assay_facts = pl.concat(fact_rows, how="diagonal_relaxed")
    calibrations = pl.DataFrame(calib_rows, schema=_CALIBRATION_SCHEMA)
    return assay_facts, calibrations


def attach_assay_facts(data: pl.DataFrame, assay_facts: pl.DataFrame) -> pl.DataFrame:
    """Left-join the per-study MaveDB ``assay_*`` keyword columns onto the SGE dataset
    by ``mavedb_urn`` (constant per gene), so the assay characteristics are queryable
    alongside every variant.

    Asserts every variant's ``mavedb_urn`` is present in ``assay_facts`` first — a left
    join would otherwise silently null-fill the ``assay_*`` columns for a study the
    metadata fetch missed (e.g. a gene added to config but not to the fetch list).
    """
    missing = set(data["mavedb_urn"].unique()) - set(assay_facts["mavedb_urn"])
    assert not missing, f"sge: mavedb_urn(s) absent from assay_facts: {sorted(missing)}"
    return data.join(
        assay_facts.select("mavedb_urn", pl.col("^assay_.*$")),
        on="mavedb_urn",
        how="left",
        # Preserve the variants' row order: a polars left join doesn't guarantee
        # output order, so without this the built parquet's row order is
        # non-deterministic (it shifts with global hash state — e.g. across test
        # orderings, which is how this surfaced). #293.
        maintain_order="left",
    )


# --------------------------------------------------------------------------- #
# Per-variant calibrated functional class, author-class harmonization, and
# function-score direction (#297). These turn the study-level `calibrations.parquet`
# + the per-study discrete author classes into clean, cross-gene-comparable per-variant
# columns, so the eval no longer re-derives them (sign-align hack, scheme-pick
# heuristic, range-membership) at score time.
# --------------------------------------------------------------------------- #

# Calibration-scheme selection policy (#297 item C). Prefer the primary ClinGen/ExCALIBR
# calibration (the VCEP PS3-BS3 framework); never a dated snapshot — e.g.
# "ExCALIBR calibration (ClinVar 2018)" pins thresholds to a stale ClinVar release.
_PRIMARY_CALIBRATION = "ExCALIBR calibration"
# A parenthesized year marks a dated-snapshot scheme ("… (ClinVar 2018)",
# "… (Feb 2026, All Variants)") — deprioritized vs the live calibration.
_DATED_SNAPSHOT_RE = re.compile(r"\(\D*\d{4}")

# Per-gene discrete author-class vocab -> harmonized {normal, intermediate, abnormal}
# (#297 item D). One (column, value-map) per gene; the six functionally_* genes share a
# column. Genes absent here (BAP1, VHL, BRCA2) ship no discrete author class.
_FUNCTIONAL_CONSEQUENCE_MAP = {
    "functionally_normal": "normal",
    "functionally_abnormal": "abnormal",
    "indeterminate": "intermediate",
}
_AUTHOR_CLASS_MAPS: dict[str, tuple[str, dict[str, str]]] = {
    "BRCA1": (
        "author_func_class",
        {"FUNC": "normal", "INT": "intermediate", "LOF": "abnormal"},
    ),
    "RAD51C": (
        "author_functional_classification",
        # Depletion screen: depleted = loss of function = abnormal; unchanged = normal;
        # `enriched` (grows faster than WT) is a distinct non-LOF phenotype -> intermediate.
        {
            "unchanged": "normal",
            "enriched": "intermediate",
            "slow depleted": "abnormal",
            "fast depleted": "abnormal",
        },
    ),
    "DDX3X": (
        "author_sge_prediction_of_variant_function_in_ndd_context",
        {"normal": "normal", "abnormal": "abnormal"},
    ),
    **{
        g: ("author_functional_consequence", _FUNCTIONAL_CONSEQUENCE_MAP)
        for g in ("BARD1", "CTCF", "PALB2", "RAD51D", "SFPQ", "XRCC2")
    },
}


def attach_author_class_harmonized(data: pl.DataFrame) -> pl.DataFrame:
    """Harmonize each study's discrete functional class to a common
    ``author_class_harmonized`` ∈ {normal, intermediate, abnormal} (#297 item D).

    Maps each gene's author-class column through :data:`_AUTHOR_CLASS_MAPS`. Asserts
    every non-null source value is mapped (loud on an unexpected category) and that no
    *unmapped* gene carries a non-null value in a known class column (so a real author
    class is never silently dropped). Genes with no discrete class (BAP1, VHL, BRCA2)
    stay null. Feeds the categorical-gene branch of :func:`attach_calibrated_class` and
    :func:`attach_function_direction`.
    """
    n = data.height
    gene_arr = np.asarray(data["gene"].to_list(), dtype=object)
    harm = np.array([None] * n, dtype=object)
    cols_to_genes: dict[str, set[str]] = {}
    for gene, (col, mapping) in _AUTHOR_CLASS_MAPS.items():
        cols_to_genes.setdefault(col, set()).add(gene)
        if col not in data.columns:
            continue
        vals = np.asarray(data[col].to_list(), dtype=object)
        for i in np.where(gene_arr == gene)[0]:
            v = vals[i]
            if v is None:
                continue
            assert v in mapping, f"{gene}: unmapped author class {v!r} in {col}"
            harm[i] = mapping[v]
    # Defensive: a gene NOT mapped to a known class column must not carry a non-null
    # value there (else a real author class would silently stay unharmonized).
    for col, genes_for_col in cols_to_genes.items():
        if col not in data.columns:
            continue
        stray = (
            data.filter(
                pl.col(col).is_not_null() & ~pl.col("gene").is_in(list(genes_for_col))
            )["gene"]
            .unique()
            .to_list()
        )
        assert not stray, f"unmapped gene(s) carry non-null {col}: {sorted(stray)}"
    return data.with_columns(
        pl.Series("author_class_harmonized", harm.tolist(), dtype=pl.Utf8)
    )


def _numeric_class_rows(scheme: pl.DataFrame, go: str) -> pl.DataFrame:
    """Rows of ``scheme`` with the given ``go_classification`` and >=1 finite range
    bound (so they can threshold ``function_score``). Categorical schemes carry
    ``[null, null]`` ranges and contribute nothing here."""
    return scheme.filter(
        (pl.col("go_classification") == go)
        & (pl.col("range_lower").is_not_null() | pl.col("range_upper").is_not_null())
    )


def _select_numeric_calibration(
    gcal: pl.DataFrame,
) -> tuple[str, pl.DataFrame, pl.DataFrame] | None:
    """Pick a gene's numeric-threshold calibration scheme by the #297 policy; return
    ``(title, normal_rows, abnormal_rows)`` or None if the gene has no numeric scheme.

    A qualifying scheme needs >=1 finite-range normal AND abnormal class. Among those,
    rank by (is the primary ``ExCALIBR calibration``, is NOT a dated snapshot,
    finite-row count, title) so the live ExCALIBR wins over its dated ClinVar snapshot
    and over investigator-provided classes; a gene whose ExCALIBR lacks a normal class
    (CTCF) falls back to the next qualifying numeric scheme (its investigator classes).
    """
    cands: list[tuple[str, pl.DataFrame, pl.DataFrame]] = []
    for key, scheme in gcal.group_by("calibration_title", maintain_order=True):
        title = key[0]
        norm = _numeric_class_rows(scheme, "normal")
        abn = _numeric_class_rows(scheme, "abnormal")
        if norm.height and abn.height:
            cands.append((title, norm, abn))
    if not cands:
        return None
    cands.sort(
        key=lambda c: (
            c[0] == _PRIMARY_CALIBRATION,
            not bool(_DATED_SNAPSHOT_RE.search(c[0] or "")),
            c[1].height + c[2].height,
            c[0],
        ),
        reverse=True,
    )
    return cands[0]


def _in_range(
    x: float, lo: float | None, hi: float | None, inc_lo: bool, inc_hi: bool
) -> bool:
    """Is score ``x`` within the (possibly half-open) range ``[lo, hi]``? A null bound
    is unbounded on that side; ``inc_*`` toggle open/closed bounds (mirrors the eval's
    range-membership test in ``calibrated_binary_label``)."""
    if lo is not None and (x < lo or (x == lo and not inc_lo)):
        return False
    if hi is not None and (x > hi or (x == hi and not inc_hi)):
        return False
    return True


def _matching_range(rows: list[dict], x: float) -> dict | None:
    """First calibration-class row (from :meth:`iter_rows(named=True)`) whose score
    range contains ``x``, else None. For ExCALIBR's disjoint graded ranges at most one
    matches; first-in-order is deterministic by the calibration table's row order."""
    return next(
        (
            r
            for r in rows
            if _in_range(
                x,
                r["range_lower"],
                r["range_upper"],
                r["inclusive_lower"],
                r["inclusive_upper"],
            )
        ),
        None,
    )


def attach_calibrated_class(
    data: pl.DataFrame, calibrations: pl.DataFrame
) -> pl.DataFrame:
    """Materialize a per-variant calibrated functional class (#297 item C).

    Adds three columns, decided once at build with an explicit policy (replacing the
    eval's most-rows heuristic):

    - ``calibrated_class`` ∈ {abnormal, intermediate, normal} or null,
    - ``calibration_scheme``: the chosen scheme title, ``"author_class"`` (categorical
      genes), or null (no calibration),
    - ``acmg_strength``: the matched range's ACMG evidence strength (SUPPORTING …
      VERY_STRONG); null for categorical / intermediate / uncalibrated.

    Per gene: prefer the numeric scheme from :func:`_select_numeric_calibration`
    (ExCALIBR-first) and label by range membership (abnormal / normal / else
    intermediate). A gene with no numeric scheme but a harmonized author class (DDX3X)
    inherits its class from ``author_class_harmonized``; a gene with neither (BRCA2)
    gets nulls. Requires :func:`attach_author_class_harmonized` first.
    """
    assert "author_class_harmonized" in data.columns, (
        "attach_calibrated_class: call attach_author_class_harmonized first"
    )
    # function_score is a non-null loader invariant; guard it here because a null would
    # become NaN below (np.asarray(..., dtype=float)) and silently match EVERY range in
    # _in_range (all comparisons against NaN are False), mislabeling the variant.
    assert data["function_score"].null_count() == 0, (
        "attach_calibrated_class: null function_score (would corrupt range membership)"
    )
    n = data.height
    fs = np.asarray(data["function_score"].to_list(), dtype=float)
    gene_arr = np.asarray(data["gene"].to_list(), dtype=object)
    ach = np.asarray(data["author_class_harmonized"].to_list(), dtype=object)
    cls = np.array([None] * n, dtype=object)
    scheme_col = np.array([None] * n, dtype=object)
    strength = np.array([None] * n, dtype=object)
    for gene in sorted(set(gene_arr)):
        idx = np.where(gene_arr == gene)[0]
        gcal = calibrations.filter(pl.col("gene") == gene)
        sel = _select_numeric_calibration(gcal) if gcal.height else None
        if sel is not None:
            title, norm, abn = sel
            norm_rows = list(norm.iter_rows(named=True))
            abn_rows = list(abn.iter_rows(named=True))
            for i in idx:
                x = fs[i]
                abn_hit = _matching_range(abn_rows, x)
                norm_hit = _matching_range(norm_rows, x)
                if abn_hit and not norm_hit:
                    cls[i], strength[i] = "abnormal", abn_hit["acmg_evidence_strength"]
                elif norm_hit and not abn_hit:
                    cls[i], strength[i] = "normal", norm_hit["acmg_evidence_strength"]
                else:
                    cls[i] = "intermediate"
                scheme_col[i] = title
        elif any(ach[i] is not None for i in idx):
            # Categorical-only gene (DDX3X): no numeric thresholds, but a harmonized
            # author class exists -> use it directly.
            for i in idx:
                if ach[i] is not None:
                    cls[i], scheme_col[i] = ach[i], "author_class"
        # else: no calibration and no author class (BRCA2) -> all null.
    return data.with_columns(
        pl.Series("calibrated_class", cls.tolist(), dtype=pl.Utf8),
        pl.Series("calibration_scheme", scheme_col.tolist(), dtype=pl.Utf8),
        pl.Series("acmg_strength", strength.tolist(), dtype=pl.Utf8),
    )


def attach_label(data: pl.DataFrame) -> pl.DataFrame:
    """Add the boolean ``label`` — the AUPRC target (#301).

    ``True`` = impactful (calibrated **abnormal**), ``False`` = **normal**, null for
    ``intermediate`` / uncalibrated (BRCA2). The build keeps only ``label``-non-null
    rows (the AUPRC-only benchmark drops the variants no classification metric uses),
    so in the shipped dataset ``label`` is always a clean bool. Requires
    :func:`attach_calibrated_class` first.
    """
    assert "calibrated_class" in data.columns, (
        "attach_label: call attach_calibrated_class first"
    )
    return data.with_columns(
        pl.when(pl.col("calibrated_class") == "abnormal")
        .then(pl.lit(True))
        .when(pl.col("calibrated_class") == "normal")
        .then(pl.lit(False))
        .otherwise(None)
        .alias("label")
    )


def _scheme_direction(norm: pl.DataFrame, abn: pl.DataFrame) -> int:
    """+1 if the abnormal calibration range sits BELOW the normal range (low score =
    abnormal = less functional, so "higher = more functional"), else -1. Range center =
    mean of finite midpoints (a single-bound range uses that bound)."""

    def center(rows: pl.DataFrame) -> float:
        vals: list[float] = []
        for r in rows.iter_rows(named=True):
            lo, hi = r["range_lower"], r["range_upper"]
            if lo is not None and hi is not None:
                vals.append((lo + hi) / 2)
            elif lo is not None:
                vals.append(lo)
            elif hi is not None:
                vals.append(hi)
        assert vals, "empty numeric range set"
        return float(np.mean(vals))

    return 1 if center(abn) < center(norm) else -1


def attach_function_direction(
    data: pl.DataFrame, calibrations: pl.DataFrame
) -> pl.DataFrame:
    """Harmonize the per-study ``function_score`` direction (#297 item B).

    Adds ``function_direction`` (+1 / -1 / null) and ``function_score_aligned`` =
    ``function_direction * function_score``, so "higher = more functional" holds across
    genes (the raw per-study ``function_score`` is kept verbatim). Direction is sourced
    from the **assay, not the model**: numeric-scheme genes from whether the abnormal
    calibration range sits below or above the normal range
    (:func:`_scheme_direction`); the categorical-only gene (DDX3X) from the sign of
    mean(function_score | abnormal) vs (| normal) over its ``author_class_harmonized``
    labels; a gene with neither (BRCA2) stays null. Requires
    :func:`attach_author_class_harmonized` first.

    Note: this corrects only the *sign*; per-study *scales* still differ (issue #297
    §3), so a pooled rank/Spearman view is valid but a pooled raw-Pearson is not.
    """
    assert "author_class_harmonized" in data.columns, (
        "attach_function_direction: call attach_author_class_harmonized first"
    )
    # function_score is a non-null loader invariant; a null would become NaN below and
    # poison the categorical-gene mean comparison (np.mean of a NaN-containing list is
    # NaN, and NaN >= NaN is False -> a spurious -1 direction).
    assert data["function_score"].null_count() == 0, (
        "attach_function_direction: null function_score (would corrupt direction)"
    )
    n = data.height
    fs = np.asarray(data["function_score"].to_list(), dtype=float)
    gene_arr = np.asarray(data["gene"].to_list(), dtype=object)
    ach = np.asarray(data["author_class_harmonized"].to_list(), dtype=object)
    direction = np.array([None] * n, dtype=object)
    for gene in sorted(set(gene_arr)):
        idx = np.where(gene_arr == gene)[0]
        gcal = calibrations.filter(pl.col("gene") == gene)
        sel = _select_numeric_calibration(gcal) if gcal.height else None
        d: int | None = None
        if sel is not None:
            _, norm, abn = sel
            d = _scheme_direction(norm, abn)
        else:
            abn_fs = [fs[i] for i in idx if ach[i] == "abnormal"]
            norm_fs = [fs[i] for i in idx if ach[i] == "normal"]
            if abn_fs and norm_fs:
                d = 1 if float(np.mean(norm_fs)) >= float(np.mean(abn_fs)) else -1
        if d is not None:
            for i in idx:
                direction[i] = d
    return data.with_columns(
        pl.Series("function_direction", direction.tolist(), dtype=pl.Int8)
    ).with_columns(
        (pl.col("function_direction") * pl.col("function_score")).alias(
            "function_score_aligned"
        )
    )


def normalize_brca1_findlay(raw: pl.DataFrame, *, mavedb_urn: str) -> pl.DataFrame:
    """Normalize the Findlay 2018 BRCA1 SGE table (already header-resolved) to the
    SGE schema.

    Emits the standard ``chrom, pos, ref, alt, gene, assay, mavedb_urn,
    function_score`` columns plus **every original column preserved verbatim under an
    ``author_`` prefix** — so no author metadata is lost and nothing collides with the
    pipeline's annotation columns. ``chrom`` is a string and ``pos`` is 1-based hg19
    (lifted to GRCh38 downstream in :func:`annotate_sge_variants`). ``function_score``
    is the common continuous score across studies (here ``author_function_score_mean``);
    ``author_func_class`` (FUNC/INT/LOF) is BRCA1's discrete classification.

    Args:
        raw: header-resolved Findlay xlsx frame.
        mavedb_urn: the dataset's canonical MaveDB accession, stamped per-variant so
            ``(gene, mavedb_urn)`` identifies the exact study (a gene can have >1).
    """
    slugs = [author_slug(c) for c in raw.columns]
    assert len(set(slugs)) == len(slugs), (
        f"BRCA1: author column slugs collide: {sorted(slugs)}"
    )
    author = raw.rename(dict(zip(raw.columns, slugs)))
    for src in _BRCA1_COORD_SOURCE:
        assert author_slug(src) in author.columns, (
            f"BRCA1: missing expected source column {src!r} ({author_slug(src)})"
        )
    out = (
        author.with_columns(
            pl.col("author_chromosome").cast(pl.Utf8).alias("chrom"),
            pl.col("author_position_hg19").cast(pl.Int64).alias("pos"),
            pl.col("author_reference").cast(pl.Utf8).alias("ref"),
            pl.col("author_alt").cast(pl.Utf8).alias("alt"),
            pl.col("author_gene").cast(pl.Utf8).alias("gene"),
            assay=pl.lit("sge"),
            mavedb_urn=pl.lit(mavedb_urn),
            function_score=pl.col("author_function_score_mean").cast(pl.Float64),
        )
        .pipe(filter_snp)
        # Standard pipeline columns first, then every preserved author column.
        .select(
            "chrom",
            "pos",
            "ref",
            "alt",
            "gene",
            "assay",
            "mavedb_urn",
            "function_score",
            pl.col("^author_.*$"),
        )
    )
    assert out["author_function_score_mean"].null_count() == 0, (
        "BRCA1: null function score (author_function_score_mean) after normalization"
    )
    # Fail loud on a blank/bad hg19 coordinate rather than dropping it silently later
    # (a null pos survives filter_snp, then liftover maps it to -1 and it vanishes).
    for coord in ("pos", "ref", "alt"):
        assert out[coord].null_count() == 0, (
            f"BRCA1: null {coord} after normalization — suspect a header-offset or a "
            "blank coordinate cell in the Findlay xlsx"
        )
    # Tolerate a null func.class (preserved author metadata, not the discriminative
    # signal) but still reject any *unexpected* category.
    classes = set(out["author_func_class"].drop_nulls().unique())
    assert classes <= {"FUNC", "INT", "LOF"}, (
        f"BRCA1: unexpected func.class values: {classes}"
    )
    return out


def read_brca1_findlay(xlsx_path: str | Path, *, mavedb_urn: str) -> pl.DataFrame:
    """Read + normalize the Findlay 2018 BRCA1 SGE supplementary xlsx.

    The sheet has two super-header rows above the real column header (row index 2),
    so it is read with ``header=2``. ``mavedb_urn`` is the dataset's canonical
    accession (see :func:`normalize_brca1_findlay`). Returns the SGE schema.
    """
    raw = pd.read_excel(xlsx_path, header=2)
    return normalize_brca1_findlay(pl.from_pandas(raw), mavedb_urn=mavedb_urn)


def load_mavedb_genomic_scoreset(
    scores_path: str | Path, *, gene: str, mavedb_urn: str
) -> pl.DataFrame:
    """Load a **genome-targeted** MaveDB SGE score-set CSV (``NC_…:g.`` hgvs_nt) to
    the SGE schema.

    The genomic ``hgvs_nt`` is parsed directly (e.g. ``NC_000002.12:g.214728667A>G``
    -> chr2 / 214728667 / A / G), so **intronic SNVs are kept** with no transcript
    mapping (unlike the transcript-targeted ``c.`` score-sets, whose intronic variants
    MaveDB's auto-map drops). Non-SNV variants (del / delins / dup / MNV) and
    null-score rows are dropped. GRCh38-native (no liftover). Every original column is
    preserved ``author_``-prefixed; the common ``function_score`` is the study's
    ``score`` column.

    Args:
        scores_path: the MaveDB ``/score-sets/{urn}/scores`` CSV.
        gene: HGNC gene symbol.
        mavedb_urn: the score-set's canonical MaveDB accession.
    """
    raw = pl.read_csv(scores_path, infer_schema_length=None)
    for col in ("hgvs_nt", "score"):
        assert col in raw.columns, (
            f"{gene}: MaveDB scores CSV missing {col!r}: {raw.columns}"
        )
    n_raw = raw.height
    nt = pl.col("author_hgvs_nt")
    # Anchor chrom extraction to the FULL single-SNV form (the same token pos/ref/alt
    # parse from below). A multi-variant `;`-joined hgvs_nt would otherwise take its
    # chrom from the first sub-variant while pos/ref/alt come from the last — a chimera;
    # the full anchor makes such rows parse to null chrom and drop cleanly.
    chrom_num = nt.str.extract(r"^NC_0*(\d+)\.\d+:g\.\d+[ACGT]>[ACGT]$", 1).cast(
        pl.Int64
    )
    out = (
        raw.rename({c: author_slug(c) for c in raw.columns})
        .with_columns(
            # Genomic SNV: NC_<chrom>.<v>:g.<pos><REF>><ALT>. Non-SNV (del/delins/dup/
            # MNV) lacks the single-base `>` form -> pos/ref/alt parse to null -> dropped.
            pl.when(chrom_num == 23)
            .then(pl.lit("X"))
            .when(chrom_num == 24)
            .then(pl.lit("Y"))
            .otherwise(chrom_num.cast(pl.Utf8))
            .alias("chrom"),
            nt.str.extract(r":g\.(\d+)[ACGT]>[ACGT]$", 1).cast(pl.Int64).alias("pos"),
            nt.str.extract(r":g\.\d+([ACGT])>[ACGT]$", 1).alias("ref"),
            nt.str.extract(r":g\.\d+[ACGT]>([ACGT])$", 1).alias("alt"),
            gene=pl.lit(gene),
            assay=pl.lit("sge"),
            mavedb_urn=pl.lit(mavedb_urn),
            # strict=False: a non-numeric score token coerces to null (then dropped by
            # the is_not_null filter below), consistent with dropping blank-score rows,
            # rather than aborting the whole build on one bad cell.
            function_score=pl.col("author_score").cast(pl.Float64, strict=False),
        )
        .filter(
            pl.col("pos").is_not_null()
            & pl.col("chrom").is_not_null()
            & pl.col("function_score").is_not_null()
        )
        .pipe(filter_snp)
        .select(
            "chrom",
            "pos",
            "ref",
            "alt",
            "gene",
            "assay",
            "mavedb_urn",
            "function_score",
            pl.col("^author_.*$"),
        )
    )
    print(f"[sge load {gene}] {n_raw} rows -> {out.height} scored genomic SNVs")
    assert out.height > 0, f"{gene}: no scored genomic SNVs parsed from hgvs_nt"
    return out


# --- transcript-targeted score-sets: recode c. HGVS -> genomic (pyhgvs + cdot) ---
# Single-base-substitution HGVS (exonic c.N, intronic c.N±M, UTR c.-N/c.*N, or g.);
# excludes del / delins / dup / ins / MNV.
SNV_HGVS_RE = re.compile(r"[ACGT]>[ACGT]$")
_NC_CONTIG_RE = re.compile(r"^NC_0*(\d+)\.")


def _nc_contig_to_chrom(contig: str) -> str | None:
    """RefSeq genomic contig (``NC_000013.11``) -> chromosome label (``13``)."""
    m = _NC_CONTIG_RE.match(contig)
    if not m:
        return None
    n = int(m.group(1))
    return "X" if n == 23 else "Y" if n == 24 else str(n)


class _NCGenome:
    """pyfaidx-FASTA wrapper translating the RefSeq ``NC_…`` contig names cdot
    transcripts use to the staged Ensembl FASTA's chromosome names (``13``)."""

    def __init__(self, fasta) -> None:
        self._fasta = fasta

    def _key(self, contig: str) -> str:
        return _nc_contig_to_chrom(contig) or contig

    def __getitem__(self, contig: str):
        return self._fasta[self._key(contig)]

    def __contains__(self, contig: str) -> bool:
        return self._key(contig) in self._fasta


def pyhgvs_cdot_mapper(genome_path: str | Path) -> Callable[[str], tuple | None]:
    """Build a ``c-hgvs -> (chrom, pos, ref, alt) | None`` mapper using pyhgvs + cdot's
    GRCh38 REST transcripts against the staged GRCh38 FASTA (transcripts are fetched +
    cached per accession). Lazy-imports the optional ``hgvs`` dependency group."""
    import pyhgvs
    from cdot.pyhgvs.pyhgvs_transcript import RESTPyHGVSTranscriptFactory
    from pyfaidx import Fasta

    get_transcript = RESTPyHGVSTranscriptFactory().get_transcript_grch38
    genome = _NCGenome(Fasta(str(genome_path)))

    def mapper(hgvs_c: str) -> tuple | None:
        try:
            contig, pos, ref, alt = pyhgvs.parse_hgvs_name(
                hgvs_c, genome, get_transcript=get_transcript
            )
        except Exception:
            return None
        chrom = _nc_contig_to_chrom(contig)
        if chrom is None or ref not in NUCLEOTIDES or alt not in NUCLEOTIDES:
            return None
        return (chrom, pos, ref, alt)

    return mapper


def recode_hgvs_c_to_genomic(
    hgvs_list: list[str], *, mapper: Callable[[str], tuple | None]
) -> pl.DataFrame:
    """Map transcript ``c.`` HGVS -> GRCh38 genomic ``(chrom, pos, ref, alt)`` via
    ``mapper`` (a ``str -> tuple | None`` callable; the production impl is
    :func:`pyhgvs_cdot_mapper`).

    This is how the **transcript-targeted** SGE score-sets keep their **intronic**
    variants: pyhgvs + cdot project ``ENST…:c.N±M…`` onto the genome (handling strand)
    — coordinates MaveDB's own cDNA->genome map drops. Returns ``[hgvs_nt, chrom, pos,
    ref, alt]`` for the SNVs that mapped; others are omitted (reported). ``mapper`` is
    injectable for tests (no network / no transcript data).
    """
    recs: list[dict] = []
    n_fail = 0
    for h in hgvs_list:
        r = mapper(h)
        if r is None:
            n_fail += 1
            continue
        chrom, pos, ref, alt = r
        recs.append(
            {"hgvs_nt": h, "chrom": chrom, "pos": int(pos), "ref": ref, "alt": alt}
        )
    print(
        f"[sge recode] {len(hgvs_list)} c. SNVs -> {len(recs)} genomic ({n_fail} unmapped)"
    )
    # Floor guard: the mapper (pyhgvs + cdot's GRCh38 REST transcripts) maps ~100% of
    # SNV-only c. HGVS. A large unmapped fraction means a systemic failure — a cdot.cc
    # outage swallowed by the mapper's `except Exception: return None`, a transcript-
    # version/build mismatch — not a few genuinely-unmappable variants. Fail loud rather
    # than silently shipping a near-empty gene (whose only other tripwire is a downstream
    # `height > 0` that one stray mapped SNV would satisfy).
    if hgvs_list:
        frac = len(recs) / len(hgvs_list)
        assert frac >= 0.5, (
            f"[sge recode] only {len(recs)}/{len(hgvs_list)} c. SNVs mapped "
            f"({frac:.0%}); expected ~100% for SNV-only input — suspect a cdot.cc/network "
            "outage or a transcript-version/build mismatch"
        )
    return pl.DataFrame(
        recs,
        schema={
            "hgvs_nt": pl.Utf8,
            "chrom": pl.Utf8,
            "pos": pl.Int64,
            "ref": pl.Utf8,
            "alt": pl.Utf8,
        },
    )


def load_mavedb_transcript_scoreset(
    scores_path: str | Path, recoded: pl.DataFrame, *, gene: str, mavedb_urn: str
) -> pl.DataFrame:
    """Load a **transcript-targeted** MaveDB SGE score-set CSV (``ENST/NM:c.`` hgvs_nt)
    to the SGE schema, using a precomputed ``recoded`` ``c.->genomic`` mapping (from
    :func:`recode_hgvs_c_to_genomic`) so intronic variants are kept.

    Every original column is preserved ``author_``-prefixed; the common
    ``function_score`` is the study's ``score``. Variants that didn't recode (or
    aren't SNVs / null-score) are dropped. GRCh38 (no liftover).
    """
    raw = pl.read_csv(scores_path, infer_schema_length=None)
    for col in ("hgvs_nt", "score"):
        assert col in raw.columns, (
            f"{gene}: MaveDB scores CSV missing {col!r}: {raw.columns}"
        )
    n_raw = raw.height
    coords = recoded.unique(subset="hgvs_nt").rename({"hgvs_nt": "author_hgvs_nt"})
    out = (
        raw.rename({c: author_slug(c) for c in raw.columns})
        .join(coords, on="author_hgvs_nt", how="inner")
        .with_columns(
            gene=pl.lit(gene),
            assay=pl.lit("sge"),
            mavedb_urn=pl.lit(mavedb_urn),
            # strict=False: a non-numeric score token coerces to null (then dropped by
            # the is_not_null filter below), consistent with dropping blank-score rows,
            # rather than aborting the whole build on one bad cell.
            function_score=pl.col("author_score").cast(pl.Float64, strict=False),
        )
        .filter(pl.col("function_score").is_not_null())
        .pipe(filter_snp)
        .select(
            "chrom",
            "pos",
            "ref",
            "alt",
            "gene",
            "assay",
            "mavedb_urn",
            "function_score",
            pl.col("^author_.*$"),
        )
    )
    print(
        f"[sge load {gene}] {n_raw} rows -> {out.height} scored SNVs (c.->g. recoded)"
    )
    assert out.height > 0, f"{gene}: no scored recoded SNVs"
    return out


def annotate_sge_variants(
    V: pl.DataFrame,
    *,
    genome: Genome,
    consequence_paths: list[str],
    chroms: list[str],
    exon_pc: pl.DataFrame,
    exon_nc: pl.DataFrame,
    tss_pc: pl.DataFrame,
    tss_nc: pl.DataFrame,
    exon_proximal_dist: int,
    tss_proximal_dist: int,
    exclude_consequences: list[str],
    consequence_groups: dict[str, list[str]],
    consequence_group_allowlist: list[str] | None = None,
    lift: bool,
    name: str = "",
) -> pl.DataFrame:
    """Liftover (optional) + ref/alt validation + consequence/distance annotation +
    HIGH-impact ``exclude_consequences`` drop + coarse ``consequence_group`` /
    ``subset`` grouping + optional group allowlist, with **no** matching or subsampling.

    Like a variant-effect curation pass (lift + ref/alt + consequences) but (a)
    drops ``exclude_consequences`` (SGE drops the HIGH-impact ones), (b)
    carries no signed ``effect`` (SGE has a direction-tied function score, not a
    QTL effect), and (c) **asserts zero ref/alt swaps**: an SGE function score is
    tied to the ref->alt substitution as the author defined it, so a swap (author
    ref != genome) signals a coordinate/strand/build problem, not a benign
    re-orientation.

    Adds (#297) the coarse ``consequence_group`` (and its ``subset`` alias) so SGE
    stratifies identically to the matched-pair datasets, then optionally filters to a
    ``consequence_group_allowlist`` (default keeps everything; the build passes
    ``[missense_variant, splicing]`` — the groups where the SGE assay actually measures
    function).

    Args:
        V: normalized SGE frame (``chrom, pos, ref, alt`` + author columns); ``pos``
            is 1-based (hg19 if ``lift`` else GRCh38).
        genome: GRCh38 reference for :func:`check_ref_alt`.
        consequence_paths / chroms: per-chrom VEP-consequence parquet paths and
            their parallel chromosome labels (only the chroms present in ``V`` are
            needed).
        exon_pc / exon_nc / tss_pc / tss_nc: nearest-feature interval frames.
        exon_proximal_dist / tss_proximal_dist: proximity thresholds for the
            ``consequence_final`` recategorization.
        exclude_consequences: VEP consequences to drop (canonical-LOF HIGH-impact).
        consequence_groups: ``{group: [consequence_final values]}`` map (the pipeline's
            shared ``consequence_groups`` config) collapsing fine consequences to the
            coarse ``consequence_group``; unmapped values keep their own value (same
            ``.replace`` semantics as :func:`trait_intervals.build_dataset`).
        consequence_group_allowlist: if not None, keep only variants whose
            ``consequence_group`` is in this list (a build-time row filter).
        lift: if True, lift hg19->GRCh38 first.
        name: label for log/assert messages.
    """
    assert V.height > 0, f"{name}: empty input frame"
    n_in = V.height
    n_strand_flip = 0
    if lift:
        # Keep the pre-lift ref to count strand-flips: for an SNV, lift_hg19_to_hg38
        # only changes `ref` when the chain maps to the minus strand (RC). This is
        # informational — a strand-RC preserves ref/alt roles, so the function score
        # is carried unchanged (see the swap note below).
        V = V.with_columns(pl.col("ref").alias("_pre_lift_ref"))
        V = V.pipe(lift_hg19_to_hg38).filter(pl.col("pos") != -1)
        n_strand_flip = V.filter(pl.col("ref") != pl.col("_pre_lift_ref")).height
        V = V.drop("_pre_lift_ref")
    n_lift = V.height
    V = V.pipe(filter_chroms)
    n_chrom = V.height
    # Two distinct allele transforms, with different consequences for an
    # alt-vs-ref quantity like the SGE function score:
    #   - liftover strand-RC (above): RCs both alleles but PRESERVES ref/alt
    #     roles (ref stays the WT allele, complemented), so the physical variant
    #     and its score are unchanged — no flip.
    #   - check_ref_alt swap (author ref != genome): re-labels which allele is
    #     ref. For SGE this is invalid, not a benign re-orientation: the score is
    #     "effect of the alt (variant) allele vs WT", so a swap would nonsensically
    #     call the WT allele the variant. We use Findlay's +strand GENOMIC
    #     `reference`/`alt` (not the mRNA-strand `transcript_*`), so ref should
    #     match the +strand genome and no swap should ever fire. Assert it.
    V = V.with_columns(pl.col("ref").alias("_pre_ref"))
    V = check_ref_alt(V, genome)
    n_ref = V.height
    n_swapped = V.filter(pl.col("ref") != pl.col("_pre_ref")).height
    V = V.drop("_pre_ref")
    print(
        f"[sge annotate {name}] attrition: in={n_in} after_lift={n_lift} "
        f"after_chrom_filter={n_chrom} after_ref_alt={n_ref} "
        f"lift_strand_flipped={n_strand_flip} ref_alt_swapped={n_swapped}"
    )
    assert n_swapped == 0, (
        f"{name}: {n_swapped} ref/alt swaps in check_ref_alt — an SGE function "
        "score is tied to ref->alt, so a swap signals a coordinate/strand/build "
        "mismatch, not a benign re-orientation"
    )
    # Anchor retention to the ORIGINAL input, not the post-lift count: a near-total
    # liftover loss makes n_lift tiny, so `n_ref >= 0.9 *
    # n_lift` would pass vacuously (0 >= 0) instead of catching the collapse.
    assert n_ref >= 0.5 * n_in, (
        f"check_ref_alt + liftover kept only {n_ref}/{n_in} variants for {name!r} — "
        "suspect a coordinate-base (0- vs 1-based), genome-build, or liftover mismatch"
    )
    V = attach_per_chrom_consequences(V, consequence_paths, chroms)
    assert V["consequence"].null_count() == 0, f"{name}: variants with null consequence"
    # add_exon/add_tss derive consequence_final from consequence_cre, so a null there
    # silently propagates; guard it at the join boundary (mirrors dart_eval.annotate).
    assert V["consequence_cre"].null_count() == 0, (
        f"{name}: variants with null consequence_cre"
    )
    # Drop the trivially-deleterious HIGH-impact consequences (canonical splice,
    # nonsense, frameshift, …) before distance recategorization.
    n_pre_excl = V.height
    V = V.filter(~pl.col("consequence").is_in(exclude_consequences))
    print(
        f"[sge annotate {name}] exclude_consequences dropped "
        f"{n_pre_excl - V.height} HIGH-impact variants ({V.height} kept)"
    )
    assert V.height > 0, f"{name}: all variants dropped by exclude_consequences"
    V = V.pipe(add_exon, exon_pc, exon_nc, exon_proximal_dist).pipe(
        add_tss, tss_pc, tss_nc, tss_proximal_dist
    )
    assert V["consequence_final"].null_count() == 0, (
        f"{name}: null consequence_final after annotation"
    )
    # Coarse consequence grouping (#297 item A): collapse consequence_final to the same
    # `consequence_group` the matched-pair datasets carry, with the SAME `.replace(...)`
    # semantics as trait_intervals.build_dataset — a consequence_final absent from the
    # map keeps its own value (missense_variant, synonymous_variant, tss_proximal,
    # 5_prime_UTR_variant, …). Then alias it to `subset` so SGE stratifies identically
    # to mendelian/complex (whose metric groups on `subset`).
    consequence_to_group = {
        c: group
        for group, consequences in consequence_groups.items()
        for c in consequences
    }
    V = V.with_columns(
        pl.col("consequence_final")
        .replace(consequence_to_group)
        .alias("consequence_group")
    )
    assert V["consequence_group"].null_count() == 0, (
        f"{name}: null consequence_group after grouping"
    )
    V = V.with_columns(pl.col("consequence_group").alias("subset"))
    # Build-time allowlist (#297 item E): keep only the consequence groups where the SGE
    # assay actually measures function (the build passes [missense_variant, splicing]);
    # the rest (synonymous, UTRs, ncRNA, distal, tss_proximal, …) are near-uninformative
    # for an SGE benchmark and are dropped for inference speed + focus. None keeps all.
    if consequence_group_allowlist is not None:
        n_pre = V.height
        by_group = dict(V.group_by("consequence_group").len().iter_rows())
        V = V.filter(pl.col("consequence_group").is_in(consequence_group_allowlist))
        dropped = {
            g: n
            for g, n in sorted(by_group.items())
            if g not in set(consequence_group_allowlist)
        }
        print(
            f"[sge annotate {name}] consequence_group allowlist "
            f"{consequence_group_allowlist} kept {V.height}/{n_pre} "
            f"(dropped {n_pre - V.height}: {dropped})"
        )
        assert V.height > 0, (
            f"{name}: all variants dropped by consequence_group_allowlist"
        )
    V = V.sort(COORDINATES)
    assert (V["pos"] > 0).all(), f"{name}: non-positive positions after annotation"
    return V
