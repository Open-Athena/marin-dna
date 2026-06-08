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
    return facts


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
    assert set(out["author_func_class"].unique()) <= {"FUNC", "INT", "LOF"}, (
        f"BRCA1: unexpected func.class values: {set(out['author_func_class'].unique())}"
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
    chrom_num = nt.str.extract(r"^NC_0*(\d+)\.", 1).cast(pl.Int64)
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
            function_score=pl.col("author_score").cast(pl.Float64),
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
            function_score=pl.col("author_score").cast(pl.Float64),
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
    lift: bool,
    name: str = "",
) -> pl.DataFrame:
    """Liftover (optional) + ref/alt validation + consequence/distance annotation +
    HIGH-impact ``exclude_consequences`` drop, with **no** matching or subsampling.

    Mirrors :func:`marin_dna.pipelines.evals.dart_eval.annotate_variants` but (a)
    drops ``exclude_consequences`` (the QTL path keeps them; SGE drops them), (b)
    carries no signed ``effect`` (SGE has a direction-tied function score, not a
    QTL effect), and (c) **asserts zero ref/alt swaps**: an SGE function score is
    tied to the ref->alt substitution as the author defined it, so a swap (author
    ref != genome) signals a coordinate/strand/build problem, not a benign
    re-orientation.

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
    assert n_ref >= 0.9 * n_lift, (
        f"check_ref_alt kept only {n_ref}/{n_lift} variants for {name!r} — suspect a "
        "coordinate-base (0- vs 1-based) or genome-build mismatch"
    )
    V = attach_per_chrom_consequences(V, consequence_paths, chroms)
    assert V["consequence"].null_count() == 0, f"{name}: variants with null consequence"
    # Drop the trivially-deleterious HIGH-impact consequences (canonical splice,
    # nonsense, frameshift, …) before distance recategorization.
    n_pre_excl = V.height
    V = V.filter(~pl.col("consequence").is_in(exclude_consequences))
    print(
        f"[sge annotate {name}] exclude_consequences dropped "
        f"{n_pre_excl - V.height} HIGH-impact variants ({V.height} kept)"
    )
    assert V.height > 0, f"{name}: all variants dropped by exclude_consequences"
    V = (
        V.pipe(add_exon, exon_pc, exon_nc, exon_proximal_dist)
        .pipe(add_tss, tss_pc, tss_nc, tss_proximal_dist)
        .sort(COORDINATES)
    )
    assert (V["pos"] > 0).all(), f"{name}: non-positive positions after annotation"
    assert V["consequence_final"].null_count() == 0, (
        f"{name}: null consequence_final after annotation"
    )
    return V
