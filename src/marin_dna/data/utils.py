from itertools import pairwise

import numpy as np
import polars as pl
from Bio import SeqIO
from Bio.Seq import Seq

from marin_dna.data.intervals import GenomicSet

HUMAN_GENOME = "GCF_000001405.40"
STANDARD_CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]
ENHANCER_CRE_CLASSES = ["dELS", "pELS"]

DEFAULT_NCRNA_BIOTYPES = [
    "lnc_RNA",
    "miRNA",
    "snoRNA",
    "tRNA",
    "snRNA",
    "rRNA",
    "antisense_RNA",
    "ncRNA",
    "scRNA",
    "vault_RNA",
    "Y_RNA",
    "scaRNA",
    "RNase_P_RNA",
    "RNase_MRP_RNA",
    "telomerase_RNA",
    "SRP_RNA",
    "piRNA",  # C. elegans specific
]


def get_array_split_pairs(L: int, n_shards: int) -> list[tuple[int, int]]:
    """
    Calculates the (start, end) slice pairs exactly like np.array_split.

    Args:
        L (int): Total length of the array/DataFrame.
        n_shards (int): Number of shards to create.

    Returns:
        list[tuple]: A list of (start, end) tuples for slicing.
    """
    base_size = L // n_shards
    remainder = L % n_shards

    # Create a list of the *size* of each shard
    shard_sizes = [base_size + 1] * remainder + [base_size] * (n_shards - remainder)

    # Calculate the cumulative sum to get the slice indices
    # e.g., [0, 5, 10, 15, 19, 23]
    indices = np.cumsum([0] + shard_sizes)

    # Create and return the (start, end) pairs
    # e.g., [(0, 5), (5, 10), (10, 15), (15, 19), (19, 23)]
    return list(pairwise(indices))


def load_annotation(path: str) -> pl.DataFrame:
    """
    Load a GTF/GFF annotation file and convert to BED-like format.

    Args:
        path (str): Path to the GTF/GFF annotation file.

    Returns:
        pl.DataFrame: DataFrame with columns [chrom, source, feature, start, end,
                      score, strand, frame, attribute]. Start coordinates are
                      converted from 1-based (GTF) to 0-based (BED). Only features
                      with strand '+' or '-' are retained.
    """
    return (
        pl.read_csv(
            path,
            has_header=False,
            separator="\t",
            comment_prefix="#",
            new_columns=[
                "chrom",
                "source",
                "feature",
                "start",
                "end",
                "score",
                "strand",
                "frame",
                "attribute",
            ],
            schema_overrides={"chrom": pl.Utf8},
        )
        .with_columns(pl.col("start") - 1)  # gtf to bed conversion
        .filter(pl.col("strand").is_in(["+", "-"]))
    )


def get_mrna_exons(ann: pl.DataFrame) -> pl.DataFrame:
    """
    Extract mRNA exons from an annotation DataFrame.

    Args:
        ann (pl.DataFrame): Annotation DataFrame from load_annotation() with
                           columns including 'feature', 'attribute', 'chrom',
                           'start', 'end', and 'strand'.

    Returns:
        pl.DataFrame: DataFrame with columns [chrom, start, end, strand, transcript_id]
                     containing only mRNA exons. Supports both transcript_biotype
                     (e.g., human) and gbkey (e.g., some species) annotations.
    """
    return (
        ann.filter(pl.col("feature") == "exon")
        .with_columns(
            pl.col("attribute")
            .str.extract(r'transcript_id "(.*?)"')
            .alias("transcript_id"),
            pl.col("attribute")
            .str.extract(r'transcript_biotype "(.*?)"')
            .alias("transcript_biotype"),
            pl.col("attribute").str.extract(r'gbkey "(.*?)"').alias("gbkey"),
        )
        # most annotations use transcript_biotype (e.g. GCF_000001405.40 Homo sapiens),
        # but some use gbkey (e.g. GCF_000691845.1 Merops nubicus)
        .filter((pl.col("transcript_biotype") == "mRNA") | (pl.col("gbkey") == "mRNA"))
        .select(["chrom", "start", "end", "strand", "transcript_id"])
    )


def _get_cds_per_transcript(ann: pl.DataFrame) -> pl.DataFrame:
    """Extract CDS regions with transcript_id for UTR computation.

    Supports both standard annotations (feature="CDS") and non-standard formats
    like C. elegans where CDS is indicated via gbkey attribute.

    Args:
        ann: Annotation DataFrame from load_annotation().

    Returns:
        DataFrame with columns [chrom, start, end, strand, transcript_id]
        containing CDS regions.
    """
    return (
        ann.with_columns(
            pl.col("attribute").str.extract(r'gbkey "(.*?)"').alias("gbkey"),
            pl.col("attribute")
            .str.extract(r'transcript_id "(.*?)"')
            .alias("transcript_id"),
        )
        # Some genomes (e.g., C. elegans, Brugia malayi) use gene names in the
        # feature column instead of "CDS", so we also check gbkey attribute
        .filter((pl.col("feature") == "CDS") | (pl.col("gbkey") == "CDS"))
        .select(["chrom", "start", "end", "strand", "transcript_id"])
    )


def _filter_ncrna_exons(ann: pl.DataFrame) -> pl.DataFrame:
    """Filter ncRNA exons excluding pseudogenes and low quality entries.

    Args:
        ann: Annotation DataFrame from load_annotation().

    Returns:
        DataFrame with ncRNA exons passing quality filters.
    """
    return (
        ann.filter(pl.col("feature") == "exon")
        .with_columns(
            pl.col("attribute")
            .str.extract(r'transcript_id "(.*?)"')
            .alias("transcript_id"),
            pl.col("attribute")
            .str.extract(r'transcript_biotype "(.*?)"')
            .alias("transcript_biotype"),
            pl.col("attribute").str.extract(r'gbkey "(.*?)"').alias("gbkey"),
            pl.col("attribute")
            .str.extract(r'gene_biotype "(.*?)"')
            .alias("gene_biotype"),
            pl.col("attribute").str.contains(r'pseudo "true"').alias("is_pseudo"),
            pl.col("attribute").str.contains(r'partial "true"').alias("is_partial"),
            pl.col("attribute")
            .str.extract(r'description "(.*?)"')
            .alias("description"),
            pl.col("attribute").str.extract(r'product "(.*?)"').alias("product"),
        )
        # Filter to allowed ncRNA biotypes
        .filter(
            pl.col("transcript_biotype").is_in(DEFAULT_NCRNA_BIOTYPES)
            | pl.col("gbkey").is_in(DEFAULT_NCRNA_BIOTYPES)
        )
        # Exclude pseudogenes
        .filter(~pl.col("is_pseudo").fill_null(False))
        .filter(~pl.col("is_partial").fill_null(False))
        .filter(
            ~pl.col("transcript_biotype").fill_null("").str.contains("(?i)pseudogenic")
        )
        .filter(~pl.col("gene_biotype").fill_null("").str.contains("(?i)pseudogene"))
        .filter(pl.col("transcript_biotype").fill_null("") != "transcript")
        .filter(pl.col("transcript_biotype").fill_null("") != "primary_transcript")
        # Exclude NMD candidates
        .filter(~pl.col("description").fill_null("").str.contains("(?i)NMD candidate"))
        .filter(~pl.col("product").fill_null("").str.contains("(?i)NMD candidate"))
        # Exclude low quality
        .filter(~pl.col("product").fill_null("").str.contains("LOW QUALITY"))
    )


def _get_functional_transcript_exons(ann: pl.DataFrame) -> pl.DataFrame:
    """Extract exons from functional transcripts (mRNA + ncRNA).

    Returns exons from mRNA transcripts and functional ncRNA transcripts,
    excluding pseudogenes and precursor RNAs. Used for promoter computation.

    Args:
        ann: Annotation DataFrame from load_annotation().

    Returns:
        DataFrame with columns [chrom, start, end, strand, transcript_id]
        containing exons from functional transcripts.
    """
    mrna_exons = get_mrna_exons(ann)
    ncrna_exons = _filter_ncrna_exons(ann).select(
        ["chrom", "start", "end", "strand", "transcript_id"]
    )
    return pl.concat([mrna_exons, ncrna_exons])


ENSEMBL_EXCLUDED_TRANSCRIPT_BIOTYPES = [
    "retained_intron",
    "nonsense_mediated_decay",
    "protein_coding_CDS_not_defined",
    "processed_pseudogene",
    "transcribed_unprocessed_pseudogene",
    "transcribed_processed_pseudogene",
    "unprocessed_pseudogene",
    "unitary_pseudogene",
    "translated_processed_pseudogene",
    "TEC",
    "artifact",
    "non_stop_decay",
]


def get_exons(ann: pl.DataFrame) -> GenomicSet:
    """Extract all exon regions from an annotation DataFrame.

    Args:
        ann: Annotation DataFrame from load_annotation().

    Returns:
        GenomicSet containing merged exon regions.
    """
    return GenomicSet(ann.filter(pl.col("feature") == "exon"))


def _extract_transcript_biotype(exons: pl.DataFrame) -> pl.Series:
    """Parse the ``transcript_biotype`` attribute from each exon row."""
    return exons.select(
        pl.col("attribute")
        .str.extract(r'transcript_biotype "([^"]+)"')
        .alias("transcript_biotype")
    )["transcript_biotype"]


def get_ensembl_functional_exons(ann: pl.DataFrame) -> GenomicSet:
    """Extract exon regions from well-supported transcripts only.

    Excludes exons from transcript biotypes that represent annotation
    artifacts or non-functional transcripts (retained introns, NMD,
    pseudogenes, TEC, etc.). This avoids incorrectly marking enhancers
    as exonic based on questionable transcript annotations.

    Args:
        ann: Annotation DataFrame from load_annotation().

    Returns:
        GenomicSet containing merged exon regions from well-supported
        transcripts.
    """
    exons = ann.filter(pl.col("feature") == "exon")
    biotype = _extract_transcript_biotype(exons)
    return GenomicSet(
        exons.filter(~biotype.is_in(ENSEMBL_EXCLUDED_TRANSCRIPT_BIOTYPES))
    )


def get_exons_for_masking(ann: pl.DataFrame) -> GenomicSet:
    """Extract exons to exclude from enhancer scanning.

    When ``transcript_biotype`` is available (Ensembl-style GTFs), excludes
    retained intron, NMD, pseudogene, and other low-quality transcript exons
    so that those regions remain scannable for potential enhancers. Falls back
    to all exons for annotations without biotype information (e.g. NCBI RefSeq),
    where retained introns are not annotated anyway.

    Args:
        ann: Annotation DataFrame from load_annotation().

    Returns:
        GenomicSet of exon regions to mask during enhancer prediction.
    """
    exons = ann.filter(pl.col("feature") == "exon")
    if len(exons) == 0:
        return GenomicSet(exons)

    biotype = _extract_transcript_biotype(exons)

    if biotype.null_count() < len(biotype) // 2:
        # Null biotype rows are kept (treated as not-in-excluded) so we don't
        # silently drop exons in mixed annotations.
        is_excluded = biotype.is_in(ENSEMBL_EXCLUDED_TRANSCRIPT_BIOTYPES).fill_null(
            False
        )
        return GenomicSet(exons.filter(~is_excluded))

    return GenomicSet(exons)


def get_cds(ann: pl.DataFrame) -> GenomicSet:
    """Extract CDS regions from an annotation DataFrame.

    Supports both standard annotations (feature="CDS") and non-standard formats
    like C. elegans where CDS is indicated via gbkey attribute.

    Args:
        ann: Annotation DataFrame from load_annotation() with columns including
            'feature', 'chrom', 'start', 'end', 'attribute'.

    Returns:
        GenomicSet containing merged CDS regions.
    """
    # Some genomes (e.g., C. elegans, Brugia malayi) use gene names in the
    # feature column instead of "CDS", so we also check gbkey attribute
    return GenomicSet(
        ann.with_columns(
            pl.col("attribute").str.extract(r'gbkey "(.*?)"').alias("gbkey"),
        ).filter((pl.col("feature") == "CDS") | (pl.col("gbkey") == "CDS"))
    )


def get_promoters_from_exons(
    exons: pl.DataFrame,
    n_upstream: int,
    n_downstream: int,
) -> GenomicSet:
    """Extract promoter regions from exon DataFrame.

    Args:
        exons: Exon DataFrame with columns [chrom, start, end, strand, transcript_id]
            from get_mrna_exons().
        n_upstream: Number of bases upstream of TSS to include.
        n_downstream: Number of bases downstream of TSS to include.

    Returns:
        GenomicSet containing merged promoter regions. For '+' strand, promoter is
        [TSS - n_upstream, TSS + n_downstream]. For '-' strand, promoter is
        [TSS - n_downstream, TSS + n_upstream].
    """
    return GenomicSet(
        exons.group_by("transcript_id")
        .agg(
            pl.col("chrom").first(),
            pl.col("start").min(),
            pl.col("end").max(),
            pl.col("strand").first(),
        )
        .with_columns(
            pl.when(pl.col("strand") == "+")
            .then(pl.col("start") - n_upstream)
            .otherwise(pl.col("end") - n_downstream)
            .alias("start")
        )
        .with_columns((pl.col("start") + n_upstream + n_downstream).alias("end"))
    )


def get_promoters(
    ann: pl.DataFrame,
    n_upstream: int,
    n_downstream: int,
    mRNA_only: bool = False,
    within_bounds: GenomicSet | None = None,
) -> GenomicSet:
    """Extract promoter regions from annotation DataFrame.

    Args:
        ann: Annotation DataFrame from load_annotation().
        n_upstream: Number of bases upstream of TSS to include.
        n_downstream: Number of bases downstream of TSS to include.
        mRNA_only: If True, only include promoters from protein-coding mRNA
            transcripts. If False (default), include promoters from mRNA and
            functional ncRNA transcripts (excluding pseudogenes and precursors).
        within_bounds: Optional GenomicSet to intersect with. If provided, the
            result will be clipped to these boundaries (e.g., chromosome bounds).

    Returns:
        GenomicSet containing merged promoter regions.
    """
    if mRNA_only:
        exons = get_mrna_exons(ann)
    else:
        exons = _get_functional_transcript_exons(ann)
    result = get_promoters_from_exons(exons, n_upstream, n_downstream)
    if within_bounds is not None:
        result = result & within_bounds
    return result


def get_5_prime_utr(ann: pl.DataFrame) -> GenomicSet:
    """Extract 5' UTR regions from an annotation DataFrame.

    Computes 5' UTR by finding exon portions upstream of CDS start.
    For + strand, this is the exon region before CDS start.
    For - strand, this is the exon region after CDS end (genomically).

    Args:
        ann: Annotation DataFrame from load_annotation().

    Returns:
        GenomicSet containing merged 5' UTR regions.
    """
    mrna_exons = get_mrna_exons(ann)
    cds = _get_cds_per_transcript(ann)

    # Get CDS boundaries per transcript
    cds_bounds = cds.group_by("transcript_id").agg(
        pl.col("start").min().alias("cds_start"),
        pl.col("end").max().alias("cds_end"),
    )

    # Join exons with CDS boundaries
    exons_with_cds = mrna_exons.join(cds_bounds, on="transcript_id", how="inner")

    # Compute 5' UTR portions
    # + strand: exon region before CDS start
    # - strand: exon region after CDS end (genomically)
    utr = GenomicSet(
        exons_with_cds.with_columns(
            pl.when(pl.col("strand") == "+")
            .then(pl.col("start"))
            .otherwise(pl.max_horizontal("start", "cds_end"))
            .alias("utr_start"),
            pl.when(pl.col("strand") == "+")
            .then(pl.min_horizontal("end", "cds_start"))
            .otherwise(pl.col("end"))
            .alias("utr_end"),
        )
        .filter(pl.col("utr_end") > pl.col("utr_start"))
        .select(
            pl.col("chrom"),
            pl.col("utr_start").alias("start"),
            pl.col("utr_end").alias("end"),
        )
    )
    return utr


def get_3_prime_utr(ann: pl.DataFrame) -> GenomicSet:
    """Extract 3' UTR regions from an annotation DataFrame.

    Computes 3' UTR by finding exon portions downstream of CDS end.
    For + strand, this is the exon region after CDS end.
    For - strand, this is the exon region before CDS start (genomically).

    Args:
        ann: Annotation DataFrame from load_annotation().

    Returns:
        GenomicSet containing merged 3' UTR regions.
    """
    mrna_exons = get_mrna_exons(ann)
    cds = _get_cds_per_transcript(ann)

    # Get CDS boundaries per transcript
    cds_bounds = cds.group_by("transcript_id").agg(
        pl.col("start").min().alias("cds_start"),
        pl.col("end").max().alias("cds_end"),
    )

    # Join exons with CDS boundaries
    exons_with_cds = mrna_exons.join(cds_bounds, on="transcript_id", how="inner")

    # Compute 3' UTR portions
    # + strand: exon region after CDS end
    # - strand: exon region before CDS start (genomically)
    utr = GenomicSet(
        exons_with_cds.with_columns(
            pl.when(pl.col("strand") == "+")
            .then(pl.max_horizontal("start", "cds_end"))
            .otherwise(pl.col("start"))
            .alias("utr_start"),
            pl.when(pl.col("strand") == "+")
            .then(pl.col("end"))
            .otherwise(pl.min_horizontal("end", "cds_start"))
            .alias("utr_end"),
        )
        .filter(pl.col("utr_end") > pl.col("utr_start"))
        .select(
            pl.col("chrom"),
            pl.col("utr_start").alias("start"),
            pl.col("utr_end").alias("end"),
        )
    )
    return utr


def get_upstream_of_CDS(
    ann: pl.DataFrame,
    dist: int,
    within_bounds: GenomicSet | None = None,
) -> GenomicSet:
    """Extract regions upstream of CDS start (5' direction).

    Returns fixed-size regions upstream of CDS boundaries per transcript.
    This provides an alternative to annotated 5' UTRs that may be more
    robust for non-model organisms with incomplete UTR annotations.

    Args:
        ann: Annotation DataFrame from load_annotation().
        dist: Number of bases upstream of CDS start to include.
        within_bounds: Optional GenomicSet to clip results to.

    Returns:
        GenomicSet containing merged upstream regions. For + strand, this is
        [cds_start - dist, cds_start]. For - strand, this is [cds_end, cds_end + dist].
    """
    cds = _get_cds_per_transcript(ann)

    # Get CDS boundaries per transcript
    cds_bounds = cds.group_by("transcript_id").agg(
        pl.col("chrom").first(),
        pl.col("start").min().alias("cds_start"),
        pl.col("end").max().alias("cds_end"),
        pl.col("strand").first(),
    )

    # Compute strand-aware upstream regions
    # + strand: upstream is [cds_start - dist, cds_start]
    # - strand: upstream is [cds_end, cds_end + dist]
    result = GenomicSet(
        cds_bounds.with_columns(
            pl.when(pl.col("strand") == "+")
            .then(pl.col("cds_start") - dist)
            .otherwise(pl.col("cds_end"))
            .alias("start"),
            pl.when(pl.col("strand") == "+")
            .then(pl.col("cds_start"))
            .otherwise(pl.col("cds_end") + dist)
            .alias("end"),
        ).select(["chrom", "start", "end"])
    )

    if within_bounds is not None:
        result = result & within_bounds
    return result


def get_downstream_of_CDS(
    ann: pl.DataFrame,
    dist: int,
    within_bounds: GenomicSet | None = None,
) -> GenomicSet:
    """Extract regions downstream of CDS end (3' direction).

    Returns fixed-size regions downstream of CDS boundaries per transcript.
    This provides an alternative to annotated 3' UTRs that may be more
    robust for non-model organisms with incomplete UTR annotations.

    Args:
        ann: Annotation DataFrame from load_annotation().
        dist: Number of bases downstream of CDS end to include.
        within_bounds: Optional GenomicSet to clip results to.

    Returns:
        GenomicSet containing merged downstream regions. For + strand, this is
        [cds_end, cds_end + dist]. For - strand, this is [cds_start - dist, cds_start].
    """
    cds = _get_cds_per_transcript(ann)

    # Get CDS boundaries per transcript
    cds_bounds = cds.group_by("transcript_id").agg(
        pl.col("chrom").first(),
        pl.col("start").min().alias("cds_start"),
        pl.col("end").max().alias("cds_end"),
        pl.col("strand").first(),
    )

    # Compute strand-aware downstream regions
    # + strand: downstream is [cds_end, cds_end + dist]
    # - strand: downstream is [cds_start - dist, cds_start]
    result = GenomicSet(
        cds_bounds.with_columns(
            pl.when(pl.col("strand") == "+")
            .then(pl.col("cds_end"))
            .otherwise(pl.col("cds_start") - dist)
            .alias("start"),
            pl.when(pl.col("strand") == "+")
            .then(pl.col("cds_end") + dist)
            .otherwise(pl.col("cds_start"))
            .alias("end"),
        ).select(["chrom", "start", "end"])
    )

    if within_bounds is not None:
        result = result & within_bounds
    return result


def get_ncrna_exons(ann: pl.DataFrame) -> GenomicSet:
    """Extract functional ncRNA exons from an annotation DataFrame.

    Extracts exons from non-coding RNA transcripts, excluding pseudogenes,
    precursor RNAs, NMD candidates, partial annotations, and low quality entries.
    Uses DEFAULT_NCRNA_BIOTYPES which includes lnc_RNA, miRNA, snoRNA, tRNA,
    snRNA, rRNA, and other functional ncRNA types.

    Args:
        ann: Annotation DataFrame from load_annotation().

    Returns:
        GenomicSet containing merged ncRNA exon regions.
    """
    return GenomicSet(_filter_ncrna_exons(ann))


def load_fasta(path: str) -> pl.DataFrame:
    """Load FASTA records into a Polars DataFrame.

    Args:
        path: Path to the FASTA file.

    Returns:
        DataFrame with String columns id and seq.
    """
    with open(path) as handle:
        return pl.DataFrame(
            [(record.id, str(record.seq)) for record in SeqIO.parse(handle, "fasta")],
            schema={"id": pl.String, "seq": pl.String},
            orient="row",
        )


def add_rc(data: pl.DataFrame) -> pl.DataFrame:
    """Append reverse-complement rows to a sequence DataFrame.

    Args:
        data: Polars DataFrame with String columns id and seq.

    Returns:
        Original sequences with a _+ ID suffix followed by reverse complements
        with a _- suffix.
    """
    missing = {"id", "seq"} - set(data.columns)
    if missing:
        raise ValueError(f"sequence DataFrame is missing columns: {sorted(missing)}")
    if data.schema["id"] != pl.String or data.schema["seq"] != pl.String:
        raise TypeError("id and seq must have String dtypes")

    forward = data.with_columns((pl.col("id") + "_+").alias("id"))
    reverse = data.with_columns(
        (pl.col("id") + "_-").alias("id"),
        pl.col("seq")
        .map_elements(
            lambda sequence: str(Seq(sequence).reverse_complement()),
            return_dtype=pl.String,
        )
        .alias("seq"),
    )
    return pl.concat([forward, reverse])
