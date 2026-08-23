"""mRNA-TSS proximity bands for projected vertebrate anchors."""

from pathlib import Path

import polars as pl

from marin_dna.data.intervals import GenomicSet
from marin_dna.data.utils import get_promoters_from_exons, load_annotation


def get_ensembl_protein_coding_exons(ann: pl.DataFrame) -> pl.DataFrame:
    """Extract protein-coding (mRNA) exons from an Ensembl GTF DataFrame.

    Args:
        ann: Annotation DataFrame from ``marin_dna.data.utils.load_annotation``.

    Returns:
        DataFrame with columns [chrom, start, end, strand, transcript_id]
        containing exons of transcripts with biotype ``"protein_coding"``.
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
        )
        .filter(pl.col("transcript_biotype") == "protein_coding")
        .select(["chrom", "start", "end", "strand", "transcript_id"])
    )


def write_mrna_tss_band_bed(
    gtf_path: str | Path, flank: int, out_bed: str | Path
) -> int:
    """Compute [TSS − flank, TSS + flank] band over Ensembl protein-coding
    transcripts; write merged BED. Returns row count of the written BED.
    """
    assert flank > 0, f"flank must be positive, got {flank}"
    ann = load_annotation(str(gtf_path))
    exons = get_ensembl_protein_coding_exons(ann)
    assert len(exons) > 0, "no protein_coding exons found — wrong GTF flavor?"
    band: GenomicSet = get_promoters_from_exons(
        exons, n_upstream=flank, n_downstream=flank
    )
    band.write_bed(str(out_bed))
    with open(out_bed) as fh:
        return sum(1 for _ in fh)
