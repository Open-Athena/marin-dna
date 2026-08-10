# EDA Rules for Mendelian and Complex Trait Variants in 3' UTR and ncRNA Regions
# ================================================================================
#
# Goal: Analyze patterns in causal variants located in 3' UTR and ncRNA regions
# to inform better data curation strategies for training DNA language models.
#
# Key Hypotheses:
# 1. 3' UTR: Causal variants may be enriched near the CDS
# 2. ncRNA: Certain types (e.g., miRNA, snoRNA) may have more causal variants
# 3. Size patterns: Causal variants may be enriched in certain ncRNA size ranges
# 4. Mendelian vs Complex traits may show different patterns
#
# Data Sources:
# - TraitGym Mendelian v2: hf://datasets/gonzalobenegas/traitgym-mendelian-v2/test.parquet
# - TraitGym Complex Traits: hf://datasets/songlab/TraitGym/complex_traits_matched_9/test.parquet
# - IMPORTANT: Filter to label=True (causal variants only)
#
# Chromosome Mapping:
# - TraitGym uses plain chromosome names (1, 2, ..., X, Y)
# - Our annotations use RefSeq names (NC_000001.11, NC_000002.12, ...)
# - We convert TraitGym -> UCSC (chr1) -> RefSeq for variant-annotation matching
#
# Coordinate System:
# - TraitGym uses VCF-style 1-based coordinates (pos column)
# - Our annotations use 0-based BED-style coordinates
# - We convert TraitGym pos to 0-based: start = pos - 1, end = pos
#
# =============================================================================
# Variant Coverage Analysis
# =============================================================================
# Not all causal variants from TraitGym can be analyzed due to annotation source
# mismatch. TraitGym uses Ensembl VEP for consequence annotation, while we use
# RefSeq (GCF_000001405.40). Variants are lost when they don't overlap with our
# RefSeq-derived 3' UTR or ncRNA exon annotations.
#
# Raw causal variant counts (unique chrom, pos, ref, alt with label=True):
# Note: Some positions have multiple causal alleles (multi-allelic sites).
#
#   Dataset         | 3' UTR variants (positions) | ncRNA variants (positions)
#   ----------------|-----------------------------|--------------------------
#   mendelian_v2    | 110 (101)                   | 159 (146)
#   complex_traits  | 52 (52)                     | 69 (69)
#
# After annotation overlap (variants / positions successfully analyzed):
#
#   Dataset         | 3' UTR analyzed        | 3' UTR lost | ncRNA analyzed         | ncRNA lost
#   ----------------|------------------------|-------------|------------------------|-------------
#   mendelian_v2    | 102 (92.7%) / 93 (92.1%) | 8 / 8       | 115 (72.3%) / 106 (72.6%) | 44 / 40
#   complex_traits  | 45 (86.5%) / 45 (86.5%)  | 7 / 7       | 20 (29.0%) / 20 (29.0%)   | 49 / 49
#
# Key observations:
# - 3' UTR coverage is good (86-92%), likely because UTR boundaries are more
#   consistent between Ensembl and RefSeq annotations.
# - ncRNA coverage is lower, especially for complex_traits (29%). This is because:
#   1. ncRNA annotations differ significantly between Ensembl and RefSeq
#   2. Some VEP "non_coding_transcript_exon_variant" may fall in intronic regions
#      of RefSeq ncRNA genes, or in transcripts not annotated in RefSeq
#   3. lncRNA annotations are particularly inconsistent across databases
#
# The analyzed variants are still representative for understanding patterns,
# but absolute counts should be interpreted with this coverage in mind.


# Dataset configuration: (repo_id, filename) tuples for huggingface_hub
EDA_DATASETS = {
    "mendelian_v2": ("gonzalobenegas/traitgym-mendelian-v2", "test.parquet"),
    "complex_traits": ("songlab/TraitGym", "complex_traits_matched_9/test.parquet"),
}


# =============================================================================
# Rule: Download datasets from HuggingFace
# =============================================================================
# Note: Some datasets (e.g., complex_traits) may require HuggingFace authentication.
# Use `huggingface-cli login` to set up credentials if needed.
rule eda_download_dataset:
    output:
        "results/eda/{dataset}/raw.parquet",
    params:
        repo_id=lambda wc: EDA_DATASETS[wc.dataset][0],
        filename=lambda wc: EDA_DATASETS[wc.dataset][1],
    run:
        from huggingface_hub import hf_hub_download
        import os
        import shutil

        # Download using huggingface_hub (handles authentication)
        downloaded_path = hf_hub_download(
            repo_id=params.repo_id,
            filename=params.filename,
            repo_type="dataset",
        )
        # Copy to output location
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        shutil.copy(downloaded_path, output[0])


# =============================================================================
# Rule: Extract 3' UTR annotations with transcript and gene info
# =============================================================================
rule eda_extract_3_prime_utr_annotations:
    input:
        annotation="results/annotation/GCF_000001405.40.gtf.gz",
    output:
        "results/eda/annotation/3_prime_utr_transcripts.parquet",
    run:
        from marin_dna.data.eda import extract_3_prime_utr_annotations

        ann = load_annotation(input.annotation)
        utr_annotations = extract_3_prime_utr_annotations(ann)
        utr_annotations.write_parquet(output[0])


# =============================================================================
# Rule: Extract ALL ncRNA annotations with biotype info (not filtered)
# =============================================================================
# Purpose: Get ALL ncRNA exon regions for overlap, then mark which biotypes
# are in our DEFAULT_NCRNA_BIOTYPES for comparison
rule eda_extract_ncrna_annotations_all:
    input:
        annotation="results/annotation/GCF_000001405.40.gtf.gz",
    output:
        "results/eda/annotation/ncrna_transcripts_all.parquet",
    run:
        from marin_dna.data.utils import DEFAULT_NCRNA_BIOTYPES

        ann = load_annotation(input.annotation)
        # Get ALL exons with biotype info (not filtered to specific biotypes)
        all_exons = ann.filter(pl.col("feature") == "exon").with_columns(
            pl.col("attribute")
            .str.extract(r'transcript_id "(.*?)"')
            .alias("transcript_id"),
            pl.col("attribute")
            .str.extract(r'transcript_biotype "(.*?)"')
            .alias("transcript_biotype"),
            pl.col("attribute").str.extract(r'gbkey "(.*?)"').alias("gbkey"),
            pl.col("attribute").str.extract(r'gene_id "(.*?)"').alias("gene_id"),
            pl.col("attribute").str.extract(r'gene "(.*?)"').alias("gene_name"),
        )
        # Determine biotype (prefer transcript_biotype, fallback to gbkey)
        all_exons = all_exons.with_columns(
            pl.when(pl.col("transcript_biotype").is_not_null())
            .then(pl.col("transcript_biotype"))
            .otherwise(pl.col("gbkey"))
            .alias("biotype")
        )
        # Filter to non-mRNA transcripts (exclude protein-coding)
        # Keep anything that's not mRNA and has a biotype
        ncrna_exons = all_exons.filter(
            pl.col("biotype").is_not_null()
            & (pl.col("transcript_biotype").fill_null("") != "mRNA")
            & (pl.col("gbkey").fill_null("") != "mRNA")
        )
        # Mark if biotype is in our default list
        ncrna_exons = ncrna_exons.with_columns(
            pl.col("biotype")
            .is_in(DEFAULT_NCRNA_BIOTYPES)
            .alias("in_default_biotypes")
        )
        # Select final columns
        result = ncrna_exons.select(
            [
                "chrom",
                "start",
                "end",
                "strand",
                "transcript_id",
                "gene_id",
                "gene_name",
                "biotype",
                "in_default_biotypes",
            ]
        )
        result.write_parquet(output[0])


# =============================================================================
# Rule: Extract CDS annotations for distance calculations
# =============================================================================
rule eda_extract_cds_annotations:
    input:
        annotation="results/annotation/GCF_000001405.40.gtf.gz",
    output:
        "results/eda/annotation/cds_regions.parquet",
    run:
        from marin_dna.data.eda import extract_cds_annotations

        ann = load_annotation(input.annotation)
        cds = extract_cds_annotations(ann)
        cds.write_parquet(output[0])


# =============================================================================
# Rule: Extract mRNA exon annotations for distance calculations
# =============================================================================
rule eda_extract_mrna_exon_annotations:
    input:
        annotation="results/annotation/GCF_000001405.40.gtf.gz",
    output:
        "results/eda/annotation/mrna_exons.parquet",
    run:
        from marin_dna.data.eda import extract_mrna_exon_annotations

        ann = load_annotation(input.annotation)
        mrna_exons = extract_mrna_exon_annotations(ann)
        mrna_exons.write_parquet(output[0])


# =============================================================================
# Rule: Filter 3' UTR variants from dataset
# =============================================================================
# Coverage: ~86-92% of raw 3' UTR variants overlap with RefSeq annotations.
# Lost variants (~8-14%) likely fall in UTR regions annotated differently
# between Ensembl (used by TraitGym VEP) and RefSeq.
rule eda_filter_3_prime_utr_variants:
    input:
        variants="results/eda/{dataset}/raw.parquet",
        utr_annotations="results/eda/annotation/3_prime_utr_transcripts.parquet",
        chrom_mapping="config/human_chrom_mapping.tsv",
    output:
        "results/eda/{dataset}/3_prime_utr_variants.parquet",
    run:
        import bioframe as bf

        # Load chromosome mapping (UCSC -> RefSeq)
        chrom_map = pl.read_csv(input.chrom_mapping, separator="\t")
        ucsc_to_refseq = dict(zip(chrom_map["ucsc"], chrom_map["refseq"]))
        # Load and filter variants
        variants = pl.read_parquet(input.variants)
        # Filter to 3' UTR variants only AND causal variants (label=True)
        variants = variants.filter(
            (pl.col("consequence") == "3_prime_UTR_variant")
            & (pl.col("label") == True)
        )
        # Convert chromosome names to RefSeq
        # TraitGym uses plain numbers (1, 2, ..., X, Y)
        variants = variants.with_columns(
            ("chr" + pl.col("chrom"))
            .replace_strict(ucsc_to_refseq, default=None)
            .alias("chrom_refseq")
        ).filter(pl.col("chrom_refseq").is_not_null())
        # Create variant intervals (VCF 1-based -> BED 0-based)
        variants = variants.with_columns(
            (pl.col("pos") - 1).alias("start"),
            pl.col("pos").alias("end"),
        )
        # Load 3' UTR annotations
        utr = pl.read_parquet(input.utr_annotations)
        # Convert to pandas for bioframe
        variants_df = variants.select(
            [
                pl.col("chrom_refseq").alias("chrom"),
                "start",
                "end",
                "pos",
                "ref",
                "alt",
                "label",
                "consequence",
            ]
        ).to_pandas()
        utr_df = utr.select(
            [
                "chrom",
                "start",
                "end",
                "strand",
                "transcript_id",
                "cds_start",
                "cds_end",
                "gene_id",
                "gene_name",
            ]
        ).to_pandas()
        # Perform overlap
        overlaps_df = bf.overlap(
            variants_df, utr_df, how="inner", suffixes=("", "_utr")
        )
        # Convert back to polars
        overlaps = pl.from_pandas(overlaps_df)
        # Rename columns
        overlaps = overlaps.rename(
            {
                "strand_utr": "strand",
                "transcript_id_utr": "transcript_id",
                "cds_start_utr": "cds_start",
                "cds_end_utr": "cds_end",
                "gene_id_utr": "gene_id",
                "gene_name_utr": "gene_name",
            }
        )
        # Compute distance to CDS
        overlaps = overlaps.with_columns(
            pl.when(pl.col("strand") == "+")
            .then(pl.col("start") - pl.col("cds_end"))
            .otherwise(pl.col("cds_start") - pl.col("start") - 1)
            .alias("distance_to_cds")
        )
        overlaps.write_parquet(output[0])


# =============================================================================
# Rule: Filter ncRNA variants from dataset (using ALL biotypes)
# =============================================================================
# Coverage: Only ~29-73% of raw ncRNA variants overlap with RefSeq annotations.
# The high loss rate (especially for complex_traits at 71%) is due to:
# 1. ncRNA annotations differ significantly between Ensembl and RefSeq
# 2. Many lncRNAs in Ensembl are not annotated in RefSeq
# 3. Some VEP "non_coding_transcript_exon_variant" may be in intronic regions
#    of RefSeq ncRNA genes or in transcripts absent from RefSeq
rule eda_filter_ncrna_variants:
    input:
        variants="results/eda/{dataset}/raw.parquet",
        ncrna_annotations="results/eda/annotation/ncrna_transcripts_all.parquet",
        chrom_mapping="config/human_chrom_mapping.tsv",
    output:
        "results/eda/{dataset}/ncrna_variants.parquet",
    run:
        import bioframe as bf

        # Load chromosome mapping (UCSC -> RefSeq)
        chrom_map = pl.read_csv(input.chrom_mapping, separator="\t")
        ucsc_to_refseq = dict(zip(chrom_map["ucsc"], chrom_map["refseq"]))
        # Load and filter variants
        variants = pl.read_parquet(input.variants)
        # Filter to non-coding transcript exon variants AND causal (label=True)
        variants = variants.filter(
            (pl.col("consequence") == "non_coding_transcript_exon_variant")
            & (pl.col("label") == True)
        )
        # Convert chromosome names to RefSeq
        variants = variants.with_columns(
            ("chr" + pl.col("chrom"))
            .replace_strict(ucsc_to_refseq, default=None)
            .alias("chrom_refseq")
        ).filter(pl.col("chrom_refseq").is_not_null())
        # Create variant intervals (VCF 1-based -> BED 0-based)
        variants = variants.with_columns(
            (pl.col("pos") - 1).alias("start"),
            pl.col("pos").alias("end"),
        )
        # Load ncRNA annotations
        ncrna = pl.read_parquet(input.ncrna_annotations)
        # Convert to pandas for bioframe
        variants_df = variants.select(
            [
                pl.col("chrom_refseq").alias("chrom"),
                "start",
                "end",
                "pos",
                "ref",
                "alt",
                "label",
                "consequence",
            ]
        ).to_pandas()
        ncrna_df = ncrna.select(
            [
                "chrom",
                "start",
                "end",
                "strand",
                "transcript_id",
                "gene_id",
                "gene_name",
                "biotype",
                "in_default_biotypes",
            ]
        ).to_pandas()
        # Perform overlap
        overlaps_df = bf.overlap(
            variants_df, ncrna_df, how="inner", suffixes=("", "_ncrna")
        )
        # Convert back to polars
        overlaps = pl.from_pandas(overlaps_df)
        # Rename columns
        overlaps = overlaps.rename(
            {
                "strand_ncrna": "strand",
                "transcript_id_ncrna": "transcript_id",
                "gene_id_ncrna": "gene_id",
                "gene_name_ncrna": "gene_name",
                "biotype_ncrna": "biotype",
                "in_default_biotypes_ncrna": "in_default_biotypes",
            }
        )
        overlaps.write_parquet(output[0])


# =============================================================================
# Rule: Extract genomic distance to CDS for 3' UTR variants
# =============================================================================
# Genomic distance includes introns if the 3' UTR spans multiple exons.
# Note: Groups by (chrom, pos), collapsing multi-allelic sites to one position.
rule eda_analyze_3_prime_utr_distance:
    input:
        "results/eda/{dataset}/3_prime_utr_variants.parquet",
    output:
        "results/eda/{dataset}/analysis/3_prime_utr_distance.parquet",
    run:
        df = pl.read_parquet(input[0])
        # Deduplicate by position: use minimum distance to CDS
        # (collapses multi-allelic sites and multi-transcript overlaps)
        df = df.group_by(["chrom", "pos"]).agg(pl.col("distance_to_cds").min())
        # Save raw distances for histogram plotting
        df.write_parquet(output[0])


# =============================================================================
# Rule: Extract mRNA distance to CDS for 3' UTR variants (excluding introns)
# =============================================================================
# mRNA distance counts only exonic bases between CDS and variant position.
# This is the distance in the mature transcript, excluding intronic sequences.
rule eda_analyze_3_prime_utr_mrna_distance:
    input:
        variants="results/eda/{dataset}/3_prime_utr_variants.parquet",
        utr_annotations="results/eda/annotation/3_prime_utr_transcripts.parquet",
    output:
        "results/eda/{dataset}/analysis/3_prime_utr_mrna_distance.parquet",
    run:
        from marin_dna.data.eda import compute_mrna_distances_for_variants

        variants = pl.read_parquet(input.variants)
        utr_ann = pl.read_parquet(input.utr_annotations)
        result_df = compute_mrna_distances_for_variants(variants, utr_ann)
        result_df.write_parquet(output[0])


# =============================================================================
# Rule: Analyze causal variant distribution by ncRNA type
# =============================================================================
rule eda_analyze_ncrna_types:
    input:
        "results/eda/{dataset}/ncrna_variants.parquet",
    output:
        "results/eda/{dataset}/analysis/ncrna_types.parquet",
    run:
        df = pl.read_parquet(input[0])
        # Count unique variants per biotype (deduplicate by chrom, pos)
        # Also track if biotype is in default list
        type_stats = (
            df.group_by(["biotype", "in_default_biotypes"])
            .agg(pl.struct("chrom", "pos").n_unique().alias("n_variants"))
            .sort("n_variants", descending=True)
        )
        # Add percentage
        total = type_stats["n_variants"].sum()
        type_stats = type_stats.with_columns(
            (pl.col("n_variants") / total * 100).alias("pct_variants")
        )
        type_stats.write_parquet(output[0])


# =============================================================================
# Rule: Extract raw ncRNA sizes for variants (for histograms)
# =============================================================================
rule eda_analyze_ncrna_sizes:
    input:
        variants="results/eda/{dataset}/ncrna_variants.parquet",
        annotations="results/eda/annotation/ncrna_transcripts_all.parquet",
    output:
        "results/eda/{dataset}/analysis/ncrna_sizes.parquet",
    run:
        variants = pl.read_parquet(input.variants)
        annotations = pl.read_parquet(input.annotations)
        # Calculate transcript size (sum of exon lengths per transcript)
        tx_sizes = annotations.group_by("transcript_id").agg(
            (pl.col("end") - pl.col("start")).sum().alias("transcript_size"),
            pl.col("biotype").first(),
            pl.col("in_default_biotypes").first(),
        )
        # Join variants with transcript sizes
        variants_with_size = variants.join(
            tx_sizes,
            on="transcript_id",
            how="left",
            suffix="_tx",
        )
        # Deduplicate: use minimum transcript size per variant
        variants_with_size = variants_with_size.group_by(["chrom", "pos"]).agg(
            pl.col("transcript_size").min()
        )
        # Save raw sizes for histogram plotting
        variants_with_size.write_parquet(output[0])


# =============================================================================
# Rule: Analyze ncRNA variant distance to nearest CDS and mRNA
# =============================================================================
# For each ncRNA variant, compute the genomic distance to:
# 1. Nearest CDS region (protein-coding sequence)
# 2. Nearest mRNA exon (any part of protein-coding transcript)
rule eda_analyze_ncrna_distance_to_coding:
    input:
        variants="results/eda/{dataset}/ncrna_variants.parquet",
        cds="results/eda/annotation/cds_regions.parquet",
        mrna="results/eda/annotation/mrna_exons.parquet",
    output:
        "results/eda/{dataset}/analysis/ncrna_distance_to_coding.parquet",
    run:
        import bioframe as bf

        variants = pl.read_parquet(input.variants)
        cds = pl.read_parquet(input.cds)
        mrna = pl.read_parquet(input.mrna)
        # Deduplicate variants by position
        variants_unique = variants.select(["chrom", "pos"]).unique()
        variants_unique = variants_unique.with_columns(
            (pl.col("pos") - 1).alias("start"),
            pl.col("pos").alias("end"),
        )
        # Convert to pandas for bioframe
        var_df = variants_unique.to_pandas()
        cds_df = cds.select(["chrom", "start", "end"]).unique().to_pandas()
        mrna_df = mrna.select(["chrom", "start", "end"]).unique().to_pandas()
        # Find nearest CDS for each variant
        nearest_cds = bf.closest(var_df, cds_df, suffixes=("", "_cds"))
        nearest_cds = pl.from_pandas(nearest_cds)
        # Compute distance to CDS (0 if overlapping)
        nearest_cds = nearest_cds.with_columns(
            pl.max_horizontal(
                pl.lit(0),
                pl.max_horizontal(
                    pl.col("start_cds") - pl.col("end"),
                    pl.col("start") - pl.col("end_cds"),
                ),
            ).alias("distance_to_cds")
        )
        # Find nearest mRNA exon for each variant
        nearest_mrna = bf.closest(var_df, mrna_df, suffixes=("", "_mrna"))
        nearest_mrna = pl.from_pandas(nearest_mrna)
        # Compute distance to mRNA (0 if overlapping)
        nearest_mrna = nearest_mrna.with_columns(
            pl.max_horizontal(
                pl.lit(0),
                pl.max_horizontal(
                    pl.col("start_mrna") - pl.col("end"),
                    pl.col("start") - pl.col("end_mrna"),
                ),
            ).alias("distance_to_mrna")
        )
        # Combine results
        result = nearest_cds.select(["chrom", "pos", "distance_to_cds"]).join(
            nearest_mrna.select(["chrom", "pos", "distance_to_mrna"]),
            on=["chrom", "pos"],
        )
        result.write_parquet(output[0])


# =============================================================================
# Plot: 3' UTR distance histograms (genomic, mRNA_min, mRNA_max)
# =============================================================================
# Each distance type gets both linear and log scale histograms.
# - genomic: distance including introns (uses max across isoforms)
# - mrna_min: mRNA distance for closest isoform to CDS
# - mrna_max: mRNA distance for furthest isoform from CDS
rule eda_plot_3_prime_utr_distance:
    input:
        "results/eda/{dataset}/analysis/3_prime_utr_mrna_distance.parquet",
    output:
        genomic_linear="results/plots/eda/{dataset}/3_prime_utr/genomic_distance_linear.svg",
        genomic_log="results/plots/eda/{dataset}/3_prime_utr/genomic_distance_log.svg",
        mrna_min_linear="results/plots/eda/{dataset}/3_prime_utr/mrna_distance_min_linear.svg",
        mrna_min_log="results/plots/eda/{dataset}/3_prime_utr/mrna_distance_min_log.svg",
        mrna_max_linear="results/plots/eda/{dataset}/3_prime_utr/mrna_distance_max_linear.svg",
        mrna_max_log="results/plots/eda/{dataset}/3_prime_utr/mrna_distance_max_log.svg",
    run:
        df = pl.read_parquet(input[0])
        # Define the three distance types to plot
        distance_configs = [
            (
                "genomic_distance_max",
                "Genomic Distance to CDS (including introns)",
                "#e74c3c",
                output.genomic_linear,
                output.genomic_log,
            ),
            (
                "mrna_distance_min",
                "mRNA Distance to CDS (min across isoforms)",
                "#3498db",
                output.mrna_min_linear,
                output.mrna_min_log,
            ),
            (
                "mrna_distance_max",
                "mRNA Distance to CDS (max across isoforms)",
                "#9b59b6",
                output.mrna_max_linear,
                output.mrna_max_log,
            ),
        ]
        for col, label, color, out_linear, out_log in distance_configs:
            distances = df[col].to_numpy()
            # Linear scale histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(distances, bins=50, color=color, alpha=0.8, edgecolor="black")
            ax.set_xlabel(f"{label} (bp)")
            ax.set_ylabel("Number of Causal Variants")
            ax.set_title(
                f"3' UTR Causal Variants: {label}\n({wildcards.dataset}, n={len(distances)})"
            )
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_linear, format="svg", bbox_inches="tight")
            plt.close()
            # Log scale histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            positive = distances[distances > 0]
            if len(positive) > 0:
                bins = np.logspace(
                    np.log10(max(1, positive.min())), np.log10(positive.max()), 30
                )
                ax.hist(
                    positive, bins=bins, color=color, alpha=0.8, edgecolor="black"
                )
                ax.set_xscale("log")
            ax.set_xlabel(f"{label} (bp, log scale)")
            ax.set_ylabel("Number of Causal Variants")
            ax.set_title(
                f"3' UTR Causal Variants: {label}\n({wildcards.dataset}, n={len(distances)})"
            )
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_log, format="svg", bbox_inches="tight")
            plt.close()


# =============================================================================
# Plot: ncRNA type distribution (with default biotype marking)
# =============================================================================
rule eda_plot_ncrna_types:
    input:
        "results/eda/{dataset}/analysis/ncrna_types.parquet",
    output:
        "results/plots/eda/{dataset}/ncrna/type_distribution.svg",
    run:
        from marin_dna.data.utils import DEFAULT_NCRNA_BIOTYPES

        df = pl.read_parquet(input[0]).to_pandas()
        fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.4)))
        y = np.arange(len(df))
        height = 0.6
        # Color by whether biotype is in default list
        colors = [
            "#2ecc71" if in_default else "#95a5a6"
            for in_default in df["in_default_biotypes"]
        ]
        bars = ax.barh(y, df["n_variants"], height, color=colors, alpha=0.8)
        # Add count labels
        max_val = df["n_variants"].max()
        for bar, count, pct in zip(bars, df["n_variants"], df["pct_variants"]):
            ax.text(
                bar.get_width() + max_val * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{count} ({pct:.1f}%)",
                va="center",
                fontsize=9,
            )
            # Add legend for colors
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(
                facecolor="#2ecc71", alpha=0.8, label="In DEFAULT_NCRNA_BIOTYPES"
            ),
            Patch(facecolor="#95a5a6", alpha=0.8, label="Not in defaults"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")
        ax.set_yticks(y)
        ax.set_yticklabels(df["biotype"])
        ax.set_xlabel("Number of Causal Variants")
        ax.set_title(f"ncRNA Causal Variants by Biotype ({wildcards.dataset})")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(output[0], format="svg", bbox_inches="tight")
        plt.close()


# =============================================================================
# Plot: ncRNA size histogram (linear and log scale)
# =============================================================================
rule eda_plot_ncrna_sizes:
    input:
        "results/eda/{dataset}/analysis/ncrna_sizes.parquet",
    output:
        linear="results/plots/eda/{dataset}/ncrna/size_linear.svg",
        log="results/plots/eda/{dataset}/ncrna/size_log.svg",
    run:
        df = pl.read_parquet(input[0])
        sizes = df["transcript_size"].drop_nulls().to_numpy()
        # Linear scale histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(sizes, bins=50, color="#9b59b6", alpha=0.8, edgecolor="black")
        ax.set_xlabel("Transcript Size (bp)")
        ax.set_ylabel("Number of Causal Variants")
        ax.set_title(
            f"ncRNA Causal Variants by Transcript Size\n({wildcards.dataset}, n={len(sizes)})"
        )
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output.linear, format="svg", bbox_inches="tight")
        plt.close()
        # Log scale histogram (x-axis)
        fig, ax = plt.subplots(figsize=(10, 6))
        positive_sizes = sizes[sizes > 0]
        if len(positive_sizes) > 0:
            bins = np.logspace(
                np.log10(max(1, positive_sizes.min())),
                np.log10(positive_sizes.max()),
                30,
            )
            ax.hist(
                positive_sizes,
                bins=bins,
                color="#9b59b6",
                alpha=0.8,
                edgecolor="black",
            )
            ax.set_xscale("log")
        ax.set_xlabel("Transcript Size (bp, log scale)")
        ax.set_ylabel("Number of Causal Variants")
        ax.set_title(
            f"ncRNA Causal Variants by Transcript Size\n({wildcards.dataset}, n={len(sizes)})"
        )
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output.log, format="svg", bbox_inches="tight")
        plt.close()


# =============================================================================
# Plot: ncRNA distance to nearest CDS and mRNA
# =============================================================================
rule eda_plot_ncrna_distance_to_coding:
    input:
        "results/eda/{dataset}/analysis/ncrna_distance_to_coding.parquet",
    output:
        cds_linear="results/plots/eda/{dataset}/ncrna/distance_to_cds_linear.svg",
        cds_log="results/plots/eda/{dataset}/ncrna/distance_to_cds_log.svg",
        mrna_linear="results/plots/eda/{dataset}/ncrna/distance_to_mrna_linear.svg",
        mrna_log="results/plots/eda/{dataset}/ncrna/distance_to_mrna_log.svg",
    run:
        df = pl.read_parquet(input[0])
        # Define distance types to plot
        distance_configs = [
            (
                "distance_to_cds",
                "Distance to Nearest CDS",
                "#e67e22",
                output.cds_linear,
                output.cds_log,
            ),
            (
                "distance_to_mrna",
                "Distance to Nearest mRNA Exon",
                "#1abc9c",
                output.mrna_linear,
                output.mrna_log,
            ),
        ]
        for col, label, color, out_linear, out_log in distance_configs:
            distances = df[col].to_numpy()
            # Linear scale histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(distances, bins=50, color=color, alpha=0.8, edgecolor="black")
            ax.set_xlabel(f"{label} (bp)")
            ax.set_ylabel("Number of Causal Variants")
            ax.set_title(
                f"ncRNA Causal Variants: {label}\n({wildcards.dataset}, n={len(distances)})"
            )
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_linear, format="svg", bbox_inches="tight")
            plt.close()
            # Log scale histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            positive = distances[distances > 0]
            if len(positive) > 0:
                bins = np.logspace(
                    np.log10(max(1, positive.min())), np.log10(positive.max()), 30
                )
                ax.hist(
                    positive, bins=bins, color=color, alpha=0.8, edgecolor="black"
                )
                ax.set_xscale("log")
            zero_count = (distances == 0).sum()
            if zero_count > 0:
                ax.axvline(x=1, color="red", linestyle="--", alpha=0.7)
                ax.text(
                    1.5,
                    ax.get_ylim()[1] * 0.9,
                    f"Overlapping: {zero_count}",
                    fontsize=10,
                    color="red",
                )
            ax.set_xlabel(f"{label} (bp, log scale)")
            ax.set_ylabel("Number of Causal Variants")
            ax.set_title(
                f"ncRNA Causal Variants: {label}\n({wildcards.dataset}, n={len(distances)})"
            )
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_log, format="svg", bbox_inches="tight")
            plt.close()


# =============================================================================
# Rule: Extract transcript-level CDS boundaries (one row per transcript)
# =============================================================================
# For each transcript, get the minimum CDS start and maximum CDS end.
# This is used to compute position-wise conservation around CDS boundaries.
rule eda_extract_transcript_cds_boundaries:
    input:
        annotation="results/annotation/GCF_000001405.40.gtf.gz",
    output:
        "results/eda/annotation/transcript_cds_boundaries.parquet",
    run:
        ann = load_annotation(input.annotation)
        # Get CDS features with transcript IDs
        cds = ann.filter(pl.col("feature") == "CDS").with_columns(
            pl.col("attribute")
            .str.extract(r'transcript_id "(.*?)"')
            .alias("transcript_id"),
        )
        # Group by transcript to get min start and max end
        cds_bounds = cds.group_by("transcript_id").agg(
            pl.col("chrom").first(),
            pl.col("strand").first(),
            pl.col("start").min().alias("cds_start"),
            pl.col("end").max().alias("cds_end"),
        )
        cds_bounds.write_parquet(output[0])


# =============================================================================
# Rule: Compute position-wise conservation around CDS boundaries
# =============================================================================
# For each position from -2000 to +2000 (excluding 0), compute the proportion
# of bases that are conserved (phyloP >= threshold) across all transcripts.
# Strand-aware: upstream (-) is 5' of CDS, downstream (+) is 3' of CDS.
rule eda_compute_cds_flanking_conservation:
    input:
        cds_bounds="results/eda/annotation/transcript_cds_boundaries.parquet",
        phylop="results/conservation/cactus241way.phyloP.bw",
        chrom_mapping="config/human_chrom_mapping.tsv",
    output:
        "results/eda/cds_flanking_conservation.parquet",
    run:
        # Load config for conservation threshold
        phylop_cutoff = config["conservation"]["phylop_cutoff"]
        # Load chromosome mapping (RefSeq -> UCSC)
        chrom_map = pl.read_csv(input.chrom_mapping, separator="\t")
        refseq_to_ucsc = dict(zip(chrom_map["refseq"], chrom_map["ucsc"]))
        # Load CDS boundaries and filter to chr1 (NC_000001.11)
        cds_bounds = pl.read_parquet(input.cds_bounds)
        cds_bounds = cds_bounds.filter(pl.col("chrom") == "NC_000001.11")
        # Map chromosome to UCSC
        chrom_ucsc = refseq_to_ucsc.get("NC_000001.11")
        # Create relative positions array
        positions = list(range(-2000, 0)) + list(range(1, 2001))
        # Open phyloP bigWig and get chromosome length
        bw = pyBigWig.open(input.phylop)
        chrom_len = bw.chroms().get(chrom_ucsc, 0)
        # Initialize counts for each position
        counts = {pos: {"n_total": 0, "n_conserved": 0} for pos in positions}
        # Process transcripts in batches to avoid memory issues
        print(f"Processing {len(cds_bounds)} transcripts...")
        batch_size = 100
        cds_list = cds_bounds.to_dicts()
        for batch_start in tqdm(range(0, len(cds_list), batch_size)):
            batch = cds_list[batch_start : batch_start + batch_size]
            # Collect all genomic positions for this batch
            batch_queries = []  # (rel_pos, genomic_pos)
            for row in batch:
                strand = row["strand"]
                cds_start = row["cds_start"]
                cds_end = row["cds_end"]
                for rel_pos in positions:
                    if rel_pos < 0:
                        # Upstream (5' of CDS)
                        if strand == "+":
                            genomic_pos = cds_start + rel_pos
                        else:
                            genomic_pos = cds_end - rel_pos - 1
                    else:
                        # Downstream (3' of CDS)
                        if strand == "+":
                            genomic_pos = cds_end + rel_pos - 1
                        else:
                            genomic_pos = cds_start - rel_pos
                            # Check bounds
                    if 0 <= genomic_pos < chrom_len:
                        batch_queries.append((rel_pos, genomic_pos))
                        # Get unique genomic positions for this batch
            unique_genomic = sorted(set(gp for _, gp in batch_queries))
            # Query phyloP values for unique positions
            phylop_cache = {}
            for gp in unique_genomic:
                vals = bw.values(chrom_ucsc, gp, gp + 1)
                phylop_cache[gp] = (
                    vals[0] if (vals and vals[0] is not None) else 0.0
                )
                # Accumulate counts
            for rel_pos, genomic_pos in batch_queries:
                counts[rel_pos]["n_total"] += 1
                if phylop_cache[genomic_pos] >= phylop_cutoff:
                    counts[rel_pos]["n_conserved"] += 1
        bw.close()
        # Build result dataframe
        result = pl.DataFrame(
            {
                "position": positions,
                "n_total": [counts[p]["n_total"] for p in positions],
                "n_conserved": [counts[p]["n_conserved"] for p in positions],
            }
        ).with_columns(
            (pl.col("n_conserved") / pl.col("n_total") * 100).alias("pct_conserved")
        )
        result.write_parquet(output[0])


# =============================================================================
# Plot: Position-wise conservation around CDS boundaries
# =============================================================================
rule eda_plot_cds_flanking_conservation:
    input:
        "results/eda/cds_flanking_conservation.parquet",
    output:
        "results/plots/eda/cds_flanking_conservation.svg",
    run:
        df = pl.read_parquet(input[0]).to_pandas()
        fig, ax = plt.subplots(figsize=(12, 6))
        # Plot the conservation profile
        ax.plot(
            df["position"],
            df["pct_conserved"],
            color="#3498db",
            linewidth=0.8,
            alpha=0.9,
        )
        # Add vertical line at x=0 (CDS boundary)
        ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
        # Add region labels
        ax.text(
            -1000,
            ax.get_ylim()[1] * 0.95,
            "Upstream (5' of CDS)",
            ha="center",
            va="top",
            fontsize=11,
            color="#2c3e50",
        )
        ax.text(
            1000,
            ax.get_ylim()[1] * 0.95,
            "Downstream (3' of CDS)",
            ha="center",
            va="top",
            fontsize=11,
            color="#2c3e50",
        )
        # Formatting
        ax.set_xlabel("Position Relative to CDS Boundary (bp)", fontsize=12)
        ax.set_ylabel("Proportion Conserved (%)", fontsize=12)
        ax.set_title(
            "Conservation Around CDS Boundaries (chr1)\n"
            f"(phyloP ≥ {config['conservation']['phylop_cutoff']})",
            fontsize=13,
        )
        ax.set_xlim(-2000, 2000)
        ax.grid(axis="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output[0], format="svg", bbox_inches="tight")
        plt.close()


# =============================================================================
# Master target: all EDA outputs for both datasets
# =============================================================================
rule all_eda:
    input:
        # Analysis outputs for both datasets
        expand(
            "results/eda/{dataset}/analysis/3_prime_utr_mrna_distance.parquet",
            dataset=EDA_DATASETS.keys(),
        ),
        expand(
            "results/eda/{dataset}/analysis/ncrna_types.parquet",
            dataset=EDA_DATASETS.keys(),
        ),
        expand(
            "results/eda/{dataset}/analysis/ncrna_sizes.parquet",
            dataset=EDA_DATASETS.keys(),
        ),
        expand(
            "results/eda/{dataset}/analysis/ncrna_distance_to_coding.parquet",
            dataset=EDA_DATASETS.keys(),
        ),
        # 3' UTR plot outputs (genomic, mrna_min, mrna_max × linear, log)
        expand(
            "results/plots/eda/{dataset}/3_prime_utr/genomic_distance_linear.svg",
            dataset=EDA_DATASETS.keys(),
        ),
        expand(
            "results/plots/eda/{dataset}/3_prime_utr/genomic_distance_log.svg",
            dataset=EDA_DATASETS.keys(),
        ),
        expand(
            "results/plots/eda/{dataset}/3_prime_utr/mrna_distance_min_linear.svg",
            dataset=EDA_DATASETS.keys(),
        ),
        expand(
            "results/plots/eda/{dataset}/3_prime_utr/mrna_distance_min_log.svg",
            dataset=EDA_DATASETS.keys(),
        ),
        expand(
            "results/plots/eda/{dataset}/3_prime_utr/mrna_distance_max_linear.svg",
            dataset=EDA_DATASETS.keys(),
        ),
        expand(
            "results/plots/eda/{dataset}/3_prime_utr/mrna_distance_max_log.svg",
            dataset=EDA_DATASETS.keys(),
        ),
        # ncRNA plot outputs
        expand(
            "results/plots/eda/{dataset}/ncrna/type_distribution.svg",
            dataset=EDA_DATASETS.keys(),
        ),
        expand(
            "results/plots/eda/{dataset}/ncrna/size_linear.svg",
            dataset=EDA_DATASETS.keys(),
        ),
        expand(
            "results/plots/eda/{dataset}/ncrna/size_log.svg",
            dataset=EDA_DATASETS.keys(),
        ),
        expand(
            "results/plots/eda/{dataset}/ncrna/distance_to_cds_linear.svg",
            dataset=EDA_DATASETS.keys(),
        ),
        expand(
            "results/plots/eda/{dataset}/ncrna/distance_to_cds_log.svg",
            dataset=EDA_DATASETS.keys(),
        ),
        expand(
            "results/plots/eda/{dataset}/ncrna/distance_to_mrna_linear.svg",
            dataset=EDA_DATASETS.keys(),
        ),
        expand(
            "results/plots/eda/{dataset}/ncrna/distance_to_mrna_log.svg",
            dataset=EDA_DATASETS.keys(),
        ),
        # CDS flanking conservation plot
        "results/plots/eda/cds_flanking_conservation.svg",
