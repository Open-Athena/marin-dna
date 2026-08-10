PLOT_REGIONS = config["functional_regions"]


rule download_phylop_conservation:
    output:
        "results/conservation/cactus241way.phyloP.bw",
    shell:
        "wget -O {output} https://hgdownload.soe.ucsc.edu/goldenPath/hg38/cactus241way/cactus241way.phyloP.bw"


# Zoonomia/Cactus 43-primate phastCons. The remote filename says "phyloP" but
# this is the phastCons track (0-1 score, threshold 0.961 calibrated to a ~3.5%
# conserved-base fraction matching phyloP-241way 2.27).
rule download_phastcons_43p_conservation:
    output:
        "results/conservation/phastCons_43p.bw",
    shell:
        "wget -O {output} https://cgl.gi.ucsc.edu/data/cactus/zoonomia-2021-track-hub/hg38/phyloPPrimates.bigWig"


rule all_functional_region_stats:
    input:
        expand(
            "results/stats/functional_regions/{g}.parquet",
            g=config["genome_subset_analysis"],
        ),


rule all_stats_plots:
    input:
        # Functional region stats per genome
        expand(
            "results/plots/functional_regions/{g}.svg",
            g=config["genome_subset_analysis"],
        ),
        # Size histograms per region (global)
        expand(
            "results/plots/functional_regions_size_histogram/{region}.svg",
            region=PLOT_REGIONS,
        ),
        # Global functional region summary
        "results/plots/functional_regions_overall.svg",
        # Annotation source plots
        "results/plots/annotation_sources.svg",
        "results/plots/annotation_sources_per_genome.svg",
        # Conservation summary
        "results/plots/conservation_summary.svg",
        "results/plots/conservation_total_bases.svg",
        "results/plots/promoter_comparison.svg",


rule all_annotation_source_stats:
    input:
        expand(
            "results/stats/annotation_sources/{g}.parquet",
            g=config["genome_subset_analysis"],
        ),
    output:
        "results/stats/annotation_sources_summary.parquet",
    run:
        dfs = []
        for g, path in zip(config["genome_subset_analysis"], input):
            df = pd.read_parquet(path)
            df["genome"] = g
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)
        # Pivot to get all sources as columns, fill missing with 0
        pivot = combined.pivot(
            index="genome", columns="source", values="tx_ratio"
        ).fillna(0)
        summary = pivot.agg(["mean", "median"]).T.reset_index()
        summary.columns = ["source", "mean", "median"]
        summary.to_parquet(output[0], index=False)


rule annotation_source_stats:
    input:
        "results/annotation/{g}.gtf.gz",
    output:
        "results/stats/annotation_sources/{g}.parquet",
    run:
        ann = load_annotation(input[0])
        transcripts = ann.filter(pl.col("feature") == "transcript")
        stats_df = transcripts.group_by("source").len().to_pandas()
        stats_df.columns = ["source", "n_transcripts"]
        stats_df["tx_ratio"] = (
            stats_df["n_transcripts"] / stats_df["n_transcripts"].sum()
        )
        stats_df.to_parquet(output[0], index=False)


rule functional_region_stats:
    input:
        expand(
            "results/intervals/{region}/{{g}}.parquet",
            region=config["functional_regions"],
        ),
    output:
        "results/stats/functional_regions/{g}.parquet",
    run:
        quantiles = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
        rows = []
        for region, path in zip(config["functional_regions"], input):
            gs = GenomicSet.read_parquet(path)
            df = gs.to_polars()
            lengths = df["end"] - df["start"]
            row = {
                "region": region,
                "n_intervals": gs.n_intervals(),
                "size_total": gs.total_size(),
                "size_mean": lengths.mean(),
                "size_std": lengths.std(),
            }
            for q in quantiles:
                row[f"size_p{int(q*100)}"] = lengths.quantile(q)
            rows.append(row)
        stats_df = pd.DataFrame(rows)
        stats_df["size_total_ratio"] = (
            stats_df["size_total"] / stats_df["size_total"].sum()
        )
        stats_df["n_intervals_ratio"] = (
            stats_df["n_intervals"] / stats_df["n_intervals"].sum()
        )
        stats_df.to_parquet(output[0], index=False)


rule calculate_conservation:
    input:
        conservation="results/conservation/cactus241way.phyloP.bw",
        chrom_mapping="config/human_chrom_mapping.tsv",
        intervals="results/intervals/{region}/GCF_000001405.40.parquet",
    output:
        "results/conservation/{region}.parquet",
    run:
        phylop_cutoff = config["conservation"]["phylop_cutoff"]
        chrom_map = pl.read_csv(input.chrom_mapping, separator="\t")
        refseq_to_ucsc = dict(zip(chrom_map["refseq"], chrom_map["ucsc"]))
        # Map RefSeq to UCSC chrom names and filter to mapped chroms
        df = (
            pl.read_parquet(input.intervals)
            .with_columns(
                pl.col("chrom")
                .replace_strict(refseq_to_ucsc, default=None)
                .alias("chrom_ucsc")
            )
            .filter(pl.col("chrom_ucsc").is_not_null())
        )
        # Total bases from interval sizes (NaN = valid but not conserved)
        total_bases = (df["end"] - df["start"]).sum()
        # Count conserved bases per interval (NaN >= cutoff is False, so excluded)
        bw = pyBigWig.open(input.conservation)
        conserved_counts = df.select(
            pl.struct(["chrom_ucsc", "start", "end"])
            .map_elements(
                lambda x: int(
                    np.sum(
                        bw.values(x["chrom_ucsc"], x["start"], x["end"], numpy=True)
                        >= phylop_cutoff
                    )
                ),
                return_dtype=pl.Int64,
            )
            .alias("conserved")
        )
        bw.close()
        conserved_bases = conserved_counts["conserved"].sum()
        pct_conserved = (
            (conserved_bases / total_bases * 100) if total_bases > 0 else 0.0
        )
        result = pl.DataFrame(
            {
                "region": [wildcards.region],
                "total_bases": [int(total_bases)],
                "conserved_bases": [int(conserved_bases)],
                "pct_conserved": [pct_conserved],
            }
        )
        result.write_parquet(output[0])


rule combine_conservation_stats:
    input:
        expand(
            "results/conservation/{region}.parquet",
            region=config["functional_regions"],
        ),
    output:
        "results/conservation/summary.parquet",
    run:
        dfs = [pl.read_parquet(path) for path in input]
        combined = pl.concat(dfs)
        combined.write_parquet(output[0])


def load_genome_labels(genomes_path: str) -> dict[str, str]:
    """Load mapping from genome accession to species name."""
    df = pd.read_parquet(genomes_path)
    return dict(zip(df["Assembly Accession"], df["Organism Name"]))


rule combine_functional_region_stats:
    input:
        expand(
            "results/stats/functional_regions/{g}.parquet",
            g=config["genome_subset_analysis"],
        ),
    output:
        "results/stats/functional_regions_combined.parquet",
    run:
        dfs = []
        for g, path in zip(config["genome_subset_analysis"], input):
            df = pd.read_parquet(path)
            df["genome"] = g
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_parquet(output[0], index=False)


rule combine_annotation_source_stats:
    input:
        expand(
            "results/stats/annotation_sources/{g}.parquet",
            g=config["genome_subset_analysis"],
        ),
    output:
        "results/stats/annotation_sources_combined.parquet",
    run:
        dfs = []
        for g, path in zip(config["genome_subset_analysis"], input):
            df = pd.read_parquet(path)
            df["genome"] = g
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_parquet(output[0], index=False)


rule plot_functional_regions_per_genome:
    input:
        stats="results/stats/functional_regions/{g}.parquet",
        genomes=config["genomes_path"],
    output:
        "results/plots/functional_regions/{g}.svg",
    run:
        df = pd.read_parquet(input.stats)
        df = df[df["region"].isin(PLOT_REGIONS)]
        genome_labels = load_genome_labels(input.genomes)
        genome_name = genome_labels.get(wildcards.g, wildcards.g)
        regions = df["region"].tolist()
        x = np.arange(len(regions))
        labels = [config["region_labels"][r] for r in regions]
        colors = [config["region_colors"][r] for r in regions]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # Panel 1: Number of intervals
        ax1 = axes[0]
        bars = ax1.bar(x, df["n_intervals"], color=colors)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha="right")
        ax1.set_ylabel("Count")
        ax1.set_title("Number of Intervals")
        ax1.grid(axis="y", alpha=0.3)
        # Panel 2: Size distribution (box plot using quantiles)
        ax2 = axes[1]
        for i, (_, row) in enumerate(df.iterrows()):
            p10, p25, p50, p75, p90 = (
                row["size_p10"],
                row["size_p25"],
                row["size_p50"],
                row["size_p75"],
                row["size_p90"],
            )
            color = colors[i]
            ax2.fill_between([i - 0.3, i + 0.3], p25, p75, alpha=0.6, color=color)
            ax2.hlines(p50, i - 0.3, i + 0.3, colors="black", linewidth=2)
            ax2.vlines(i, p10, p25, colors="black", linewidth=1)
            ax2.vlines(i, p75, p90, colors="black", linewidth=1)
            ax2.hlines(p10, i - 0.15, i + 0.15, colors="black", linewidth=1)
            ax2.hlines(p90, i - 0.15, i + 0.15, colors="black", linewidth=1)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha="right")
        ax2.set_ylabel("Interval Size (bp)")
        ax2.set_title("Size Distribution (p10-p90)")
        ax2.grid(axis="y", alpha=0.3)
        # Panel 3: Total size
        ax3 = axes[2]
        bars = ax3.bar(x, df["size_total"] / 1e6, color=colors)
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, rotation=45, ha="right")
        ax3.set_ylabel("Total Size (Mb)")
        ax3.set_title("Total Size")
        ax3.grid(axis="y", alpha=0.3)
        fig.suptitle(f"Functional Region Statistics - {genome_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(output[0], format="svg", bbox_inches="tight")
        plt.close()


rule plot_functional_regions_overall:
    input:
        stats="results/stats/functional_regions_combined.parquet",
    output:
        "results/plots/functional_regions_overall.svg",
    run:
        df = pd.read_parquet(input.stats)
        df = df[df["region"].isin(PLOT_REGIONS)]
        regions = PLOT_REGIONS
        x = np.arange(len(regions))
        labels = [config["region_labels"][r] for r in regions]
        colors = [config["region_colors"][r] for r in regions]
        # Aggregate stats across genomes (using median interval size = size_p50)
        agg = (
            df.groupby("region")
            .agg(
                n_intervals_mean=("n_intervals", "mean"),
                n_intervals_std=("n_intervals", "std"),
                size_total_mean=("size_total", "mean"),
                size_total_std=("size_total", "std"),
                size_median_mean=("size_p50", "mean"),
                size_median_std=("size_p50", "std"),
            )
            .reindex(regions)
        )
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # Panel 1: Number of intervals
        ax1 = axes[0]
        ax1.bar(
            x,
            agg["n_intervals_mean"],
            yerr=agg["n_intervals_std"],
            color=colors,
            capsize=3,
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha="right")
        ax1.set_ylabel("Count")
        ax1.set_title("Number of Intervals (mean ± std)")
        ax1.grid(axis="y", alpha=0.3)
        # Panel 2: Median interval size
        ax2 = axes[1]
        ax2.bar(
            x,
            agg["size_median_mean"],
            yerr=agg["size_median_std"],
            color=colors,
            capsize=3,
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha="right")
        ax2.set_ylabel("Median Interval Size (bp)")
        ax2.set_title("Median Interval Size (mean ± std)")
        ax2.grid(axis="y", alpha=0.3)
        # Panel 3: Total size
        ax3 = axes[2]
        ax3.bar(
            x,
            agg["size_total_mean"] / 1e6,
            yerr=agg["size_total_std"] / 1e6,
            color=colors,
            capsize=3,
        )
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, rotation=45, ha="right")
        ax3.set_ylabel("Total Size (Mb)")
        ax3.set_title("Total Size (mean ± std)")
        ax3.grid(axis="y", alpha=0.3)
        fig.suptitle(
            "Functional Region Statistics (Aggregated Across Genomes)", fontsize=14
        )
        plt.tight_layout()
        plt.savefig(output[0], format="svg", bbox_inches="tight")
        plt.close()


rule plot_functional_regions_size_histogram:
    input:
        intervals=expand(
            "results/intervals/{{region}}/{g}.parquet",
            g=config["genome_subset_analysis"],
        ),
    output:
        "results/plots/functional_regions_size_histogram/{region}.svg",
    run:
        # Collect all interval sizes across genomes
        all_sizes = []
        for path in input.intervals:
            gs = GenomicSet.read_parquet(path)
            df = gs.to_polars()
            sizes = (df["end"] - df["start"]).to_list()
            all_sizes.extend(sizes)
        all_sizes = np.array(all_sizes)
        fig, ax = plt.subplots(figsize=(10, 6))
        # Log-scale histogram
        bins = 50
        ax.hist(all_sizes, bins=bins, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Interval Size (bp)")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Size Distribution - {wildcards.region} (n={len(all_sizes):,})"
        )
        ax.grid(axis="y", alpha=0.3)
        # Add summary statistics
        stats_text = (
            f"Median: {np.median(all_sizes):,.0f} bp\n"
            f"Mean: {np.mean(all_sizes):,.0f} bp\n"
            f"Total: {np.sum(all_sizes)/1e9:.2f} Gb"
        )
        ax.text(
            0.98,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        plt.tight_layout()
        plt.savefig(output[0], format="svg", bbox_inches="tight")
        plt.close()


rule plot_annotation_sources:
    input:
        "results/stats/annotation_sources_summary.parquet",
    output:
        "results/plots/annotation_sources.svg",
    run:
        df = pd.read_parquet(input[0])
        df = df.sort_values("mean", ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.4)))
        y_pos = range(len(df))
        ax.barh(y_pos, df["mean"], alpha=0.7, label="Mean")
        ax.scatter(df["median"], y_pos, color="red", zorder=5, label="Median", s=50)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(df["source"])
        ax.set_xlabel("Proportion of Transcripts")
        ax.set_title("Annotation Sources (Mean Proportion Across Genomes)")
        ax.legend()
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output[0], format="svg", bbox_inches="tight")
        plt.close()


rule plot_annotation_sources_per_genome:
    input:
        stats="results/stats/annotation_sources_combined.parquet",
        genomes=config["genomes_path"],
    output:
        "results/plots/annotation_sources_per_genome.svg",
    run:
        df = pd.read_parquet(input.stats)
        genome_labels = load_genome_labels(input.genomes)
        pivot = df.pivot(
            index="genome", columns="source", values="tx_ratio"
        ).fillna(0)
        # Sort by curated sources (BestRefSeq first, then RefSeq) in descending order
        curated_cols = [c for c in ["BestRefSeq", "RefSeq"] if c in pivot.columns]
        if curated_cols:
            pivot = pivot.sort_values(curated_cols, ascending=False)
        pivot.index = pivot.index.map(lambda g: genome_labels.get(g, g))
        fig, ax = plt.subplots(figsize=(12, max(8, len(pivot) * 0.3)))
        bottom = np.zeros(len(pivot))
        colors = plt.cm.Set3(np.linspace(0, 1, len(pivot.columns)))
        for col, color in zip(pivot.columns, colors):
            ax.barh(pivot.index, pivot[col], left=bottom, label=col, color=color)
            bottom += pivot[col].values
        ax.set_xlabel("Proportion of Transcripts")
        ax.set_title("Annotation Sources by Species (Sorted by Curated Sources)")
        ax.legend(loc="lower right", bbox_to_anchor=(1.2, 0))
        ax.set_xlim(0, 1)
        plt.tight_layout()
        plt.savefig(output[0], format="svg", bbox_inches="tight")
        plt.close()


rule plot_conservation_summary:
    input:
        "results/conservation/summary.parquet",
    output:
        "results/plots/conservation_summary.svg",
    run:
        phylop_cutoff = config["conservation"]["phylop_cutoff"]
        df = pd.read_parquet(input[0]).sort_values("pct_conserved", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = [config["region_labels"][r] for r in df["region"]]
        colors = [config["region_colors"][r] for r in df["region"]]
        bars = ax.bar(labels, df["pct_conserved"], color=colors, edgecolor="black")
        for bar, pct in zip(bars, df["pct_conserved"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        ax.set_xlabel("Functional Region")
        ax.set_ylabel("Bases Conserved (%)")
        ax.set_title(
            f"Conservation by Functional Region (phyloP >= {phylop_cutoff})"
        )
        ax.set_ylim(0, max(df["pct_conserved"]) * 1.15)
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output[0], format="svg", bbox_inches="tight")
        plt.close()


rule plot_conservation_total_bases:
    input:
        "results/conservation/summary.parquet",
    output:
        "results/plots/conservation_total_bases.svg",
    run:
        phylop_cutoff = config["conservation"]["phylop_cutoff"]
        df = pd.read_parquet(input[0]).sort_values(
            "conserved_bases", ascending=False
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = [config["region_labels"][r] for r in df["region"]]
        colors = [config["region_colors"][r] for r in df["region"]]
        conserved_mb = df["conserved_bases"] / 1e6
        bars = ax.bar(labels, conserved_mb, color=colors, edgecolor="black")
        for bar, mb in zip(bars, conserved_mb):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{mb:.1f}M",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        ax.set_xlabel("Functional Region")
        ax.set_ylabel("Conserved Bases (millions)")
        ax.set_title(
            f"Total Conserved Bases by Functional Region (phyloP >= {phylop_cutoff})"
        )
        ax.set_ylim(0, max(conserved_mb) * 1.15)
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output[0], format="svg", bbox_inches="tight")
        plt.close()


rule plot_promoter_comparison:
    input:
        expand(
            "results/conservation/{region}.parquet",
            region=config["promoter_comparison_regions"],
        ),
    output:
        "results/plots/promoter_comparison.svg",
    run:
        # Labels specific to this plot
        labels = {
            "promoters_mRNA/256/256": "mRNA ±256",
            "promoters_mRNA/2048/2048": "mRNA ±2048",
            "promoters/256/256": "All ±256",
            "promoters/2048/2048": "All ±2048",
        }
        colors = {
            "promoters_mRNA/256/256": "#e78ac3",
            "promoters_mRNA/2048/2048": "#c994b0",
            "promoters/256/256": "#fbb4ae",
            "promoters/2048/2048": "#d9a09b",
        }
        phylop_cutoff = config["conservation"]["phylop_cutoff"]
        # Read and combine conservation data
        dfs = [pl.read_parquet(path) for path in input]
        df = pl.concat(dfs).to_pandas()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        region_labels = [labels[r] for r in df["region"]]
        region_colors = [colors[r] for r in df["region"]]
        # Panel 1: Total bases
        ax1 = axes[0]
        total_mb = df["total_bases"] / 1e6
        bars1 = ax1.bar(
            region_labels, total_mb, color=region_colors, edgecolor="black"
        )
        for bar, mb in zip(bars1, total_mb):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{mb:.1f}M",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        ax1.set_ylabel("Total Bases (millions)")
        ax1.set_title("Total Bases")
        ax1.set_ylim(0, max(total_mb) * 1.15)
        ax1.grid(axis="y", alpha=0.3)
        plt.sca(ax1)
        plt.xticks(rotation=45, ha="right")
        # Panel 2: Conservation %
        ax2 = axes[1]
        bars2 = ax2.bar(
            region_labels,
            df["pct_conserved"],
            color=region_colors,
            edgecolor="black",
        )
        for bar, pct in zip(bars2, df["pct_conserved"]):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        ax2.set_ylabel("Bases Conserved (%)")
        ax2.set_title(f"Conservation (phyloP ≥ {phylop_cutoff})")
        ax2.set_ylim(0, max(df["pct_conserved"]) * 1.15)
        ax2.grid(axis="y", alpha=0.3)
        plt.sca(ax2)
        plt.xticks(rotation=45, ha="right")
        fig.suptitle("Promoter Region Comparison", fontsize=14)
        plt.tight_layout()
        plt.savefig(output[0], format="svg", bbox_inches="tight")
        plt.close()
