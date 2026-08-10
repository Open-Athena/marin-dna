"""Generalized interval projection across genomes via mmseqs2 nucleotide search.

Produces `results/intervals/{name}/{g}.parquet` for each genome `g`. On the
source genome, the file is a chrom-normalized copy of the configured source
parquet. On target genomes, the file is the result of aligning the source
intervals onto `g` with mmseqs2 (nucleotide mode) and keeping the best hit
per query by bit score.

Configured via the `interval_mappings` block in config.yaml. For v1 every
mapping uses `HUMAN_GENOME` as its source; the wildcard constraints below
assume that, and should be generalized if more source genomes are added.

Choice of aligner (mmseqs2 over minimap2) follows issue #120, where mmseqs2
at `-s 7.5` is Pareto-optimal on the low-cost end of the recall-vs-compute
frontier for hg38↔mm10 cCRE orthologs (~70% R@1 at ~97% P@1 on the
phyloP_241m-conserved subset, vs. minimap2's ~40% R@1).
"""


rule genome_fa:
    """Whole-genome FASTA; required as the mmseqs2 target input."""
    input:
        "results/genome/{g}.2bit",
    output:
        "results/genome/{g}.fa",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitToFa {input} {output}"


rule intervals_source_unified:
    """Stage the source-genome intervals at the unified path.

    For CRE-derived sources, chroms are normalized from UCSC-stripped (bare
    digit, e.g. "1", "X") to the RefSeq NC_* form used by the per-genome 2bit
    files, via config/human_chrom_mapping.tsv. Same remap pattern as
    intervals_recipe_v17/v18 in intervals.smk.
    """
    input:
        source_parquet=lambda w: MAPPINGS[w.name]["source_parquet"],
        chrom_mapping=local("config/human_chrom_mapping.tsv"),
    output:
        f"results/intervals/{{name}}/{HUMAN_GENOME}.parquet",
    run:
        cfg = MAPPINGS[wildcards.name]
        style = cfg.get("source_chrom_style")
        assert style in (
            None,
            "ucsc_stripped",
        ), f"unsupported source_chrom_style {style!r}; expected None or 'ucsc_stripped'"
        df = pl.read_parquet(input.source_parquet)
        if style == "ucsc_stripped":
            chrom_map = pl.read_csv(input.chrom_mapping, separator="\t")
            simple_to_refseq = dict(
                zip(chrom_map["ucsc"].str.replace("chr", ""), chrom_map["refseq"])
            )
            df = df.with_columns(pl.col("chrom").replace_strict(simple_to_refseq))
        df.select(["chrom", "start", "end"]).write_parquet(output[0])


rule make_alignment_query_bed:
    """4-col BED from the source intervals with symmetric flank.

    Flank positive expands each side; flank 0 (v1 default) passes the
    intervals through unchanged. End is clamped to the chrom size; degenerate
    intervals (hi <= lo) are dropped.

    The 4th column is the source interval id (`chrom:start-end`) so hits can
    be traced back to their query.
    """
    input:
        source_parquet=lambda w: (
            f"results/intervals/{w.name}/"
            f"{MAPPINGS[w.name]['source_genome']}.parquet"
        ),
        chrom_sizes=lambda w: (
            f"results/chrom_sizes/{MAPPINGS[w.name]['source_genome']}.tsv"
        ),
    output:
        temp("results/interval_alignment/{name}/{g}.query.bed"),
    run:
        cfg = MAPPINGS[wildcards.name]
        flank = int(cfg.get("flank_bp", 0))
        sizes: dict[str, int] = {}
        with open(input.chrom_sizes) as f:
            for line in f:
                c, s = line.rstrip("\n").split("\t")
                sizes[c] = int(s)
        df = pl.read_parquet(input.source_parquet)
        kept = 0
        with open(output[0], "w") as fout:
            for row in df.iter_rows(named=True):
                chrom = row["chrom"]
                lo = max(0, row["start"] - flank)
                hi = row["end"] + flank
                if chrom in sizes:
                    hi = min(sizes[chrom], hi)
                if hi <= lo:
                    continue
                name = f"{chrom}:{row['start']}-{row['end']}"
                fout.write(f"{chrom}\t{lo}\t{hi}\t{name}\n")
                kept += 1
        print(
            f"  {wildcards.name} → {wildcards.g}: {kept}/{df.height} queries "
            f"(flank={flank})"
        )


rule extract_alignment_query_fasta:
    """Per-query FASTA from the source 2bit."""
    input:
        twobit=lambda w: f"results/genome/{MAPPINGS[w.name]['source_genome']}.2bit",
        bed="results/interval_alignment/{name}/{g}.query.bed",
    output:
        temp("results/interval_alignment/{name}/{g}.query.fa"),
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitToFa {input.twobit} {output} -bed={input.bed}"


rule align_intervals_mmseqs2:
    """Build mmseqs2 query + target DBs, run nucleotide search, emit hits TSV.

    Consolidated into one rule because mmseqs2 produces a fan of sidecar
    files per DB (`.index`, `.source`, `.lookup`, `_h*`, ...) that are
    awkward to declare individually, and the pipeline's S3 default-storage
    backend only uploads declared outputs. Keeping the DBs in a local
    `mktemp -d` tmpdir sidesteps the upload entirely — only the final
    `{g}.hits.tsv` gets stored, and a rerun rebuilds the DBs from the
    (cached) input FASTAs.

    Search flags mirror snakemake/analysis/dELS_orthologs (issue #120):
    - `--search-type 3` forces nucleotide mode (no auto-detect surprises).
    - `--strand 2` searches forward + reverse complement targets.
    - `--mask-lower-case 1` excludes soft-masked repeats from k-mer seeding.
    - `-s` (sensitivity) and `--max-accept` come from the mapping config.
    - `--split-memory-limit` lets the rule fit a small-RAM box at the cost
      of wall time; whole-mammal nucleotide search needs ~50-80 GB resident
      unsplit, so large targets require either a big instance or a small
      split.

    `tstart,tend,bits,evalue,fident,qcov,tcov` is the column set consumed
    by `marin_dna.pipelines.alignment.mmseqs2.parse_mmseqs2_hits`.
    """
    input:
        query="results/interval_alignment/{name}/{g}.query.fa",
        target="results/genome/{g}.fa",
    output:
        tsv=temp("results/interval_alignment/{name}/{g}.hits.tsv"),
    conda:
        "../envs/mmseqs2.yaml"
    threads: workflow.cores
    resources:
        mem_mb=lambda w: MAPPINGS[w.name].get("mem_mb", 14000),
    params:
        sensitivity=lambda w: MAPPINGS[w.name].get("sensitivity", 7.5),
        max_accept=lambda w: MAPPINGS[w.name].get("max_accept", 1),
        split_memory_limit=lambda w: MAPPINGS[w.name].get("split_memory_limit", "12G"),
    shell:
        """
        TMP=$(mktemp -d -t mmseqs2_{wildcards.name}_{wildcards.g}_XXXX)
        trap 'rm -rf "$TMP"' EXIT

        mmseqs createdb {input.target} $TMP/targetDB --mask-lower-case 1
        mmseqs createdb {input.query} $TMP/queryDB --mask-lower-case 1

        mmseqs search \
            $TMP/queryDB $TMP/targetDB $TMP/resultDB $TMP/search_tmp \
            --search-type 3 \
            --strand 2 \
            --mask-lower-case 1 \
            --split-memory-limit {params.split_memory_limit} \
            -s {params.sensitivity} \
            --max-accept {params.max_accept} \
            --threads {threads}

        mmseqs convertalis \
            $TMP/queryDB $TMP/targetDB $TMP/resultDB {output.tsv} \
            --format-output "query,target,tstart,tend,bits,evalue,fident,qcov,tcov"
        """


rule project_intervals_mmseqs2:
    """Project mmseqs2 hits to target-coord intervals and keep one per query."""
    input:
        tsv="results/interval_alignment/{name}/{g}.hits.tsv",
    output:
        "results/intervals/{name}/{g}.parquet",
    wildcard_constraints:
        g=f"(?!{HUMAN_GENOME}).*",
    run:
        from marin_dna.pipelines.alignment.mmseqs2 import (
            best_hit_per_query,
            parse_mmseqs2_hits,
            project_hits_to_intervals,
        )

        hits = parse_mmseqs2_hits(input.tsv)
        projected = project_hits_to_intervals(hits)
        best = best_hit_per_query(projected)
        (
            best.select(["chrom", "start", "end"])
            .sort(["chrom", "start", "end"])
            .write_parquet(output[0])
        )
        print(
            f"  {wildcards.name} → {wildcards.g}: "
            f"{best.height:,} mapped intervals "
            f"({projected.height:,} total alignments before best-hit dedup)"
        )


rule all_mapped_intervals_mammals_seg20:
    """Bulk target for cloud runs: every configured `interval_mappings`
    entry × every genome in the `mammals_seg20` set. Useful to produce
    the per-species parquets that recipe v30 consumes without triggering
    the full `rule all` HF-upload end-to-end build.
    """
    input:
        expand(
            "results/intervals/{name}/{g}.parquet",
            name=list(MAPPINGS.keys()),
            g=genome_sets.get("mammals_seg20", []),
        ),


rule mapped_intervals_parquet_to_recipe_bed:
    """Adapter: convert unified-namespace mapped parquet to the bed.gz path
    the existing dataset pipeline expects.

    Wildcard-constrained to configured mapping names, so this rule never
    collides with the legacy `intervals_recipe_v{N}` rules (v1..v19), which
    produce their `.bed.gz` directly.
    """
    input:
        "results/intervals/{name}/{g}.parquet",
    output:
        "results/intervals/recipe/{name}/{g}.bed.gz",
    run:
        GenomicSet.read_parquet(input[0]).write_bed(output[0])
