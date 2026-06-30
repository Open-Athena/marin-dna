# Dataset Creation Pipeline

This pipeline takes a list of genomes and creates training datasets by extracting genomic intervals, creating sliding windows, and uploading to HuggingFace.

## What it does

1. **Download genomes** - Downloads genome assemblies and GTF annotations from NCBI
2. **Extract intervals** - Identifies genomic regions based on recipe:
   - v1: Promoters (256bp upstream + 256bp downstream of TSS)
   - v2: mRNA exons + promoters (with flanking regions)
   - v3: CDS regions only
3. **Create windows** - Generates sliding windows over extracted regions
4. **Extract sequences** - Retrieves DNA sequences for each window
5. **Create training parquets** - All sequences from all genomes go to training (no chromosome-based split)
6. **Create validation parquet** - Conservation-aware subsample of human sequences where base case encodes phyloP scores (uppercase iff phyloP >= threshold, lowercase otherwise)
7. **Merge and shard** - Combines data from multiple genomes and splits into shards
8. **Upload** - Uploads to HuggingFace Hub

## Genomic Region Extraction

The pipeline uses functions from `marin_dna.data.utils` to extract genomic regions. These functions handle cross-species annotation differences (e.g., `transcript_biotype` vs `gbkey` attributes).

### Available Region Types

| Function | Description | Returns |
|----------|-------------|---------|
| `get_cds(ann)` | Coding sequences | GenomicSet |
| `get_promoters(ann, up, down, mRNA_only)` | Promoter regions around TSS | GenomicSet |
| `get_5_prime_utr(ann)` | 5' untranslated regions | GenomicSet |
| `get_3_prime_utr(ann)` | 3' untranslated regions | GenomicSet |
| `get_ncrna_exons(ann, biotypes)` | Functional ncRNA exons | GenomicSet |
| `get_mrna_exons(ann)` | mRNA exons with transcript_id | pl.DataFrame |

### ncRNA Filtering Criteria

The `get_ncrna_exons` function includes functional ncRNAs while excluding non-functional annotations.

#### Annotation Attributes

NCBI annotations use two key attributes to classify transcripts:

| Attribute | Description | Example values |
|-----------|-------------|----------------|
| `gbkey` | High-level category | mRNA, ncRNA, misc_RNA, tRNA, rRNA, precursor_RNA |
| `transcript_biotype` | Specific transcript type | lnc_RNA, miRNA, snoRNA, transcript, primary_transcript |

Per-attribute cross-species counts and the functional/non-functional designations are tracked in [issue #36](https://github.com/Open-Athena/marin-dna/issues/36).

#### Included Biotypes

`DEFAULT_NCRNA_BIOTYPES`: lnc_RNA, miRNA, snoRNA, tRNA, snRNA, rRNA, antisense_RNA, ncRNA, scRNA, vault_RNA, Y_RNA, scaRNA, RNase_P_RNA, RNase_MRP_RNA, telomerase_RNA, SRP_RNA, piRNA

#### Excluded Categories

**By transcript_biotype (human exon counts):**
- `transcript` (167k exons) - misc_RNA, uncertain/uncharacterized transcripts
- `primary_transcript` (2.1k exons) - precursor RNAs before processing
- `V/J/D/C_gene_segment` (1.6k exons) - immunoglobulin segments (not ncRNA)
- `pseudogenic_tRNA`, `pseudogenic_rRNA` - pseudogenic ncRNAs (C. elegans specific)

**By quality attributes:**
- `pseudo "true"` (~15k exons in human) - annotated pseudogenes
- `partial "true"` (~1.3k exons) - incomplete annotations
- `gene_biotype` containing "pseudogene" - pseudogene annotations
- `description` or `product` containing "NMD candidate" (~1.8k exons) - nonsense-mediated decay targets
- `product` containing "LOW QUALITY" - explicitly flagged low quality

#### Cross-Species Consistency

The biotype/gbkey system is broadly consistent across species. Edge cases (e.g. C. elegans has a large piRNA set and uses the generic `ncRNA` biotype for unclassified non-coding RNAs) are tracked in [issue #36](https://github.com/Open-Athena/marin-dna/issues/36).

### Promoter Options

`get_promoters(ann, n_upstream, n_downstream, mRNA_only=False)`:
- `mRNA_only=True`: Promoters from protein-coding mRNA transcripts only
- `mRNA_only=False` (default): Promoters from mRNA + functional ncRNA transcripts

### Cross-Species Compatibility

Functions handle both annotation styles:
- `transcript_biotype` / `gene_biotype` (human, mouse, most species)
- `gbkey` (some NCBI genomes like C. elegans, Merops nubicus)

For CDS extraction, both `feature == "CDS"` and `gbkey == "CDS"` are checked to handle C. elegans-style annotations where gene names appear in the feature column.

### UTR Extraction

The `get_5_prime_utr` and `get_3_prime_utr` functions compute UTR regions by finding exon portions outside the CDS boundaries for each transcript:

- **5' UTR**: Exon regions before CDS start (+ strand) or after CDS end (- strand)
- **3' UTR**: Exon regions after CDS end (+ strand) or before CDS start (- strand)

**CDS exclusion**: Regions that are CDS in *any* transcript are excluded from UTRs, even if they are UTR in another transcript. This handles alternative transcripts where the same genomic region may be coding in one isoform but non-coding in another.

**Verification identity**: By definition, mRNA exons consist of CDS + 5' UTR + 3' UTR, so:

```
mRNA - CDS = 5' UTR | 3' UTR
```

This identity can be used to verify the correctness of UTR extraction:

```python
mrna_exons = GenomicSet(get_mrna_exons(ann))
cds = get_cds(ann)
utr5 = get_5_prime_utr(ann)
utr3 = get_3_prime_utr(ann)

# These should be equal (or nearly equal)
mrna_minus_cds = mrna_exons - cds
utr_union = utr5 | utr3

# Check differences
diff1 = mrna_minus_cds - utr_union  # Should be ~0
diff2 = utr_union - mrna_minus_cds  # Should be 0
```

**Edge cases**: A small number of intervals satisfy `mRNA - CDS ⊃ 5' UTR | 3' UTR` but not strict equality — typically 1bp gaps within CDS (small introns or frameshift annotations) or unusual annotations on unplaced scaffolds. Some genomes lack UTR or ncRNA annotations entirely (common in non-model organisms with coding-focused annotations); the pipeline handles these via empty `GenomicSet`s, which work correctly in all downstream operations. Per-genome details and analysis tables are tracked in [issue #36](https://github.com/Open-Athena/marin-dna/issues/36).

### Annotation Sources

NCBI genome annotations come from multiple sources with varying quality:

| Source | Description | Transcript Prefix |
|--------|-------------|-------------------|
| **BestRefSeq** | Curated, high-confidence | NM_, NR_ |
| **Gnomon** | Computational predictions | XM_, XR_ |
| **RefSeq** | Reference sequences | NM_, NR_ |
| **cmsearch** | RNA structure-based | - |
| **tRNAscan-SE** | tRNA predictions | - |
| **Curated Genomic** | Manual curation | - |

NCBI also provides MANE Select / RefSeq Select tags for "preferred" or "canonical" transcripts, but availability is limited to human and mouse.

**Current approach:** all transcripts are used regardless of source or selection tags. Filtering to curated transcripts would eliminate most data for non-model organisms. Cross-species distribution stats, per-region outlier-length analyses, and per-species curated-transcript availability are tracked in [issue #36](https://github.com/Open-Athena/marin-dna/issues/36).

To compute annotation source statistics:
```bash
uv run snakemake all_annotation_source_stats --cores 4
```

## Setup

Python dependencies are managed by the main project (see `../../../README.md` for installation).

Conda must be available for bioinformatics CLI tools.

### Storage

Pipeline results are stored in S3 (`s3://oa-bolinas/snakemake/training_dataset/dataset_creation/`). A default Snakemake profile at `workflow/profiles/default/config.yaml` configures S3 storage, conda, and cores automatically.

You need AWS credentials with S3 access:
- **On EC2**: Attach an IAM role with `AmazonS3FullAccess` to the instance
- **On your laptop**: Run `aws configure` with an IAM user's access key

## Configuration

Edit `config/config.yaml` to customize the pipeline:

### Required Parameters

- **`genomes_path`** - Path to filtered genome list (parquet file from genome_selection pipeline)

- **`intervals`** - Interval types to generate
  - `intervals.training`: list of `{recipe}/{window_size}/{overlap}` strings; built across the cartesian product of *every* `genome_set`.
  - `intervals.training_per_genome_set` (optional): mapping `{genome_set: [recipes...]}` for recipes that only apply to a specific genome_set (e.g., the cCRE recipes `v30`-`v33` only apply to the 20 mammals in `mammals_seg20`). For any genome_set listed here, the given recipes **replace** (not extend) the cartesian-product training recipes — so the listed set won't also get the default `intervals.training` recipes built.
  - `intervals.validation`: list of recipes used to build the conservation-aware human validation parquet.
  - Format: `{recipe}/{window_size}/{overlap}`
  - Examples:
    - `v1/512/256` - Promoters with 512bp windows and 256bp overlap
    - `v2/512/256` - mRNA + promoters with 512bp windows and 256bp overlap
    - `v3/512/256` - CDS regions with 512bp windows and 256bp overlap
  - Recipes:
    - **v1**: Promoters only (256bp upstream + 256bp downstream from TSS)
    - **v2**: mRNA exons + promoters
    - **v3**: CDS regions only

- **`genome_sets`** - Groupings for separate datasets. Each entry has a `name`, plus *either*:
  - `rank_key` + `rank_value` for taxonomic filtering (e.g., `{rank_key: "class", rank_value: "Mammalia"}`), or
  - `accessions`: an explicit list of Assembly Accessions (e.g., the 20 species in `mammals_seg20`).
  - Creates separate datasets for each group.

- **`output_hf_prefix`** - HuggingFace repository prefix (e.g., "username/dataset-name")

### Optional Parameters

- **`validation`** - Configuration for the conservation-aware validation set
  - `max_samples`: Maximum number of sequences to subsample (default: 16384)
  - `genome`: Genome accession to draw validation sequences from (default: GCF_000001405.40, human)
  - `conservation_bigwig`: Path to phyloP BigWig file for conservation scores
  - `phylop_threshold`: phyloP score threshold for uppercase encoding (default: 2.27)
  - `seed`: Random seed for subsampling (default: 42)

- **`shuffle_seed`** - Random seed for reproducibility (default: 42)

- **`n_shards`** - Number of shards to split dataset into (default: 64)
  - More shards = better parallelization

- **`add_rc`** - Add reverse complement sequences (default: true)
  - Doubles dataset size by including reverse complements

## Usage

```bash
# Ensure you have a genome list from the genome_selection pipeline
# Copy or link it to config/genomes.parquet

# Edit configuration file: config/config.yaml

# Run pipeline
uv run snakemake
```

## Interval projection across genomes (`interval_mappings`)

The pipeline supports defining an interval set on one source genome and projecting it onto other genomes via local pairwise alignment. This makes it possible to train on regions — like ENCODE cCRE enhancers — that only exist as a native annotation in a small number of species, while still getting per-species sequences.

Each configured mapping produces `results/intervals/{name}/{g}.parquet` for every genome. On the source genome the file is a chrom-normalized copy of the configured source parquet; on other genomes it is the result of alignment + best-hit-per-query projection. Downstream rules (windowing, FASTA extraction, sharding, HF upload) treat these the same as annotation-derived interval sets — origin is not visible past the projection step.

**Configuration** (`interval_mappings` block in `config.yaml`):

```yaml
interval_mappings:
    - name: ELS_conserved_20_mmseqs2_s75
      source_parquet: results/cre/ELS_conserved_20.parquet
      source_chrom_style: ucsc_stripped # bare-digit chroms; remap to RefSeq NC_*
      source_genome: GCF_000001405.40
      mapper: mmseqs2
      sensitivity: 7.5
      max_accept: 1
      split_memory_limit: "12G"  # override on big-mem cloud instances
      mem_mb: 14000
      flank_bp: 0
```

Naming convention: underscored, semantic `{source_name}_{mapper}_{preset}` (no dots — they're fragile in HF dataset IDs and Snakemake wildcards). Name a variant with different flags as its own entry, e.g. `ELS_conserved_20_mmseqs2_s75_flank100`.

**Referencing a mapping in the dataset config:** add its name wherever you would list a legacy recipe, e.g. `intervals.training: ["ELS_conserved_20_mmseqs2_s75/255/128"]`, pair with the `human_mouse` (or any suitable) genome_set, and the full windows → fasta → shards → HF upload flow runs unchanged.

**Resources:** mmseqs2 nucleotide search against a whole mammalian target genome needs ~50-80 GB resident at the full index, so real runs use a big-memory cloud instance (r6i.8xlarge, 256 GB). `split_memory_limit` lets a smaller box fit at the cost of wall time; the defaults shown above target the 15 GB dev box.

**Status:** v1 (issue [#123](https://github.com/Open-Athena/marin-dna/issues/123)): mmseqs2 only, flank 0, `-s 7.5 --max-accept 1`. Run time: ~2 min wall (createdb 7 s, search 110 s, convertalis 0.3 s) on r6i.8xlarge (32 vCPU, 256 GB RAM), ~26 GB peak RSS. Recall/precision benchmarks against alternative aligners are tracked in [#120](https://github.com/Open-Athena/marin-dna/issues/120) and [#123](https://github.com/Open-Athena/marin-dna/issues/123). Sensitivity sweep, flank sweep, soft-mask filtering, and alternative aligners (e.g. lastz for the high-recall end of the frontier) are left for future iterations.

### Source-curation sweep around v30 (recipes v31/v32/v33)

Following [#136](https://github.com/Open-Athena/marin-dna/issues/136), where projection-based curation (v30) outperformed segmentation-based curation, three sibling recipes vary the upstream cCRE conservation filter while keeping the projection method (mmseqs2 `-s 7.5`), target genome set (`mammals_seg20`), and downstream windowing/exon-masking identical to v30. Each is just a thin wrapper over a different `interval_mappings` entry.

| Recipe | Conservation track | Per-base cutoff | Per-cCRE filter | Source mapping |
|---|---|---|---|---|
| v30 (baseline) | phyloP-241way | ≥2.27 | ≥20 bp | `ELS_conserved_20_mmseqs2_s75` |
| v31 | phyloP-241way | ≥2.27 | ≥50 bp | `ELS_conserved_50_mmseqs2_s75` |
| v32 | phastCons-43p | ≥0.961 | ≥20 bp | `ELS_phastCons_43p_conserved_20_mmseqs2_s75` |
| v33 | phastCons-43p | ≥0.961 | ≥50 bp | `ELS_phastCons_43p_conserved_50_mmseqs2_s75` |

phastCons-43p is the Zoonomia 43-primate phastCons track; 0.961 is calibrated to a ~3.46% conserved-base fraction matching phyloP-241way 2.27 at the per-base level. The per-cCRE filter is absolute conserved bp (`pct_conserved × size ≥ N`); 20 bp is ~7% of the median ELS length (272 bp), 50 bp ~18%. Caveat: phastCons-43p selects primate-conserved cCREs by construction, so v32/v33 may project worse to non-primate mammals than v30/v31.

## Output

Datasets are uploaded to HuggingFace Hub at the specified `output_hf_prefix`.

Dataset naming format: `{output_hf_prefix}-genome_set-{genome_set}-intervals-{recipe}_{window_size}_{overlap}`

Example: `bolinas-dna/genomes-v5-genome_set-mammals-intervals-v1_255_128`

Local intermediate files are stored in `results/` (not committed to git).

### HuggingFace dataset cards (README.md)

Every uploaded repo ships a per-repo `README.md` dataset card. The card text is
generated by `marin_dna.pipelines.training_dataset.hf_readme`
(`build_training_readme` / `build_validation_readme`) and pushed by the
`hf_upload_training_readme` / `hf_upload_validation_readme` rules with a
*separate* `hf upload … README.md` call — `hf upload-large-folder` (used for the
shards) silently skips top-level files, which is why the cards need their own
upload step.

Each card documents the genome set, the region recipe, the schema, the
sequence **case-encoding** (training = soft-masked repeats; validation = phyloP
conservation — opposite conventions), and an **exact sample count**. The count
is read from the per-genome parquet footers (`count_parquet_rows`), not from
HuggingFace's auto-generated `num_examples`, which is frequently wrong for
sharded `JSONL.zst`. To edit a recipe's description, update the `RECIPE_BLURBS`
/ `GENOME_SET_BLURBS` tables in that module (covered by
`tests/pipelines/training_dataset/test_hf_readme.py`).
