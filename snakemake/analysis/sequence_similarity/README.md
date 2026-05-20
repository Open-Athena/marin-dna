# Sequence Similarity Analysis for Train/Test Leakage Detection

This pipeline analyzes sequence similarity between training and validation splits across different taxonomic scales to understand potential data leakage in genomic language model training.

## Background

When training genomic language models on multi-species datasets, traditional train/test splits (e.g., holding out a chromosome) may not adequately prevent data leakage. Similar sequences can exist:

1. **Within a genome**: Duplicated genes, paralogs, repetitive elements
2. **Across species**: Orthologs, conserved regulatory elements

This is a well-known problem in both protein and genomic foundation models.

### Prior Work and References

| Model/Paper | Approach | Thresholds |
|------------|----------|------------|
| **ESM2** (protein) | Remove train seqs matching val | 50% identity |
| **Fungal DNA LM** (Chao et al.) | Minimap2 alignment, remove train seqs | Coverage ≥5% AND Identity ≥30% |
| **hashFrag** | BLAST + union-find clustering | Alignment score threshold |

**Key References:**

- **ESM2**: [Lin et al., Science 2023](https://www.science.org/doi/10.1126/science.ade2574)
  > "All train sequences which match a validation sequence with 50% sequence identity under this search are removed from the train set."

- **Fungal DNA Language Model**: [Chao, Kuan-Hao et al., bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.09.19.677475v1)
  > "We aligned every training sequence against both the validation and test sets using Minimap2 (v2.28-r1209; assembly-to-assembly mode, '-x asm'). We removed any training sequence for which both coverage ≥ 5% and identity ≥ 30% in any alignment."

- **hashFrag**: [de Boer Lab, bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.01.22.634321v1)
  > Analysis of homology-based data leakage in genome-trained sequence models.

## Goals

1. **Quantify leakage**: Measure what percentage of training/validation sequences have similar sequences across splits at various thresholds.

2. **Compare across taxonomic scales**: Using datasets with increasing phylogenetic diversity (humans → primates → mammals → vertebrates → animals), understand how cross-species similarity contributes to potential leakage.

3. **Inform filtering thresholds**: Determine appropriate identity/coverage thresholds for filtering similar sequences in future dataset creation.

## Implementation

This pipeline uses [MMseqs2](https://github.com/soedinglab/MMseqs2) `search` to find direct pairwise alignments between validation (query) and training (target) sequences. Key flags:

- **`--mask-lower-case 1`**: excludes soft-masked repeats (from RepeatMasker) from k-mer seeding, so that similarity reflects genuine homology rather than shared transposable elements.
- **`--strand 2`**: searches both forward and reverse complement strands, so reverse complements are detected without needing to pre-canonicalize sequences.
- **`--search-type 3`**: forces nucleotide mode (MMseqs2 otherwise auto-detects from character frequencies, which can misclassify).

### Pipeline Flow

```
1. Download datasets from HuggingFace
   └── Convert to FASTA format
   └── Filter out reverse complement rows (_- suffix)

2. MMseqs2 Search Analysis:
   └── Create separate query (val) and target (train) databases
   └── Search val against train at multiple identity × coverage thresholds
   └── Count direct train hits per val sequence

3. Generate summary statistics and visualizations
```

### Reverse Complement Handling

DNA sequences and their reverse complements encode the same information. The HuggingFace datasets contain both strands (original with `_+` suffix, reverse complement with `_-` suffix). This pipeline:

1. **Filters out `_-` rows at download time** — keeps only the forward strand sequences.
2. **Uses `--strand 2`** in mmseqs2 search — searches both strands automatically, so a val sequence will match a train sequence regardless of which strand was stored.

This is simpler and more correct than the previous approach of canonicalizing sequences (picking the lexicographically smaller of seq/RC), which required careful case-preserving comparisons to avoid destroying soft-masking.

## Installation

### Dependencies

MMseqs2 is managed automatically by Snakemake:
- **MMseqs2**: Installed via conda environment (`workflow/envs/mmseqs2.yaml`), configured in the default profile

You only need [uv](https://docs.astral.sh/uv/) and the project's Python dependencies:

```bash
uv sync
```

### Verify Installation

```bash
uv run python -c "from datasets import load_dataset; print('OK')"
uv run snakemake --version
```

## Usage

### Quick Start

```bash
cd snakemake/analysis/sequence_similarity

# Dry-run (validate workflow without executing)
uv run snakemake -n

# 1. Run synthetic tests (masking + strand)
uv run snakemake test_masking --resources mem_mb=512

# 2. Run sanity check (val self-similarity, promoters only)
uv run snakemake sanity_check

# 3. Run the full pipeline (sanity check + leakage analysis + plots)
uv run snakemake

# Or run just the leakage analysis:
uv run snakemake analysis
```

> **Note**: The default profile configures `--use-conda` automatically to install MMseqs2 via conda.
> Conda envs are cached after the first run.

### Tests

Run the synthetic masking + strand tests to verify MMseqs2 flags work correctly:

```bash
uv run snakemake test_masking --resources mem_mb=512
```

#### Repeat masking test

Genomic sequences contain repetitive elements (transposons, satellites, etc.) that can cause
spurious similarity between unrelated sequences. The input data is expected to be
**pre-masked by RepeatMasker**, where lowercase letters indicate repetitive regions and
uppercase letters indicate non-repetitive ("unique") sequence.

MMseqs2's `--mask-lower-case 1` flag excludes lowercase regions from k-mer matching, so
that search reflects genuine homology rather than shared repeats.

The test generates 6 synthetic 256bp sequences (`results/tests/synthetic.fasta`) and
tests them with `mmseqs search`, with and without lowercase masking:

| Sequence | Description |
|----------|-------------|
| `genuine_A` | Random uppercase DNA (non-repetitive) |
| `genuine_B` | `genuine_A` with ~10% point mutations |
| `repeat_only_A` | Random uppercase flanks + **lowercase** repeat (~200bp) in middle |
| `repeat_only_B` | **Different** uppercase flanks + **same** lowercase repeat |
| `mixed_A` | Copy of `genuine_A` with a ~50bp lowercase repeat inserted |
| `mixed_B` | Copy of `genuine_B` with the same ~50bp lowercase repeat inserted |

The "repeat" block is a random high-complexity lowercase sequence (simulating a transposable
element fragment), not a simple dinucleotide repeat (see gotchas below).

**Expected results** (A sequences as query, B sequences as target):

| Pair | `--mask-lower-case 0` | `--mask-lower-case 1` |
|------|----------------------|----------------------|
| `genuine_A` → `genuine_B` (true homologs) | Hit | Hit |
| `repeat_only_A` → `repeat_only_B` (share only repeat) | Hit | **No hit** |
| `mixed_A` → `mixed_B` (homologs + repeat) | Hit | Hit |

The key assertion: with masking on, sequences that share **only** a repeat (and have
unrelated flanking regions) should **not** match, while sequences with genuine homology
should still match regardless of masking.

#### Strand test

Validates that `--strand 2` finds reverse complement matches:

| Search mode | Query | Target | Expected |
|-------------|-------|--------|----------|
| `--strand 1` (forward only) | seq_A | RC(A) | **No hit** |
| `--strand 2` (both strands) | seq_A | RC(A) | **Hit** |

This validates the core assumption that we can drop sequence canonicalization and rely on
mmseqs2 `--strand 2` to search both strands.

#### MMseqs2 gotchas discovered during development

1. **`linclust` silently ignores `--mask-lower-case`.** Only `mmseqs cluster` and
   `mmseqs search` (which use the cascade prefilter + alignment pipeline) honor the flag.
   The pipeline uses `mmseqs search` (not `linclust`) to ensure consistent repeat masking.

2. **`--mask-lower-case` must be passed to both `createdb` and `search`.** The
   `createdb` step stores masking metadata in the database; the `search` step
   uses it during k-mer seeding. Passing the flag only to `search` has no effect.

3. **`mmseqs search` needs `--search-type 3` to reliably force nucleotide mode.** MMseqs2
   auto-detects nucleotide vs protein from character frequencies, but the heuristic can
   misclassify. Passing `--search-type 3` explicitly avoids this.

4. **Simple dinucleotide repeats (e.g., `atatat...`) are not suitable test sequences.**
   They produce only 2 unique k-mers (for any k), which makes them behave unpredictably
   in k-mer-based algorithms. The test uses a random high-complexity lowercase block
   (simulating a realistic transposable element fragment) to isolate `--mask-lower-case`
   behaviour from MMseqs2's built-in compositional bias filtering (`--mask`).

5. **`--mask` (compositional bias / tandem repeat masking) is a separate parameter** from
   `--mask-lower-case`. The former uses an algorithm similar to DUST/SEG to detect and mask
   low-complexity regions at runtime; the latter uses pre-existing case information in the
   input FASTA. Both default to 0 in `linclust` but `--mask` defaults to 1 in `cluster`.

### Sanity Check

Before running the full analysis, run the sanity check to verify the pipeline works:

```bash
uv run snakemake sanity_check
```

This searches validation sequences against themselves (promoters only), including self-hits.

#### Sliding window structure

The dataset creation pipeline generates 256bp windows with 128bp step around each TSS. For a typical TSS this produces 3 windows:

```
TSS
 |
 (-256, 0)   (-128, 128)   (0, 256)
    A             B            C
         128bp overlap   128bp overlap
```

Adjacent windows share 128/256 = 50% of their bases. For a 3-window group, the expected match counts (including self) are:
- **A**: matches self + B = **2**
- **B**: matches self + A + C = **3**
- **C**: matches self + B = **2**

Some TSSes produce more than 3 windows when multiple TSSes are close together and their windows overlap, forming longer chains.

#### Expected results

- **At cov ≤ 0.5**: the 50% sliding window overlap passes the coverage threshold, so ~97% of sequences should have matches (self + at least one neighbor). The remaining ~3% are sequences that are 100% lowercase (entirely repeat-masked) — mmseqs2 cannot seed any k-mers from them, so even the self-hit fails.

- **At cov = 0.7**: the 50% overlap does not pass the 70% coverage threshold, so only sequences with genuine similarity to other loci (e.g. segmental duplications, gene families) will match beyond themselves.

- **Identity 0.3 vs 0.5**: results are identical — there is no additional signal below 50% identity for 256bp DNA.

#### Sanity check results (`humans_promoters`, 7,015 val sequences)

| Identity | Coverage | Seqs with matches | % with matches | Median | Mean | Max |
|----------|----------|-------------------|----------------|--------|------|-----|
| 0.3 | 0.3 | 6,827 | 97.3% | 2 | 3.9 | 92 |
| 0.5 | 0.3 | 6,827 | 97.3% | 2 | 3.9 | 92 |
| 0.3 | 0.5 | 6,827 | 97.3% | 2 | 3.4 | 67 |
| 0.5 | 0.5 | 6,827 | 97.3% | 2 | 3.4 | 67 |
| 0.3 | 0.7 | 6,827 | 97.3% | 1 | 1.7 | 35 |
| 0.5 | 0.7 | 6,827 | 97.3% | 1 | 1.7 | 35 |

The 188 sequences (2.7%) with zero matches are all 100% lowercase (entirely repeat-masked).

At cov ≤ 0.5, the median of 2 reflects a typical edge window in a 3-window TSS group (self + one neighbor). The mean is higher because center windows match 3 (self + two neighbors), and longer chains from closely-spaced TSSes contribute more. At cov = 0.7, the median drops to 1 (self-hit only) because the 50% sliding window overlap no longer passes the coverage threshold. The remaining matches above 1 at cov = 0.7 come from genuinely similar sequences at different loci (e.g. segmental duplications, gene families).

### Configuration

Edit `config/config.yaml` to customize:

```yaml
# Datasets to analyze
datasets:
  # Promoters (intervals-v1)
  - name: humans_promoters
    hf_path: bolinas-dna/genomes-v4-genome_set-humans-intervals-v1_256_128
  - name: primates_promoters
    hf_path: bolinas-dna/genomes-v4-genome_set-primates-intervals-v1_256_128
  - name: mammals_promoters
    hf_path: bolinas-dna/genomes-v4-genome_set-mammals-intervals-v1_256_128
  # CDS (intervals-v5)
  - name: humans_cds
    hf_path: bolinas-dna/genomes-v4-genome_set-humans-intervals-v5_256_128
  - name: primates_cds
    hf_path: bolinas-dna/genomes-v4-genome_set-primates-intervals-v5_256_128
  - name: mammals_cds
    hf_path: bolinas-dna/genomes-v4-genome_set-mammals-intervals-v5_256_128

# MMseqs2 parameters (identity × coverage grid)
mmseqs2:
  identity_thresholds: [0.3, 0.5]
  coverage_thresholds: [0.3, 0.5, 0.7]

# Analysis settings
analysis:
  sequence_column: seq
```

### Running Specific Steps

```bash
# Only download data
uv run snakemake results/data/humans_promoters/metadata.parquet

# Only run search for one dataset/threshold combo
uv run snakemake results/search/humans_promoters/0.5/0.5/hits.tsv
```

## Output

### Directory Structure

```
results/
├── data/
│   └── {dataset}/
│       ├── train.fasta              # Training sequences (forward strand only)
│       ├── validation.fasta         # Validation sequences (forward strand only)
│       └── metadata.parquet         # Sequence IDs and split labels
│
├── sanity_check/                    # Validation self-similarity (promoters only)
│   └── humans_promoters/
│       ├── {identity}/{coverage}/
│       │   ├── resultDB*            # Binary result database
│       │   ├── hits.tsv             # Pairwise hits
│       │   └── stats.parquet        # Per-threshold statistics
│       └── summary.parquet
│
├── search/                          # Leakage analysis
│   ├── {dataset}/
│   │   ├── queryDB*                 # Validation sequence database
│   │   ├── targetDB*                # Training sequence database
│   │   └── {identity}/{coverage}/
│   │       ├── resultDB*            # Binary result database
│   │       ├── hits.tsv             # Pairwise hits
│   │       └── stats.parquet        # Per-threshold statistics
│   └── summary.parquet
│
├── tests/                           # Synthetic tests
│   ├── synthetic.fasta              # 6 synthetic sequences
│   ├── search_query.fasta           # Query sequences (A) for search test
│   ├── search_target.fasta          # Target sequences (B) for search test
│   ├── search_hits_masklc0.tsv      # Search hits without masking
│   ├── search_hits_masklc1.tsv      # Search hits with masking
│   ├── search_masking_test_summary.txt  # Search test pass/fail assertions
│   ├── strand_query.fasta           # Query for strand test
│   ├── strand_target.fasta          # Target (RC) for strand test
│   ├── strand1_hits.tsv             # --strand 1 results (expect empty)
│   ├── strand2_hits.tsv             # --strand 2 results (expect hit)
│   └── strand_test_summary.txt      # Strand test pass/fail assertions
│
└── plots/
    ├── train_matches_median.svg     # Median train matches heatmap
    ├── train_matches_mean.svg       # Mean train matches heatmap
    └── pct_filtered_train.svg       # % train seqs filtered heatmap
```

### Interpreting Results

The key metric is **train_matches**: how many training sequences are direct alignment hits for each validation sequence at a given identity × coverage threshold. This directly answers the question "which train sequences are similar to this val sequence?" — exactly what's needed for filtering.

## Results

Leakage analysis tables and observations live in [issue #28](https://github.com/Open-Athena/marin-dna/issues/28). The pipeline emits per-threshold statistics to `results/search/summary.parquet`.

## Filtering workflow

1. Run `mmseqs search` with val as query, train as target
2. Collect train IDs with hits
3. Remove them from the training set

## Next Steps

1. **Expand to additional taxonomic scales** (vertebrates, animals) to see how leakage
   scales with further phylogenetic distance.

2. **Implement filtering in the dataset creation pipeline**: after creating train/validation
   splits, remove training sequences that exceed similarity thresholds against validation.
   This follows the ESM2/Chao et al. approach: preserve validation set, filter training.

See [Issue #28](https://github.com/Open-Athena/marin-dna/issues/28) for the full implementation plan.

## References

1. Lin, Z. et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123-1130. https://doi.org/10.1126/science.ade2574

2. Chao, K.-H. et al. (2025). Predicting dynamic expression patterns in budding yeast with a fungal DNA language model. *bioRxiv*. https://doi.org/10.1101/2025.09.19.677475

3. Rafi, A.M. et al. (2025). Detecting and avoiding homology-based data leakage in genome-trained sequence models. *bioRxiv*. https://doi.org/10.1101/2025.01.22.634321

4. Steinegger, M. & Söding, J. (2017). MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. *Nature Biotechnology*, 35(11), 1026-1028.

5. Li, H. (2018). Minimap2: pairwise alignment for nucleotide sequences. *Bioinformatics*, 34(18), 3094-3100.
