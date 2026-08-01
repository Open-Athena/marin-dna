# exp429: paired SAE responses across CDS and splice variants

This permanent, unmerged experiment begins the paired-variant feature map in
issue #429, informing the durable research question in issue #288. The first tier
tests whether reference-to-alternate changes in the frozen m5.1 block-10 SAE
distinguish coding and splice consequence subclasses. The analysis and results
are tracked in GitHub issue #429.

The first panel is intentionally limited to chromosome 21. Dataset positions
remain 1-based in the frozen panel and are converted to 0-based coordinates only
at the FASTA boundary during sequence extraction.

## Frozen panel

Download the pinned chr21 shard once, then build the deterministic balanced
panel:

```bash
uv run hf download songlab/hg38-variant-consequences 21.parquet \
  --repo-type dataset \
  --revision eb3022cc6797b9369cca16af72ff3c4197df343a \
  --local-dir ../../scratch/issue429/source

EXPERIMENT_COMMIT="$(git rev-parse HEAD)" uv run python sample_panel.py \
  --input ../../scratch/issue429/source/21.parquet \
  --output ../../scratch/issue429/retrieval/panel/panel.parquet \
  --fasta ../../scratch/issue418/reference/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz
```

The sampler uses deterministic 1 Mb genomic-block splits and selects 1,024
discovery, 512 validation, and 512 held-out variants from each of 11
`consequence_cre` classes: missense, synonymous, start loss, stop gain/loss,
splice region, polypyrimidine tract, donor region, donor, acceptor, and donor fifth base. This first pinned source is SNV-only, so frameshifts require a later indel panel rather than being silently approximated here. Before exact
selection, every oversampled candidate
must have an A/C/G/T-only 255 bp GRCh38 window and an exact center-REF match;
invalid candidates are recorded in the manifest and deterministically replaced
within the same class/split. Reuse the written panel and its manifest unchanged
when comparing future SAE layers, training budgets, seeds, or dictionaries. To
build it on the CPU worker from the same pinned commit, run:

```bash
COMMIT="$(git rev-parse HEAD)"
uv run python launch.py panel --commit "$COMMIT" --execute
```

## Tests

```bash
uv lock
uv sync --frozen
uv run pytest
uv run ruff check .
```

## GPU extraction

`extract.py` preserves forward and reverse-complement results separately. For
each orientation it writes sparse reference/alternate SAE activations and a
dense raw-residual delta array. The registered first pass scores the edited
nucleotide for every SAE feature; after discovery/validation selects candidates,
a second bounded pass will save their full local position profiles. This staging
avoids materializing a feature-by-position tensor for every feature and variant.
`manifest.json` pins the model revision, block,
SAE weights and configuration, panel/source hashes, protocol, and every output
hash so the same panel can be compared across later SAE versions.

The checked-in Sky task mounts the frozen panel, issue #418 SAE, and GRCh38
reference. `launch.py` always uses `sky launch`, including on a warm cluster, so
the pinned checkout and environment setup cannot be skipped. Print the command
without mutating cloud state:

```bash
COMMIT="$(git rev-parse HEAD)"
uv run python launch.py extract --commit "$COMMIT"
```

The current #288 session already authorizes one CPU and one GPU concurrently.
Launch with the commit containing the extractor:

```bash
uv run python launch.py extract --commit "$COMMIT" --execute
```

The task uses one EC2 A10G, validates CUDA during setup, auto-stops after 30
idle minutes, and writes results under
`~/exp429-artifacts/extraction/`. Download that directory
before the cluster is torn down.

The base model runs under `torch.inference_mode()` in bfloat16 and the SAE
encoder in float32. Batches are constructed directly because every example is a
fixed 255-bp pair and extraction must preserve the requested hook cache. A
Transformers `Trainer` would not simplify this loop. `torch.compile` is
deliberately disabled: eager execution is validated for the pinned dynamic hook
path, whereas the earlier compile attempt did not preserve that cache correctly.
The A10G launch uses batch size 32; GPU utilization is checked during the first
few minutes. After discovery/validation selection, the immediate candidate-only
spatial pass saves ref and alt activations for the 18 frozen feature IDs at
relative positions -15 through +15:

```bash
COMMIT="$(git rev-parse HEAD)"
uv run python launch.py spatial --commit "$COMMIT" --execute
```

`analyze_spatial.py` reverses the RC position axis into genomic coordinates
before constructing signed-mean or max-absolute FWD/RC views. Feature ID,
direction, and signed/absolute transform remain frozen from the focal analysis.
Within each candidate it uses validation blocks only to choose among the focal
base, the strongest oriented response in the ±15 bp window, and the integrated
local response; it then uses validation again to choose one orientation-specific
candidate per class. The untouched test blocks are read once for final AUPRC,
1,000 genomic-block bootstrap intervals, and class-minus-background position
profiles. Its CLI takes `--panel`, `--focal-analysis-dir`, `--spatial-dir`,
`--output-dir`, and optional `--bootstrap-samples`.

After the spatial result is frozen, `inspect_candidates.py` uses validation AP
only to select strong positive-direction candidates, then inspects sequence
grammar on discovery variants only. For each variant it orients the 31 bp
REF/ALT context by the FWD or RC contribution that produced the max-absolute
score. It compares the top 128 responses with the remaining variants from the
same consequence class, exporting contexts, base frequencies, substitution
frequencies, and an enrichment heatmap. This is a hypothesis-generation step;
held-out test scores are carried through from the prior analysis and are not
used to select sequences.

`annotate_coding_candidates.py` then reuses the tested Ensembl-109
transcript/codon reconstruction from exp428 for the stop-gain and synonymous
discovery contexts. It reports annotation coverage, codon phase, transcript-
oriented substitutions, and unambiguous REF→ALT codon pairs for the top versus
remaining same-class responses.

`design_perturbations.py` turns the discovery-derived interpretations into a
bounded causal panel. For each acceptor, donor-fifth-base, stop-gain, and
synonymous candidate it retains 16 strongest contexts plus 16 deterministic
rank-spaced same-class controls. Splice contexts receive all three substitutions
at response-oriented positions -12 through +4. Coding contexts receive every
other transcript-oriented codon, labeled as synonymous, missense, or stop gain.

`extract_perturbations.py` evaluates the five frozen feature IDs 3312, 4281,
6072, 11681, and 11698 at positions -15 through +15 in FWD and RC. It
deduplicates the 14,592 paired sequence states to 7,424 unique model forwards,
then preserves row-level reference and alternate indices for paired analysis.
The same validated eager hook path runs in bfloat16 for the base model and
float32 for the SAE.

```bash
COMMIT="$(git rev-parse HEAD)"
uv run python launch.py perturbations --commit "$COMMIT" --execute
```

After retrieving and hash-validating `perturbation-extraction-r1`, run the
frozen causal analysis. It reports splice position-saturation curves and
within-context codon selectivity with source-context bootstrap intervals; no
feature, reducer, position, or codon state is selected from these outputs.

```bash
export PERTURBATION_ANALYSIS_COMMIT="$(git rev-parse HEAD)"
uv run python analyze_perturbations.py \
  --panel ../../scratch/issue429/retrieval/perturbation-design-r1/perturbation_panel.parquet \
  --design-manifest ../../scratch/issue429/retrieval/perturbation-design-r1/manifest.json \
  --extraction-dir ../../scratch/issue429/retrieval/perturbation-extraction-r1 \
  --output-dir ../../scratch/issue429/perturbation-analysis-r1
```

## Held-out analysis

After retrieving and hash-validating the extraction directory, run the sparse
individual-feature, SAE-only multiclass-probe, context, and plot
analysis locally:

```bash
export ANALYSIS_COMMIT="$(git rev-parse HEAD)"
uv run python analyze.py \
  --extraction-dir ../../scratch/issue429/retrieval/extraction \
  --panel ../../scratch/issue429/retrieval/panel/panel.parquet \
  --fasta ../../scratch/issue418/reference/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz \
  --output-dir ../../scratch/issue429/analysis \
  --probe-jobs 1
```

Feature and transform selection use discovery/validation blocks only. Test AUPRC
and macro-F1 are read once, with AUPRC intervals bootstrapped over held-out 1 Mb
blocks. An interval is omitted when a class has positives in fewer than two test
blocks because spatial uncertainty is not identifiable. The script reports FWD and RC separately, uses signed same-ID mean as the primary
aggregate, reports `max(|ΔFWD|, |ΔRC|)` as the declared sensitivity view, then
writes activating
reference/alternate contexts for the selected SAE features.

For the SAE-only multiclass check, `sky.analysis.yaml` uses the same warm
8-vCPU CPU class and parallelizes only the deterministic one-vs-rest class fits. Every
orientation/transform receives the same fixed 100 SGD epochs:

```bash
uv run python launch.py analyze --commit "$COMMIT"
uv run python launch.py analyze --commit "$COMMIT" --execute
```

Retrieve `~/exp429-artifacts/analysis/` before terminating
the cluster.
