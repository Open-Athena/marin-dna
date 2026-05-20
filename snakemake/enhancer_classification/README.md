# Enhancer Classification

Binary enhancer classifier. For design decisions, results, and iteration
history see [#96](https://github.com/Open-Athena/marin-dna/issues/96).

This pipeline also includes a **per-bin enhancer segmentation** formulation
(issue [#115](https://github.com/Open-Athena/marin-dna/issues/115)) that
shares the conserved-enhancer definition with the classifier but predicts one
logit per 128 bp bin inside a 16384 bp window.

## Dataset output schema

### Classification (255 bp windows)

| Column | Type | Description |
|--------|------|-------------|
| `genome` | str | Species (`"homo_sapiens"` or `"mus_musculus"`) |
| `chrom` | str | Chromosome (Ensembl naming: `"1"`, `"X"`, etc.) |
| `start` | int | 0-based start coordinate |
| `end` | int | End coordinate (exclusive) |
| `strand` | str | `"+"` (forward) or `"-"` (reverse complement) |
| `seq` | str | DNA sequence (255 bp) |
| `label` | int | 1 = enhancer, 0 = non-enhancer |

### Segmentation (16384 bp windows, 128 × 128 bp bins)

| Column | Type | Description |
|--------|------|-------------|
| `genome` | str | Species |
| `chrom` | str | Chromosome |
| `start` | int | 0-based window start |
| `end` | int | Window end (= start + 16384) |
| `strand` | str | `"+"` or `"-"` (RC augmentation reverses `labels`) |
| `seq` | str | DNA sequence (16384 bp) |
| `labels` | list[uint8] | Per-bin label (length 128); 1 if ≥50 % of the bin overlaps a conserved enhancer |

## Code layout

| File | Description |
|------|-------------|
| `src/marin_dna/enhancer_classification/{dataset,model,train}.py` | 255 bp binary classifier |
| `src/marin_dna/enhancer_segmentation/{dataset,model,train}.py` | Per-bin segmenter (Conv1d head on encoder) |
| `src/marin_dna/enhancer_segmentation/labeling.py` | `label_windows_by_bin_overlap` — bin-level labels from enhancer intervals |
| `workflow/rules/model.smk` | Classifier training rules |
| `workflow/rules/segmentation.smk` | Segmentation data build + training rules |

## Prerequisites

- AWS credentials configured (EC2 IAM role or `aws configure`)
- **Single GPU required** for training (multi-GPU not supported)
- All available CPU cores are used for data loading (`threads: workflow.cores`)

## Usage

```bash
uv sync --group enhancer-classification
uv run snakemake
```

## Configuration

See `config/config.yaml`.
