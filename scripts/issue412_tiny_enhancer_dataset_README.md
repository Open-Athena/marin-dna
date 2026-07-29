---
tags:
  - biology
  - genomics
  - dna
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.jsonl
---

# Tiny Zoonomia non-promoter cCRE tutorial sample

This dataset contains 256 unchanged rows from the public
[`marin-dna/zoonomia-v1-v3_ccre_non_promoter`](https://huggingface.co/datasets/marin-dna/zoonomia-v1-v3_ccre_non_promoter)
cross-mammal enhancer dataset. It exists only to keep MarinDNA's local CPU
training tutorial fast; it is not an independent biological dataset or a
representative benchmark.

## Contents

`data/train.jsonl` contains the first 256 records from source shard
`data/train/shard_0000.jsonl.zst` at immutable source revision
`862485aa18eed53a53e693ba4c2eb45e0afc5087`. The source dataset was globally
shuffled with seed 42 before sharding.

Every record retains the source schema:

| Field | Description |
| --- | --- |
| `query_name` | Human anchor identifier. |
| `species` | Zoonomia target species. |
| `t_chrom` | Target chromosome. |
| `t_start`, `t_end` | 0-based, half-open target interval; always 255 bp. |
| `t_strand` | Projection strand (`+` or `-`). |
| `t_src_size` | Target chromosome size. |
| `sequence` | Strand-aware 255 bp DNA sequence. |
| `augmentation` | Original (`+`) or reverse-complement (`-`) augmentation. |

## Provenance

The deterministic builder copies and validates the rows without transforming
their contents:

[`scripts/issue412_build_tiny_enhancer_dataset.py`](https://github.com/Open-Athena/marin-dna/blob/cc528f3cb493a6703743339212e5233186c6559d/scripts/issue412_build_tiny_enhancer_dataset.py)

The source dataset card documents the ENCODE cCRE V4 non-promoter labels,
Zoonomia projection, reverse-complement augmentation, and upstream pipeline.

