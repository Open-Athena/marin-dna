---
license: apache-2.0
library_name: transformers
tags:
  - biology
  - genomics
  - dna
---

# MarinDNA m5.1 1B base model

MarinDNA m5.1 is a 1.12B-parameter, nucleotide-level causal language model developed with [Marin](https://github.com/marin-community/marin). This is the final m5.1 base-model checkpoint at step 59,158 from run [`dna-bolinas-mix-v0.9-p1B-i24-exp135-zoonomia-m5.1-bef41e`](https://wandb.ai/eric-czech/marin/runs/dna-bolinas-mix-v0.9-p1B-i24-exp135-zoonomia-m5.1-bef41e), released with the [Genomic Language Model Optimization](https://openathena.ai/blog/genomic-lm-optimization/) blog post.

## Model details

| Field | Value |
|---|---|
| Architecture | `Qwen3ForCausalLM`-compatible decoder-only Transformer |
| Parameters | 1,120,772,224 |
| Layers | 19 |
| Hidden / intermediate size | 1,920 / 7,680 |
| Attention heads / KV heads | 15 / 15 |
| Context | 256 tokens: one BOS token followed by up to 255 DNA bases |
| Checkpoint | Final m5.1 checkpoint, step 59,158 |
| Approximate token exposure | 166.0B nucleotide tokens over the inherited training lineage |
| Stored weight dtype | float32 |
| License | Apache-2.0 |

The source checkpoint is
`gs://marin-us-east5/checkpoints/dna-bolinas-mix-v0.9-p1B-i24-exp135-zoonomia-m5.1-bef41e/hf/step-59158`.
The commit-pinned [training script](https://github.com/marin-community/marin/blob/a41a83fdddfdef85a75e39b56c32949518e3f578/experiments/dna/exp135_bolinas_mix_sweep.py#L514-L657) defines the experiment.

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "bolinas-dna/marin-dna-exp135-m5.1"

tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(repo_id)
```

The float32 checkpoint is approximately 4.18 GiB. Convert or load it in a lower-precision dtype only if that is appropriate for your downstream use.

## Tokenizer and input format

The bundled tokenizer is a case-insensitive, single-nucleotide tokenizer with seven tokens:

| Token | ID |
|---|---:|
| `[PAD]` | 0 |
| `[UNK]` | 1 |
| `[BOS]` | 2 |
| `a` | 3 |
| `c` | 4 |
| `g` | 5 |
| `t` | 6 |

Pass raw DNA strings containing `A`, `C`, `G`, and `T` without spaces or separators. The tokenizer lowercases input and automatically prepends `[BOS]`; the model has no EOS token and the tokenizer does not append one. Characters outside the four canonical bases map to `[UNK]`. Because BOS occupies one of the 256 model positions, inputs are limited to 255 DNA bases.

## Training lineage and data

m5.1 is a continued-training lineage with approximately 166.0B total inherited and newly seen nucleotide tokens:

1. approximately 42.0B inherited tokens from the pre-cooldown point of a uniform three-region mixture (CDS, upstream, and downstream);
2. approximately 62.0B tokens from a continued uniform three-region mixture; and
3. approximately 62.0B tokens from a uniform five-region mixture adding enhancer and ncRNA sequence.

The five-region inventory below describes the resources used across the m5.1 lineage. It does **not** mean that all five datasets contributed to every training phase.

### Training datasets

- [CDS](https://huggingface.co/datasets/bolinas-dna/genomes-v5-genome_set-animals-intervals-v5_255_128/tree/ffe3e78c99868077c65ad6568e1445d80e480794)
- [Upstream](https://huggingface.co/datasets/bolinas-dna/genomes-v5-genome_set-animals-intervals-v1_255_128/tree/d93209847b02a0c9be5c03591a0a5e56ee09c35d)
- [Downstream](https://huggingface.co/datasets/bolinas-dna/genomes-v5-genome_set-animals-intervals-v15_255_128/tree/b009afaab756937d75b8da3b1271ad8f0cec0b4d)
- [Enhancer](https://huggingface.co/datasets/bolinas-dna/zoonomia-v1-v3_ccre_non_promoter/tree/862485aa18eed53a53e693ba4c2eb45e0afc5087)
- [ncRNA](https://huggingface.co/datasets/bolinas-dna/zoonomia-v1-v3_ncrna_exon/tree/3e48d9ae7c604b99ccfc8bd07e391b960c1ea21a)

### Matched training-validation probes

These matched datasets were validation probes, not training data.

- [CDS validation](https://huggingface.co/datasets/bolinas-dna/genomes-v5-validation-intervals-v5_255_255/tree/daff592f213aaa1cab1711d477a79ff6b1bc4ef4)
- [Upstream/promoter validation](https://huggingface.co/datasets/bolinas-dna/genomes-v5-validation-intervals-v1_255_255/tree/a761bc0b663a9827303f3112e4667d53d5326fac)
- [Downstream validation](https://huggingface.co/datasets/bolinas-dna/genomes-v5-validation-intervals-v15_255_255/tree/d7b27eecd68453934ebb3e7e6e78d5401789faa5)
- [Enhancer validation](https://huggingface.co/datasets/bolinas-dna/zoonomia-v1-val_enhancer/tree/d40d1e067b2a56ac812af122de029eb79cab1106)
- [ncRNA validation](https://huggingface.co/datasets/bolinas-dna/zoonomia-v1-val_ncrna/tree/76a18c1bbf07ac9bd064722431bbdab894b9e6c6)

## Intended uses

This checkpoint is intended for genomic language-model research, including sequence likelihood analyses, representation extraction, variant-effect research, and reproducible comparison with the experiments reported in the accompanying blog. It is a base model, not a task-specific predictor.

## Evaluation and caveats

The downstream evaluation-only datasets are [Mendelian variant effects](https://huggingface.co/datasets/bolinas-dna/evals_mendelian_traits/tree/4aed58e50c5dea0b878a665007af2ef9e5108e9f) and [saturation genome editing (SGE)](https://huggingface.co/datasets/bolinas-dna/evals_sge/tree/225d3d1ea32a4af547891b13c33b5e92a5aae849). They were not used as training data. Mendelian and SGE use different construction, matching, and aggregation protocols, so their score levels must not be compared directly.

The supported interpretation is documented in the [blog analysis hub](https://github.com/Open-Athena/marin-dna/issues/361) and the [collaborator-review dossier](https://github.com/Open-Athena/marin-dna/issues/370). In particular, the paired analysis does not support a claim that m5.1 significantly outperforms Evo 2 40B or AlphaGenome; those comparisons are statistical ties under the reported Mendelian analysis.

Additional limitations:

- The 255-base input window is short relative to many regulatory and structural genomic effects.
- The tokenizer represents only the four canonical DNA bases directly; other symbols become `[UNK]`.
- Performance depends on the downstream scoring or probing protocol and can vary substantially by variant category.
- This research checkpoint has not been validated for clinical diagnosis or medical decision-making.

## Provenance and reproducibility

- [Genomic Language Model Optimization blog](https://openathena.ai/blog/genomic-lm-optimization/)
- [Blog analysis and figures tracker](https://github.com/Open-Athena/marin-dna/issues/361)
- [Figures 5–11 review dossier and supported interpretations](https://github.com/Open-Athena/marin-dna/issues/370)
- [Commit-pinned training script](https://github.com/marin-community/marin/blob/a41a83fdddfdef85a75e39b56c32949518e3f578/experiments/dna/exp135_bolinas_mix_sweep.py#L514-L657)

The tracked release manifest records source and destination inventories, byte sizes, checksums, model configuration, tokenizer details, dataset revisions, and the deterministic inference reference.
