# Carbon species conditioning on Mendelian VEP

## TL;DR

This exploratory pilot reports 2 prompt conditions on the `tss_proximal` subset.
No pass/fail outcome or testing hierarchy is assigned.

## Fixed contract

- Model: `HuggingFaceBio/Carbon-3B` at `95c3c68fc77fdf70b1582031bacf9d7753f72cf2`.
- Dataset: `marin-dna/evals_mendelian_traits` `train` at `4aed58e50c5dea0b878a665007af2ef9e5108e9f`.
- Reference: GRCh38 Ensembl release 115 soft-masked primary assembly.
- Window: 8,192 bp, truncated to 8,190 bp at Carbon's 6-mer boundary.
- Score: mean causal token log likelihood in bf16; FWD/RC LLR average; `score = -llr`.
- Bootstrap: 1,000 seeded match-group draws.

## Prompt preflight

- Selected grammar: `corpus_card` with template `<species>{species}<dna>`.
- Rejected grammar: `model_card` with template `<{species}><dna>`.
- Tokenizer revision: `95c3c68fc77fdf70b1582031bacf9d7753f72cf2`.

- `untagged`: `<dna>`; prefix IDs `[151669]`.
- `correct`: `<species>vertebrate_mammalian<dna>`; prefix IDs `[27, 42490, 29, 64832, 64116, 717, 8666, 10480, 151669]`.

## Absolute AUPRC

| subset | untagged | correct | rows | groups | macro | warning |
| --- | --- | --- | --- | --- | --- | --- |
| tss_proximal | 0.1775 | 0.1738 | 2050 | 205 | yes |  |
| _macro_avg_ | 0.1775 | 0.1738 | 2050 | 205 | yes |  |

Subsets with fewer than 30 match groups are reported and excluded from the macro average.

## Paired AUPRC differences

| comparison | subset | delta | ci_low | ci_high | n_groups | low_sample |
| --- | --- | --- | --- | --- | --- | --- |
| correct_minus_untagged | tss_proximal | -0.0037 | -0.0269 | 0.0204 | 205 |  |
| correct_minus_untagged | _macro_avg_ | -0.0037 | -0.0269 | 0.0204 | 205 |  |

Intervals are paired bootstrap intervals over identical rows and shared match-group draws.
An interval crossing zero is not evidence that two prompt conditions are equivalent.

## Exclusions and deviations

- No match groups were excluded during window validation.
- No scorer-contract deviations were recorded; dataset rows that failed the window contract are listed above.

## Runtime

| condition | rows | devices | elapsed | peak GPU | peak RSS |
| --- | --- | --- | --- | --- | --- |
| untagged | 2050 | NVIDIA GH200 480GB | 213.0 s | 11.26 GiB | 4.26 GiB |
| correct | 2050 | NVIDIA GH200 480GB | 243.6 s | 11.29 GiB | 4.81 GiB |

## Exact commands

```bash
cd snakemake/analysis/carbon_conditioning_vep
uv run --locked --group genome-s3 snakemake --profile workflow/profiles/default smoke
uv run --locked --group genome-s3 snakemake -n --profile workflow/profiles/default
uv run --locked --group genome-s3 snakemake --profile workflow/profiles/default
```
