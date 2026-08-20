# Carbon species conditioning on Mendelian VEP

## TL;DR

This exploratory pilot reports 3 prompt conditions on the `all_development` scope.
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
- `far_wrong`: `<species>fungi<dna>`; prefix IDs `[27, 42490, 29, 78606, 72, 151669]`.

## Absolute AUPRC

| subset | untagged | correct | far_wrong | rows | groups | macro | warning |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3_prime_UTR_variant | 0.3576 | 0.3872 | 0.4220 | 770 | 77 | yes |  |
| 5_prime_UTR_variant | 0.3728 | 0.3540 | 0.3543 | 2100 | 210 | yes |  |
| distal | 0.1419 | 0.1548 | 0.1619 | 580 | 58 | yes |  |
| mature_miRNA_variant | 0.0759 | 0.0914 | 0.0853 | 40 | 4 | no | low sample |
| missense_variant | 0.4463 | 0.4377 | 0.4411 | 5800 | 580 | yes |  |
| non_coding_transcript_exon_variant | 0.1011 | 0.1079 | 0.1090 | 1150 | 115 | yes |  |
| splicing | 0.6865 | 0.6709 | 0.6861 | 3190 | 319 | yes |  |
| synonymous_variant | 0.5820 | 0.5550 | 0.5432 | 460 | 46 | yes |  |
| tss_proximal | 0.1775 | 0.1738 | 0.1524 | 2050 | 205 | yes |  |
| _macro_avg_ | 0.3582 | 0.3552 | 0.3587 | 16100 | 1610 | yes |  |

Subsets with fewer than 30 match groups are reported and excluded from the macro average.

## Paired AUPRC differences

| comparison | subset | delta | ci_low | ci_high | n_groups | low_sample |
| --- | --- | --- | --- | --- | --- | --- |
| correct_minus_untagged | 3_prime_UTR_variant | 0.0297 | -0.0151 | 0.0809 | 77 |  |
| correct_minus_untagged | 5_prime_UTR_variant | -0.0187 | -0.0517 | 0.0114 | 210 |  |
| correct_minus_untagged | distal | 0.0129 | -0.0261 | 0.0434 | 58 |  |
| correct_minus_untagged | mature_miRNA_variant | 0.0156 | -0.0968 | 0.0713 | 4 | yes |
| correct_minus_untagged | missense_variant | -0.0086 | -0.0168 | -0.0008 | 580 |  |
| correct_minus_untagged | non_coding_transcript_exon_variant | 0.0068 | -0.0062 | 0.0229 | 115 |  |
| correct_minus_untagged | splicing | -0.0155 | -0.0347 | 0.0025 | 319 |  |
| correct_minus_untagged | synonymous_variant | -0.0270 | -0.1080 | 0.0407 | 46 |  |
| correct_minus_untagged | tss_proximal | -0.0037 | -0.0292 | 0.0208 | 205 |  |
| correct_minus_untagged | _macro_avg_ | -0.0030 | -0.0183 | 0.0088 | 1610 |  |
| far_wrong_minus_untagged | 3_prime_UTR_variant | 0.0644 | 0.0219 | 0.1095 | 77 |  |
| far_wrong_minus_untagged | 5_prime_UTR_variant | -0.0184 | -0.0451 | 0.0114 | 210 |  |
| far_wrong_minus_untagged | distal | 0.0200 | -0.0196 | 0.0574 | 58 |  |
| far_wrong_minus_untagged | mature_miRNA_variant | 0.0095 | -0.0763 | 0.0384 | 4 | yes |
| far_wrong_minus_untagged | missense_variant | -0.0052 | -0.0116 | 0.0011 | 580 |  |
| far_wrong_minus_untagged | non_coding_transcript_exon_variant | 0.0079 | -0.0049 | 0.0245 | 115 |  |
| far_wrong_minus_untagged | splicing | -0.0004 | -0.0129 | 0.0123 | 319 |  |
| far_wrong_minus_untagged | synonymous_variant | -0.0388 | -0.1306 | 0.0338 | 46 |  |
| far_wrong_minus_untagged | tss_proximal | -0.0251 | -0.0523 | -0.0028 | 205 |  |
| far_wrong_minus_untagged | _macro_avg_ | 0.0005 | -0.0137 | 0.0124 | 1610 |  |
| far_wrong_minus_correct | 3_prime_UTR_variant | 0.0347 | -0.0157 | 0.0837 | 77 |  |
| far_wrong_minus_correct | 5_prime_UTR_variant | 0.0003 | -0.0243 | 0.0309 | 210 |  |
| far_wrong_minus_correct | distal | 0.0071 | -0.0191 | 0.0450 | 58 |  |
| far_wrong_minus_correct | mature_miRNA_variant | -0.0061 | -0.0504 | 0.0141 | 4 | yes |
| far_wrong_minus_correct | missense_variant | 0.0034 | -0.0038 | 0.0109 | 580 |  |
| far_wrong_minus_correct | non_coding_transcript_exon_variant | 0.0010 | -0.0123 | 0.0150 | 115 |  |
| far_wrong_minus_correct | splicing | 0.0151 | -0.0001 | 0.0313 | 319 |  |
| far_wrong_minus_correct | synonymous_variant | -0.0118 | -0.0803 | 0.0501 | 46 |  |
| far_wrong_minus_correct | tss_proximal | -0.0214 | -0.0476 | 0.0001 | 205 |  |
| far_wrong_minus_correct | _macro_avg_ | 0.0036 | -0.0087 | 0.0152 | 1610 |  |

Intervals are paired bootstrap intervals over identical rows and shared match-group draws.
An interval crossing zero is not evidence that two prompt conditions are equivalent.

## Exclusions and deviations

- No match groups were excluded during window validation.
- No scorer-contract deviations were recorded; dataset rows that failed the window contract are listed above.

## Runtime

| condition | rows | devices | elapsed | peak GPU | peak RSS |
| --- | --- | --- | --- | --- | --- |
| untagged | 16140 | NVIDIA GH200 480GB | 1617.2 s | 11.26 GiB | 4.88 GiB |
| correct | 16140 | NVIDIA GH200 480GB | 1683.8 s | 11.29 GiB | 4.67 GiB |
| far_wrong | 16140 | NVIDIA GH200 480GB | 1698.9 s | 11.28 GiB | 4.36 GiB |

## Exact commands

```bash
cd snakemake/analysis/carbon_conditioning_vep
uv run --locked --group genome-s3 snakemake --configfile config/full_development.yaml --profile workflow/profiles/default smoke
uv run --locked --group genome-s3 snakemake -n --configfile config/full_development.yaml --profile workflow/profiles/default
uv run --locked --group genome-s3 snakemake --configfile config/full_development.yaml --profile workflow/profiles/default
```
