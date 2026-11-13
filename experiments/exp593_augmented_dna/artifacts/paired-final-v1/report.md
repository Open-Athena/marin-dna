Final comparison

| Initialization | New inputs (B) | Primary NLL | FIM NLL | Conservation AUROC, bare | Conservation AUROC, FIM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text initialization | 0.000 | 5.625091 | 9.973624 | 0.483963 | 0.509323 |
| Prior DNA216B; half LR | 0.000 | 0.947501 | 17.804113 | 0.645721 | 0.499817 |
| Text initialization | 54.040 | 1.287671 | 1.259378 | 0.553040 | 0.521774 |
| Prior DNA216B; half LR | 54.040 | 1.198421 | 1.208104 | 0.535583 | 0.510895 |
| Text initialization | 108.079 | 1.275218 | 1.215574 | 0.548676 | 0.524048 |
| Prior DNA216B; half LR | 108.079 | 1.176901 | 1.169002 | 0.568909 | 0.551041 |
| Text initialization | 162.119 | 1.265534 | 1.183576 | 0.548096 | 0.540833 |
| Prior DNA216B; half LR | 162.119 | 1.149386 | 1.135152 | 0.569611 | 0.555130 |
| Text initialization | 216.159 | 1.249087 | 1.170395 | 0.556778 | 0.559250 |
| Prior DNA216B; half LR | 216.159 | 1.112691 | 1.086264 | 0.586044 | 0.564697 |

<details>
<summary>Final causal losses and repeat partitions</summary>

| Metric | Text initialization | Prior DNA216B; half LR |
| --- | ---: | ---: |
| Historical primary NLL | 1.249087 | 1.112691 |
| Softmasked repeat NLL | 1.050748 | 0.516164 |
| Nonsoftmasked NLL | 1.269785 | 1.174941 |
| 10:1 nonsoftmasked:softmasked evaluation NLL | 1.267523 | 1.168138 |
| Disjoint panel NLL | 1.245488 | 1.095639 |

</details>

<details>
<summary>Final FIM presentation and selection breakdowns</summary>

| Metric | Text initialization | Prior DNA216B; half LR |
| --- | ---: | ---: |
| Overall: NLL | 1.170395 | 1.086264 |
| Overall: base accuracy | 0.420155 | 0.512267 |
| Overall: greedy exact match | 0.111111 | 0.166667 |
| Overall: casefolded greedy exact match | 0.111111 | 0.166667 |
| minimal: NLL | 1.172869 | 1.088322 |
| minimal: base accuracy | 0.420178 | 0.512908 |
| minimal: greedy exact match | 0.111111 | 0.166667 |
| minimal: casefolded greedy exact match | 0.111111 | 0.166667 |
| formatted: NLL | 1.167921 | 1.084207 |
| formatted: base accuracy | 0.420131 | 0.511627 |
| formatted: greedy exact match | 0.111111 | 0.166667 |
| formatted: casefolded greedy exact match | 0.111111 | 0.166667 |
| short: NLL | 1.141507 | 1.069342 |
| short: base accuracy | 0.490885 | 0.531250 |
| short: greedy exact match | 0.333333 | 0.500000 |
| short: casefolded greedy exact match | 0.333333 | 0.500000 |
| long: NLL | 1.255283 | 1.135990 |
| long: base accuracy | 0.417479 | 0.511549 |
| long: greedy exact match | 0.000000 | 0.000000 |
| long: casefolded greedy exact match | 0.000000 | 0.000000 |
| random: NLL | 1.217030 | 1.131835 |
| random: base accuracy | 0.419061 | 0.511563 |
| random: greedy exact match | 0.117647 | 0.176471 |
| random: casefolded greedy exact match | 0.117647 | 0.176471 |
| guided: NLL | 0.948751 | 0.869675 |
| guided: base accuracy | 0.543011 | 0.591398 |
| guided: greedy exact match | 0.000000 | 0.000000 |
| guided: casefolded greedy exact match | 0.000000 | 0.000000 |

</details>

<details>
<summary>Final FIM completion-length breakdowns</summary>

| Metric | Text initialization | Prior DNA216B; half LR |
| --- | ---: | ---: |
| 1 base: NLL | 1.151422 | 1.085739 |
| 1 base: base accuracy | 0.512500 | 0.533333 |
| 1 base: greedy exact match | 0.500000 | 0.500000 |
| 2 bases: NLL | 1.138496 | 1.043391 |
| 2 bases: base accuracy | 0.481884 | 0.534420 |
| 2 bases: greedy exact match | 0.500000 | 0.500000 |
| 3 bases: NLL | 1.135262 | 1.082355 |
| 3 bases: base accuracy | 0.490591 | 0.528226 |
| 3 bases: greedy exact match | 0.000000 | 0.500000 |
| 16 bases: NLL | 1.242823 | 1.165812 |
| 16 bases: base accuracy | 0.426411 | 0.498992 |
| 16 bases: greedy exact match | 0.000000 | 0.000000 |
| 32 bases: NLL | 1.280488 | 1.186486 |
| 32 bases: base accuracy | 0.396382 | 0.439145 |
| 32 bases: greedy exact match | 0.000000 | 0.000000 |
| 64 bases: NLL | 1.252250 | 1.176459 |
| 64 bases: base accuracy | 0.416992 | 0.455404 |
| 64 bases: greedy exact match | 0.000000 | 0.000000 |
| 128 bases: NLL | 1.224829 | 1.111950 |
| 128 bases: base accuracy | 0.430859 | 0.498437 |
| 128 bases: greedy exact match | 0.000000 | 0.000000 |
| 256 bases: NLL | 1.325818 | 1.075289 |
| 256 bases: base accuracy | 0.374099 | 0.520583 |
| 256 bases: greedy exact match | 0.000000 | 0.000000 |
| 512 bases: NLL | 1.241033 | 1.067064 |
| 512 bases: base accuracy | 0.427607 | 0.523438 |
| 512 bases: greedy exact match | 0.000000 | 0.000000 |

</details>

<details>
<summary>Final functional development panels</summary>

| Metric | Text initialization | Prior DNA216B; half LR |
| --- | ---: | ---: |
| Conservation bare: AUROC | 0.556778 | 0.586044 |
| Conservation bare: average precision | 0.538924 | 0.572613 |
| Conservation fim: AUROC | 0.559250 | 0.564697 |
| Conservation fim: average precision | 0.533928 | 0.541369 |
| splice_donor: NLL | 0.306771 | 0.217375 |
| splice_donor: base accuracy | 0.906250 | 0.890625 |
| splice_donor: greedy exact match | 0.812500 | 0.843750 |
| splice_acceptor: NLL | 0.251076 | 0.183234 |
| splice_acceptor: base accuracy | 0.937500 | 0.921875 |
| splice_acceptor: greedy exact match | 0.906250 | 0.906250 |
| random_control: NLL | 1.228392 | 1.198503 |
| random_control: base accuracy | 0.414062 | 0.382812 |
| random_control: greedy exact match | 0.171875 | 0.125000 |

</details>

<details>
<summary>Interpretation, denominators and provenance</summary>

Conditions differ in initialization, prior DNA exposure and learning rates.
These frozen development diagnostics do not establish downstream benchmark gains or isolate the effect of augmentation.
The two zero-input rows are actual production diagnostics before the first new-stage update.
The DNA initializer already has 216B prior DNA inputs; zero means no additional exp593 inputs.
Those rows identify the imported parent checkpoints and are not new exp593 permanent checkpoints or additional milestone handoff models.
The verified initial scores expose readout trade-offs that quarter-to-final trajectories alone can hide.
Both conditions match all eight counters at every quarter and at the final horizon.
Primary and disjoint causal panels each contain 512 examples.
The historical case partitions contain 396,290 softmasked and 3,797,502 nonsoftmasked targets, including 5,286 unknown-N targets counted by native case.
Uppercase means unmasked; it does not independently establish nonrepetitive biology.
The 10:1 value is an evaluation metric; training has no such repeat reweighting.
FIM has 1,024 presentations and 36 greedy completions (four per completion length).
The guided greedy subset has only two examples, so its exact-match value is especially coarse.
FIM NLL is the defined example-macro statistic; base accuracy is token-micro.
The complete source artifacts retain all 124 fixed metrics and 53 denominator fields.
The functional panels contain 512 balanced conservation examples and 128 junction presentations: 32 donor, 32 acceptor and 64 random controls.
This generator reads verified source artifacts locally; it performs no new inference, remote revalidation or acceptance decision.
Full-array readability does not establish independent JAX restore or inference parity.
Sparse norm coverage means at least one finite 20-scalar snapshot per expected 1,000-update point, not every gradient or replayed log row.

- dna216b, before adaptation: `s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/checkpoints/exp586-plantcad2-d768-pretrained-lrm1-seed0-v1/2026.10.08/checkpoints/step-412290`; source canonical metadata SHA-256 `275e68e8a521333a4dd5c7bf2aa8f9c79c11517677a8af47ce7abc1aa5004a89`.
- text, before adaptation: `s3://marin-us-east-02a/marin/grug/rav-ladder-d768-v2/2026.08.18/checkpoints/step-11420`; source canonical metadata SHA-256 `01cbe71d30793c7a66dd7fe1f54a50e1198490a58cc364a28b5cf27a26e76cf4`.
- production, quarter 1: `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/training/dna-exp593-augmented-d768-pretrained-seed0-v1/checkpoints/step-156559`; metadata SHA-256 `40fd5b04572ab299ec5c150923fab8c9db22784664b5cac59011fa9be5dcd891`.
- dna216b, quarter 1: `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/training/dna-exp593-augmented-d768-dna216b-lrm0p5-seed0-v1/checkpoints/step-156559`; metadata SHA-256 `ed5205b65d0d9c2021ed8f40c3b9f5ad4656687a45caeea97db34cdf5db67688`.
- production, quarter 2: `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/training/dna-exp593-augmented-d768-pretrained-seed0-v1/checkpoints/step-313074`; metadata SHA-256 `221876446be4148cf2553b62af11778c8355923efc3a80ddbebd477dd86d48ad`.
- dna216b, quarter 2: `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/training/dna-exp593-augmented-d768-dna216b-lrm0p5-seed0-v1/checkpoints/step-313074`; metadata SHA-256 `1dbe5c6d6b5bfb93c864235391513f963970e2f736c9282fefc5891fc6c364a0`.
- production, quarter 3: `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/training/dna-exp593-augmented-d768-pretrained-seed0-v1/checkpoints/step-469566`; metadata SHA-256 `a2ee3bcfa78a181cb17bc21c90077a27bb3c2e8b850a3c8040a99d686fe35efe`.
- dna216b, quarter 3: `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/training/dna-exp593-augmented-d768-dna216b-lrm0p5-seed0-v1/checkpoints/step-469566-retry-93646d85fc44411d9a21bac9dce18a29`; metadata SHA-256 `5b9e92fd73c6b71ea98ac3cfd36805f93738e9c789163846bafb5a8f12c98bed`.
- production, final: `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/training/dna-exp593-augmented-d768-pretrained-seed0-v1/checkpoints/step-626011`; metadata SHA-256 `9a10fe1715400ef9495cf003fe0fa20d6ab740b9ed655d0d0a9bc60595b6ca6e`.
- dna216b, final: `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/training/dna-exp593-augmented-d768-dna216b-lrm0p5-seed0-v1/checkpoints/step-626011`; metadata SHA-256 `ee340e80cec5113c94c902e8ad14cb3bb49a1b979f6b35871842b4354225107f`.

</details>
