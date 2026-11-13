# exp593 FIM annealing and old-distribution control results

Three annotation-guided FIM continuations and one original-1T-distribution control completed the fixed additional-input schedules. The FIM example probability rose from 0 to 30% over the first 20% of each experimental schedule, then stayed at 30%; the control used no FIM. All four restored full model and optimizer state, and no evaluation metric altered training.

Guided FIM landmarks cover 909,773 existing train rows from 23 of the original 65 species; no outside species or new sequence rows entered this training mix. The attempted Carietinum annotation reference had no matching train contigs and contributes no FIM examples.

| Arm | Source checkpoint step | Source epoch-equivalents | Additional input tokens | Final epoch-equivalents | Primary loss: source → final | FIM base accuracy: source → final | FIM loss: source → final | Sampled median MFU | Final checkpoint | W&B |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- | --- |
| 181b FIM | `step-346791` | 8.41 | 216.000B | 18.40 | 1.0238 → 0.9556 | 0.3382 → 0.5438 | 1.1142 → 0.9684 | 5.7% | `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/annealing/training/dna-exp593-anneal-181b-guidedfim30-equalbase-seed0-v2/checkpoints/step-972256` | [run](https://forge.coreweave.com/wandb/eric-czech/marin/runs/dna-exp593-anneal-181b-guidedfim30-equalbase-seed0-v2) |
| 364b FIM | `step-693582` | 16.82 | 168.000B | 24.59 | 0.9965 → 0.9486 | 0.3145 → 0.5268 | 1.1148 → 0.9852 | 5.8% | `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/annealing/training/dna-exp593-anneal-364b-guidedfim30-equalbase-seed0-v2/checkpoints/step-1179790` | [run](https://forge.coreweave.com/wandb/eric-czech/marin/runs/dna-exp593-anneal-364b-guidedfim30-equalbase-seed0-v2) |
| 545b FIM | `step-1040373` | 25.23 | 120.000B | 30.79 | 0.9698 → 0.9384 | 0.3510 → 0.5465 | 1.1019 → 0.9741 | 7.0% | `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/annealing/training/dna-exp593-anneal-545b-guidedfim30-equalbase-seed0-v2/checkpoints/step-1387775` | [run](https://forge.coreweave.com/wandb/eric-czech/marin/runs/dna-exp593-anneal-545b-guidedfim30-equalbase-seed0-v2) |
| control | `step-1040373` | 25.23 | 120.000B | 30.79 | 0.9698 → 0.9137 | 0.3461 → 0.3287 | 1.1019 → 1.0990 | 6.1% | `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/annealing/training/dna-exp593-anneal-545b-originaldata-equalbase-seed0-v1/checkpoints/step-1269254` | [run](https://forge.coreweave.com/wandb/eric-czech/marin/runs/dna-exp593-anneal-545b-originaldata-equalbase-seed0-v1) |

Epoch-equivalents divide cumulative input tokens by the original 21,615,869,952-base corpus; guided sampling and FIM prompt tokens mean these are not literal complete passes through the corpus. MFU is the median of up to 1,000 downsampled W&B update-history records across each completed stage, excluding first-compilation records. It is an analytic padded-bucket FLOP estimate, not a hardware profiler reading; the per-arm 10th–90th percentile range shows its variability.

Matched 545b training-loss medians from 1,000 downsampled W&B history records per arm, in the same added-input windows:

| Added-input window | FIM train/loss | Control train/loss | FIM samples | Control samples |
| --- | ---: | ---: | ---: | ---: |
| 0–6B | 0.9918 | 0.9628 | 47 | 57 |
| 54–66B | 0.9922 | 0.9387 | 107 | 89 |
| 114–120B | 0.9356 | 0.8897 | 44 | 46 |

The FIM branch is flat across the early and middle windows, then falls in the late window; the control falls across all three. These losses score each arm's own training presentation, so their levels cannot isolate FIM alone. The validation comparison below uses the same fixed evaluation data and each arm's exact source baseline.

Matched fixed validation changes from each arm's exact source baseline (both schedules continued unchanged):

| Added tokens | 545b FIM Δ primary loss | Control Δ primary loss | 545b FIM Δ FIM loss |
| ---: | ---: | ---: | ---: |
| 5.24B | +0.0195 | -0.0006 | -0.0254 |
| 10.49B | +0.0235 | -0.0006 | -0.0216 |
| 15.73B | +0.0279 | -0.0008 | -0.0438 |
| 20.97B | +0.0281 | +0.0002 | -0.0528 |
| 26.21B | +0.0313 | -0.0011 | -0.0448 |
| 31.46B | +0.0314 | -0.0044 | -0.0693 |
| 36.70B | +0.0284 | -0.0075 | -0.0665 |
| 41.94B | +0.0248 | -0.0097 | -0.0635 |
| 47.19B | +0.0224 | -0.0121 | -0.0741 |
| 52.43B | +0.0182 | -0.0156 | -0.0779 |
| 57.67B | +0.0167 | -0.0175 | -0.0900 |
| 62.91B | +0.0112 | -0.0208 | -0.0864 |
| 68.16B | +0.0079 | -0.0239 | -0.0873 |
| 73.40B | +0.0045 | -0.0287 | -0.0963 |
| 78.64B | -0.0001 | -0.0316 | -0.1039 |
| 83.89B | -0.0036 | -0.0350 | -0.1062 |
| 89.13B | -0.0082 | -0.0366 | -0.1067 |
| 94.37B | -0.0114 | -0.0415 | -0.1151 |
| 99.61B | -0.0149 | -0.0445 | -0.1202 |
| 104.86B | -0.0200 | -0.0469 | -0.1186 |
| 110.10B | -0.0240 | -0.0504 | -0.1248 |
| 115.34B | -0.0277 | -0.0540 | -0.1218 |

At the first fixed 20.97B-token panel, every task used 2048 examples under each presentation:

| Arm | FIM task scores improved from source | Raw task scores improved from source |
| --- | ---: | ---: |
| 181b | 21/21 | 3/21 |
| 364b | 21/21 | 3/21 |
| 545b | 21/21 | 7/21 |
| 545b-control | 12/21 | 15/21 |

For the matched 545b branches, the FIM branch had the larger source-relative change on 21/21 FIM tasks and 6/21 raw tasks. The arms differ in the entire data presentation, including causal-window crops, and in qualified GPU placement.

At the final checkpoints, compare each arm's score change against its own step-zero readout:

| Presentation | FIM improved from own source | Control improved from own source | FIM larger change | Control larger change | Ties |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw | 17/21 | 17/21 | 14/21 | 7/21 | 0/21 |
| fim | 21/21 | 11/21 | 21/21 | 0/21 | 0/21 |

The two step-zero readouts from the common checkpoint differ slightly (maximum absolute task-score gap 0.00463); each paired change above subtracts that arm's own baseline. All task panels use the same frozen 2,048-example membership. The arms also differ in causal-window crops and GPU placement, so this is a descriptive comparison. The control's complete original-data batches total 119,999,561,728 inputs, 0.00037% short of the FIM arm's exact 120B horizon. Both use the same inherited LR rule and decay shape; the horizon rounding changes their LR factor at the final matched token count by 1.99e-06 of the original peak.
The 545B source checkpoint has a historical saved BF16 output-head / FP32-master casting mismatch in 18,468,864 elements. Both branches preserved that exact native source state; their first ordinary optimizer update reconciled the compute head. This shared source quirk does not make their step-zero task readouts numerically identical.

At the fixed curriculum-end boundary, each panel again used 2048 examples for each task and presentation:

| Arm | Added input tokens | FIM task scores improved from source | Raw task scores improved from source |
| --- | ---: | ---: | ---: |
| 181b | 43.200B | 21/21 | 2/21 |
| 364b | 33.600B | 21/21 | 4/21 |
| 545b | 24.000B | 21/21 | 6/21 |
| 545b-control | 24.000B | 10/21 | 14/21 |

At the second fixed 41.94B-token panel:

| Arm | FIM task scores improved from source | Raw task scores improved from source |
| --- | ---: | ---: |
| 181b | 21/21 | 3/21 |
| 364b | 21/21 | 2/21 |
| 545b | 21/21 | 8/21 |
| 545b-control | 12/21 | 15/21 |

<details><summary>Later fixed 21-task panels</summary>

| Arm | Added input tokens | FIM task scores improved from source | Raw task scores improved from source |
| --- | ---: | ---: | ---: |
| 181b | 62.915B | 21/21 | 4/21 |
| 181b | 83.886B | 21/21 | 5/21 |
| 181b | 104.858B | 21/21 | 16/21 |
| 181b | 125.829B | 21/21 | 18/21 |
| 181b | 146.801B | 21/21 | 19/21 |
| 181b | 167.773B | 21/21 | 19/21 |
| 181b | 188.744B | 21/21 | 18/21 |
| 181b | 209.715B | 21/21 | 18/21 |
| 364b | 62.915B | 21/21 | 7/21 |
| 364b | 83.886B | 21/21 | 7/21 |
| 364b | 104.858B | 21/21 | 13/21 |
| 364b | 125.829B | 21/21 | 15/21 |
| 364b | 146.801B | 21/21 | 16/21 |
| 364b | 167.772B | 21/21 | 16/21 |
| 545b | 62.915B | 21/21 | 12/21 |
| 545b | 83.886B | 21/21 | 16/21 |
| 545b | 104.858B | 21/21 | 17/21 |
| 545b-control | 62.915B | 11/21 | 16/21 |
| 545b-control | 83.886B | 11/21 | 16/21 |
| 545b-control | 104.858B | 8/21 | 17/21 |

</details>

<details><summary>181b: checkpoint, mix, validation and task results</summary>

- Final global optimizer step: 972,256; additional non-padding inputs: 216,000,000,000.
- Final FIM selection probability: 0.300; cumulative annotation-guided examples: 0.270; cumulative FIM scored-target fraction: 0.00297; reverse-complement examples: 0.500.
- Sampled MFU: median 5.69%, 10th–90th percentile 1.96%–8.22% from 1000 update records; final instantaneous reading 2.13%.
- Primary loss: 1.0238 → 0.9556; nonsoftmasked: 1.0928 → 1.0282; softmasked: 0.3626 → 0.2605; nonsoftmasked:softmasked 10:1: 1.0853 → 1.0203.
- FIM base accuracy: 0.3382 → 0.5438; scored-base FIM loss: 1.1142 → 0.9684; short exact match: 0.1250 → 0.2656; suffix benefit: -0.0234 → 0.0090.
- Checkpoint metadata SHA-256: `b87bebc6f5a06a8539d14e5bb72b2b82b7fd85aaefe8ffb7dc03e54f41d96edf`; native manifest SHA-256: `e79ad55ce6da6d072a493260d0a86888521f76dde1475e35013936ea5f1f0aa2`; state arrays: 128.
- Independent full-state restore audit: `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/annealing/native_audits/dna-exp593-anneal-181b-guidedfim30-equalbase-seed0-v2/step-972256-b87bebc6f5a06a8539d14e5bb72b2b82b7fd85aaefe8ffb7dc03e54f41d96edf.json` (SHA-256 `4cd37b93082bbcaa1b7fa4b28a3c7ffcfbdc3bb7196ac9b8a699509f04c7a1e3`), with all arrays finite, model/master consistency, and exact optimizer clock.

| Task | Metric | Source raw | Final raw | Source FIM | Final FIM |
| --- | --- | ---: | ---: | ---: | ---: |
| conservation_andropogoneae | auroc | 0.7138 | 0.7540 | 0.5138 | 0.7205 |
| conservation_poaceae_non_tis | auroc | 0.7402 | 0.7735 | 0.5434 | 0.7533 |
| conservation_poaceae_tis | auroc | 0.5876 | 0.6584 | 0.4474 | 0.6803 |
| core_maize_acceptor | average_precision | 0.9133 | 0.9023 | 0.8398 | 0.9354 |
| core_maize_donor | average_precision | 0.9027 | 0.9020 | 0.8597 | 0.9164 |
| core_maize_tis | average_precision | 0.7917 | 0.7756 | 0.7464 | 0.9326 |
| core_maize_tts | average_precision | 0.7982 | 0.7905 | 0.7543 | 0.8589 |
| core_tomato_acceptor | average_precision | 0.9729 | 0.9753 | 0.9120 | 0.9781 |
| core_tomato_donor | average_precision | 0.9739 | 0.9794 | 0.9618 | 0.9798 |
| core_tomato_tis | average_precision | 0.8485 | 0.8592 | 0.8526 | 0.9021 |
| core_tomato_tts | average_precision | 0.8640 | 0.8780 | 0.8224 | 0.9167 |
| maize_af | spearman | 0.1545 | 0.1776 | -0.1056 | 0.0809 |
| motif_maize_acceptor | accuracy | 0.7666 | 0.8574 | 0.0005 | 0.9761 |
| motif_maize_donor | accuracy | 0.7603 | 0.8613 | 0.0000 | 0.9585 |
| motif_maize_tis | accuracy | 0.4072 | 0.5938 | 0.0078 | 0.8906 |
| motif_maize_tts | accuracy | 0.2866 | 0.4951 | 0.0020 | 0.5332 |
| motif_tomato_acceptor | accuracy | 0.6562 | 0.7168 | 0.0166 | 0.9175 |
| motif_tomato_donor | accuracy | 0.6558 | 0.7129 | 0.0005 | 0.8955 |
| motif_tomato_tis | accuracy | 0.3267 | 0.4292 | 0.0024 | 0.8564 |
| motif_tomato_tts | accuracy | 0.2090 | 0.2515 | 0.0010 | 0.3916 |
| sv_impact | average_precision | 0.8127 | 0.8624 | 0.4438 | 0.8047 |

</details>

<details><summary>364b: checkpoint, mix, validation and task results</summary>

- Final global optimizer step: 1,179,790; additional non-padding inputs: 168,000,000,000.
- Final FIM selection probability: 0.300; cumulative annotation-guided examples: 0.270; cumulative FIM scored-target fraction: 0.00297; reverse-complement examples: 0.500.
- Sampled MFU: median 5.79%, 10th–90th percentile 2.35%–8.20% from 1000 update records; final instantaneous reading 2.22%.
- Primary loss: 0.9965 → 0.9486; nonsoftmasked: 1.0676 → 1.0210; softmasked: 0.3153 → 0.2540; nonsoftmasked:softmasked 10:1: 1.0598 → 1.0131.
- FIM base accuracy: 0.3145 → 0.5268; scored-base FIM loss: 1.1148 → 0.9852; short exact match: 0.0781 → 0.2188; suffix benefit: -0.0078 → 0.0033.
- Checkpoint metadata SHA-256: `0cac8a7f6a903012c9011c0ff87b4af068a9a6db7dd263d8e8c90fad3ebc5a22`; native manifest SHA-256: `e79ad55ce6da6d072a493260d0a86888521f76dde1475e35013936ea5f1f0aa2`; state arrays: 128.
- Independent full-state restore audit: `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/annealing/native_audits/dna-exp593-anneal-364b-guidedfim30-equalbase-seed0-v2/step-1179790-0cac8a7f6a903012c9011c0ff87b4af068a9a6db7dd263d8e8c90fad3ebc5a22.json` (SHA-256 `6952c86a8b0c6be8b675af8e747a6bc08f80c55044c2ba8c2b056cdec271d1eb`), with all arrays finite, model/master consistency, and exact optimizer clock.

| Task | Metric | Source raw | Final raw | Source FIM | Final FIM |
| --- | --- | ---: | ---: | ---: | ---: |
| conservation_andropogoneae | auroc | 0.7352 | 0.7650 | 0.5103 | 0.7222 |
| conservation_poaceae_non_tis | auroc | 0.7498 | 0.7770 | 0.5418 | 0.7689 |
| conservation_poaceae_tis | auroc | 0.6141 | 0.6519 | 0.4554 | 0.6547 |
| core_maize_acceptor | average_precision | 0.9079 | 0.8929 | 0.8610 | 0.9288 |
| core_maize_donor | average_precision | 0.9138 | 0.8965 | 0.8673 | 0.9009 |
| core_maize_tis | average_precision | 0.7874 | 0.7699 | 0.7552 | 0.9302 |
| core_maize_tts | average_precision | 0.7925 | 0.7871 | 0.7658 | 0.8570 |
| core_tomato_acceptor | average_precision | 0.9762 | 0.9751 | 0.9151 | 0.9768 |
| core_tomato_donor | average_precision | 0.9771 | 0.9783 | 0.9346 | 0.9795 |
| core_tomato_tis | average_precision | 0.8522 | 0.8634 | 0.8137 | 0.9069 |
| core_tomato_tts | average_precision | 0.8701 | 0.8747 | 0.8248 | 0.9083 |
| maize_af | spearman | 0.1509 | 0.1643 | -0.0703 | 0.0690 |
| motif_maize_acceptor | accuracy | 0.8013 | 0.8647 | 0.0000 | 0.9712 |
| motif_maize_donor | accuracy | 0.8076 | 0.8613 | 0.0000 | 0.9590 |
| motif_maize_tis | accuracy | 0.4751 | 0.5942 | 0.0015 | 0.9116 |
| motif_maize_tts | accuracy | 0.3711 | 0.5146 | 0.0024 | 0.5542 |
| motif_tomato_acceptor | accuracy | 0.6768 | 0.7085 | 0.0010 | 0.9043 |
| motif_tomato_donor | accuracy | 0.6841 | 0.7183 | 0.0010 | 0.8975 |
| motif_tomato_tis | accuracy | 0.3643 | 0.4370 | 0.0020 | 0.8740 |
| motif_tomato_tts | accuracy | 0.2378 | 0.2510 | 0.0010 | 0.3960 |
| sv_impact | average_precision | 0.8396 | 0.8701 | 0.4328 | 0.8170 |

</details>

<details><summary>545b: checkpoint, mix, validation and task results</summary>

- Final global optimizer step: 1,387,775; additional non-padding inputs: 120,000,000,000.
- Final FIM selection probability: 0.300; cumulative annotation-guided examples: 0.270; cumulative FIM scored-target fraction: 0.00297; reverse-complement examples: 0.500.
- Sampled MFU: median 6.98%, 10th–90th percentile 3.59%–8.39% from 1000 update records; final instantaneous reading 6.37%.
- Primary loss: 0.9698 → 0.9384; nonsoftmasked: 1.0412 → 1.0104; softmasked: 0.2852 → 0.2487; nonsoftmasked:softmasked 10:1: 1.0334 → 1.0025.
- FIM base accuracy: 0.3510 → 0.5465; scored-base FIM loss: 1.1019 → 0.9741; short exact match: 0.1719 → 0.2500; suffix benefit: -0.0015 → 0.0096.
- Checkpoint metadata SHA-256: `37f9b77c0e71472084ca8506b61ac4189056a50cea4dd865518f1886d09f5694`; native manifest SHA-256: `f27968148ca1b6336449eadf85af931e893203c6489ca6138c14b565b705d49a`; state arrays: 128.
- Independent full-state restore audit: `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/annealing/native_audits/dna-exp593-anneal-545b-guidedfim30-equalbase-seed0-v2/step-1387775-37f9b77c0e71472084ca8506b61ac4189056a50cea4dd865518f1886d09f5694.json` (SHA-256 `8dd28753ece67d57844a960efaef48234a4948665b3cc8072d7b0aa8cc7e4324`), with all arrays finite, model/master consistency, and exact optimizer clock.

| Task | Metric | Source raw | Final raw | Source FIM | Final FIM |
| --- | --- | ---: | ---: | ---: | ---: |
| conservation_andropogoneae | auroc | 0.7521 | 0.7646 | 0.5129 | 0.7173 |
| conservation_poaceae_non_tis | auroc | 0.7623 | 0.7739 | 0.5454 | 0.7557 |
| conservation_poaceae_tis | auroc | 0.6260 | 0.6566 | 0.4569 | 0.6631 |
| core_maize_acceptor | average_precision | 0.8971 | 0.8900 | 0.8288 | 0.9282 |
| core_maize_donor | average_precision | 0.8908 | 0.8770 | 0.8915 | 0.9096 |
| core_maize_tis | average_precision | 0.7731 | 0.7629 | 0.7476 | 0.9223 |
| core_maize_tts | average_precision | 0.7837 | 0.7754 | 0.7579 | 0.8324 |
| core_tomato_acceptor | average_precision | 0.9757 | 0.9782 | 0.9393 | 0.9791 |
| core_tomato_donor | average_precision | 0.9774 | 0.9787 | 0.9410 | 0.9807 |
| core_tomato_tis | average_precision | 0.8549 | 0.8626 | 0.8335 | 0.9014 |
| core_tomato_tts | average_precision | 0.8731 | 0.8808 | 0.8282 | 0.9213 |
| maize_af | spearman | 0.1441 | 0.1657 | -0.0894 | 0.0873 |
| motif_maize_acceptor | accuracy | 0.8403 | 0.8691 | 0.0986 | 0.9771 |
| motif_maize_donor | accuracy | 0.8350 | 0.8687 | 0.0005 | 0.9595 |
| motif_maize_tis | accuracy | 0.5078 | 0.6055 | 0.0034 | 0.9048 |
| motif_maize_tts | accuracy | 0.4214 | 0.5293 | 0.0010 | 0.5415 |
| motif_tomato_acceptor | accuracy | 0.6973 | 0.7192 | 0.0020 | 0.9077 |
| motif_tomato_donor | accuracy | 0.6963 | 0.7197 | 0.0005 | 0.8931 |
| motif_tomato_tis | accuracy | 0.3730 | 0.4307 | 0.0039 | 0.8535 |
| motif_tomato_tts | accuracy | 0.2393 | 0.2554 | 0.0117 | 0.3965 |
| sv_impact | average_precision | 0.8563 | 0.8678 | 0.4361 | 0.8049 |

</details>

<details><summary>545b-control: checkpoint, mix, validation and task results</summary>

- Final global optimizer step: 1,269,254; additional non-padding inputs: 119,999,561,728.
- Final FIM selection probability: 0.000; cumulative annotation-guided examples: 0.000; cumulative FIM scored-target fraction: 0.00000; reverse-complement examples: 0.500.
- Sampled MFU: median 6.05%, 10th–90th percentile 2.43%–6.18% from 1000 update records; final instantaneous reading 6.09%.
- Primary loss: 0.9698 → 0.9137; nonsoftmasked: 1.0412 → 0.9850; softmasked: 0.2852 → 0.2303; nonsoftmasked:softmasked 10:1: 1.0334 → 0.9772.
- FIM base accuracy: 0.3461 → 0.3287; scored-base FIM loss: 1.1019 → 1.0990; short exact match: 0.1562 → 0.1094; suffix benefit: -0.0037 → -0.0214.
- Checkpoint metadata SHA-256: `02641b40cf4fa61678180436262fbaa2ae54192335a3ecb326fe33c002a7114a`; native manifest SHA-256: `e79ad55ce6da6d072a493260d0a86888521f76dde1475e35013936ea5f1f0aa2`; state arrays: 128.
- Independent full-state restore audit: `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/annealing/native_audits/dna-exp593-anneal-545b-originaldata-equalbase-seed0-v1/step-1269254-02641b40cf4fa61678180436262fbaa2ae54192335a3ecb326fe33c002a7114a.json` (SHA-256 `a94036e015f2d2821ae757ad58bbf0fcc8243da7a2d656d5a1b0bd330489b3f9`), with all arrays finite, model/master consistency, and exact optimizer clock.

| Task | Metric | Source raw | Final raw | Source FIM | Final FIM |
| --- | --- | ---: | ---: | ---: | ---: |
| conservation_andropogoneae | auroc | 0.7520 | 0.7785 | 0.5138 | 0.4941 |
| conservation_poaceae_non_tis | auroc | 0.7621 | 0.7686 | 0.5453 | 0.4889 |
| conservation_poaceae_tis | auroc | 0.6260 | 0.6493 | 0.4566 | 0.5478 |
| core_maize_acceptor | average_precision | 0.8973 | 0.8849 | 0.8293 | 0.8737 |
| core_maize_donor | average_precision | 0.8908 | 0.8719 | 0.8907 | 0.8541 |
| core_maize_tis | average_precision | 0.7734 | 0.7584 | 0.7458 | 0.7477 |
| core_maize_tts | average_precision | 0.7837 | 0.7640 | 0.7568 | 0.7602 |
| core_tomato_acceptor | average_precision | 0.9757 | 0.9778 | 0.9398 | 0.9439 |
| core_tomato_donor | average_precision | 0.9774 | 0.9780 | 0.9408 | 0.9481 |
| core_tomato_tis | average_precision | 0.8546 | 0.8629 | 0.8322 | 0.8281 |
| core_tomato_tts | average_precision | 0.8732 | 0.8782 | 0.8277 | 0.8347 |
| maize_af | spearman | 0.1487 | 0.1505 | -0.0890 | 0.0746 |
| motif_maize_acceptor | accuracy | 0.8418 | 0.8799 | 0.1006 | 0.0015 |
| motif_maize_donor | accuracy | 0.8350 | 0.8818 | 0.0010 | 0.0005 |
| motif_maize_tis | accuracy | 0.5073 | 0.6157 | 0.0034 | 0.0005 |
| motif_maize_tts | accuracy | 0.4233 | 0.5425 | 0.0015 | 0.0239 |
| motif_tomato_acceptor | accuracy | 0.6973 | 0.7114 | 0.0024 | 0.0010 |
| motif_tomato_donor | accuracy | 0.6968 | 0.7065 | 0.0005 | 0.0005 |
| motif_tomato_tis | accuracy | 0.3750 | 0.4072 | 0.0034 | 0.0024 |
| motif_tomato_tts | accuracy | 0.2397 | 0.2593 | 0.0088 | 0.0947 |
| sv_impact | average_precision | 0.8568 | 0.8579 | 0.4386 | 0.4397 |

</details>

Training, native audit reports, and exact Iris writer completion were verified independently for each branch.
