---
topic: carbon-species-conditioning-vep
issue: https://github.com/Open-Athena/marin-dna/issues/486
description: Frozen Carbon-3B species-prompt conditioning on development-safe Mendelian VEP
author: gonzalobenegas
---

# Carbon species conditioning VEP: Task Logbook

## Current TL;DR

- `CARBON-SC-001` selected the corpus-card grammar on an AWS A10G.
- The corrected label-blind four-condition A10G smoke passed.
- The retained labeled pilot compares untagged versus correct conditioning on 2,050 development-set promoter rows.
- Correct conditioning produced AUPRC 0.1738 versus 0.1775 untagged, with paired delta -0.0037 and 95% match-group bootstrap interval [-0.0269, 0.0204].
- The full two-arm follow-up retained all 16,140 development/train variants and 1,614 complete match groups.
- Correct conditioning produced macro AUPRC 0.3552 versus 0.3582 untagged, with paired delta -0.0030 and 95% match-group bootstrap interval [-0.0183, 0.0088].
- The full Lambda GH200 cluster is terminated; SkyPilot estimates $2.28 cost against the approved $3.00 cap.
- Compact metric, runtime, checksum, and summary artifacts are retained locally; raw score parquets and the reproducible staged input remain local and were not uploaded to S3.

## Scope

- Goal: Compare frozen Carbon-3B Mendelian VEP rankings under untagged, correct mammalian, near-wrong vertebrate, and far-wrong fungal prompts.
- Primary metrics: Per-subset and macro AUPRC under every condition; paired AUPRC differences with 95% match-group bootstrap intervals.
- Constraints: Mendelian `train` split only; Carbon-3B revision `95c3c68fc77fdf70b1582031bacf9d7753f72cf2`; 8,192 bp windows; token-level full-sequence likelihood; FWD/RC average; no held-out labels.
- Coordinating issue: https://github.com/Open-Athena/marin-dna/issues/486

## Baseline

- Date: 2026-08-20
- Code refs: repository base `d40a56ac83ac414bc5c31625bc3996007edbd407`; Carbon scorer `10bbc4b35f6e26d2a8767342576ff65108028bf5`.
- Baseline numbers: Pending prompt preflight and model inference.

## Hypothesis Queue

### Active

- `CARBON-SC-001`: One released metadata grammar produces higher continuation accuracy with correct low-resource species tags than without tags.
  Next test: Run the fixed sequence-recovery preflight on fungi, protozoa, and invertebrate rows.

### Blocked

- None.

### Falsified / Dead End

- `CARBON-SC-002` improvement form: The scoped promoter pilot did not support an AUPRC benefit from correct mammalian conditioning.
  The full-development follow-up also did not support a macro AUPRC benefit, with paired delta -0.0030 and 95% interval [-0.0183, 0.0088].
  Both paired intervals cross zero, so they also do not establish equivalence or overall harm.
  Do not expand conditioning comparisons without a new rationale or substantially more power.

### Promoted

- None.

## Decision Log

- 2026-08-20: Use `GenerTeam/sequence-recovery` revision `3ac1de1be0e4c55dd180c719c1f3805a2cdb9be9` for the prompt preflight.
- 2026-08-20: Keep prompt preflight upstream of Mendelian data loading in the Snakemake DAG.
- 2026-08-20: Preserve Carbon's official token path: truncate each strand from 8,192 bp to 8,190 bp, prepend the condition prefix, and compute masked mean causal token log likelihood.
- 2026-08-20: Freeze every condition-prefix token-ID sequence from the pinned Carbon-3B tokenizer and block inference if any ID changes.
- 2026-08-20: Use one AWS `g5.2xlarge` A10G instance, score one variant per batch, check current pricing immediately before launch, and automatically tear it down after the smoke job.
- 2026-08-20: Limit the first labeled pilot to `tss_proximal` promoter variants and the untagged-versus-correct comparison.
- 2026-08-20: Prefer Lambda GH200 for the promoter pilot if capacity is available; stage inputs through Sky instead of forwarding AWS credentials.
- 2026-08-20: Keep a one-variant inference batch on GH200 because the measured eight-variant batch reduced throughput and raised peak allocation to 44.76 GiB.
- 2026-08-20: Retain only the untagged-versus-correct comparison for the full development split and stop after it because neither the promoter nor full-development paired result supports an improvement.

## Negative Results Index

- `CARBON-SC-002`: Correct mammalian conditioning did not improve AUPRC on 2,050 development-only promoter variants; paired delta -0.0037, 95% CI [-0.0269, 0.0204].
- `CARBON-SC-002`: Correct mammalian conditioning did not improve macro AUPRC across 16,140 development-only variants; paired delta -0.0030, 95% CI [-0.0183, 0.0088].

## Entry Log

### 2026-08-20 17:04 UTC - CARBON-SC-001 implementation start

- Hypothesis: Carbon's released low-resource species-tag direction can distinguish the two documented prompt grammars before Mendelian labels are loaded.
- Commit Hash: `d40a56ac83ac414bc5c31625bc3996007edbd407` (starting point).
- Command: Read issue #486, the pinned Carbon model and corpus cards, Carbon `vep_eval.py` and `sequence_recovery.py`, and the local `evals_v2` contracts.
- Config: Carbon-3B `95c3c68fc77fdf70b1582031bacf9d7753f72cf2`; Mendelian `4aed58e50c5dea0b878a665007af2ef9e5108e9f`; sequence recovery `3ac1de1be0e4c55dd180c719c1f3805a2cdb9be9`; Ensembl 115 GRCh38 soft-masked primary assembly.
- Result: The pinned token scorer truncates DNA to a multiple of six and averages the causal target-token log probabilities under the attention mask.
  The model card documents `<species_value><dna>`, while the corpus card documents `<species>species_value<dna>`.
- Interpretation: The grammar ambiguity must remain a hard gate before the Mendelian window rule.
- Next action: Implement the isolated project, tests, and dry-run.

### 2026-08-20 17:43 UTC - Frozen tokenizer contract

- Hypothesis: The pinned tokenizer represents each candidate metadata scaffold as text-mode tokens followed by exactly one DNA-mode opener.
- Commit Hash: `d40a56ac83ac414bc5c31625bc3996007edbd407` (starting point; working tree uncommitted).
- Command: Load only the tokenizer from Carbon-3B revision `95c3c68fc77fdf70b1582031bacf9d7753f72cf2` and encode all eight grammar-condition prefixes with `return_token_mask=True`.
- Result: `<dna>` encodes as `[151669]`; all tagged prefixes end in `151669`; and every preceding token remains in text mode.
  Exact ID vectors are frozen in `tokenizer_snapshot.py` and covered by a drift-blocking test.
- Interpretation: Prompt evidence will retain both strings and IDs, and inference cannot proceed under an unexpected tokenizer representation.
- Next action: Run the project-local tests and Snakemake dry-run.

### 2026-08-20 17:38 UTC - Local contract verification

- Hypothesis: The isolated package enforces the frozen prompt, coordinate, allele, score-orientation, grouping, paired-bootstrap, and label-blind smoke contracts, and Snakemake can construct the full dependency graph without accessing data.
- Commit Hash: `d40a56ac83ac414bc5c31625bc3996007edbd407` (starting point; working tree uncommitted).
- Command: From `snakemake/analysis/carbon_conditioning_vep`, run `uv run --locked pytest` under the shared-node guard.
  Then run `uv run --locked snakemake -n` under the same guard and parse `sky/run.yaml` without launching it.
- Result: All 21 project tests passed in 4.38 seconds.
  That test process unexpectedly peaked at 1,016,944 KiB RSS because the test collection imports the ML stack, so the full command was not repeated locally.
  The two subsequently touched lightweight test modules passed all 6 tests at 126,680 KiB peak RSS.
  The first dry-run exposed a wildcard expansion error in a JSON-valued parameter; after the fix, the dry-run constructed the intended 15-job full DAG and exited successfully at 117,720 KiB peak RSS.
  The Sky task parses as YAML and defaults to the `smoke` target.
  No model inference, Mendelian metric computation, S3 result write, or remote job was launched.
- Interpretation: The implementation is ready for the issue-required paid remote smoke, starting with the prompt grammar gate and stopping after the label-blind four-condition score matrix.
- Next action: Obtain explicit approval for the paid remote smoke, launch it, and inspect grammar evidence, score rows, devices, runtime, and peak memory before requesting separate approval for the full matrix.

### 2026-08-20 17:52 UTC - Lower-cost smoke resource

- Hypothesis: Carbon-3B bf16 scoring with one variant, represented by four allele-strand prompts, fits an A10G while preserving the exact likelihood path.
- Commit Hash: `d40a56ac83ac414bc5c31625bc3996007edbd407` (starting point; working tree uncommitted).
- Command: Compare current SkyPilot catalogs for AWS A10G and Lambda A10/A6000/RTX6000 offerings and verify provider authentication.
- Result: Lambda offers a 48 GiB A6000 at $1.09/hour, but the Lambda VM would need temporary AWS credentials to access the private S3 inputs and outputs.
  AWS us-east-2 offers a 1-GPU A10G `g5.2xlarge` spot instance with 32 GiB host memory at $0.388/hour in the current catalog.
- Interpretation: Keep the existing same-account AWS S3/IAM boundary, pin `g5.2xlarge`, use spot without an on-demand fallback, and abort manually if the fresh catalog price exceeds $0.40/hour because SkyPilot 0.12 cannot encode that ceiling.
- Next action: Dry-run the changed smoke target, then launch and actively inspect its first minutes.

### 2026-08-20 18:03 UTC - A10G smoke startup findings

- Hypothesis: The locked workflow can execute Carbon's released preflight and the label-blind smoke on a 24 GB-class A10G.
- Commit Hash: `d40a56ac83ac414bc5c31625bc3996007edbd407` (starting point; working tree uncommitted).
- Command: Launch `sky/run.yaml` as cluster `carbon-conditioning-vep-smoke`, inspect startup, and allow the configured five-minute autodown after failures.
- Result: The first spot VM launched at no more than $0.40/hour and confirmed an NVIDIA A10G with 23,028 MiB VRAM.
  Setup first failed because the image had `uv` 0.12.5 instead of the project's required 0.11.31.
  After pinning `uv`, the unconstrained lock installed PyTorch 2.13.0 with CUDA 13.0, which could not use the VM's driver 535.216.01 and CUDA 12.2 interface.
  PyTorch was then locked to 2.12.1 from the official CUDA 12.6 wheel index; Carbon-3B loaded on the GPU, and the released sequence-recovery slice exposed ambiguous bases rejected by the local preflight.
  Carbon's pinned runner passes those bases through verbatim, so the local truncation now matches that behavior and has a regression test.
  The first cluster autodowned successfully.
  A corrected spot relaunch found no `g5.2xlarge` capacity in any us-east-2 zone and created no instance.
- Interpretation: The code-level startup failures are corrected and reproducibly locked.
  Use the requested A10G on-demand shape for the next smoke rather than waiting or polling for spot capacity.
- Next action: Dry-run and launch one on-demand `g5.2xlarge`, inspect the remote tests and smoke artifacts, then tear it down automatically.

### 2026-08-20 18:13 UTC - Held-out split access stopped and remediated

- Hypothesis: Requesting `split="train"` through the pinned Mendelian dataset builder reads only development labels.
- Commit Hash: `d40a56ac83ac414bc5c31625bc3996007edbd407` (starting point; working tree uncommitted).
- Command: Run the on-demand A10G smoke and monitor the dataset-build logs before model scoring.
- Result: All 22 then-current tests passed remotely, and the prompt preflight selected the corpus-card grammar with correct-tag accuracy 0.470833 versus untagged accuracy 0.463889 across 24 sequence-recovery rows.
  During `build_windows`, Hugging Face reported generating both 16,140 train rows and 9,490 test rows even though `load_dataset(..., split="train")` was requested.
  The job was canceled before Carbon Mendelian VEP inference began, but the dataset library had already materialized held-out test labels in its cache and uploaded train-derived window intermediates.
  The A10G instance was terminated, and the three affected S3 intermediates were deleted while the independent prompt-preflight artifacts were retained.
  The loader now downloads exactly `train.parquet` from the pinned dataset revision with `hf_hub_download`, rejects any other split or filename, and has focused regression coverage.
- Interpretation: A split argument is not a sufficient access boundary for repository-backed dataset builders that discover or prepare multiple files.
  Development workflows must select the exact allowed source file before parsing.
- Next action: Dry-run and retry the approved A10G label-blind smoke, verify that logs contain no test-split generation, inspect all four score/runtime artifacts, and terminate the instance.

### 2026-08-20 18:22 UTC - Corrected A10G smoke passed

- Hypothesis: Carbon-3B bf16 prompt-conditioned VEP scoring fits safely on one 24 GB-class A10G with a one-variant inference batch.
- Commit Hash: `d40a56ac83ac414bc5c31625bc3996007edbd407` (starting point; working tree uncommitted).
- Command: Launch the corrected `smoke` DAG on one on-demand AWS `g5.2xlarge`, run the project test suite, build development windows from the exact pinned `train.parquet`, score eight label-blind variants under all four prompts, validate the S3 artifacts, and terminate the cluster.
- Result: All 24 project tests passed remotely.
  The window build emitted no test-split generation and the scorer received no label, subset, match-group, source, trait, ClinVar, or consequence columns.
  Every condition produced eight aligned rows with finite likelihood atoms, the frozen model revision, the selected corpus-card grammar, and the expected prefix.
  Per-condition scoring took 9.033 to 9.079 seconds after model files were cached; the first condition took 53.450 seconds including its model download.
  Peak allocated GPU memory was 11.235 to 11.263 GiB on the NVIDIA A10G.
  The Sky job succeeded, uploaded all smoke artifacts, and the instance was explicitly terminated immediately afterward.
- Interpretation: A10G has ample memory for the locked one-variant batch and is the supported resource for the full development-only matrix.
  Smoke scores are an operational validation only because labels were intentionally absent.
- Next action: Obtain separate approval before launching the full four-condition development matrix and computing AUPRC or paired bootstrap results.

### 2026-08-20 18:49 UTC - GH200 benchmark and promoter-pilot scope

- Hypothesis: A Lambda GH200 reduces total scoring cost despite costing $2.29 per hour versus $1.21 per hour for the AWS A10G.
- Commit Hash: `d40a56ac83ac414bc5c31625bc3996007edbd407` (starting point; working tree uncommitted).
- Command: Stage the existing eight-row label-blind smoke input through Sky, launch one Lambda `gpu_1x_gh200` with no AWS credentials, run all project tests, and score the correct condition with the locked batch size of one.
  Query the development window artifact with a three-column parquet projection to identify the requested promoter subset.
- Result: Named-region launches in Lambda `us-east-1` and `us-east-3` first failed without creating an instance.
  A provider-wide retry acquired a GH200 in `us-east-3`.
  All 24 tests passed on ARM64 with PyTorch 2.12.1+cu126.
  The eight scoring batches took 1.55 seconds on GH200 versus 6.7 seconds on A10G, a 4.3-fold throughput increase at a 1.89-fold hourly-price increase.
  The runtime record included the first model download and reported 31.02 seconds, 11.29 GiB peak allocated GPU memory, and an NVIDIA GH200 device.
  ARM dependency installation took 3 minutes 56 seconds because `polars-bio` built from source.
  The development `tss_proximal` subset contains 2,050 rows, 205 positives, and 205 complete match groups.
  The one-minute autodown removed the instance before a cached rerun; no cluster remains.
- Interpretation: Preliminary batch progress suggests GH200 lowers inference-only cost by about 2.3-fold for this scorer.
  The narrowed two-condition promoter pilot projects to about 13 minutes of scoring and $0.50 of GH200 inference, plus about $0.19 of cold-start setup observed in this run.
  The benchmark did not retain the GH200 score parquet, so cross-device score and rank agreement remain unverified.
- Next action: Validate the revised promoter-only DAG and obtain approval for the approximately $0.70 Lambda GH200 pilot before computing labels or metrics.

### 2026-08-20 19:08 UTC - GH200 batch sizing and dependency cleanup

- Hypothesis: Using eight variants per GH200 inference call improves throughput enough to reduce the promoter-pilot cost further.
- Command: Launch a second approved label-blind Lambda GH200 smoke with an eight-row batch, run the full locked test suite, score once after a cold model download, repeat from the local model cache, compare the cached output with the A10G smoke, and terminate the cluster.
- Result:
  The dependency split avoided the earlier ARM64 `polars-bio` source build; 95 packages prepared in 34.63 seconds.
  All 25 tests passed in 9.70 seconds.
  The cold scoring command took 32.48 seconds, including a 23-second model download, while the cached command took 6.00 seconds.
  The single eight-row inference batch itself took 3.70 seconds and allocated 44.76 GiB, compared with 1.55 seconds and 11.29 GiB for eight one-row batches on the same GH200.
  Against the A10G smoke, the maximum absolute score difference was 0.002263 and the eight-row score Spearman correlation was 0.6905.
  The benchmark cluster terminated successfully; another user's live GH200 cluster was left untouched.
- Interpretation:
  Large prompt batches do not improve this scorer's GH200 throughput, so the promoter pilot retains the one-variant batch.
  The cross-device comparison is diagnostic rather than a pilot result; both promoter conditions will be scored on the same GH200 and paired on identical rows.
  The two-condition pilot remains approximately 13–15 GPU-minutes, or $0.50–$0.60 at $2.29 per hour, with a conservative operational cap near $0.90.
- Next action: Stage and dry-run the promoter-only Lambda task without forwarding AWS credentials, then request explicit approval before any labeled scoring.

### 2026-08-20 19:20 UTC - Promoter inputs staged and Lambda task dry-ran

- Hypothesis: The validated development artifact can be filtered to the promoter pilot on the coordinator and transferred to Lambda without AWS credentials or held-out data on the GPU VM.
- Command:
  Add provenance-checked `stage-windows` support, stage the pinned preflight and development-window artifacts, inspect only the promoter metadata columns, dry-run the local-storage Snakemake DAG, and dry-run the Lambda Sky task.
- Result:
  The first staging audit peaked at 1.27 GiB because the CLI eagerly imported Torch for a non-scoring command.
  Command-local model imports and 256-row parquet streaming reduced the repeated staging audit to 365 MiB peak RSS.
  The staged parquet is 13 MiB and contains exactly 2,050 `train` rows, 205 positives, and 205 complete ten-row `tss_proximal` match groups; it has no exclusions.
  The local-storage DAG contains only two score jobs, two absolute-metric jobs, one paired delta, one report, and the aggregate target.
  Sky's no-cost dry-run selected one Lambda `gpu_1x_gh200` in `us-east-1`, with 64 vCPUs, 432 GB memory, one GH200, and a current price of $2.29 per hour.
- Interpretation:
  The paid VM needs only the 13 MiB promoter window parquet and small preflight files; it does not need AWS credentials or the genomic reference.
  The expected charge remains $0.50–$0.60, with explicit approval requested up to $1.00 for the 20-minute command timeout, three-minute autodown, and transfer overhead.
- Next action: Obtain explicit approval for up to $1.00, launch the two-condition promoter pilot, retrieve and validate the result bundle, terminate the cluster, and publish the evidence upward.

### 2026-08-20 19:40 UTC - First promoter run succeeded but transfer failed

- Hypothesis: The committed Lambda task at `8f8afb58f33c42440fb1babe777d04ef92893eee` completes both promoter conditions within the $1.00 approval cap and leaves enough autodown time to retrieve results through the documented SSH alias.
- Command: Launch `sky/run-gh200-pilot.yaml` as `carbon-conditioning-vep-gh200-pilot`, follow job 1, rsync `results/promoter_pilot`, and down the cluster.
- Result:
  Lambda `us-east-1` lacked capacity; Sky acquired the same $2.29/hour GH200 shape in `us-east-3`.
  All 27 tests passed.
  Correct scoring completed 2,050 rows in 3 minutes 30 seconds, untagged scoring completed them in 3 minutes 27 seconds, and all seven DAG steps succeeded.
  Plain `rsync` could not resolve the cluster name because SkyPilot 0.12 keeps generated cluster hosts in per-cluster files under `~/.sky/generated/ssh/` instead of the default SSH configuration.
  The three-minute autodown terminated the VM before the corrected transfer command was constructed, so the ephemeral result files were lost.
- Interpretation:
  The experimental computation passed, but the run is not usable without its score and metric artifacts.
  A rerun should keep the combined cost below the approved $1.00 cap because each inference pass was about half the projected duration.
- Next action: Pass the generated per-cluster SSH configuration explicitly to rsync, use a two-minute autodown, dry-run, snapshot the correction, and rerun once.

### 2026-08-20 19:58 UTC - Retained promoter pilot result

- Hypothesis: Correct mammalian conditioning improves Carbon-3B AUPRC over an untagged prompt on the development-only `tss_proximal` Mendelian subset.
- Commit Hash: `2de69ae1`.
- Command: Launch `sky/run-gh200-pilot.yaml` as `carbon-conditioning-vep-gh200-pilot`, run all locked tests and the seven-job promoter DAG on one Lambda GH200, retrieve `results/promoter_pilot` with SkyPilot's generated SSH configuration, validate every score and metric contract locally, and explicitly terminate the cluster.
- Config: Carbon-3B `95c3c68fc77fdf70b1582031bacf9d7753f72cf2`; corpus-card prompts `<dna>` and `<species>vertebrate_mammalian<dna>`; Mendelian development `train` rows only; subset `tss_proximal`; 2,050 rows; 205 positives; 205 complete ten-row match groups; 8,190 scored DNA bases per strand; bf16; batch size one; FWD/RC average; 1,000 seeded paired match-group bootstrap draws.
- Result:
  All 27 locked tests passed in 9.71 seconds before scoring.
  Untagged AUPRC was 0.177510 with absolute 95% bootstrap interval [0.138739, 0.227481].
  Correct-conditioned AUPRC was 0.173788 with absolute 95% bootstrap interval [0.137731, 0.221405].
  The paired correct-minus-untagged delta was -0.003722 with 95% interval [-0.026868, 0.020443].
  Both conditions contained the same 2,050 unique variants and identical labels, subsets, and match groups; every likelihood atom was finite and satisfied the score identities; no window rows were excluded.
  Correct scoring took 243.6 seconds with 11.29 GiB peak allocated GPU memory and 4.81 GiB peak RSS; untagged scoring took 213.0 seconds with 11.26 GiB peak allocated GPU memory and 4.26 GiB peak RSS on an NVIDIA GH200 480GB.
  The corrected SSH transfer succeeded, and explicit teardown left no cluster named `carbon-conditioning-vep-gh200-pilot`.
  SkyPilot's local cost report estimates each 14-minute cluster at $0.54, or $1.08 combined; this exceeded the approved $1.00 cap by $0.08 because the first successful run's artifacts were lost to the original retrieval bug.
  Local artifact validation exited zero in 1.08 seconds at 207,848 KiB peak RSS.
- Interpretation:
  This scoped pilot provides no evidence that correct mammalian conditioning improves promoter-variant ranking over no conditioning.
  The confidence interval includes zero, so the result does not demonstrate equivalence or a detrimental effect either.
  The planned broader four-condition matrix should not be launched on the basis of this result.
- Next action: Preserve the compact result artifacts in an annotated experiment snapshot and request human review before any GitHub publication, S3 upload, or additional compute.

### 2026-08-20 20:37 UTC - Full-development run prepared

- Hypothesis: The promoter-only result may not represent Carbon-3B's conditioning effect across all Mendelian consequence subsets, so compare the same untagged and correct prompts on the complete development split.
- Commit Hash: `d42215f5`.
- Command:
  Stage the canonical validated train-window artifact with `sky/stage-gh200-full.sh` under the shared-node guard.
  Run `uv run --locked pytest tests/test_pipeline.py tests/test_report.py`, the exact local-storage Snakemake dry-run with `config/full_development.yaml`, the SkyPilot dry-run, and the repository pre-commit suite.
- Config: 16,140 train rows; 1,614 positives; 1,614 complete ten-row match groups; all observed consequence subsets; untagged and `vertebrate_mammalian` corpus-card prompts; Carbon-3B `95c3c68fc77fdf70b1582031bacf9d7753f72cf2`; bf16; one variant per inference batch; 1,000 seeded match-group bootstrap draws; Lambda GH200 at $2.29 per hour; 70-minute command timeout; two-minute autodown; $3.00 approved cap.
- Result:
  Bounded staging validated the exact row, positive, group, prompt, dataset, split, reference, and assembly contracts and peaked at 294,684 KiB RSS.
  The first focused test run exposed a legacy fixture with only one positive across two synthetic match groups after the stronger staging check was added.
  Correcting the fixture to the real one-positive-per-group invariant produced 8 passing focused tests.
  The exact Snakemake dry-run contains two score jobs, two absolute-metric jobs, one paired-delta job, one report job, and the aggregate target, with no upstream window or preflight jobs.
  The SkyPilot dry-run selected one Lambda `gpu_1x_gh200` in `us-east-1` at $2.29 per hour.
  The full pre-commit suite passed after its formatter changed one Python file.
  No paid resource was created during preparation.
- Interpretation: The follow-up is isolated under `results/full_development`, preserves the promoter snapshot, and can run on Lambda without AWS credentials.
- Next action: Launch `carbon-conditioning-vep-gh200-full`, verify all 29 remote tests and initial throughput, retrieve the result bundle immediately, and terminate the instance.

### 2026-08-20 21:43 UTC - Full-development result retained

- Hypothesis: Correct mammalian conditioning improves Carbon-3B macro AUPRC over an untagged prompt across the complete development-only Mendelian variant set.
- Commit Hash: `d42215f5` (run implementation); `47610800` (launch record).
- Command:
  Launch `sky/run-gh200-full.yaml` as `carbon-conditioning-vep-gh200-full`, fall back from unavailable Lambda `us-east-1` capacity to `us-east-3`, run all locked tests and the exact seven-job DAG, retrieve `results/full_development` with SkyPilot's generated SSH configuration, validate the result bundle independently, and explicitly terminate the cluster.
- Config: Carbon-3B `95c3c68fc77fdf70b1582031bacf9d7753f72cf2`; corpus-card prompts `<dna>` and `<species>vertebrate_mammalian<dna>`; exact pinned Mendelian development `train.parquet`; all 16,140 development rows; 1,614 positives; 1,614 complete ten-row match groups; nine observed consequence subsets; 8,190 scored DNA bases per strand; bf16; batch size one; FWD/RC average; 1,000 seeded match-group bootstrap draws; one Lambda GH200 at $2.29 per hour.
- Result:
  All 29 locked tests passed in 9.93 seconds before scoring, and all seven Snakemake jobs succeeded.
  Untagged macro AUPRC was 0.358200 with absolute 95% bootstrap interval [0.338822, 0.387681].
  Correct-conditioned macro AUPRC was 0.355182 with absolute 95% bootstrap interval [0.333180, 0.385529].
  The paired correct-minus-untagged macro delta was -0.003018 with 95% interval [-0.018293, 0.008793] across 1,610 eligible match groups; the four-group mature-miRNA subset was excluded from the macro by the prespecified minimum-group rule.
  The exploratory missense subset was the only unadjusted subset interval excluding zero, favoring untagged conditioning with delta -0.008599 and 95% interval [-0.016770, -0.000791].
  Both conditions contained the same 16,140 unique variants and identical labels, subsets, and match groups; every likelihood atom was finite and satisfied the score identities; no window rows were excluded.
  Correct scoring took 1,683.85 seconds with 11.29 GiB peak allocated GPU memory and 4.67 GiB peak RSS; untagged scoring took 1,617.17 seconds with 11.26 GiB peak allocated GPU memory and 4.88 GiB peak RSS on an NVIDIA GH200 480GB.
  Independent local recomputation reproduced both absolute metric tables and the paired bootstrap table within 1e-13 and exited zero in 34.69 seconds at 274,808 KiB peak RSS.
  The corrected SSH transfer succeeded, explicit teardown left no cluster named `carbon-conditioning-vep-gh200-full`, and SkyPilot's local cost report estimates 59 minutes 49 seconds and $2.28 against the approved $3.00 cap.
  No held-out labels or predictions were accessed, and no result was uploaded to S3 or published to GitHub.
- Interpretation:
  The complete development set provides no evidence that correct mammalian conditioning improves overall Carbon-3B ranking over no conditioning.
  The macro interval includes zero, so the experiment does not establish equivalence or an overall detrimental effect.
  The isolated missense interval is exploratory and uncorrected for the nine subset comparisons, so it does not change the primary conclusion.
  Do not launch the broader four-condition matrix without a new hypothesis.
- Next action: Preserve the compact metrics, runtimes, summary, exclusions, and checksum manifest in an annotated local snapshot, and request human review before any GitHub publication or durable raw-score upload.
