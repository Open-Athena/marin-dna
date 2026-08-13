---
name: wandb-reporting
description: "Use W&B runs, reports, and artifacts consistently for experiments, benchmarks, and task results with dense numeric output."
---

# W&B Reporting

Use W&B when scalar series, plots, large comparison tables, or raw artifacts are too dense for issue comments or logbooks. Keep GitHub as the narrative layer and W&B as the data and report layer.

## Project Policy

- Choose project scope by the type signature of the work.
- Default to the `marin` project for pretraining runs.
- Use a new project for materially different work, such as kernel development or a new RL variant.
- Put runs requiring explicit run-to-run comparison in the same W&B project.
- Decide scope before launching; runs cannot reliably move across projects later.

## Run Naming And Metadata

- Use the same experiment or task ID in W&B run names, logbook entries, and issue comments.
- For a MarinDNA experiment issue `exp<N>`, include `dna-exp<N>` in every run name and use a stable group such as `dna-exp<N>-v<version>` for related arms or sweeps.
- Prefer artifacts for raw CSV or JSON outputs that feed published tables.

## Reporting

- Link W&B runs and reports from the coordinating issue and logbook.
- Summarize only the decision-relevant numbers in GitHub; link W&B for dense tables and plots.
- Before publishing a claim, verify expected row counts, key uniqueness, and de-duplication or aggregation logic.
- Keep report titles and sections aligned with issue and logbook labels.
- Keep W&B artifacts below 10 MB. Store larger artifacts with the experiment and link them.

## Completion Checklist

- Relevant runs are in the intended project.
- Run names map back to issue or logbook experiment IDs.
- The primary comparison table or chart is linked from the issue.
- Raw artifacts needed to reproduce the table are uploaded or linked.
- Claims in GitHub match the final W&B values.
