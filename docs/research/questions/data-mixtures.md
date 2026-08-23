# How to optimize pretraining data mixtures?

> [!NOTE]
> **TL;DR:** No validated genomic mixture-optimization method exists yet; proxy-swarm regression is promising but depends on rank transfer, objective signal, finite-data repetition, and leakage controls, so confidence is low and the next step is a bounded proxy study.

## Question

How should we optimize genomic pretraining data mixtures to improve downstream genomic performance at target scale?
This includes deciding how to partition the data, what objective to optimize, and which selection strategies are reliable and compute-efficient.
The intended outcome is a reproducible selection method, not a single set of weights.

## Current answer

No validated method for optimizing genomic pretraining mixtures exists yet.
The current evidence shows that mixture components matter: region specialists differ sharply by downstream class, and mammalian species density affects some regions but not others.
It does not show how to select a joint mixture.

Proxy-swarm regression is a plausible candidate.
Its usefulness depends on four unverified conditions: the downstream objective must have measurable signal in small models; mixture rankings must transfer to target scale; finite high-value components must tolerate the repetitions implied by target weights; and the component manifest, overlap policy, and leakage rules must remain fixed.

The optimization target is also unresolved.
Macro-average VEP, minimax performance, zero-shot scores, global probes, per-subset probes, and regulatory tasks can favor different recipes.
Selecting whichever readout looks best after the swarm would invalidate the optimization claim.

Confidence is low.
The next useful work is a bounded proxy-noise study with fixed anchor mixtures and repeated seeds, followed by held-out-mixture prediction.
A broad swarm or target-scale launch is not justified until rank stability, repetition feasibility, and leakage controls pass those gates.

<details>
<summary>Related work</summary>

- [RegMix](https://arxiv.org/abs/2407.01492) samples mixtures, trains small proxy models, fits a surrogate, and optimizes predicted performance.
  [Olmix](https://arxiv.org/abs/2602.12237) studies this family more systematically.
  These methods motivate offline proxy swarms and held-out-mixture validation, but their rank-transfer behavior cannot be assumed to carry over to genomic objectives.
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) shows diminishing returns from repeated data.
  Target-scale epoching is therefore part of mixture feasibility even when repetition is mild in a proxy run.

</details>

<details>
<summary>Related experiments</summary>

- [#232](https://github.com/Open-Athena/marin-dna/issues/232) trained six region specialists and a background arm.
  Home-region specialists won their matched Mendelian classes, showing that biological-domain components can have distinct downstream utility; the experiment did not optimize a combined mixture.
- [#255](https://github.com/Open-Athena/marin-dna/issues/255) compared 108-family and 19-order mammalian cohorts at matched compute across five regions.
  Species density was neutral for some regions and harmful for others, showing an interaction between taxonomic mixture and biological domain; one seed and different epoch counts limit surrogate use.

</details>

<details>
<summary>Possible directions</summary>

- Define a versioned, non-overlapping component manifest with unique-token budgets, leakage rules, and realized exposure counts.
- Predeclare one primary downstream objective and readout; report other tasks and subsets separately rather than selecting the best result after training.
- Measure the proxy noise floor with repeated anchor mixtures before launching a swarm.
- Compare simple expert mixtures with held-out-mixture predictions from regularized absolute, rank, or pairwise surrogates.
- Constrain candidate weights by target-scale repetition, then test whether proxy rankings transfer to an intermediate or target scale.
- Add conservation tier, evolutionary breadth, or phase-specific weights only after the simpler domain mixture is identifiable.

</details>
