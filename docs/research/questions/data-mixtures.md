# How to optimize pretraining data mixtures?

## TL;DR

No validated genomic mixture-optimization method exists yet. Proxy-swarm regression is promising but depends on small-to-large-scale rank transfer, objective signal, finite-data repetition, and leakage controls; confidence is low, and the next gap is a bounded proxy study before any broad or paid sweep.

## Question

How should we optimize genomic pretraining data mixtures to improve downstream genomic performance at target scale? This includes deciding how to partition the data, what objective to optimize, and which optimization strategies are reliable and compute-efficient.

This issue tracks evidence across multiple possible approaches; swarm-based regression with small proxy models is one especially promising candidate, but it is not the definition of the research question. The intended outcome is a reproducible way to choose mixtures, not only a single set of weights.

## Current answer

No validated method for optimizing genomic pretraining mixtures exists yet. The current evidence shows that mixture components matter: region specialists differ sharply by downstream class, and mammalian species density affects some regions but not others. It does not show how to select a joint mixture.

Proxy-swarm regression is a plausible candidate, not the answer. Its usefulness depends on four unverified conditions: the downstream objective must have measurable signal in small models; mixture rankings must transfer to target scale; finite high-value components must tolerate the repetitions implied by target weights; and the component manifest, overlap policy, and leakage rules must remain fixed.

The optimization target is also unresolved. Macro-average VEP, minimax performance, zero-shot scores, global probes, per-subset probes, and later regulatory tasks can favor different recipes. Selecting whichever readout looks best after the swarm would invalidate the optimization claim.

Confidence is low. The next useful work is a bounded proxy-noise study with fixed anchor mixtures and repeated seeds, followed by held-out-mixture prediction. A broad swarm or target-scale launch is not justified until rank stability, repetition feasibility, and leakage controls pass those gates.

<details>
<summary>Related work</summary>

- The [Open Athena pretraining data community meeting slides](https://www.figma.com/deck/LcfkIgrKJUHV1DrJ40XbXb/Open-Athena-Pretraining-Data-Community-Meeting) define a source → normalize → deduplicate/decontaminate → bucket → optimize → train/evaluate pipeline and treat a mixture as a sampling policy over fixed buckets. The implication is that component definitions and overlap policy are part of the optimization contract. The remaining gap is a versioned MarinDNA manifest with unique-token budgets and leakage audits.
- [RegMix](https://arxiv.org/abs/2407.01492) samples mixtures, trains small proxy models, fits a surrogate, and optimizes predicted performance. [Olmix](https://arxiv.org/abs/2602.12237) studies this family more systematically. These methods motivate offline proxy swarms and held-out-mixture validation. Their proxy sizes, swarm sizes, and rank-transfer behavior cannot be assumed to carry over to genomic objectives.
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) shows diminishing returns from repeated data. This makes target-scale epoching part of mixture feasibility: a component that is barely repeated in a proxy may be heavily repeated in the final run. The missing observable is performance as a function of realized repetition for each genomic component.
- Candidate surrogate families include linear/log-linear models, regularized interactions, tree models, and rank or pairwise objectives. Flexible models require held-out mixtures or nested validation; in-swarm fit alone does not support mixture selection.
- Candidate outputs should include intended and realized weights, unique tokens and repetitions per component, per-subset downstream scores, uncertainty, held-out ranking, and target-scale regret. Without those fields, a selected weight vector is not a reproducible optimization result.

</details>

<details>
<summary>Related experiments</summary>

- [#232](https://github.com/Open-Athena/marin-dna/issues/232) trained six region specialists and a background arm. Home-region specialists won their matched Mendelian classes, showing that biological-domain mixture components can have distinct downstream utility; it did not optimize a combined mixture.
- [#255](https://github.com/Open-Athena/marin-dna/issues/255) compared 108-family and 19-order mammalian cohorts at matched compute across five regions. Species density was neutral for some regions and harmful for others, showing interaction between taxonomic mixture and biological domain; one seed and different epoch counts limit surrogate use.

</details>

## Possible directions

### Core questions

1. How should the training corpus be partitioned into mixture components?
2. Which downstream objective should define a "good" mixture?
3. Which optimization approach, or combination of approaches, should be used?
4. For proxy-based approaches, can the objective be measured with enough signal at small scale?
5. For surrogate approaches, should the model predict absolute performance, within-swarm rankings, or pairwise differences between mixtures?
6. How should finite data and target-scale repetition be represented?
7. Do selected mixtures remain effective as model size and training-token budget increase?
8. Optionally, what evolutionary breadth or taxonomic mixture should the training data cover?
9. Should weights be stationary throughout training, or differ between broad pretraining and a cooldown or midtraining phase?

### Candidate optimization approaches

The research question does not presuppose one optimization method. Candidate approaches include, but are not limited to:

1. **Expert-designed mixtures and controlled ablations:** useful as baselines and for testing a small number of strong biological hypotheses.
2. **Direct search:** random, structured, Bayesian, or evolutionary search over mixture weights when the number of components and training cost permit.
3. **Offline proxy-swarm regression:** train small models on sampled mixtures, fit a surrogate, and optimize its predicted downstream utility, as in RegMix and Olmix.
4. **Online or adaptive reweighting:** update mixture weights during training using learning dynamics, validation signals, or estimated domain utility.
5. **Scaling-law, analytical, or hybrid methods:** model how data quantity, repetition, model scale, and domain interactions affect utility, possibly combined with proxy or online measurements. Structured saturation and overexposure models such as the Domain Saturation-Penalty (DSP) form highlighted in the community slides are concrete candidates alongside black-box regressors.

Comparisons should account for the compute spent selecting the mixture, not only the compute of the final training run. Different approaches may also be appropriate at different stages or scales.

### Candidate mixture axes

Treat mixture construction as a grid over at least two axes.

#### Axis 1: biological domain

Use the five v4 specialist partitions from the latest per-region experiment ([exp255](https://github.com/Open-Athena/marin-dna/blob/6370a321fcd08eb0a55bcb19cdd1b0586f382118/experiments/exp255_per_region_order.py#L4-L12)) as the provisional starting taxonomy:

- coding sequence (`v4_cds`);
- non-promoter cCRE (`v4_ccre_non_promoter`);
- 3′ UTR (`v4_utr3`);
- ncRNA exon (`v4_ncrna_exon`); and
- TSS region plus 5′ UTR (`v4_tss_region_and_utr5`).

These are the five specialist v4 partitions; `v4_bg` is the separate background partition. Retain background as an optional mixture component or control rather than calling it a sixth functional region.

#### Axis 2: putative data quality

Partition each biological domain into pinned conservation tiers, such as high, medium, and low conservation, rather than assuming only a binary conserved/unconserved split. The exact conservation statistic, species panel, thresholds, and missing-value policy remain part of this research question.

Conservation should initially be described as a *putative quality proxy*, not as ground-truth data quality. The experiment should be able to discover that less-conserved data are useful, either in general or for particular downstream tasks.

The resulting components are cells such as:

```text
CDS × high conservation
CDS × medium conservation
CDS × low conservation
non-promoter cCRE × high conservation
...
TSS region + 5′ UTR × low conservation
```

#### Axis 3 (optional): evolutionary scale

A third axis could control the evolutionary distances represented in training: for example, how much data should come from primates, mammals, vertebrates, plants, or other clades. This is scientifically important but substantially expands the search space, so it can remain an optional follow-up direction.

The example taxonomic groups are nested: primates are mammals, and mammals are vertebrates. Do not assign independent mixture weights to overlapping groups. Use one of two explicit formulations:

1. **Disjoint clade buckets:** for example, primates, non-primate mammals, non-mammalian vertebrates, plants, and any additional non-overlapping groups supported by the data.
2. **Hierarchical allocation:** first allocate weight among broad clades, then allocate each clade's weight across its descendant groups and the biological-domain-by-conservation cells.

This axis is also confounded by the number of available species and their phylogenetic redundancy. Raw sequence counts should not make a densely sampled clade dominate by construction. Compare species-sampling policies such as all available species, one species per family, and one species per order, and record both token exposure and phylogenetic coverage.

Avoid beginning with the full domain-by-conservation-by-clade Cartesian product. It may create too many small or weakly identifiable cells. Start with a factorized or hierarchical parameterization and add cross-axis interactions only when proxy data support estimating them.

### Static versus phase-specific mixtures

The community slides distinguish a broad-coverage Phase 0 from a cooldown or midtraining Phase 1, using the same buckets with different weights and fitting those weights for a particular model size and token budget. This is a schedule decision, not another biological component axis. A phase-aware formulation would optimize vectors `p^(0)` and `p^(1)` together with pinned phase budgets `R^(0)` and `R^(1)`.

For component `j` with `N_j` unique tokens, total target-scale exposure becomes

```math
e_j = \frac{R^{(0)}p_j^{(0)} + R^{(1)}p_j^{(1)}}{N_j}.
```

For components with nonzero total exposure, the late-phase share can also be recorded as

```math
r_j = \frac{R^{(1)}p_j^{(1)}}{R^{(0)}p_j^{(0)} + R^{(1)}p_j^{(1)}}.
```

Start with a stationary mixture as the identifiable baseline. Add a phase-specific arm only if proxy or intermediate-scale evidence can distinguish schedule effects, and compare schedules at matched total per-component exposure so that a cooldown effect is not confused with simply seeing more copies of a scarce component.
### What should the mixture optimize?

A plausible initial follow-up issue could focus on the project's current Mendelian-trait variant-effect-prediction (VEP) evaluation and treat [DART-Eval](https://arxiv.org/abs/2412.05430) as a separate target. DART-Eval covers regulatory DNA in zero-shot, probing, and fine-tuning settings, and may favor a different data mixture from Mendelian VEP. That difference is scientifically useful and should not be hidden by immediately collapsing every evaluation into one score.

Let `s_t(p)` be AUPRC for trait or evaluation subset `t` after training on mixture `p`. Candidate objectives include:

1. **Macro average:** `U_mean(p) = (1/T) sum_t s_t(p)`

   This rewards average performance and gives each trait equal weight.

2. **Worst-subset or minimax performance:** `U_min(p) = min_t s_t(p)`

   This protects against a mixture that improves the mean by sacrificing one trait, but it may be dominated by the noisiest or smallest subset.

3. **Robust compromise**

   Use a predeclared soft minimum, lower quantile, or `U_mean(p) - lambda D(p)` objective, where `D(p)` measures performance dispersion, to trade some mean performance for consistency without allowing one noisy subset to determine the entire mixture.

4. **Baseline-relative constrained improvement**

   Let `p_0` be a pinned baseline or expert proposal. Optimize a predeclared primary utility while requiring every guardrail subset to remain within a tolerance of its baseline score:

   ```math
   \max_p U_{\mathrm{primary}}(p) - \eta D(p,p_0)
   \quad \text{subject to} \quad
   s_t(p) \geq s_t(p_0) - \epsilon_t \;\; \text{for all guardrail subsets } t.
   ```

   This adapts the A2B pattern in the community slides: guard all tasks, improve selected targets, and stay local to the proposal. It may be easier to interpret than collapsing primary and guardrail tasks into one scalar, but the tolerances, distance, and primary target must be preregistered.
#### Which evaluation readout defines the objective?

The optimization metric must also specify how model performance is read out. Candidate definitions are:

- zero-shot variant scores;
- one global linear probe shared across applicable subsets; and
- separate per-subset linear probes.

Avoid defining the objective as the post hoc "best of zero-shot and probe." That gives every mixture multiple chances to win and makes the selected objective depend on evaluation noise. Better options are to:

- preregister one readout as primary and report the others as secondary;
- define a fixed composite of normalized zero-shot and probe scores; or
- treat the readouts as a multi-objective problem and report a Pareto frontier.

Probe training data, hyperparameter searches, and random seeds must be identical across mixtures. The test split used to report the final comparison must not be used to fit mixture weights.

### Can proxy models provide signal at small scale?

A possible follow-up experiment could audit signal over a small set of deliberately different anchor mixtures—for example natural proportions, uniform cells, CDS-heavy, regulatory-heavy, high-conservation-heavy, and low-conservation-heavy—before considering a large swarm.

For each proxy scale and evaluation readout, estimate:

- between-mixture variation;
- seed-to-seed and probe-to-probe variation;
- confidence intervals for per-subset and aggregate scores;
- separation from a random or trivial predictor;
- stability over training checkpoints; and
- Spearman/Kendall rank agreement across proxy sizes and token budgets.

The mixture signal must be distinguishable from run and evaluation noise. If a metric is effectively random at proxy scale, it should not be used as a regression target merely because it is meaningful at large scale. Options are to increase proxy size or duration, use a lower-variance intermediate metric, or stop and record that the proposed optimization is not currently cost-effective.

The community slides sharpen the intermediate-metric option into a concrete open problem: build low-cost, smooth evaluations that emerge at swarm scale and correlate with the intended target-scale objective, because many hard benchmarks show no signal in small proxies. A smooth proxy metric may be used for screening only after its cross-mixture correlation and rank transfer are evaluated on held-out mixtures at one or more larger scales; smoothness alone is not evidence of relevance.

### Proxy-swarm targets: absolute versus relative performance

Possible surrogate targets include:

1. **Absolute performance:** predict the measured score for each mixture.
2. **Centered or ranked performance:** predict a mixture's rank or its score after centering within a common swarm/evaluation batch.
3. **Pairwise preference:** predict which of two mixtures performs better, preferably using matched evaluation and training seeds where possible.

Relative targets may suppress shared run-level noise and align directly with the goal of selecting the best mixture. They also discard effect-size information and can become incomparable across independently trained swarms. Include the same anchor mixtures in every swarm batch so batches can be calibrated, and evaluate all target formulations on held-out mixtures.

Surrogate quality should be judged by selection behavior, not training fit:

- rank correlation on held-out mixtures;
- recall among the top `k` mixtures;
- regret relative to the best observed held-out mixture;
- calibration or error for absolute-score models; and
- stability of the proposed optimum under bootstrap resampling of proxy runs.

### Simulating target-scale epoching

For target training budget `R` tokens, component `j` with `N_j` unique tokens and proposed weight `p_j` has an approximate exposure count

```math
e_j(p) = \frac{R p_j}{N_j}.
```
This exposes a mismatch between a short proxy run and a long target run. Possible treatments include:

1. **Hard repetition constraints:** `p_j <= e_j^max N_j / R`

   Optimize the learned surrogate only over mixtures that do not exceed a predeclared maximum exposure for any component.

2. **Repetition-aware utility**

   Penalize or discount weights whose implied target-scale epochs enter a diminishing-return regime. The penalty must be fixed from an independent repetition study or estimated in a nested training split, not tuned on the final evaluation.

3. **Simulated target exposure in proxy training**

   Construct proxy sampling schedules that reproduce the target run's component-level epoch counts or repetition pattern at reduced compute, then test whether this improves rank transfer.

The first approach follows Olmix's practical recommendation to enforce finite-data feasibility during mixture optimization. The third most directly captures the "simulated epoching" hypothesis. A follow-up experiment could compare them rather than assuming that scarce high-conservation cells must simply be downweighted in the regression itself.

Report the unconstrained optimum as a diagnostic. A large gap between the unconstrained and feasible optima is evidence that data availability, not only estimated utility, is determining the final recipe.

### Implementation ideas for follow-up issues

The following are a menu of possible implementation and analysis directions, not a prescribed study or ordered roadmap. The proxy-swarm path is currently the most developed because it appears especially promising, not because this issue has selected it over the alternatives. Each item could become a separate EDA or experiment issue.

#### Possible comparison: Benchmark optimization approaches

- Compare selected approaches against natural and expert-designed mixtures.
- Match or report both mixture-selection compute and final-training compute.
- Evaluate whether different approaches converge on similar weights.
- Record whether an approach remains practical as the number of mixture components grows.

#### Possible foundation: Define the data and evaluation contract

- Pin the biological-domain and conservation-tier definitions.
- Resolve overlaps, duplicates, and empty or tiny cells.
- Inventory unique tokens per cell.
- Pin train, probe-training, validation, and final test partitions.
- Audit direct sequence overlap and locus leakage into VEP and DART-Eval.
- Define a baseline mixture and the target training-token budget.

#### Possible experiment: Measure the proxy noise floor

- Train anchor mixtures with enough repeated seeds to separate mixture effects from run noise.
- Compare zero-shot, global-probe, and per-subset-probe signal.
- Select a primary objective and proxy scale before the full swarm.
- Stop or revise the proxy design if rankings are not reproducible.

#### Possible experiment: Train a mixture swarm

- Sample feasible mixtures from the simplex, combining broad exploration with perturbations around the baseline mixture.
- Include fixed anchor mixtures in every batch.
- Keep architecture, tokenizer, optimizer, training-token budget, evaluation cadence, and data preprocessing matched across runs.
- Record both sampled weights and realized token counts; assert that sampling noise does not materially change the intended mixtures.
- Choose swarm size from a documented power/sample-complexity analysis rather than copying a fixed run count from NLP.

Dense versus sparse Dirichlet swarms should be treated as a design choice to validate. With a grid of correlated genomic cells, sparse mixtures may expose cell effects more clearly, while dense mixtures may better resemble feasible target recipes.

#### Possible analysis: Fit and validate surrogates

Compare simple baselines before more flexible regressors:

- linear and log-linear models;
- a regularized model with interactions between domain and conservation axes;
- a tree-based model such as LightGBM when the swarm is large enough; and
- rank or pairwise models for the relative-performance hypothesis.

Use held-out mixtures or nested cross-validation for all model and hyperparameter selection. A flexible regressor with excellent in-swarm fit but poor held-out ranking is not useful.

When optimizing the surrogate:

- enforce the target-scale data constraints;
- consider a pinned KL or distance penalty from the baseline mixture to avoid unsupported extrapolation;
- quantify uncertainty in both the proposed weights and predicted utility; and
- return several diverse, near-optimal candidates when the optimum is flat.

#### Possible experiment: Test scale transfer

A confirmatory comparison could train matched models on:

1. natural data proportions;
2. the current expert-designed baseline;
3. the proxy-predicted macro-optimal mixture;
4. the proxy-predicted robust/minimax mixture; and
5. a deliberately different or randomly selected held-out mixture.

One risk-reducing option is to validate at an intermediate scale not used to fit the surrogate before comparing the best-supported candidates at the intended target scale. Architecture, tokenizer, optimizer, total training tokens, evaluation protocol, and number of seeds should be matched wherever feasible.

This research-question issue does not authorize paid large-scale training or a broad swarm. Any follow-up experiment proposing such a launch must obtain explicit approval.

#### Possible follow-up: Test objective specificity with DART-Eval

A separate follow-up issue could repeat or reuse the mixture analysis with DART-Eval regulatory tasks to ask whether:

- the VEP-optimized mixture transfers to regulatory tasks;
- a separately optimized DART-Eval mixture favors regulatory, UTR, ncRNA, or conservation cells differently; and
- a stable Pareto compromise exists between the two evaluation families.

Do not treat failure to transfer as a failure of mixture optimization; it may show that the optimal data recipe is downstream-objective dependent.

### Open design choices for follow-up issues

| Design choice | Candidate options | Possible evidence |
| --- | --- | --- |
| Optimization approach | expert/ablation; direct search; proxy-swarm regression; DSP-style saturation model; online/adaptive; scaling-law or hybrid | downstream utility, selection compute, robustness, and scale transfer |
| Mixture components | domain only; conservation only; domain × conservation | cell sizes, identifiability, held-out performance |
| Primary VEP utility | macro AUPRC; minimax; robust compromise; baseline-relative constrained improvement | uncertainty and sensitivity to small/noisy subsets |
| Evaluation readout | zero-shot; global probe; per-subset probes; fixed composite | proxy-scale signal and target-scale rank transfer |
| Surrogate target | absolute; centered/rank; pairwise | held-out rank, regret, and stability |
| Repetition handling | hard cap; utility penalty; simulated epoching; DSP-style saturation and overexposure terms | transfer under the target token budget |
| Weight schedule | stationary; Phase 0 plus cooldown or midtraining | held-out schedule transfer at matched total exposure |
| Evolutionary axis | omit initially; disjoint clade weights; hierarchical allocation | proxy signal, phylogenetic coverage, and target-scale transfer |
| Proxy configuration | model size, token budget, checkpoint | reproducibility and rank agreement with intermediate scale |
| Swarm design | dense; sparse; baseline-local; hybrid | coverage, regression fit, and selected-mixture regret |
| Final scope | one universal mixture; task-specific mixtures; Pareto family | Mendelian VEP and later DART-Eval results |

### Possible outputs from follow-up work

Depending on which follow-up issues are pursued, useful outputs could include:

- A versioned manifest defining the selected data cells and their unique-token budgets.
- A leakage audit covering evaluation loci, homologous/projected sequences, chromosomes, and duplicates.
- A clearly stated primary objective and evaluation protocol for a particular experiment.
- Reproducible configs for anchor, swarm, held-out-mixture, and scale-transfer runs.
- A machine-readable table with intended and realized mixture weights, training metadata, per-subset scores, aggregate objectives, and uncertainty.
- Held-out comparisons of surrogate families and target formulations.
- Repetition-feasibility plots or tables at the intended target budget.
- An intermediate/target-scale report comparing optimized and baseline mixtures at matched compute.
- Evidence supporting adoption, revision, or rejection of a mixture-selection method at the tested scales.

### Evidence that would advance the question

This research question can accumulate evidence across several independent issues. Meaningful progress would include one or more of:

- defining interpretable, versioned mixture components;
- identifying an evaluation objective with measurable proxy-scale signal;
- determining whether absolute, ranked, or pairwise surrogate targets work best on unseen mixtures;
- understanding how target-scale repetition changes feasible mixtures;
- measuring whether mixture rankings transfer across model scales or token budgets;
- learning whether different downstream objectives favor different mixtures; or
- establishing that proxy-regression mixture selection is not reliable or cost-effective in the tested regime.

### Boundaries

- Assuming conservation is synonymous with data quality.
- Selecting a mixture from final test-set performance.
- Treating the post hoc best of zero-shot and probing evaluations as a valid preregistered objective.
- Copying RegMix or Olmix proxy/swarm sizes without measuring genomic proxy-scale signal.
- Launching an exhaustive combinatorial sweep over the mixture grid.
- Selecting a smooth proxy metric without validating its held-out cross-mixture relationship to the target-scale objective.
- Fitting a stationary mixture and then changing the target run's cooldown or midtraining schedule without representing the resulting component exposures.
- Assuming that DART-Eval and Mendelian VEP must be optimized jointly in the same follow-up issue.
- Claiming that one optimized mixture will be best for every downstream task.

### Interpretations to confirm

This draft makes the following non-blocking interpretations of the transcription:

1. "Mendelian traits barne effect prediction" means the current **Mendelian-trait variant-effect-prediction** evaluation, summarized by AUPRC across trait subsets.
2. "OlmoMix" refers to **Olmix**, the data-mixing framework developed in the OLMo ecosystem, rather than OLMoE or a dataset named OLMo-mix.
3. "Simulated epoching" means accounting for the number and pattern of times each finite data component would be repeated at the target training budget. If it refers to a specific named method, add that citation and align the experimental arm with its definition.
4. The initial biological-domain axis is the five v4 specialist partitions used by exp255: CDS, non-promoter cCRE, 3′ UTR, ncRNA exon, and TSS region plus 5′ UTR. Background remains a separate candidate component/control.
5. DART-Eval is a separate possible follow-up target rather than necessarily part of the same scalar objective as Mendelian VEP.
6. Evolutionary scale is an optional follow-up axis. If included, nested taxonomic groups must be converted to disjoint buckets or modeled hierarchically so the same data are not counted under multiple weights.
7. Phase-specific mixing means separate weight vectors over the same semantic cells for broad pretraining and cooldown or midtraining; phase is not an additional biological bucket axis.
