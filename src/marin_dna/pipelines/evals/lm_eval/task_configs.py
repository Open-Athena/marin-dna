# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DNA variant-effect-prediction (VEP) ``EvalTaskConfig`` constants.

The general-purpose ``EvalTaskConfig`` and ``convert_to_levanter_task_config``
remain in marin (see ``experiments.evals.task_configs``).
"""

from marin.evaluation.evaluation_config import EvalTaskConfig

# Bumped to version 1 in #179: PairwiseAccuracy + per-variant FWD/RC averaging
# replaces the old per-row AUPRC. Cached eval results from version 0 are not
# comparable, and the dataset moves from bolinas-dna/evals-traitgym_mendelian_v2_*
# to bolinas-dna/evals_mendelian_traits_harness_255 (snakemake/evals/ output).
MENDELIAN_TRAITS_255 = EvalTaskConfig(
    "mendelian_traits_255", 1, task_alias="mendelian_traits_255"
)
