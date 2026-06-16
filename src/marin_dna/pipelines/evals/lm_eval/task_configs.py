# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DNA variant-effect-prediction (VEP) ``EvalTaskConfig`` constants.

The general-purpose ``EvalTaskConfig`` and ``convert_to_levanter_task_config``
remain in marin (see ``experiments.evals.task_configs``).
"""

from marin.evaluation.evaluation_config import EvalTaskConfig

# The ``1`` is ``num_fewshot``, not a version field: this VEP task overrides
# construct_requests and prepends no few-shot context (the #225 online-vs-offline
# parity — where offline uses no few-shot at all — confirms it's a no-op for
# scoring). #179 (ab)used it as a cache-busting bump (0 → 1) when it changed the
# metric + dataset. #225 migrated the online metric to AUPRC + per-variant FWD/RC
# averaging (was PairwiseAccuracy); the dataset is
# bolinas-dna/evals_mendelian_traits_harness_255 (snakemake/evals/ output).
MENDELIAN_TRAITS_255 = EvalTaskConfig(
    "mendelian_traits_255", 1, task_alias="mendelian_traits_255"
)
