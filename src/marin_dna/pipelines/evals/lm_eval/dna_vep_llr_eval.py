# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LLR-based variant effect prediction (VEP) task for lm-eval-harness.

For each row we compute ``LLR = log P(alt|ctx) - log P(ref|ctx)`` and then
``score = llr_transform(LLR)``. Rows are grouped per variant into FWD / RC /
AVG scores; PairwiseAccuracy ± SE is computed per strand and per subset via
:func:`marin_dna.pipelines.evals.metrics.compute_pairwise_metrics` — the same
helper offline ``snakemake/analysis/evals_v2/`` calls. The lm-eval headline
scalar is ``_global_/avg/pairwise_accuracy``.
"""

import logging
from collections import defaultdict
from collections.abc import Callable

import datasets
import pandas as pd
from lm_eval.api.instance import Instance
from lm_eval.api.task import Task

from marin_dna.pipelines.evals.metrics import (
    GLOBAL_SUBSET,
    MACRO_AVG_SUBSET,
    compute_pairwise_metrics,
)

_logger = logging.getLogger(__name__)

# Match `compute_pairwise_metrics`' default `n_min` and the matched-pair
# leaderboards' convention (#161/#162/#172).
_MIN_PAIRS_PER_SUBSET = 30

# Strand → wandb key segment. "+"/"-" would render as math operators in the
# Workspace search bar; slashes group the panel.
_STRAND_TAGS = {"+": "fwd", "-": "rc"}
_AVG_TAG = "avg"


class _PairwiseAccuracyAggregation:
    """Group per-row LLR scores by variant, compute per-subset PA per strand.

    For each variant: emit ``score_avg`` (mean of the per-strand scores; equals
    the single-strand value if only one strand row was present), plus
    ``score_fwd`` / ``score_rc`` for 2-strand datasets. Per-subset rows with
    ``n_pairs`` < :data:`_MIN_PAIRS_PER_SUBSET` are dropped; ``_global_`` and
    ``_macro_avg_`` are always emitted.
    """

    def __init__(
        self, results_store: dict, metric_name: str, task_name: str | None = None
    ):
        self.results_store = results_store
        self.metric_name = metric_name
        self.task_name = task_name

    def __call__(
        self,
        items: list[tuple[float, float, str | None, tuple, int, str]],
    ) -> float:
        # Reset so a repeated `__call__` (lm-eval can re-evaluate within the
        # same Task instance) doesn't leak prior keys into the tracker push.
        self.results_store.clear()

        by_variant: dict[tuple, dict] = defaultdict(dict)
        meta_by_variant: dict[tuple, dict] = {}
        expected_strands: set[str] | None = None
        for score, target, subset, variant_id, match_group, strand in items:
            assert strand in _STRAND_TAGS, (
                f"variant {variant_id} has unknown strand={strand!r}; "
                f"expected one of {sorted(_STRAND_TAGS)}"
            )
            v = by_variant[variant_id]
            assert strand not in v, (
                f"variant {variant_id} has duplicate strand={strand!r} rows"
            )
            v[strand] = float(score)
            m = meta_by_variant.get(variant_id)
            if m is None:
                meta_by_variant[variant_id] = {
                    "target": target,
                    "subset": subset,
                    "match_group": match_group,
                }
            else:
                assert (m["target"], m["subset"], m["match_group"]) == (
                    target,
                    subset,
                    match_group,
                ), (
                    f"variant {variant_id} has inconsistent meta: "
                    f"{m} vs target={target}, subset={subset}, match_group={match_group}"
                )

        # Strand set must be uniform across variants — a mixed dataset would
        # silently make `score_avg` mean different things across rows.
        for variant_id, v in by_variant.items():
            strands = frozenset(v)
            if expected_strands is None:
                expected_strands = set(strands)
            assert strands == expected_strands, (
                f"variant {variant_id} has strands={sorted(strands)}; "
                f"expected {sorted(expected_strands)}"
            )
        assert expected_strands, "no items to aggregate"
        emit_per_strand = len(expected_strands) > 1

        rows: list[dict] = []
        for variant_id, v in by_variant.items():
            meta = meta_by_variant[variant_id]
            strand_scores = [v[s] for s in expected_strands]
            row = {
                "label": int(meta["target"]),
                "subset": str(meta["subset"]),
                "match_group": int(meta["match_group"]),
                f"score_{_AVG_TAG}": sum(strand_scores) / len(strand_scores),
            }
            if emit_per_strand:
                for s in expected_strands:
                    row[f"score_{_STRAND_TAGS[s]}"] = v[s]
            rows.append(row)

        df = pd.DataFrame(rows)
        # fwd, rc, avg ordering — matches how dashboards typically list them.
        per_strand_cols = (
            [f"score_{_STRAND_TAGS[s]}" for s in _STRAND_TAGS if s in expected_strands]
            if emit_per_strand
            else []
        )
        score_columns = per_strand_cols + [f"score_{_AVG_TAG}"]

        metrics = compute_pairwise_metrics(
            dataset=df[["label", "subset", "match_group"]],
            scores=df[score_columns],
            score_columns=score_columns,
            n_min=_MIN_PAIRS_PER_SUBSET,
        )
        for row in metrics.to_dict("records"):
            subset_name = row["subset"]
            n_pairs = int(row["n_pairs"])
            if (
                subset_name not in (GLOBAL_SUBSET, MACRO_AVG_SUBSET)
                and n_pairs < _MIN_PAIRS_PER_SUBSET
            ):
                continue
            strand_tag = row["score_type"].removeprefix("score_")
            key = f"{subset_name}/{strand_tag}/{self.metric_name}"
            self.results_store[key] = float(row["value"])
            self.results_store[f"{key}_se"] = float(row["se"])

        # lm-eval only propagates the scalar return to wandb; the per-subset
        # cells we computed above only surface if we push them ourselves.
        self._push_per_subset_to_tracker()

        global_avg = metrics[
            (metrics["subset"] == GLOBAL_SUBSET)
            & (metrics["score_type"] == f"score_{_AVG_TAG}")
        ]
        assert not global_avg.empty, (
            "compute_pairwise_metrics did not emit a _global_ row for score_avg"
        )
        return float(global_avg["value"].iloc[0])

    def _push_per_subset_to_tracker(self) -> None:
        """Log ``results_store`` as wandb history at the tracker's current step.

        ``log`` (not ``log_summary``) so cells land in both the run history
        (workspace charts) and the summary panel. ``step=None`` lets the
        backend fill in its current step — required when this aggregator
        runs inside a training-loop eval (a literal ``step=0`` would trip
        levanter's "cowardly refusing to log past steps" guard).
        """
        try:
            import levanter.tracker
        except ImportError:
            return
        prefix = "lm_eval"
        if self.task_name:
            prefix = f"{prefix}/{self.task_name}"
        payload = {
            f"{prefix}/{key}": value for key, value in self.results_store.items()
        }
        try:
            levanter.tracker.log(payload, step=None)
        except Exception as exc:
            # NoopTracker / no current tracker / serialization issue: log at
            # debug so silent dashboard failures are still discoverable.
            _logger.debug("levanter.tracker.log failed: %s", exc)


METRIC_REGISTRY: dict[str, dict] = {
    "pairwise_accuracy": {
        "aggregation_cls": _PairwiseAccuracyAggregation,
        "higher_is_better": True,
    },
}


# The aggregation averages transformed scores per variant, not raw LLR. For
# transforms that commute with averaging (``identity``, ``negate``) this is
# equivalent to averaging LLR and then transforming — matching the offline
# ``marin_dna.model.runner`` path. ``abs`` does not commute and will diverge.
LLR_TRANSFORMS: dict[str, Callable[[float], float]] = {
    "identity": lambda x: x,
    "negate": lambda x: -x,
    "abs": abs,
}


class DnaVepLlrEvalTask(Task):
    """LLR eval task for variant effect prediction with per-strand + AVG aggregation.

    Parameterized by dataset and metrics via YAML config.

    Dataset rows must have ``[chrom, pos, ref, alt, context, ref_completion,
    alt_completion, target, match_group]``. Optional: ``subset`` (str) — metrics
    computed per distinct value plus ``_global_`` and ``_macro_avg_``.
    Optional: ``strand`` (str, ``"+"`` or ``"-"``) — when present the same variant
    appears once per strand (see
    :func:`marin_dna.pipelines.evals.materialize.materialize_sequences`).

    The lm-eval headline scalar is ``_global_/avg/{metric}``.

    YAML config fields:
        dataset_path: HuggingFace dataset path
        dataset_name: HuggingFace dataset config name (optional)
        test_split: dataset split to evaluate on
        metrics: list of metric names from ``METRIC_REGISTRY``
        llr_transform: identity | negate | abs (default: identity)
    """

    VERSION = 1
    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(
        self, data_dir=None, cache_dir=None, download_mode=None, config=None
    ) -> None:
        # Task.__init__ calls self.download() before setting self._config, and
        # wraps `config` via TaskConfig({**config}) which doesn't populate
        # fields correctly. Read what we need straight from the dict.
        cfg = config or {}
        self._task_name = cfg.get("task")
        self.DATASET_PATH = cfg.get("dataset_path") or self.DATASET_PATH
        self.DATASET_NAME = cfg.get("dataset_name") or self.DATASET_NAME
        self._metrics = cfg.get("metrics") or ["pairwise_accuracy"]
        self._test_split = cfg.get("test_split") or "test"
        unknown = [m for m in self._metrics if m not in METRIC_REGISTRY]
        if unknown:
            raise ValueError(
                f"Unknown metrics {unknown}. Must be one of {list(METRIC_REGISTRY)}"
            )

        transform_name = cfg.get("llr_transform") or "identity"
        if transform_name not in LLR_TRANSFORMS:
            raise ValueError(
                f"Unknown llr_transform: {transform_name}. Must be one of {list(LLR_TRANSFORMS.keys())}"
            )
        self._llr_transform = LLR_TRANSFORMS[transform_name]
        self._subset_results: dict[str, float] = {}

        super().__init__(
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
            config=config,
        )

    @property
    def task_name(self) -> str:
        # Required by lm-eval's get_subtask_list (only defined on
        # ConfigurableTask by default; we subclass plain Task).
        return self._task_name

    def download(self, data_dir=None, cache_dir=None, download_mode=None) -> None:
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
        )

    def has_training_docs(self) -> bool:
        return False

    def has_validation_docs(self) -> bool:
        return False

    def has_test_docs(self) -> bool:
        return True

    def test_docs(self):
        return self.dataset[self._test_split]

    def doc_to_text(self, doc) -> str:
        return doc["context"]

    def doc_to_target(self, doc):
        raise NotImplementedError(
            "DnaVepLlrEvalTask overrides construct_requests; use with num_fewshot=0."
        )

    def construct_requests(self, doc, ctx, **kwargs):
        # Drop chat-template kwargs `build_all_requests` passes that `Instance`
        # doesn't accept.
        metadata = kwargs.get("metadata", (None, None, None))
        return [
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(ctx, doc["ref_completion"]),
                idx=0,
                metadata=metadata,
            ),
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(ctx, doc["alt_completion"]),
                idx=1,
                metadata=metadata,
            ),
        ]

    def process_results(self, doc, results):
        log_prob_ref = results[0][0]
        log_prob_alt = results[1][0]
        llr = log_prob_alt - log_prob_ref
        score = self._llr_transform(llr)
        # Defensive casts: HF can return numpy scalars in dataset rows; the
        # variant_id tuple must be hashable for per-variant collapse.
        variant_id = (
            str(doc["chrom"]),
            int(doc["pos"]),
            str(doc["ref"]),
            str(doc["alt"]),
        )
        # 1-row-per-variant datasets predate the strand column; default to "+".
        strand = str(doc.get("strand", "+"))
        return {
            metric: (
                score,
                doc["target"],
                doc.get("subset"),
                variant_id,
                int(doc["match_group"]),
                strand,
            )
            for metric in self._metrics
        }

    def aggregation(self):
        return {
            metric: METRIC_REGISTRY[metric]["aggregation_cls"](
                results_store=self._subset_results,
                metric_name=metric,
                task_name=self._task_name,
            )
            for metric in self._metrics
        }

    def higher_is_better(self):
        return {
            metric: METRIC_REGISTRY[metric]["higher_is_better"]
            for metric in self._metrics
        }
