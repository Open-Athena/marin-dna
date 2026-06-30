# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LLR-based variant effect prediction (VEP) task for lm-eval-harness.

For each (variant, strand) row we compute the raw ``LLR = log P(alt|ctx) -
log P(ref|ctx)``. Rows are grouped per variant; the raw LLR is averaged across
the FWD/RC strand rows and *then* transformed (``score = llr_transform(mean
LLR)`` — averaging raw LLR before the transform, matching the offline
``evals_v2`` ``{protocol}_avg`` semantics). The **point** AUPRC is then computed
per strand and per subset via
:func:`marin_dna.pipelines.evals.metrics.compute_auprc_metrics` (called with
``n_bootstrap=0``) — the same helper offline ``snakemake/analysis/evals_v2/``
calls, so the point values match the offline parquet. The cluster-bootstrap SE
is **not** computed here: this in-training metric tracks the AUPRC trend, and the
authoritative SE lives in the offline ``evals_v2`` leaderboard (skipping the
per-eval-step bootstrap keeps the eval cheap). The lm-eval headline scalar is
``_global_/avg/auprc``.
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
    compute_auprc_metrics,
)

_logger = logging.getLogger(__name__)

# Match `compute_auprc_metrics`' default `n_min` and the matched-pair
# leaderboards' convention (#161/#162/#172): the minimum number of
# ``match_group``s a subset needs for its per-subset cell to be reported and to
# qualify for the ``_macro_avg_`` average.
_MIN_GROUPS_PER_SUBSET = 30

# Strand → wandb key segment. "+"/"-" would render as math operators in the
# Workspace search bar; slashes group the panel.
_STRAND_TAGS = {"+": "fwd", "-": "rc"}
_AVG_TAG = "avg"


def _collapse_variants(
    items: list[tuple[float, float, str | None, tuple, int, str]],
    llr_transform: Callable[[float], float],
) -> tuple[pd.DataFrame, list[str]]:
    """Collapse per-(variant, strand) raw-LLR rows into one row per variant.

    Each item is ``(llr, target, subset, variant_id, match_group, strand)`` with
    the **raw** LLR (no transform applied yet). For each variant we average the
    raw LLR across its strand rows and *then* apply ``llr_transform`` — matching
    the offline ``evals_v2`` ``{protocol}_avg`` semantics (average raw LLR first,
    then transform; so ``abs_llr_avg = |(llr_fwd + llr_rc)/2|``). For 2-strand
    datasets we also emit per-strand ``score_fwd`` / ``score_rc``, each the
    transform of that strand's raw LLR.

    All the per-variant consistency invariants are asserted here (fail loud near
    the bug rather than feeding a silently-wrong frame into the metric).

    Returns ``(df, score_columns)`` where ``df`` has columns
    ``[label, subset, match_group] + score_columns`` and ``score_columns`` is
    ``["score_fwd", "score_rc", "score_avg"]`` for 2-strand datasets, or just
    ``["score_avg"]`` for 1-strand datasets (where avg == the single strand).
    """
    by_variant: dict[tuple, dict] = defaultdict(dict)
    meta_by_variant: dict[tuple, dict] = {}
    for llr, target, subset, variant_id, match_group, strand in items:
        assert strand in _STRAND_TAGS, (
            f"variant {variant_id} has unknown strand={strand!r}; "
            f"expected one of {sorted(_STRAND_TAGS)}"
        )
        v = by_variant[variant_id]
        assert strand not in v, (
            f"variant {variant_id} has duplicate strand={strand!r} rows"
        )
        v[strand] = float(llr)
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
    expected_strands: set[str] | None = None
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
        strand_llrs = [v[s] for s in expected_strands]
        # Average RAW LLR across strands, then transform (offline `_avg` order).
        llr_avg = sum(strand_llrs) / len(strand_llrs)
        row = {
            "label": int(meta["target"]),
            "subset": str(meta["subset"]),
            "match_group": int(meta["match_group"]),
            f"score_{_AVG_TAG}": float(llr_transform(llr_avg)),
        }
        if emit_per_strand:
            for s in expected_strands:
                row[f"score_{_STRAND_TAGS[s]}"] = float(llr_transform(v[s]))
        rows.append(row)

    df = pd.DataFrame(rows)
    # fwd, rc, avg ordering — matches how dashboards typically list them.
    per_strand_cols = (
        [f"score_{_STRAND_TAGS[s]}" for s in _STRAND_TAGS if s in expected_strands]
        if emit_per_strand
        else []
    )
    score_columns = per_strand_cols + [f"score_{_AVG_TAG}"]
    return df, score_columns


class _AuprcAggregation:
    """Group per-row raw LLR by variant, compute per-subset AUPRC per strand.

    For each variant: average raw LLR across the strand rows (FWD + RC) then
    transform to ``score_avg``; for 2-strand datasets also emit ``score_fwd`` /
    ``score_rc``. Point AUPRC is then computed per subset via
    :func:`compute_auprc_metrics` (``n_bootstrap=0``; no SE — the offline
    ``evals_v2`` parquet is the authoritative SE source). Per-subset rows with ``n_groups`` <
    :data:`_MIN_GROUPS_PER_SUBSET` are dropped from the tracker push; ``_global_``
    and ``_macro_avg_`` are always emitted. The headline scalar returned to
    lm-eval is the ``_global_`` / ``score_avg`` AUPRC.
    """

    def __init__(
        self,
        results_store: dict,
        metric_name: str,
        llr_transform: Callable[[float], float],
        task_name: str | None = None,
    ):
        self.results_store = results_store
        self.metric_name = metric_name
        self.llr_transform = llr_transform
        self.task_name = task_name

    def __call__(
        self,
        items: list[tuple[float, float, str | None, tuple, int, str]],
    ) -> float:
        # Reset so a repeated `__call__` (lm-eval can re-evaluate within the
        # same Task instance) doesn't leak prior keys into the tracker push.
        self.results_store.clear()

        df, score_columns = _collapse_variants(items, self.llr_transform)

        # n_bootstrap=0 → point AUPRC only (no SE): the in-training metric tracks
        # the AUPRC trend; the cluster-bootstrap SE is computed offline in
        # evals_v2 (the authoritative leaderboard). Skipping the per-eval-step
        # bootstrap keeps this eval cheap. The point values still match the
        # offline parquet (validated in #225).
        metrics = compute_auprc_metrics(
            dataset=df[["label", "subset", "match_group"]],
            scores=df[score_columns],
            score_columns=score_columns,
            n_bootstrap=0,
            n_min=_MIN_GROUPS_PER_SUBSET,
        )
        # Two distinct n_min gates (intentional, not redundant): the `n_min` arg
        # to compute_auprc_metrics gates which subsets enter `_macro_avg_`; this
        # loop gate decides which per-subset cells get pushed to the tracker —
        # WandB has no display-time threshold, so we drop tiny subsets here
        # (offline keeps every per-subset row in the parquet and filters at the
        # dashboard). `_global_` / `_macro_avg_` are always pushed.
        for row in metrics.to_dict("records"):
            subset_name = row["subset"]
            n_groups = int(row["n_groups"])
            if (
                subset_name not in (GLOBAL_SUBSET, MACRO_AVG_SUBSET)
                and n_groups < _MIN_GROUPS_PER_SUBSET
            ):
                continue
            strand_tag = row["score_type"].removeprefix("score_")
            key = f"{subset_name}/{strand_tag}/{self.metric_name}"
            self.results_store[key] = float(row["value"])
            # Point-only online (se is NaN); push _se only if a finite SE exists,
            # so this still works if a caller ever re-enables the bootstrap.
            se = float(row["se"])
            if pd.notna(se):
                self.results_store[f"{key}_se"] = se

        # lm-eval only propagates the scalar return to wandb; the per-subset
        # cells we computed above only surface if we push them ourselves.
        self._push_per_subset_to_tracker()

        global_avg = metrics[
            (metrics["subset"] == GLOBAL_SUBSET)
            & (metrics["score_type"] == f"score_{_AVG_TAG}")
        ]
        assert not global_avg.empty, (
            "compute_auprc_metrics did not emit a _global_ row for score_avg"
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
    "auprc": {
        "aggregation_cls": _AuprcAggregation,
        "higher_is_better": True,
    },
}


# Applied to the per-variant **averaged raw LLR** (and to each per-strand raw
# LLR) inside ``_collapse_variants`` — i.e. after averaging, matching the
# offline ``evals_v2`` ``{protocol}_avg`` semantics where
# ``abs_llr_avg = |(llr_fwd + llr_rc)/2|``. These mirror
# ``marin_dna.pipelines.evals.metrics.SCORE_PROTOCOLS`` (``negate`` ≡ ``minus_llr``,
# ``abs`` ≡ ``abs_llr``; ``identity`` is online-only) — if a protocol is
# added/changed there for the offline leaderboard, mirror it here or the
# in-training metric silently diverges from the parquet. (``negate`` =
# ``minus_llr`` is the protocol used for the mendelian dataset.)
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
    appears once per strand.

    The lm-eval headline scalar is ``_global_/avg/{metric}``.

    YAML config fields:
        dataset_path: HuggingFace dataset path
        dataset_name: HuggingFace dataset config name (optional)
        dataset_revision: HuggingFace dataset commit/revision to pin (optional
            but recommended — keeps the dataset from silently changing under
            the eval)
        test_split: dataset split to evaluate on
        metrics: list of metric names from ``METRIC_REGISTRY``
        llr_transform: identity | negate | abs (default: identity)
    """

    VERSION = 1
    DATASET_PATH = None
    DATASET_NAME = None
    DATASET_REVISION = None

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
        self.DATASET_REVISION = cfg.get("dataset_revision") or self.DATASET_REVISION
        self._metrics = cfg.get("metrics") or ["auprc"]
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
            revision=self.DATASET_REVISION,
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
        # Emit the **raw** LLR; `_collapse_variants` averages raw LLR across
        # FWD/RC per variant and applies `llr_transform` afterwards (matching
        # offline `{protocol}_avg`). Transforming here would only be equivalent
        # for transforms that commute with averaging (`negate`), not `abs`.
        llr = log_prob_alt - log_prob_ref
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
                llr,
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
                llr_transform=self._llr_transform,
                task_name=self._task_name,
            )
            for metric in self._metrics
        }

    def higher_is_better(self):
        return {
            metric: METRIC_REGISTRY[metric]["higher_is_better"]
            for metric in self._metrics
        }
