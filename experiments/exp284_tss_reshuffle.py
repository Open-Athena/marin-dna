# Copyright The Marin Authors / Bolinas Authors
# SPDX-License-Identifier: Apache-2.0

"""Epoch-reshuffling stopgap on the TSS-region 0.25B specialist (issue #284).

A single-arm follow-up to #255 / #256 (exp255). exp255 trains 0.25B order-cohort
region specialists, each on **one** training dataset epoched many times at fixed
5K × 8192 compute. The smallest arm — ``v4_tss_region_and_utr5_order`` (the
promoter / TSS region, 1,965,838 post-RC rows → **20.8 epochs**) — is the extreme
of that regime, and the subject here.

The problem (marin#2869): levanter's data path applies a single, fixed
block-shuffle permutation and does NOT reshuffle on restart, so a single-dataset
specialist sees the identical batch sequence every epoch (~21× here). The stopgap
(Eric Czech's gist) slices that one tokenized dataset into ``NUM_SLICES`` disjoint
block-shuffled components and re-mixes them, so ``MixtureDataset``'s per-block
dataset-id permutation re-interleaves the slices fresh each epoch — batch
composition almost never repeats — via the cheap block-shuffle (no global
per-epoch reshuffle). See ``marin_dna.levanter.slice_mix``.

This experiment is exp255's ``v4_tss_region_and_utr5_order`` arm with the
slice-mix data path turned ON and **nothing else changed** — same dataset
(``bolinas-dna/zoonomia-v1-v4_tss_region_and_utr5-order``, reusing exp255's
tokenize cache), geometry, optimizer, 8192 batch, 5K steps, the 5 ``val_*``
LL-gap recipes, and the ``mendelian_traits_255`` AUPRC eval. The matched
no-reshuffle baseline is exp255's existing run for this arm (#256, no re-run).
Readout: per-step (10 HF checkpoints, ep ~2 → ~21) mendelian AUPRC on the 5′UTR /
tss_proximal subsets + the ``val_tss_pc`` LL gap, ON vs OFF — does reshuffle help,
and does the benefit grow with epoch count?

How the slice-mix is wired (see ``_run_train_with_marin_dna_imports``): the
executor graph and ``_build_data_mixture`` are byte-identical to exp255 — a
standard single-component cache-backed config. The reshuffle is a *worker-side*
transform applied after ``materialize`` resolves the tokenize-cache path and
before levanter builds the data loader: ``inject_slice_mix`` loads the one
training component (via levanter's own ``build_caches`` + ``build_token_datasets``,
which route through the DNA-patched ``dataset_for_component`` so per-token loss
weights are preserved), slices it into ``NUM_SLICES`` block-shuffled
``DirectDatasetComponent``s, keeps the validation components untouched, and sets
``shuffle=False``. ``run_levanter_train_lm`` then re-materializes (a safe no-op on
the now placeholder-free config — ``materialize`` is idempotent and treats the
live sliced datasets as inert leaves) and keeps all marin orchestration (HF
export, run id, output-path baking).

In-training validation is held fixed — the same **family-projection** recipes
exp255 uses:
  * 5 region-specific ``zoonomia-v1-val_*`` recipes (LL gap) from PR #171,
    each tokenized functional + nonfunctional (incl. the gene-centric ``val_tss_pc``).
  * ``mendelian_traits_255`` lm-eval task (PR #186) — per consequence subset +
    Global + Macro Avg + FWD/RC strand averaging. Online metric is AUPRC +
    per-variant FWD/RC averaging (#226); in-training headline ``_global_/avg/auprc``,
    offline evals_v2 leaderboard ``_macro_avg_/avg/auprc`` (#161).

HF checkpoint saving is enabled by setting ``hf_save_steps`` to match the eval
cadence (every 500 steps). ``hf_save_path`` is auto-set by marin's
``_update_config_to_use_out_path`` to ``<output_path>/hf``. ``hf_save_dtype``
stays ``None``, preserving the param dtype (fp32 under our
``jmp.get_policy("p=f32,c=bfloat16")`` policy) — load at bf16 via
``torch_dtype=torch.bfloat16``.

Hardware: ``v6e-4`` + ``PDP=1024`` is the cost-optimal target (script defaults;
PR #250). ``PDP`` is the per-chip microbatch: on the 4-chip v6e-4 slice the full
per-chip batch is 8192/4 = 2048, which OOMs v6e's ~31 GB HBM, so ``PDP=1024``
grad-accumulates 2 microbatches of 1024/chip (~24 GB, fits) with the **effective
batch unchanged at 8192**, and is a no-op on any 8-chip slice, so a re-route
needs only ``-e TPU_TYPE``. v6e is preemptible → babysit/relaunch; resume is free
(the slice-mix is rebuilt deterministically from ``SLICE_MIX_SEED`` on the
worker, and neither TPU type nor PDP is in the output-config hash).

Download/tokenize pattern: option 1 (``HfTokenizeConfig(id=<hf-name>)``). The
training partition + 5 val recipes reuse exp255/exp232's tokenizations (identical
names → cache hit), so launch in the region where those caches live (exp255's
TSS arm ran in ``europe-west4``) to avoid a re-tokenize.

Launch from a CPU box with an iris tunnel open (see ``experiments/README.md``):

    uv run iris --cluster=marin job run \\
        --no-wait --user gonzalo --job-name exp284-tss-reshuffle-v6e4 \\
        --cpu 1 --memory 2g --extra marin --region europe-west4 \\
        -e WANDB_API_KEY "$(grep -A2 api.wandb.ai ~/.netrc | grep password | awk '{print $2}')" \\
        -e HF_HUB_DOWNLOAD_TIMEOUT 120 -e UV_LOCK_TIMEOUT 7200 \\
        -e TPU_TYPE v6e-4 -e PDP 1024 -e TPU_RAM 300g \\
        -- python experiments/exp284_tss_reshuffle.py

Optional knobs: ``-e SLICE_MIX_NUM_SLICES 8`` (default), ``-e SLICE_MIX_SEED 0``.

Reference templates:
  * Direct parent — ``experiments/exp255_per_region_order.py`` (this is that
    script's TSS arm with the slice-mix worker-side data path; geometry /
    optimizer / eval / dataset unchanged).
  * Tracking issue — #284 (epoch reshuffling); upstream marin#2869; baseline #255 / #256.
  * Slice-mix helper — ``src/marin_dna/levanter/slice_mix.py``.
  * Eval-task wiring — ``experiments/parity/exp179_eval_only.py``.
"""

import logging
import os
from datetime import timedelta
from functools import lru_cache

import jmp
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.evaluation.evaluation_config import convert_to_levanter_task_config
from marin.execution import (
    ExecutorStep,
    ensure_versioned,
    executor_main,
    this_output_path,
)
from marin.execution.remote import remote
from marin.processing.tokenize import lm_mixture_data_config, tokenize
from marin.processing.tokenize.tokenize import HfTokenizeConfig
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

from marin_dna.levanter.defaults import dna_effective_seq_len
from marin_dna.levanter.formats import DNALmDatasetFormat
from marin_dna.pipelines.evals.lm_eval.task_configs import MENDELIAN_TRAITS_255

# =============================================================================
# Constants — issue #284 epoch-reshuffling TSS arm (clone of exp255).
# =============================================================================

EXP_ISSUE = 284
VERSION = "v0.1"
TOKENIZER = "bolinas-dna/tokenizer-char-bos"
DNA_BASE_SEQ_LEN = 255  # bp (256 - 1 for BOS)

# The single TSS-region order-cohort partition — exp255's smallest, most-epoched
# arm (1,965,838 post-RC rows → 20.8 epochs at 5K×8192; #230 / #233). Reusing the
# *same* dataset + tokenize cache as exp255 keeps the no-reshuffle baseline (#256)
# a byte-identical match — the only difference between the arms is the worker-side
# slice-mix data path (see ``_run_train_with_marin_dna_imports``). The arm KEY is
# kept identical to exp255's so the training tokenize cache hits exp255's; the
# wandb run name + checkpoint path get a distinct ``-reshuffle`` suffix in
# ``_build_train_step``. SWEEP_DATASETS still selects the arm (one entry here).
TRAIN_DATASETS: dict[str, str] = {
    "v4_tss_region_and_utr5_order": "bolinas-dna/zoonomia-v1-v4_tss_region_and_utr5-order",
}

# Five region-specific validation recipes from PR #171, tokenized functional +
# nonfunctional for the LL-gap signal. Drops val_utr5 / val_promoter (both
# subsumed by the gene-centric ±255 bp val_tss_pc); keeps val_enhancer
# (chromatin-side, not subsumed). These are the FAMILY-projection val sets,
# byte-identical to exp187/exp232 — the eval is held fixed across cohorts, and
# the identical names reuse exp187/exp232's tokenization caches (no re-tokenize).
VAL_DATASETS: tuple[tuple[str, str], ...] = (
    ("val_cds", "bolinas-dna/zoonomia-v1-val_cds"),
    ("val_utr3", "bolinas-dna/zoonomia-v1-val_utr3"),
    ("val_ncrna", "bolinas-dna/zoonomia-v1-val_ncrna"),
    ("val_enhancer", "bolinas-dna/zoonomia-v1-val_enhancer"),
    ("val_tss_pc", "bolinas-dna/zoonomia-v1-val_tss_pc"),
)

# Training masks lowercase positions to 1% loss weight. Zoonomia datasets use
# ``sequence`` as the text field (vs the older ``seq`` default for genomes-v5).
TRAIN_FORMAT = DNALmDatasetFormat(text_key="sequence", lowercase_weight=0.01)

# Validation tokenization specs — the two terms of the LL gap (matches exp160 /
# exp166; deliberately skips the "default" matched-to-training variant).
VAL_SPECS: tuple[tuple[str, DNALmDatasetFormat], ...] = (
    ("functional", DNALmDatasetFormat(uppercase_weight=1.0, lowercase_weight=0.0)),
    ("nonfunctional", DNALmDatasetFormat(uppercase_weight=0.0, lowercase_weight=1.0)),
)

# Qwen3-0.25B head_dim=128 (1152 / 9 = 128). The ``~255M`` rung of exp109's
# calibrated scaling ladder (1152→255M, 1408→476M, 1920→1.12B); the 1B that
# exp187 ran sits two rungs up at hidden=1920. Geometry from marin's
# ``CompletedAdamHHeuristic._build_model_config(1152)``: num_layers=12,
# intermediate=hidden×4, n_heads=hidden//128. Identical to exp232.
HIDDEN_DIM = 1152
INTERMEDIATE_DIM = HIDDEN_DIM * 4  # 4608
NUM_HEADS = HIDDEN_DIM // 128  # 9
NUM_LAYERS = 12

BATCH_SIZE = 8192
# TPU type is env-overridable (``TPU_TYPE``, comma-separated for fallbacks). The
# agreed target for this experiment is ``v6e-4`` (default): 4 chips at ~2× the MFU
# of v6e-8 → ~half the chip-hours for similar wall-clock, and v6e is abundant in
# ``us-east5-b`` (data-local to the us-east5 cache) while v5p stays capacity-
# starved (PR #250). Re-route with ``-e TPU_TYPE v6e-8`` / ``v5p-8`` / ``v4-8``
# (all 8-chip; launch v4-8 with ``--region us-central2``). Resume is free across
# re-routes — neither TPU type nor PDP is in the output-config hash (PR #250).
TPU_TYPES: tuple[str, ...] = tuple(
    t.strip() for t in os.getenv("TPU_TYPE", "v6e-4").split(",") if t.strip()
)

# Per-device microbatch (`per_device_parallelism`), env-overridable via ``PDP``;
# default 1024. On the 4-chip v6e-4 slice the full per-chip batch is 8192/4 =
# 2048, which OOMs v6e's ~31 GB HBM (~43 GB activations); ``PDP=1024`` grad-
# accumulates 2 microbatches of 1024/chip (~24 GB, fits) with the **effective
# batch unchanged at 8192** (train_batch_size fixed → mathematically identical
# step). PDP=1024 is a no-op on any 8-chip slice (8192/8 = 1024/chip already =
# one microbatch), so it's safe on every 4-8-chip pool — a re-route needs only
# ``-e TPU_TYPE``. levanter already runs ``gradient_checkpointing=True``; the OOM
# is activations, not weights, so the microbatch is the lever.
PER_DEVICE_PARALLELISM: int = int(os.getenv("PDP", "1024"))

# Host RAM request for the TPU pod, env-overridable via ``TPU_RAM``. The default
# ``300g`` is inherited from exp166's 1B/4B but is oversized for 0.25B — host RAM
# only buffers the data loader (the model + optimizer states live in TPU HBM), so
# a few tens of GB suffice. Under a busy cluster a 300g ask can fail to schedule
# ("Scheduler: Insufficient memory (need 300.0GB, ...)"), so override low (e.g.
# ``-e TPU_RAM 40g``) to fit available nodes. NOTE: ``resources`` is **not** in
# the marin executor's output-path hash (only ``versioned()`` values + step deps
# are — see ``collect_dependencies_and_version``), so changing this is
# resume-safe — a relaunched arm still loads its existing checkpoint.
TPU_RAM: str = os.getenv("TPU_RAM", "300g")

# 5K steps × 8192 batch × 256 tokens/seq ≈ 10.5B tokens — identical compute to
# exp255's TSS arm (#256), the matched no-reshuffle baseline. The TSS -order set
# is 1,965,838 post-RC rows, so 5K×8192 is ~20.8 epochs — the heavy-epoching
# regime where epoch reshuffling should matter most (issue #284).
NUM_TRAIN_STEPS = 5_000

# --- Epoch-reshuffling (slice-mix) knobs (issue #284 / marin#2869) ---
# Number of disjoint slices the single training dataset is split into. The mixture
# re-interleaves these K block-shuffled slices with a fresh per-block permutation
# each epoch, so batch composition almost never repeats across epochs (Eric's
# stopgap; see ``marin_dna.levanter.slice_mix``). K=8 gives rich cross-epoch
# interleaving for the ~1.97M-row TSS arm; env-overridable for tuning.
NUM_SLICES: int = int(os.getenv("SLICE_MIX_NUM_SLICES", "8"))
# Seed for the per-slice block-shuffle permutations. Resume-stable: the worker
# rebuilds identical slices on restart, and the mixture's own per-epoch
# permutation is keyed off the (resume-stable) trainer seed.
SLICE_MIX_SEED: int = int(os.getenv("SLICE_MIX_SEED", "0"))

# Optimizer hparams from marin PR #5530's exp166 transferred-hparam table
# (resolved by ``CompletedAdamHHeuristic`` for B=8192, T=5.73e10 — exp166's
# full-epoch horizon). Inherited verbatim from exp232 / exp187.
#
# **CAVEAT** — our T = 5_000 × 8192 × 256 ≈ 1.05e10, ~5× smaller than exp166's
# horizon. The DNA-calibrated heuristic would resolve slightly different lr /
# beta2 / epsilon for our regime. We use exp166's values here as a deliberate
# starting point; the AdamH heuristic is size-independent (keyed on (B, T), both
# unchanged at 0.25B), so the order arms share exp232's / exp166's optimizer
# config — only the training species cohort differs. If reviewers want regime-
# correct hparams, re-resolve via marin's CompletedAdamHHeuristic for
# (B=8192, T=1.05e10) and pin the resolved values here. Tracked as Q1 in #232.
LEARNING_RATE = 0.00430097
BETA1 = 0.66756
BETA2 = 0.952222
EPSILON = 6.77142e-15
MAX_GRAD_NORM = 0.995188
Z_LOSS_WEIGHT = 4.312883184368223e-06
INITIALIZER_RANGE = 0.02
WEIGHT_DECAY = 0.1  # exp160 / exp135 default; the heuristic doesn't tune WD
WARMUP_FRACTION = 0.1
DECAY_FRACTION = 0.2
LR_SCHEDULE = "linear"
MIN_LR_RATIO = 0.0

# Eval + checkpoint cadence. EVALS_PER_RUN = 10 → first eval at step 500. We
# match HF-checkpoint cadence to eval cadence so every in-training eval has a
# reloadable HF artifact for offline analysis (e.g. re-running mendelian eval
# against a specific step, or the matched-epoch read vs the family baseline).
EVALS_PER_RUN = 10
CHECKPOINTS_PER_RUN = 10
# Levanter resume-checkpoint cadence. Dropped 1h -> 10min: under frequent v6e
# preemption (busy-hours windows can be ~15-20 min, much of it startup), a 1h
# interval meant a relaunched arm got preempted before saving any progress and
# kept resuming the same step (ccre_order stuck at 2500). 10min lets a ~25-min
# window bank forward progress. Resume-safe — the checkpointer config is not in
# the marin output-path hash (only ``versioned()`` values + deps are). HF
# checkpoints stay every 500 steps (``hf_save_steps``) for analysis parity w/ cds.
CHECKPOINT_TIME_INTERVAL = timedelta(minutes=10)

WANDB_PROJECT = "marin"
WANDB_GROUP = f"dna-exp{EXP_ISSUE}-{VERSION}"

_EXPECTED_VOCAB_SIZE_WARNING = (
    f"Tokenizer {TOKENIZER!r} not found in _KNOWN_VOCAB_SIZES"
)
logging.getLogger("marin.processing.tokenize.data_configs").addFilter(
    lambda record: _EXPECTED_VOCAB_SIZE_WARNING not in record.getMessage()
)


# =============================================================================
# Environment overrides
# =============================================================================


def _selected_datasets() -> dict[str, str]:
    """Return the subset of TRAIN_DATASETS named in SWEEP_DATASETS (or all)."""
    raw = os.getenv("SWEEP_DATASETS")
    if not raw:
        return dict(TRAIN_DATASETS)
    requested = tuple(s.strip() for s in raw.split(","))
    invalid = [n for n in requested if n not in TRAIN_DATASETS]
    if invalid:
        raise ValueError(
            f"Invalid SWEEP_DATASETS {invalid}; available: {sorted(TRAIN_DATASETS)}"
        )
    return {n: TRAIN_DATASETS[n] for n in requested}


# =============================================================================
# Builders
# =============================================================================


@lru_cache(maxsize=1)
def _model_seq_len() -> int:
    """Model context size = base DNA seq len + special tokens (BOS)."""
    return dna_effective_seq_len(DNA_BASE_SEQ_LEN, TOKENIZER)


# Inherited verbatim from exp160 — small orchestrator footprint for the
# tokenize step (heavy work runs on zephyr workers).
_TOKENIZE_RESOURCES = ResourceConfig.with_cpu(cpu=1, ram="12g", disk="10g")


def _tokenize(
    name: str, dataset: str, dataset_format: DNALmDatasetFormat
) -> ExecutorStep:
    """Tokenize one HF dataset (option-1 path, same as exp160/exp166)."""
    config = HfTokenizeConfig(
        id=dataset,
        cache_path=this_output_path(),
        tokenizer=ensure_versioned(TOKENIZER),
        format=dataset_format,
    )
    return ExecutorStep(
        name=os.path.join("tokenized", name),
        description=f"Tokenize {dataset!r} with the {TOKENIZER} tokenizer.",
        fn=remote(
            tokenize,
            resources=_TOKENIZE_RESOURCES,
            # bolinas-dna doesn't define a ``cpu`` extra; route to ``marin``
            # which transitively installs marin + jax + jmp + tokenizers.
            pip_dependency_groups=["marin"],
            env_vars={
                "TRANSFORMERS_NO_TORCH": "1",
                "TRANSFORMERS_NO_TORCHVISION": "1",
                "USE_TORCH": "0",
                "TORCH_DISABLE_GLOBAL_DEPS": "1",
                # huggingface_hub's default read_timeout=10s is too short for
                # bigger parquet manifests; bump to 120s.
                "HF_HUB_DOWNLOAD_TIMEOUT": "120",
                # Many concurrent zephyr workers share a uv cache; first build
                # serializes lm-eval (URL dep). Default 300s isn't enough.
                "UV_LOCK_TIMEOUT": "7200",
            },
        ),
        config=config,
    )


def _build_data_mixture(strategy: str, dataset: str) -> LmDataConfig:
    """One training component + cross-product of validation recipes × specs.

    The training component is keyed on ``strategy`` (which carries the ``_order``
    cohort suffix), so its tokenize cache name is distinct from exp232's family
    ``v4_<region>`` cache; the val components keep the exp187/exp232 names so
    their caches are reused (the eval is held fixed across cohorts).
    """
    components: dict[str, ExecutorStep] = {
        strategy: _tokenize(
            f"marin_dna-zoonomia-v1-{strategy}-char-bos", dataset, TRAIN_FORMAT
        ),
    }
    for region, val_dataset in VAL_DATASETS:
        for suffix, fmt in VAL_SPECS:
            key = f"{region}_{suffix}"
            components[key] = _tokenize(
                f"marin_dna-zoonomia-v1-{key}-char-bos", val_dataset, fmt
            )
    return lm_mixture_data_config(
        components=components,
        weights={strategy: 1.0},
    )


def _build_model_config() -> Qwen3Config:
    """Qwen3-0.25B head_dim=128 (hidden=1152, layers=12, heads=9).

    The ~255M rung of exp109's calibrated ladder; geometry from marin's
    ``CompletedAdamHHeuristic._build_model_config(1152)`` (num_layers=12,
    intermediate=4608, n_heads=9). Identical to exp232.
    """
    return Qwen3Config(
        hidden_dim=HIDDEN_DIM,
        intermediate_dim=INTERMEDIATE_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_HEADS,
        max_seq_len=_model_seq_len(),
        rope=Llama3RotaryEmbeddingsConfig(),
        initializer_range=INITIALIZER_RANGE,
    )


def _build_optimizer() -> AdamConfig:
    return AdamConfig(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        beta1=BETA1,
        beta2=BETA2,
        epsilon=EPSILON,
        max_grad_norm=MAX_GRAD_NORM,
        warmup=WARMUP_FRACTION,
        decay=DECAY_FRACTION,
        lr_schedule=LR_SCHEDULE,
        min_lr_ratio=MIN_LR_RATIO,
    )


def _eval_harness_config() -> LmEvalHarnessConfig:
    """Wire the mendelian eval (AUPRC + Global + Macro Avg + FWD/RC strand
    averaging; online metric migrated from PA in #226). Same shape as exp179_eval_only.py."""
    return LmEvalHarnessConfig(
        task_spec=convert_to_levanter_task_config([MENDELIAN_TRAITS_255]),
    )


def _checkpointer(num_train_steps: int) -> CheckpointerConfig:
    return CheckpointerConfig(
        save_interval=CHECKPOINT_TIME_INTERVAL,
        keep=[dict(every=max(1, num_train_steps // CHECKPOINTS_PER_RUN))],
    )


def _hf_save_steps(num_train_steps: int) -> int:
    """Match HF-checkpoint cadence to eval cadence — every eval has a paired
    reloadable HF artifact under ``<this_output_path>/hf/step-<N>/``."""
    return max(1, num_train_steps // EVALS_PER_RUN)


def _run_train_with_marin_dna_imports(pod_config: TrainLmOnPodConfig) -> None:
    """Worker entrypoint: bake in the marin_dna imports + inject the slice-mix.

    Iris's ``Entrypoint.from_callable`` cloudpickles ``__main__`` by-value,
    capturing function bytecode but NOT re-importing modules on the worker.
    Module-top imports in this script (``marin_dna.levanter.formats``,
    ``marin_dna.pipelines.evals.lm_eval.task_configs``) never fire on the TPU
    pod, so the side-effecting registrations (``@register_subclass("dna")`` for
    ``DNALmDatasetFormat`` AND the monkey-patch of ``dataset_for_component``; the
    lm-eval task-manager patch) never install. Trainer init then fails with
    ``ValueError: Unknown format DNALmDatasetFormat(...)`` when deserializing
    the data config (exp255 smoke5 confirmed this). Exp179_eval_only.py:91-96
    has the same pattern. Fix: bake the imports into the worker function body.

    Epoch-reshuffling injection (issue #284). ``_build_data_mixture`` produced a
    standard exp255-style single-component cache-backed config (so the executor
    graph + provenance stay unchanged). Here, on the worker, we turn the
    reshuffle ON: ``materialize`` first resolves the tokenize-cache placeholder
    to a concrete path (the cache already exists — built by the upstream tokenize
    step), then ``inject_slice_mix`` loads that one training component and
    rewrites it into ``NUM_SLICES`` block-shuffled ``DirectDatasetComponent``s
    that ``MixtureDataset`` re-interleaves fresh each epoch. ``materialize`` must
    run before the injection (the cache must exist to be sliced) and after the
    formats import (the slice build routes through the DNA-patched
    ``dataset_for_component``, preserving per-token loss weights).
    ``run_levanter_train_lm`` then runs normally — its own internal
    ``materialize`` is a safe no-op on the now placeholder-free config
    (idempotent; the live sliced datasets are inert leaves to the config walk),
    and it keeps the full marin orchestration (HF export, run id, path baking).
    """
    import marin_dna.levanter.formats  # noqa: F401  # registers "dna" + patches dataset_for_component
    import marin_dna.pipelines.evals.lm_eval  # noqa: F401  # patches lm_eval TaskManager
    from marin.execution.executor import materialize

    from marin_dna.levanter.slice_mix import inject_slice_mix

    pod_config = materialize(pod_config)
    pod_config = inject_slice_mix(
        pod_config, num_slices=NUM_SLICES, seed=SLICE_MIX_SEED
    )
    run_levanter_train_lm(pod_config)


def _train_remote_env_vars() -> dict[str, str]:
    """Env vars baked into the train ``remote()`` call.

    Iris workers don't inherit ``-e`` flags from the launcher's ``iris job run``
    command — the orchestrator passes them to its own process tree, but child
    tasks (the TPU pod running ``run_levanter_train_lm``) get a fresh env.
    Per ``experiments/README.md:59-67`` and the working pattern in
    ``experiments/parity/exp179_eval_only.py:124-130``, capture the key vars
    from the launcher env here and bake them into the remote spec.

    Missing ``WANDB_API_KEY`` here was the cause of ``/gonzalo/exp187-smoke2``
    failing with ``wandb.UsageError: No API key configured`` after ~45 min of
    successful tokenize work.
    """
    env: dict[str, str] = {
        "HF_HUB_DOWNLOAD_TIMEOUT": "120",
        "UV_LOCK_TIMEOUT": "7200",
    }
    # Reject empty WANDB_API_KEY — the launch command's shell ordering can
    # silently substitute an empty value (``VAR=$(...) cmd -e VAR "$VAR"``
    # resolves ``$VAR`` *before* the prefix assignment). Wandb's UsageError
    # in that case is delayed until trainer init on the TPU pod, after ~45
    # min of tokenize work has succeeded — costly. Fail loud at launch time
    # instead.
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if wandb_key:
        env["WANDB_API_KEY"] = wandb_key
    else:
        raise RuntimeError(
            "WANDB_API_KEY is not set in the launcher's environment "
            "(or is empty). The recommended launch pattern is to use "
            "inline ``$(...)`` substitution in ``-e WANDB_API_KEY ...`` "
            "per experiments/README.md — not a ``VAR=... cmd`` prefix, "
            "which has a shell-ordering bug."
        )
    return env


def _build_train_step(strategy: str, dataset: str) -> ExecutorStep:
    steps_per_eval = max(1, NUM_TRAIN_STEPS // EVALS_PER_RUN)
    run_name = f"dna-exp{EXP_ISSUE}-zoonomia-v1-0p25b-{strategy}-reshuffle-{VERSION}"
    tags = (
        "dna",
        "marin_dna",
        f"exp{EXP_ISSUE}",
        "per-region",
        "zoonomia_v1_v4",
        "cohort=order",
        "species=19_order",
        "epoch_reshuffle=slice_mix",
        f"num_slices={NUM_SLICES}",
        VERSION,
        f"region={strategy}",
        "scale=0.25b",
        f"bs={BATCH_SIZE}",
        f"steps={NUM_TRAIN_STEPS}",
    )

    # iris's Entrypoint.from_callable cloudpickles __main__ by-value, so a
    # module-top import of marin_dna.pipelines.evals.lm_eval doesn't fire on the
    # TPU pod. Per the eval-task wiring pattern in exp179_eval_only.py:91-96,
    # the in-function noqa import on the train worker happens inside
    # marin.training.training.run_levanter_train_lm — see that function for the
    # lm-eval task-manager monkeypatch's installation. We don't need to do
    # anything special here; the convert_to_levanter_task_config([MENDELIAN_TRAITS_255])
    # call serializes the task name + class path, which run_levanter_train_lm
    # resolves on the worker after it imports our task module.

    inner = TrainLmConfig(
        data=_build_data_mixture(strategy, dataset),
        model=_build_model_config(),
        train_seq_len=_model_seq_len(),
        z_loss_weight=Z_LOSS_WEIGHT,
        optimizer=_build_optimizer(),
        eval_harness=_eval_harness_config(),
        eval_harness_steps=steps_per_eval,
        # HF checkpoint cadence — one save per eval step. ``hf_save_path`` is
        # auto-set by marin's ``_update_config_to_use_out_path`` to
        # ``<output_path>/hf`` (since ``output_path`` is set on the pod config
        # below); we only need to override the default ``hf_save_steps=10_000``.
        # ``hf_save_dtype`` stays None → preserves param dtype (fp32 under our
        # ``p=f32,c=bfloat16`` jmp policy); downstream consumers can cast at
        # load time via ``from_pretrained(..., torch_dtype=torch.bfloat16)``.
        hf_save_steps=_hf_save_steps(NUM_TRAIN_STEPS),
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project=WANDB_PROJECT,
                tags=list(tags),
                group=WANDB_GROUP,
                name=run_name,
                replicate_path=this_output_path(),
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=BATCH_SIZE,
            per_device_parallelism=PER_DEVICE_PARALLELISM,
            num_train_steps=NUM_TRAIN_STEPS,
            steps_per_eval=steps_per_eval,
            checkpointer=_checkpointer(NUM_TRAIN_STEPS),
            mesh=MeshConfig(axes={"replica": 1, "data": -1, "model": 1}),
            allow_nondivisible_batch_size=True,
        ),
    )
    pod_config = TrainLmOnPodConfig(
        train_config=inner,
        resources=ResourceConfig.with_tpu(TPU_TYPES, ram=TPU_RAM),
        output_path=this_output_path(),
    )
    # The remote() call here IS the TPU worker — Fray's RemoteCallable submits
    # the job with these resources verbatim. exp179_eval_only.py:140-145 mirrors
    # this pattern; the old exp160_parity / exp166 pattern of CPU-orchestrator +
    # TPU-pod-config relies on marin executor magic that no longer fires after
    # the PR #182 / #186 rebase (smoke4 ran on a CPU worker and JAX raised
    # "No accelerator found" at trainer.initialize). pip_dependency_groups
    # explicitly installs ``tpu`` so libtpu is present — without it JAX
    # silently falls back to CPU and we get the same error.
    return ExecutorStep(
        name=os.path.join("checkpoints", run_name),
        fn=remote(
            _run_train_with_marin_dna_imports,
            resources=ResourceConfig.with_tpu(TPU_TYPES, ram=TPU_RAM),
            pip_dependency_groups=["marin", "tpu"],
            env_vars=_train_remote_env_vars(),
        ),
        config=pod_config,
    )


def main() -> None:
    selected = _selected_datasets()
    steps = [
        _build_train_step(strategy, dataset) for strategy, dataset in selected.items()
    ]
    executor_main(
        steps=steps,
        description=(
            f"DNA Bolinas exp{EXP_ISSUE} epoch-reshuffling TSS 0.25B (slice-mix) — "
            f"{len(selected)}/{len(TRAIN_DATASETS)} arm(s) {VERSION}"
        ),
    )


if __name__ == "__main__":
    main()
