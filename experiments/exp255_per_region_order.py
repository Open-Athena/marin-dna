# Copyright The Marin Authors / Bolinas Authors
# SPDX-License-Identifier: Apache-2.0

"""Order-deduplicated per-region Qwen3-0.25B gLMs — v4_cds & v4_ccre_non_promoter (issue #255).

Species diversity-vs-quantity ablation: a two-arm clone of exp232 that swaps the
*species cohort* of the training data and holds everything else — model,
optimizer, compute, eval — byte-identical. exp232 trains six 0.25B region
specialists on the **108-family** zoonomia-v1 projection; here we retrain the two
**largest** partitions — ``v4_cds`` and ``v4_ccre_non_promoter`` — on the
**19-order** cohort instead (``zoonomia-v1-v4_<region>-order``, one species per
placental order, a strict subset of the v1 projection; #230 / #233), at the same
5K steps × 8192 batch. exp232's own family ``v4_cds`` / ``v4_ccre_non_promoter``
arms are the matched baseline — no re-run.

The order arms see ~3-4× more epochs at fixed compute, because the -order sets
are ~17.5% the size of the family partitions:
  * ``v4_cds-order`` 10,089,622 post-RC rows → 4.06 ep (family 0.71)
  * ``v4_ccre_non_promoter-order`` 15,419,526 post-RC rows → 2.66 ep (family 0.47)
That epoch asymmetry is the headline confound (#255): a tie-or-win for the order
cohort is the informative outcome. Both these arms and exp232's family arms save
10 per-step HF checkpoints, so the diversity effect is *additionally* read at
matched epochs / per-arm optimal-stop offline — no extra training.

In-training validation is held fixed — the **family-projection** recipes,
unchanged from exp232; only the *training* species cohort varies:
  * 5 region-specific ``zoonomia-v1-val_*`` recipes (LL gap) from PR #171,
    each tokenized functional + nonfunctional. Drops ``val_utr5`` / ``val_promoter``
    in favor of the gene-centric ``val_tss_pc``.
  * ``mendelian_traits_255`` lm-eval task (PR #186) — per consequence subset +
    Global + Macro Avg + FWD/RC strand averaging. Online metric is AUPRC +
    per-variant FWD/RC averaging (#226); in-training headline ``_global_/avg/auprc``,
    offline evals_v2 leaderboard ``_macro_avg_/avg/auprc`` (#161).

HF checkpoint saving is enabled by setting ``hf_save_steps`` to match the eval
cadence (every 500 steps). ``hf_save_path`` is auto-set by marin's
``_update_config_to_use_out_path`` to ``<output_path>/hf`` whenever
``TrainLmOnPodConfig.output_path`` is set, so we don't pass it explicitly.
``hf_save_dtype`` stays at the default ``None``, preserving the param dtype
(fp32 under our ``jmp.get_policy("p=f32,c=bfloat16")`` policy) — losslessly
downstream-loadable at bf16 via ``torch_dtype=torch.bfloat16``. Mechanism
inherited from exp187/exp232 (verified end-to-end in exp187's smoke test).

Hardware: ``v6e-4`` + ``PDP=1024`` in ``us-east5`` is the cost-optimal target
(script defaults). v6e-4 is 4 chips at ~2× the MFU of v6e-8 → ~half the
chip-hours for similar wall-clock, and v6e is abundant in ``us-east5-b``
(data-local to the us-east5 cache) while v5p stays capacity-starved — PR #250's
operational findings. ``PDP`` is the per-chip microbatch: on the 4-chip v6e-4
slice the full per-chip batch is 8192/4 = 2048, which OOMs v6e's ~31 GB HBM, so
``PDP=1024`` grad-accumulates 2 microbatches of 1024/chip (~24 GB, fits) with the
**effective batch unchanged at 8192** (mathematically identical step). ``PDP=1024``
is a no-op on any 8-chip slice (8192/8 = 1024/chip already), so it's safe on every
4-8-chip pool — a re-route needs only ``-e TPU_TYPE``. v6e is preemptible →
babysit/relaunch per PR #250 (resume is free: neither TPU type nor PDP is in the
output-config hash).

Download/tokenize pattern: option 1 (``HfTokenizeConfig(id=<hf-name>)``). The 5
val recipes reuse exp187/exp232's tokenizations (identical names → cache hit);
only the 2 -order training partitions tokenize anew (distinct ``-order`` cache
names so they never collide with the family caches).

Launch from a CPU box with an iris tunnel open, one job per arm (see
``experiments/README.md``):

    SWEEP_DATASETS=v4_cds_order uv run iris --cluster=marin job run \\
        --no-wait --user gonzalo --job-name exp255-cds-order-v6e4 \\
        --cpu 1 --memory 2g --extra marin --region us-east5 \\
        -e WANDB_API_KEY "$(grep -A2 api.wandb.ai ~/.netrc | grep password | awk '{print $2}')" \\
        -e HF_HUB_DOWNLOAD_TIMEOUT 120 -e UV_LOCK_TIMEOUT 7200 \\
        -e SWEEP_DATASETS v4_cds_order -e TPU_TYPE v6e-4 -e PDP 1024 \\
        -- python experiments/exp255_per_region_order.py

Reference templates:
  * Direct parent — ``experiments/exp232_per_region.py`` (this is a clone with the
    two family→order dataset swaps; geometry / optimizer / eval unchanged).
  * Tracking issue — #255 (diversity-vs-quantity); species-subset axis — #233 / #230.
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
# Constants — issue #255 order-cohort per-region experiment.
# =============================================================================

EXP_ISSUE = 255
VERSION = "v0.1"
TOKENIZER = "bolinas-dna/tokenizer-char-bos"
DNA_BASE_SEQ_LEN = 255  # bp (256 - 1 for BOS)

# The two LARGEST v4 region partitions, restricted to the 19-ORDER species cohort
# (#230 / #233): one representative species per placental order, a strict subset
# of the 108-family v1 projection (reuses the projection — no re-halLiftover).
# Same v4 bp-priority / window-majority labeling as exp232 (#227 / #228) — only
# the species set differs. SWEEP_DATASETS selects the arm; the matched family
# baselines are exp232's v4_cds / v4_ccre_non_promoter arms (no re-run).
#
# The arm KEY carries the ``_order`` cohort suffix (not just the repo value), so
# every key-derived name — the training tokenize cache, the wandb run name, and
# the checkpoint path — is structurally distinct from exp232's family
# ``v4_<region>`` arms, rather than relying on a special-cased string. (Even with
# identical names there'd be no functional clash: the marin executor hashes the
# dataset ``id`` into each output path and the family repos differ from the
# ``-order`` repos — distinct keys just make it self-documenting.)
TRAIN_DATASETS: dict[str, str] = {
    "v4_cds_order": "bolinas-dna/zoonomia-v1-v4_cds-order",
    "v4_ccre_non_promoter_order": "bolinas-dna/zoonomia-v1-v4_ccre_non_promoter-order",
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

# 5K steps × 8192 batch × 256 tokens/seq ≈ 10.5B tokens per arm — same compute as
# exp232 (we hold compute fixed, not data). The -order sets are ~17.5% the size of
# the family partitions, so the order arms see far more epochs than the family
# baseline: v4_cds-order (10,089,622 rows) 4.06 ep vs family 0.71;
# v4_ccre_non_promoter-order (15,419,526 rows) 2.66 ep vs family 0.47. The epoch
# asymmetry is the headline confound (#255), read out offline at matched epochs
# via the per-step HF checkpoints both cohorts save.
NUM_TRAIN_STEPS = 5_000

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
CHECKPOINT_TIME_INTERVAL = timedelta(hours=1)

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
    """Wrap ``run_levanter_train_lm`` with the in-function marin_dna imports the
    TPU worker needs.

    Iris's ``Entrypoint.from_callable`` cloudpickles ``__main__`` by-value,
    capturing function bytecode but NOT re-importing modules on the worker.
    Module-top imports in this script (``marin_dna.levanter.formats``,
    ``marin_dna.pipelines.evals.lm_eval.task_configs``) never fire on the TPU
    pod, so the side-effecting registrations (``@register_subclass("dna")``
    for ``DNALmDatasetFormat``; the lm-eval task-manager patch) never
    install. Trainer init then fails with
    ``ValueError: Unknown format DNALmDatasetFormat(...)`` when deserializing
    the data config (smoke5 confirmed this).

    Exp179_eval_only.py:91-96 has the same pattern for the eval path. The
    fix is the same here: bake the imports into the function body that the
    TPU worker actually runs.
    """
    import marin_dna.levanter.formats  # noqa: F401  # registers "dna" LmDatasetFormat
    import marin_dna.pipelines.evals.lm_eval  # noqa: F401  # patches lm_eval TaskManager

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
    run_name = f"dna-exp{EXP_ISSUE}-zoonomia-v1-0p25b-{strategy}-{VERSION}"
    tags = (
        "dna",
        "marin_dna",
        f"exp{EXP_ISSUE}",
        "per-region",
        "zoonomia_v1_v4",
        "cohort=order",
        "species=19_order",
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
            f"DNA Bolinas exp{EXP_ISSUE} order-cohort per-region Qwen3-0.25B sweep — "
            f"{len(selected)}/{len(TRAIN_DATASETS)} arms {VERSION}"
        ),
    )


if __name__ == "__main__":
    main()
