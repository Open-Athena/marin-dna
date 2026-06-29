"""HF inference harness for CLM variant/sequence scoring.

Vendored from biofoundation/inference.py at commit 834dd4c (May 2026),
CLM-only and rewritten to operate directly on HF objects — no
``Tokenizer`` / ``CausalLM`` abstract base classes, no adapter wrappers.

**Expected model/tokenizer interface (duck-typed):**

- ``model``: a callable that, given ``input_ids``, returns an object with
  a ``.logits`` attribute (shape ``[B, L, V]``). When called with
  ``output_hidden_states=True``, must also return ``.hidden_states`` — a
  tuple of layer outputs indexable with ``[-1]`` (last) and
  ``[len // 2]`` (middle). HF ``AutoModelForCausalLM`` satisfies this
  natively. Non-HF models (e.g. Evo2) can be wrapped to expose the same
  surface — see ``pipelines/evals/evo2.py`` for an example.

- ``tokenizer``: any object exposing ``.encode(text) -> list[int]``,
  ``.bos_token_id``, ``.eos_token_id``. HF ``PreTrainedTokenizerBase``
  satisfies this. The tokenizer is only used by the data transform step
  (``marin_dna.data.transforms``), not the forward pass.
"""

from __future__ import annotations

import tempfile
from functools import partial
from typing import Any, Callable, Literal

import datasets
import numpy as np
import torch.nn as nn
from transformers import Trainer, TrainingArguments

import torch

from marin_dna.data.dna import NUCLEOTIDES
from marin_dna.data.genome import Genome
from marin_dna.data.transforms import (
    _get_nucleotide_token_ids,
    _get_special_token_counts,
    in_seq_var_pos,
    transform_ll_clm,
    transform_llr_clm,
    transform_variant_marginal_clm,
    transform_window_embedding,
)
from marin_dna.model.scoring import (
    compute_ll_clm,
    compute_marginal_clm,
    compute_variant_score_bundle,
    compute_window_embedding,
)


def run_inference(
    model: nn.Module,
    tokenizer: Any,
    dataset: datasets.Dataset,
    compute_fn: Callable[..., Any],
    data_transform_fn: Callable[..., dict[str, Any]] | None = None,
    data_transform_on_the_fly: bool = False,
    data_transform_kwargs: dict[str, Any] | None = None,
    inference_kwargs: dict[str, Any] | None = None,
) -> Any:
    processed_dataset = _process_dataset(
        dataset,
        tokenizer,
        data_transform_fn,
        data_transform_on_the_fly,
        data_transform_kwargs,
    )
    return _run_inference(
        _ModelComputeFnWrapper(model, compute_fn),
        processed_dataset,
        **(inference_kwargs or {}),
    )


def _run_strand_aware(
    model: nn.Module,
    tokenizer: Any,
    dataset: datasets.Dataset,
    *,
    compute_fn: Callable[..., Any],
    transform_fn: Callable[..., dict[str, Any]],
    transform_kwargs: dict[str, Any] | None = None,
    rc_avg: bool = False,
    **kwargs: Any,
) -> Any:
    """Run inference once on the forward strand; if ``rc_avg=True``, also run
    on the reverse-complemented strand and return the element-wise mean.

    ``strand`` is bound into ``transform_fn`` via ``partial``. Averaging is
    element-wise on numpy arrays, so it works for arbitrary output shape.
    """

    def _one(strand: Literal["+", "-"]) -> Any:
        return run_inference(
            model,
            tokenizer,
            dataset,
            compute_fn=compute_fn,
            data_transform_fn=partial(
                transform_fn, strand=strand, **(transform_kwargs or {})
            ),
            **kwargs,
        )

    fwd = _one("+")
    if not rc_avg:
        return fwd
    rc = _one("-")
    return (np.asarray(fwd) + np.asarray(rc)) / 2


run_ll_clm = partial(
    run_inference,
    compute_fn=compute_ll_clm,
    data_transform_fn=transform_ll_clm,
)


def run_variant_score_bundle(
    model: nn.Module,
    tokenizer: Any,
    dataset: datasets.Dataset,
    genome: Genome,
    window_size: int,
    rc: bool = False,
    return_embeddings: bool = False,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """Run the variant-score bundle (LLR + next-token JSD) in one forward pass.

    Two scores per variant:

    - LLR (log-likelihood ratio)
    - ``next_token_jsd_mean`` — per-position 4-nucleotide softmax JSD between
      REF and ALT next-token predictions, averaged over downstream positions
      (called ``down_jsd_mean`` in Open-Athena/marin-dna#175).

    Both scores share a single 4-nuc log_softmax in the kernel. The
    nucleotide token IDs come from ``_get_nucleotide_token_ids`` (handles
    BOS / n_prefix offset).

    With ``return_embeddings=True`` the same forwards also emit the entire-window
    mean-pooled last-layer hidden states for ref and alt (issue #318): the kernel
    is asked for ``output_hidden_states`` and pools token positions
    ``[n_prefix, n_prefix + window_size)`` — the ``window_size`` DNA positions,
    BOS excluded (the #314 ``entire_window`` extent). These pool bounds are
    strand-independent (entire-window covers the same genomic block on FWD and
    RC), so they are bound into the kernel once; ``var_pos`` still differs per
    strand. Each per-strand array widens from ``[B, 2]`` to ``[B, 2 + 2D]``
    (``D`` = model hidden size): columns ``[2:2+D]`` = ``emb_ref``,
    ``[2+D:2+2D]`` = ``emb_alt`` (fp32). The driver FWD+RC-averages and casts to
    f16.

    The token-level variant position is computed here per-strand and
    bound into the compute_fn as a Python int — constant within the
    inference call, no graph break under torch.compile. Per
    ``_get_variant_window`` the in-sequence pos is ``window_size // 2``
    on the FWD strand and ``window_size - 1 - window_size // 2`` on RC
    (equal for odd ``window_size``, off-by-one for even); after
    tokenization both shift by ``n_prefix``. We don't fuse FWD+RC into
    ``_run_strand_aware`` because ``var_pos`` differs across strands for
    even ``window_size``.

    Args:
        model: HF-shaped causal LM (see module docstring for interface).
        tokenizer: Tokenizer for the model.
        dataset: Dataset with variant information (chrom, pos, ref, alt).
        genome: Genome object for sequence extraction.
        window_size: Window size for sequence context.
        rc: If True, also score the reverse-complemented window for
            each variant and return both strands' arrays. Doubles
            inference cost. Callers compose the per-strand arrays
            (averaging, applying ``minus_llr``/``abs_llr``, etc.).
        return_embeddings: If True, widen each per-strand array to
            ``[B, 2 + 2D]`` with the entire-window-pooled ``emb_ref``/``emb_alt``
            (issue #318; see the function summary). ``output_hidden_states`` makes
            the forward heavier — pair with a smaller ``per_device_eval_batch_size``
            (and optionally ``eval_accumulation_steps``) in ``inference_kwargs``.
        **kwargs: Additional arguments passed to run_inference.

    Returns:
        Dict ``{"fwd": A}`` when ``rc=False``, or ``{"fwd": A, "rc": A}`` when
        ``rc=True``. Each array ``A`` is ``[B, 2]`` (``return_embeddings=False``)
        or ``[B, 2 + 2D]`` (``True``): column 0 is LLR, column 1 is
        ``next_token_jsd_mean``, and (when on) ``[2:2+D]`` / ``[2+D:2+2D]`` are
        the fp32 pooled ``emb_ref`` / ``emb_alt``.
    """
    n_prefix, _ = _get_special_token_counts(tokenizer)
    nuc_ids_dict = _get_nucleotide_token_ids(tokenizer)
    nuc_token_ids = torch.tensor(
        [nuc_ids_dict[nuc] for nuc in NUCLEOTIDES], dtype=torch.long
    )
    # Entire-window pool bounds: the window_size DNA token positions, excluding
    # the n_prefix BOS/special tokens. Strand-independent (entire-window pooling
    # covers the same genomic block on FWD and RC; mean is order-invariant), so
    # bound into the kernel once — only var_pos differs per strand.
    pool_lo = n_prefix
    pool_hi = n_prefix + window_size

    def _one(strand: Literal["+", "-"]) -> Any:
        var_pos = in_seq_var_pos(window_size, strand) + n_prefix
        return run_inference(
            model,
            tokenizer,
            dataset,
            compute_fn=partial(
                compute_variant_score_bundle,
                var_pos=var_pos,
                nuc_token_ids=nuc_token_ids,
                return_embeddings=return_embeddings,
                pool_lo=pool_lo,
                pool_hi=pool_hi,
            ),
            data_transform_fn=partial(
                transform_llr_clm, genome=genome, window_size=window_size, strand=strand
            ),
            **kwargs,
        )

    out: dict[str, np.ndarray] = {"fwd": np.asarray(_one("+"))}
    if rc:
        out["rc"] = np.asarray(_one("-"))
    return out


def run_variant_marginal(
    model: nn.Module,
    tokenizer: Any,
    dataset: datasets.Dataset,
    genome: Genome,
    window_size: int,
    rc: bool = False,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """Per-site 4-allele marginal ``log p(x)`` per strand → ``{"fwd": [N, 4], ...}``.

    The opt-in entropy/calibration route (issue #269), parallel to and
    independent of the default LLR+JSD bundle (``run_variant_score_bundle``),
    which is left untouched. Each row's marginal over ``{A,C,G,T}`` at the
    variant position is produced by ``compute_marginal_clm`` (1 prefix + 4
    cached-suffix forwards — the bundle's prefix-sharing generalized to all four
    alleles). Entropy (``entropy_from_marginal``), ref log-prob
    (``marginal[:, ref]``) and the three LLRs (``marginal[:, alt] −
    marginal[:, ref]``) are CPU reductions on the returned ``[N, 4]``. For FWD+RC,
    average the two per-strand log-marginals first with ``rc_average_marginal``
    (a plain mean after the complement realignment) and apply the reduction to
    that — so the calibrated entropy is ``H`` of the FWD+RC-averaged marginal and
    each LLR is the mean of the two per-strand LLRs.

    Mirrors ``run_variant_score_bundle``: ``var_pos`` is computed per strand and
    bound into the compute_fn as a Python int (constant within the call → no
    torch.compile graph break), and FWD/RC are separate ``Trainer.predict``
    passes (``var_pos`` differs across strands for even ``window_size``).

    Args:
        model: HF-shaped causal LM (see module docstring for the interface).
        tokenizer: Tokenizer for the model.
        dataset: Dataset with site information (chrom, pos, ref); ``alt`` is not
            required — the kernel scores all four alleles.
        genome: Genome object for sequence extraction.
        window_size: Window size for sequence context.
        rc: If True, also score the reverse-complemented window per site and
            return its marginal. Doubles inference cost.
        **kwargs: Additional arguments passed to ``run_inference``.

    Returns:
        ``{"fwd": [N, 4]}`` when ``rc=False``, else
        ``{"fwd": [N, 4], "rc": [N, 4]}``. Columns are ``log p(A), log p(C),
        log p(G), log p(T)`` in ``NUCLEOTIDES`` order on the requested strand
        (the RC marginal is over RC-strand alleles, i.e. column ``i`` is the
        forward-strand complement of ``NUCLEOTIDES[i]``; ``rc_average_marginal``
        realigns this before averaging, and a per-strand ref/LLR gather must
        account for it on RC).
    """
    n_prefix, _ = _get_special_token_counts(tokenizer)
    nuc_ids_dict = _get_nucleotide_token_ids(tokenizer)
    nuc_token_ids = torch.tensor(
        [nuc_ids_dict[nuc] for nuc in NUCLEOTIDES], dtype=torch.long
    )

    def _one(strand: Literal["+", "-"]) -> Any:
        var_pos = in_seq_var_pos(window_size, strand) + n_prefix
        return run_inference(
            model,
            tokenizer,
            dataset,
            compute_fn=partial(
                compute_marginal_clm,
                var_pos=var_pos,
                nuc_token_ids=nuc_token_ids,
            ),
            data_transform_fn=partial(
                transform_variant_marginal_clm,
                genome=genome,
                window_size=window_size,
                strand=strand,
            ),
            **kwargs,
        )

    out: dict[str, np.ndarray] = {"fwd": np.asarray(_one("+"))}
    if rc:
        out["rc"] = np.asarray(_one("-"))
    return out


def _center_token_bounds(
    window_size: int,
    n_center_bp: int,
    n_prefix: int,
    strand: Literal["+", "-"],
) -> tuple[int, int]:
    """Token ``[lo, hi)`` slice for the central ``n_center_bp`` DNA positions.

    Strand-aware so the forward and reverse-complement windows pool the **same**
    genomic block: an RC window index ``k`` maps to forward index
    ``window_size - 1 - k``, so the forward block ``[c0, c0+n)`` is covered on RC
    by ``[window_size - n_center_bp - c0, …)``. When ``window_size - n_center_bp``
    is even the two starts coincide; when odd they differ by one (the mirror of
    ``in_seq_var_pos`` for the variant kernel — a shared slice would leave the two
    strands pooling 1-bp-offset windows). ``n_prefix`` offsets past BOS/special
    prefix tokens.
    """
    c0_fwd = (window_size - n_center_bp) // 2
    c0 = c0_fwd if strand == "+" else window_size - n_center_bp - c0_fwd
    lo = n_prefix + c0
    return lo, lo + n_center_bp


def run_window_embeddings(
    model: nn.Module,
    tokenizer: Any,
    dataset: datasets.Dataset,
    genome: Genome,
    window_size: int,
    *,
    n_center_bp: int = 100,
    layer_index: int = -1,
    rc: bool = True,
    **kwargs: Any,
) -> np.ndarray:
    """Center-pooled, FWD+RC-averaged window embeddings via the HF Trainer harness.

    The embedding-UMAP readout (issue #246). Mirrors ``run_variant_score_bundle``
    but reads hidden states instead of logits: each region's ``window_size``
    context is tokenized (``transform_window_embedding``), the center
    ``n_center_bp`` token positions of one layer are mean-pooled
    (``compute_window_embedding``), and the forward and reverse-complement
    strands are averaged. The center-pool token bounds are strand-aware
    (``_center_token_bounds``) so both strands pool the *same* genomic block,
    even when ``window_size - n_center_bp`` is odd.

    Pass ``model`` as a base ``AutoModel`` (not ``AutoModelForCausalLM``) so the
    forward skips the LM head — ``compute_window_embedding`` reads
    ``last_hidden_state`` (``layer_index == -1``). ``**kwargs`` flow to
    ``run_inference`` (e.g. ``data_transform_on_the_fly=True``,
    ``inference_kwargs={...}`` carrying ``per_device_eval_batch_size`` /
    ``bf16_full_eval`` / ``torch_compile`` / ``dataloader_num_workers``).

    Returns ``[N, D]`` (``N = len(dataset)``, ``D`` = model hidden size).
    """
    assert n_center_bp <= window_size, (
        f"n_center_bp {n_center_bp} exceeds window_size {window_size}"
    )
    n_prefix, _ = _get_special_token_counts(tokenizer)

    # One Trainer.predict pass per strand with strand-aware center bounds, so
    # FWD and RC pool the same genomic block before averaging. (Cannot use
    # _run_strand_aware: it shares one compute_fn across strands, but the center
    # bounds must differ by strand for odd window_size - n_center_bp.)
    def _one(strand: Literal["+", "-"]) -> np.ndarray:
        tok_lo, tok_hi = _center_token_bounds(
            window_size, n_center_bp, n_prefix, strand
        )
        return np.asarray(
            run_inference(
                model,
                tokenizer,
                dataset,
                compute_fn=partial(
                    compute_window_embedding,
                    tok_lo=tok_lo,
                    tok_hi=tok_hi,
                    layer_index=layer_index,
                ),
                data_transform_fn=partial(
                    transform_window_embedding,
                    genome=genome,
                    window_size=window_size,
                    strand=strand,
                ),
                **kwargs,
            )
        )

    fwd = _one("+")
    if not rc:
        return fwd
    rc_emb = _one("-")
    assert fwd.shape == rc_emb.shape, (
        f"FWD/RC embedding shape mismatch: {fwd.shape} vs {rc_emb.shape}"
    )
    return (fwd + rc_emb) / 2


def _run_inference(
    model: nn.Module,
    dataset: datasets.Dataset,
    **kwargs: Any,
) -> Any:
    """Run inference on a dataset using a trained model via HF Trainer.

    Args:
        model: A trained PyTorch model that can be used with the HuggingFace Trainer.
        dataset: HuggingFace dataset to run inference on. The dataset should be
            compatible with the model's expected input format.
        **kwargs: Additional keyword arguments to pass to TrainingArguments.
            Common options include:
            - per_device_eval_batch_size: Batch size for evaluation
            - dataloader_num_workers: Number of workers for data loading
            - torch_compile: Whether to use torch.compile for faster inference
            - bf16_full_eval: Whether to use bf16 for evaluation

    Returns:
        The model's predictions on the dataset. The exact format depends on the
        model and dataset, but typically includes probabilities or embeddings.

    Notes:
        Pads the dataset up to a multiple of ``per_device_eval_batch_size`` by
        repeating the last example, then slices the padded predictions off
        before returning. With ``torch_compile=True``, this keeps every batch
        at the same shape, so dynamo compiles a single graph for the cell
        instead of a second one for the otherwise-partial trailing batch
        (which on 1B Qwen3 + A10G cost ~15 s recompile per pass).
    """
    n_real = len(dataset)
    batch_size = kwargs.get("per_device_eval_batch_size", 1)
    pad_n = (batch_size - n_real % batch_size) % batch_size if batch_size > 1 else 0
    if pad_n > 0:
        # Repeat the last real example pad_n times. The padded predictions
        # are sliced off below — we only consume the first n_real rows.
        dataset = dataset.select(list(range(n_real)) + [n_real - 1] * pad_n)

    # HF Trainer requires an output_dir even for .predict() (it never
    # writes to it on the predict path). Use a tempdir scoped to this
    # call so it's cleaned up deterministically.
    with tempfile.TemporaryDirectory() as output_dir:
        training_args = TrainingArguments(
            output_dir=output_dir,
            **(kwargs or {}),
        )
        trainer = Trainer(model=model, args=training_args)
        predictions = trainer.predict(test_dataset=dataset).predictions

    if pad_n > 0:
        predictions = predictions[:n_real]
    return predictions


class _ModelComputeFnWrapper(nn.Module):
    """Adapt ``compute_fn(model, batch)`` into an ``nn.Module.forward`` so
    HF Trainer can call it as if it were the model."""

    def __init__(self, model: nn.Module, compute_fn: Callable[..., Any]):
        super().__init__()
        self.model = model
        self.compute_fn = compute_fn

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.compute_fn(self.model, *args, **kwargs)


def _process_dataset(
    dataset: datasets.Dataset,
    tokenizer: Any,
    data_transform_fn: Callable[..., dict[str, Any]] | None = None,
    data_transform_on_the_fly: bool = False,
    data_transform_kwargs: dict[str, Any] | None = None,
) -> datasets.Dataset:
    if data_transform_fn is None:
        return dataset
    if data_transform_kwargs is None:
        data_transform_kwargs = {}
    data_transform_fn = partial(data_transform_fn, tokenizer=tokenizer)
    if data_transform_on_the_fly:
        return dataset.with_transform(
            _make_batch_transform(data_transform_fn),
            **data_transform_kwargs,
        )
    return dataset.map(
        data_transform_fn,
        **data_transform_kwargs,
    )


def _make_batch_transform(
    transform_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> Callable[[dict[str, list[Any]]], dict[str, list[Any]]]:
    def batch_transform_fn(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # Convert batch format to list of examples
        examples = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        # Apply transform to each example
        transformed_examples = [transform_fn(example) for example in examples]
        # Convert back to batch format
        return {
            key: [ex[key] for ex in transformed_examples]
            for key in transformed_examples[0].keys()
        }

    return batch_transform_fn
