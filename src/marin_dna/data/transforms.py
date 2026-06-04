"""Variant-window extraction + tokenization transforms for CLM scoring.

Vendored from biofoundation/data.py at commit 834dd4c (May 2026), CLM-only
(MLM variants dropped) and rewritten to call HF tokenizer methods directly
(``tokenizer.encode(...)``, ``tokenizer.bos_token_id``) — no ``Tokenizer``
abstract base, no ``HFTokenizer`` wrapper.

Each ``transform_*_clm`` takes a single VCF-style example dict
(``{"chrom", "pos", "ref", "alt", ...}`` for variants; ``{"seq", "pos"}``
for sequence-level) plus a tokenizer (HF ``PreTrainedTokenizerBase`` or
any object exposing ``.encode(text)``, ``.bos_token_id``, ``.eos_token_id``,
and ``.mask_token_id``) and returns a dict of model-ready tensors.
"""

from __future__ import annotations

import functools
from typing import Any, Literal

import torch

from marin_dna.data.dna import NUCLEOTIDES, complement_base, reverse_complement


def _maybe_rc(seq: str, pos: int, strand: Literal["+", "-"]) -> tuple[str, int]:
    """If ``strand == "-"``, reverse-complement ``seq`` and map ``pos`` to
    its position in the RC string. Otherwise return inputs unchanged."""
    if strand == "-":
        seq = reverse_complement(seq)
        pos = len(seq) - 1 - pos
    return seq, pos


def in_seq_var_pos(window_size: int, strand: Literal["+", "-"] = "+") -> int:
    """In-sequence variant position for a centered window on FWD or RC strand.

    FWD: ``window_size // 2``. RC: ``window_size - 1 - window_size // 2``.
    Equal for odd ``window_size``; differ by 1 for even.
    """
    return window_size // 2 if strand == "+" else window_size - 1 - window_size // 2


def _get_variant_window(
    example: dict[str, Any],
    genome: Any,
    window_size: int,
    strand: Literal["+", "-"] = "+",
) -> tuple[str, int]:
    """Extract a window around a variant position from the genome.

    The forward (``strand="+"``) window splits as
    ``left_flank | REF | right_flank``, with
    ``left_flank = window_size // 2`` bp and
    ``right_flank = window_size - window_size // 2 - 1`` bp. Odd
    ``window_size`` is symmetric (e.g. 127 + 1 + 127 = 255); even
    ``window_size`` puts the extra base in the left flank (e.g. 2 + 1 + 1 = 4).

    With ``strand="-"``, the same genomic interval is returned reverse-
    complemented. The variant moves to index ``in_seq_var_pos(window_size, "-")``
    (equal to the forward index for odd ``window_size``; shifted by 1 for
    even). The base at that index equals ``complement(example["ref"])``.

    For FWD/RC strand averaging, odd ``window_size`` gives symmetric left/right
    context lengths across strands and is the cleanest choice; even sizes are
    supported but the variant's left-context length differs by 1 between strands.

    Args:
        example: Dictionary containing 'chrom', 'pos', 'ref' keys
        genome: Genome object to extract sequence from
        window_size: Size of the window in bp
        strand: ``"+"`` for the forward strand (default), ``"-"`` for the
            reverse-complemented window.

    Returns:
        Tuple of (sequence, position_within_window)
    """
    center_index = example["pos"] - 1  # 1-based to 0-based
    pos = in_seq_var_pos(window_size, "+")
    start = center_index - pos
    end = start + window_size
    seq = genome(example["chrom"], start, end, strand=strand).upper()
    assert len(seq) == window_size
    pos = in_seq_var_pos(window_size, strand)
    if strand == "-":
        assert seq[pos] == complement_base(example["ref"])
    else:
        assert seq[pos] == example["ref"]
    return seq, pos


@functools.cache
def _get_special_token_counts(tokenizer: Any) -> tuple[int, int]:
    """``(n_prefix, n_suffix)`` — special tokens auto-prepended / appended.

    Some tokenizers define ``bos_token_id`` / ``eos_token_id`` but don't
    auto-insert them (HF GPT-2-style); a behavioural probe of ``encode("A")``
    is needed to verify the actual policy.
    """
    try:
        bos_id: int | None = tokenizer.bos_token_id
    except AttributeError:
        bos_id = None
    try:
        eos_id: int | None = tokenizer.eos_token_id
    except AttributeError:
        eos_id = None

    encoded = tokenizer.encode("A")
    n_prefix = 1 if bos_id is not None and encoded[:1] == [bos_id] else 0
    n_suffix = 1 if eos_id is not None and encoded[-1:] == [eos_id] else 0
    return n_prefix, n_suffix


@functools.cache
def _get_nucleotide_token_ids(tokenizer: Any) -> dict[str, int]:
    """Token IDs for the 4 DNA nucleotides under this tokenizer."""
    n_prefix, _ = _get_special_token_counts(tokenizer)
    return {nuc: tokenizer.encode(nuc)[n_prefix] for nuc in NUCLEOTIDES}


def transform_llr_clm(
    example: dict[str, Any],
    tokenizer: Any,
    genome: Any,
    window_size: int,
    strand: Literal["+", "-"] = "+",
) -> dict[str, Any]:
    """SNV-explicit variant-window transform for ``compute_variant_score_bundle``.

    Returns ``{"input_ids": [L] (ref only), "alt_token_id": int}``. The
    alt sequence is identical to ref except at one position, so storing
    only the ref tokens + the alt nucleotide token ID is a strictly
    smaller representation than ``[2, L]`` (~50% dataset memory). The
    kernel reconstructs the alt suffix on the fly.

    The input dictionary follows VCF semantics where ``pos`` is a 1-based
    coordinate and ``ref``/``alt`` are single nucleotides. With
    ``strand="-"``, the window is reverse-complemented and ``alt`` is
    complemented; the variant ends up at the RC-strand DNA index inside
    the window.

    SNV-only: the schema can't represent indels or multi-base substitutions.
    """
    seq, _pos = _get_variant_window(example, genome, window_size, strand=strand)
    alt = example["alt"] if strand == "+" else complement_base(example["alt"])
    input_ids = torch.tensor(tokenizer.encode(seq))
    alt_token_id = _get_nucleotide_token_ids(tokenizer)[alt]
    return dict(input_ids=input_ids, alt_token_id=alt_token_id)


def transform_window_embedding(
    example: dict[str, Any],
    tokenizer: Any,
    genome: Any,
    window_size: int,
    strand: Literal["+", "-"] = "+",
) -> dict[str, Any]:
    """Tokenize the model-context window centered on a region (issue #246).

    The region ``[start, end)`` (0-based half-open) is expanded to
    ``window_size`` bp centered on its midpoint and fetched on ``strand``
    (``"-"`` returns the reverse complement). Returns ``{"input_ids": [L]}``.

    The caller (``marin_dna.model.runner.run_window_embeddings``) computes the
    center-pool token bounds **strand-aware** so both strands pool the same
    genomic block before averaging. No variant logic
    here — unlike ``transform_llr_clm`` there is no per-strand position to track.
    """
    chrom = str(example["chrom"])
    start = int(example["start"])
    end = int(example["end"])
    center = (start + end) // 2
    ctx_start = center - window_size // 2
    seq = genome(chrom, ctx_start, ctx_start + window_size, strand).upper()
    # Genome pads off-chromosome flanks with N to the full width; a wrong length
    # would mean a coordinate/extraction bug — fail loud rather than ship a
    # ragged batch (which would surface as an opaque collation error instead).
    assert len(seq) == window_size, (
        f"expected {window_size} bp at {chrom}:{ctx_start}, got {len(seq)}"
    )
    return {"input_ids": torch.tensor(tokenizer.encode(seq))}


def transform_reflogprob_clm(
    example: dict[str, Any],
    tokenizer: Any,
    strand: Literal["+", "-"] = "+",
) -> dict[str, Any]:
    """Transform a sequence example for reference log probability CLM inference.

    Produces a ``[4, L]`` tensor with all four nucleotides substituted at the
    specified position. The ``ref`` field is the index into ``NUCLEOTIDES``
    (``A``/``C``/``G``/``T``) of the reference base at that position.

    With ``strand="-"``, the input sequence is reverse-complemented and the
    position is mapped to ``len(seq) - 1 - pos``. ``ref`` is then the index
    of the complemented base (because ``seq[pos]`` after RC is the complement
    of the original) — no extra complementation logic needed at the lookup.
    """
    assert example["seq"][example["pos"]] in NUCLEOTIDES
    seq, pos = _maybe_rc(example["seq"], example["pos"], strand)
    input_ids = torch.tensor(tokenizer.encode(seq))
    nuc_ids = _get_nucleotide_token_ids(tokenizer)
    n_prefix, _ = _get_special_token_counts(tokenizer)
    tokenized_pos = pos + n_prefix
    new_input_ids = input_ids.unsqueeze(0).repeat(len(NUCLEOTIDES), 1)
    for i, nuc in enumerate(NUCLEOTIDES):
        new_input_ids[i, tokenized_pos] = nuc_ids[nuc]
    ref = NUCLEOTIDES.index(seq[pos])
    return dict(input_ids=new_input_ids, ref=ref)


def transform_ll_clm(
    example: dict[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    """Prepare an example for CLM sequence-level log-likelihood scoring.

    The raw ``seq`` is uppercased before tokenization, so the tokenizer
    always sees the case it was trained on. The original case is
    preserved in ``is_upper`` for the loss-weight breakdown. This matters
    for case-sensitive (e.g. byte-level) tokenizers such as Evo2's vortex
    CharLevelTokenizer, where ``'a'`` and ``'A'`` map to different token
    ids; for case-insensitive DNA tokenizers like Marin's it is a no-op.

    BOS/EOS handling follows the tokenizer's own policy — we don't
    second-guess it. We detect at most one BOS at the start and one EOS
    at the end (if the tokenizer's ``bos_token_id`` / ``eos_token_id``
    matches there) and mark those positions ``is_upper=False``.
    ``is_upper`` is source-aligned: ``is_upper[i]`` describes the
    character that produced ``input_ids[i]`` (False for special tokens).
    ``compute_ll_clm`` performs the source->target shift when scoring.

    Special-token *targets* (e.g. EOS, when the tokenizer auto-appends
    one) end up with ``is_upper=False`` and so contribute to
    ``ll_sum_lower`` in ``compute_ll_clm`` — a ~1-token bias on
    LL(non-functional) for EOS-trained models, worth knowing if you
    compare absolute LL(non-functional) values across models with vs.
    without EOS.

    Returns:
        input_ids: [L] long tensor.
        is_upper:  [L] bool tensor — True iff ``input_ids[i]`` came from an
                   uppercase character of ``example["seq"]``.
    """
    seq = example["seq"]
    full_ids = tokenizer.encode(seq.upper())

    n_prefix, n_suffix = _get_special_token_counts(tokenizer)
    body_len = len(full_ids) - n_prefix - n_suffix
    assert body_len == len(seq), (
        "Char-level tokenization required for case-breakdown LL "
        f"(body={body_len} tokens vs {len(seq)} chars; "
        "either non-char-level or unexpected special tokens)."
    )

    is_upper = [False] * n_prefix + [c.isupper() for c in seq] + [False] * n_suffix
    return dict(
        input_ids=torch.tensor(full_ids, dtype=torch.long),
        is_upper=torch.tensor(is_upper, dtype=torch.bool),
    )
