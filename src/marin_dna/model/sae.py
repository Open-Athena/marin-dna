"""m5.1 activation extraction and genomic-coordinate alignment for SAEs.

The first SAE milestone needs two correctness properties before any training:

* observing a post-block residual stream must not change model logits; and
* every observed vector or next-base score must retain its 0-based, half-open
  reference coordinate, including for reverse-complemented inputs.

This module is intentionally specific to the m5.1 checkpoint.  Its fixed model
and tokenizer dimensions make silent use of a wrong checkpoint, EOS-bearing
tokenizer, padded batch, or incorrectly sized genomic window fail immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

M51_NUM_BLOCKS = 19
M51_HIDDEN_SIZE = 1920
M51_WINDOW_BP = 255
M51_BOS_TOKENS = 1
M51_SEQUENCE_TOKENS = M51_BOS_TOKENS + M51_WINDOW_BP


@dataclass(frozen=True)
class M51GenomicWindow:
    """Reference interval and source strand for one full m5.1 input window."""

    chrom: str
    start: int
    end: int
    strand: Literal["+", "-"]

    def __post_init__(self) -> None:
        assert self.chrom, "chrom must be non-empty"
        assert self.start >= 0, f"window start must be non-negative, got {self.start}"
        assert self.end > self.start, (
            f"window end {self.end} must exceed start {self.start}"
        )
        assert self.end - self.start == M51_WINDOW_BP, (
            f"m5.1 windows must span {M51_WINDOW_BP} bp, got "
            f"{self.chrom}:{self.start}-{self.end} ({self.end - self.start} bp)"
        )
        assert self.strand in ("+", "-"), (
            f"strand must be '+' or '-', got {self.strand!r}"
        )


@dataclass(frozen=True)
class ReferenceCoordinateBatch:
    """Per-position 0-based, half-open reference intervals in model order."""

    chroms: tuple[str, ...]
    strands: tuple[Literal["+", "-"], ...]
    starts: Tensor
    ends: Tensor

    def __post_init__(self) -> None:
        batch_size = len(self.chroms)
        assert len(self.strands) == batch_size
        assert self.starts.shape == (batch_size, M51_WINDOW_BP), (
            f"coordinate starts must have shape ({batch_size}, {M51_WINDOW_BP}), "
            f"got {tuple(self.starts.shape)}"
        )
        assert self.ends.shape == self.starts.shape
        assert self.starts.dtype == torch.long
        assert self.ends.dtype == torch.long
        assert torch.equal(self.ends, self.starts + 1), (
            "every stored nucleotide coordinate must be a 1-bp half-open interval"
        )


@dataclass(frozen=True)
class M51ActivationBatch:
    """BOS-stripped post-block residuals and their reference coordinates."""

    activations: Tensor
    coordinates: ReferenceCoordinateBatch
    block_index: int

    def __post_init__(self) -> None:
        batch_size = len(self.coordinates.chroms)
        assert self.activations.shape == (
            batch_size,
            M51_WINDOW_BP,
            M51_HIDDEN_SIZE,
        ), (
            "BOS-stripped m5.1 activations must have shape "
            f"({batch_size}, {M51_WINDOW_BP}, {M51_HIDDEN_SIZE}), got "
            f"{tuple(self.activations.shape)}"
        )
        assert 0 <= self.block_index < M51_NUM_BLOCKS
        assert torch.isfinite(self.activations).all(), "activations contain NaN or inf"

    @property
    def report_block(self) -> int:
        """Human-facing 1-based block number corresponding to ``block_index``."""

        return self.block_index + 1


@dataclass(frozen=True)
class M51NextBaseBatch:
    """Next-base logits, target IDs, and target reference coordinates.

    Entry ``logits[b, i]`` came from model token position ``i`` and predicts
    ``target_ids[b, i] == input_ids[b, i + 1]``.  The attached coordinate is
    therefore the following input base, never the base at the logit's own token
    position.
    """

    logits: Tensor
    target_ids: Tensor
    coordinates: ReferenceCoordinateBatch

    def __post_init__(self) -> None:
        batch_size = len(self.coordinates.chroms)
        assert self.logits.ndim == 3
        assert self.logits.shape[:2] == (batch_size, M51_WINDOW_BP)
        assert self.target_ids.shape == (batch_size, M51_WINDOW_BP)
        assert torch.isfinite(self.logits).all(), "next-base logits contain NaN or inf"


@dataclass(frozen=True)
class FrozenM51:
    """Frozen, evaluation-mode m5.1 model and its validated tokenizer."""

    model: nn.Module
    tokenizer: Any


def _m51_blocks(model: nn.Module) -> nn.ModuleList:
    """Return m5.1's Qwen3 decoder blocks, failing on a wrong model layout."""

    base_model = getattr(model, "model", None)
    blocks = getattr(base_model, "layers", None)
    assert isinstance(blocks, nn.ModuleList), (
        "expected m5.1 Qwen-style decoder blocks at model.model.layers"
    )
    assert len(blocks) == M51_NUM_BLOCKS, (
        f"m5.1 must have {M51_NUM_BLOCKS} decoder blocks, got {len(blocks)}"
    )
    return blocks


def validate_m51_model(model: nn.Module) -> None:
    """Assert that ``model`` has the architecture expected by the SAE pipeline."""

    config = getattr(model, "config", None)
    assert config is not None, "model has no Hugging Face config"
    hidden_size = getattr(config, "hidden_size", None)
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    assert hidden_size == M51_HIDDEN_SIZE, (
        f"m5.1 hidden_size must be {M51_HIDDEN_SIZE}, got {hidden_size}"
    )
    assert num_hidden_layers == M51_NUM_BLOCKS, (
        f"m5.1 num_hidden_layers must be {M51_NUM_BLOCKS}, got {num_hidden_layers}"
    )
    _m51_blocks(model)


def _validate_m51_tokenizer(tokenizer: Any) -> None:
    """Assert one BOS + 255 bases and no EOS/padding for a full window."""

    assert tokenizer.bos_token_id is not None, "m5.1 tokenizer must define BOS"
    assert tokenizer.pad_token_id is not None, "m5.1 tokenizer must define PAD"
    encoded = tokenizer(
        "A" * M51_WINDOW_BP,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    assert input_ids.shape == (1, M51_SEQUENCE_TOKENS), (
        "m5.1 tokenizer must encode 255 bp as one BOS + 255 nucleotide tokens; "
        f"got shape {tuple(input_ids.shape)}"
    )
    assert input_ids[0, 0].item() == tokenizer.bos_token_id
    assert not torch.eq(input_ids[:, 1:], tokenizer.bos_token_id).any()
    assert torch.all(attention_mask == 1), (
        "full m5.1 window unexpectedly contains padding"
    )
    assert not torch.eq(input_ids, tokenizer.pad_token_id).any(), (
        "full m5.1 window unexpectedly contains a PAD token"
    )


def load_frozen_m51(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> FrozenM51:
    """Load the local Hugging Face m5.1 checkpoint for read-only extraction.

    ``checkpoint_path`` is expected to be the locally staged form of the
    registry's GCS checkpoint.  Staging is deliberately outside this function:
    downloads and accelerator allocation are explicit experiment operations.
    """

    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.exists(), (
        f"m5.1 checkpoint does not exist: {checkpoint_path}"
    )
    tokenizer: Any = AutoTokenizer.from_pretrained(checkpoint_path)
    _validate_m51_tokenizer(tokenizer)
    model: nn.Module = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    validate_m51_model(model)
    model.to(device)
    model.requires_grad_(False)
    model.eval()
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    return FrozenM51(model=model, tokenizer=tokenizer)


def _post_block_hidden(output: Any) -> Tensor:
    """Extract the residual tensor from the m5.1 decoder block hook output."""

    hidden = output[0] if isinstance(output, (tuple, list)) else output
    assert isinstance(hidden, Tensor), (
        f"m5.1 decoder block returned {type(output).__name__}, expected Tensor or tuple[Tensor]"
    )
    return hidden


class M51PostBlockCapture:
    """Read-only forward hook for one m5.1 post-block residual stream.

    The capture can span a streaming loop.  Call :meth:`pop` exactly once after
    every model forward so a stale activation cannot be paired with a new batch.
    """

    def __init__(self, model: nn.Module, block_index: int) -> None:
        validate_m51_model(model)
        assert 0 <= block_index < M51_NUM_BLOCKS, (
            f"block_index must be in [0, {M51_NUM_BLOCKS}), got {block_index}"
        )
        self._block = _m51_blocks(model)[block_index]
        self.block_index = block_index
        self._handle: torch.utils.hooks.RemovableHandle | None = None
        self._captured: list[Tensor] = []

    @property
    def report_block(self) -> int:
        """Human-facing 1-based block number corresponding to ``block_index``."""

        return self.block_index + 1

    def __enter__(self) -> M51PostBlockCapture:
        assert self._handle is None, "capture hook is already registered"

        def _capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            self._captured.append(_post_block_hidden(output).detach())

        self._handle = self._block.register_forward_hook(_capture)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        assert self._handle is not None, "capture hook was not registered"
        self._handle.remove()
        self._handle = None
        if exc_type is None:
            assert not self._captured, (
                "unconsumed activation remains; call capture.pop() after every forward"
            )
        else:
            self._captured.clear()

    def pop(self) -> Tensor:
        """Return the one activation emitted since the previous ``pop``."""

        assert self._handle is not None, "capture hook is not registered"
        assert len(self._captured) == 1, (
            f"expected exactly one block activation, captured {len(self._captured)}; "
            "run one model forward between capture.pop() calls"
        )
        return self._captured.pop()


def _validate_m51_inputs(
    input_ids: Tensor,
    attention_mask: Tensor,
    windows: Sequence[M51GenomicWindow],
    *,
    bos_token_id: int,
    pad_token_id: int,
) -> None:
    batch_size = len(windows)
    assert batch_size > 0, "m5.1 activation batch must be non-empty"
    assert not input_ids.is_floating_point(), "input_ids must contain integer token IDs"
    assert bos_token_id != pad_token_id, "m5.1 BOS and PAD token IDs must differ"
    assert input_ids.shape == (batch_size, M51_SEQUENCE_TOKENS), (
        f"input_ids must have shape ({batch_size}, {M51_SEQUENCE_TOKENS}), got "
        f"{tuple(input_ids.shape)}"
    )
    assert attention_mask.shape == input_ids.shape
    assert attention_mask.device == input_ids.device
    assert torch.all(attention_mask == 1), (
        "padded tokens must not enter SAE activations"
    )
    assert torch.all(input_ids[:, 0] == bos_token_id), (
        "every m5.1 input must begin with BOS"
    )
    assert not torch.eq(input_ids[:, 1:], bos_token_id).any(), (
        "BOS may appear only at token position 0"
    )
    assert not torch.eq(input_ids, pad_token_id).any(), (
        "PAD token found in fixed-length m5.1 input"
    )


def reference_coordinates(
    windows: Sequence[M51GenomicWindow],
) -> ReferenceCoordinateBatch:
    """Map 255 model-order nucleotide indices to reference intervals."""

    assert windows, "coordinate batch must be non-empty"
    offsets = torch.arange(M51_WINDOW_BP, dtype=torch.long)
    starts = torch.stack(
        [
            window.start + offsets if window.strand == "+" else window.end - 1 - offsets
            for window in windows
        ]
    )
    ends = starts + 1
    for row, window in enumerate(windows):
        assert torch.all(starts[row] >= window.start)
        assert torch.all(ends[row] <= window.end)
        assert torch.unique(starts[row]).numel() == M51_WINDOW_BP
    return ReferenceCoordinateBatch(
        chroms=tuple(window.chrom for window in windows),
        strands=tuple(window.strand for window in windows),
        starts=starts,
        ends=ends,
    )


def build_m51_activation_batch(
    captured: Tensor,
    input_ids: Tensor,
    attention_mask: Tensor,
    windows: Sequence[M51GenomicWindow],
    *,
    block_index: int,
    bos_token_id: int,
    pad_token_id: int,
) -> M51ActivationBatch:
    """Validate and strip BOS from one captured m5.1 residual-stream batch."""

    _validate_m51_inputs(
        input_ids,
        attention_mask,
        windows,
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
    )
    batch_size = len(windows)
    assert captured.shape == (
        batch_size,
        M51_SEQUENCE_TOKENS,
        M51_HIDDEN_SIZE,
    ), (
        "raw post-block residual must have shape "
        f"({batch_size}, {M51_SEQUENCE_TOKENS}, {M51_HIDDEN_SIZE}), got "
        f"{tuple(captured.shape)}"
    )
    assert torch.isfinite(captured).all(), "raw post-block residual contains NaN or inf"
    activations = captured[:, M51_BOS_TOKENS:, :]
    assert activations.shape[1] == M51_WINDOW_BP
    return M51ActivationBatch(
        activations=activations,
        coordinates=reference_coordinates(windows),
        block_index=block_index,
    )


def build_m51_next_base_batch(
    logits: Tensor,
    input_ids: Tensor,
    attention_mask: Tensor,
    windows: Sequence[M51GenomicWindow],
    *,
    bos_token_id: int,
    pad_token_id: int,
) -> M51NextBaseBatch:
    """Align causal logits with the following input bases and coordinates."""

    _validate_m51_inputs(
        input_ids,
        attention_mask,
        windows,
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
    )
    batch_size = len(windows)
    assert logits.ndim == 3
    assert logits.shape[:2] == (batch_size, M51_SEQUENCE_TOKENS), (
        f"logits must start with shape ({batch_size}, {M51_SEQUENCE_TOKENS}), "
        f"got {tuple(logits.shape)}"
    )
    # Logit i predicts input token i+1.  Keep BOS's logit (it predicts the first
    # genomic base) and drop the final logit (its target lies off-window).
    next_base_logits = logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    assert next_base_logits.shape[1] == target_ids.shape[1] == M51_WINDOW_BP
    return M51NextBaseBatch(
        logits=next_base_logits,
        target_ids=target_ids,
        coordinates=reference_coordinates(windows),
    )


def run_m51_with_activations(
    frozen: FrozenM51,
    input_ids: Tensor,
    attention_mask: Tensor,
    windows: Sequence[M51GenomicWindow],
    *,
    block_index: int,
) -> tuple[Any, M51ActivationBatch]:
    """Run frozen m5.1 once and return its output plus aligned activations."""

    model = frozen.model
    tokenizer = frozen.tokenizer
    validate_m51_model(model)
    assert not model.training, "m5.1 must be in eval mode during activation extraction"
    assert all(not parameter.requires_grad for parameter in model.parameters()), (
        "m5.1 parameters must be frozen during SAE activation extraction"
    )
    _validate_m51_inputs(
        input_ids,
        attention_mask,
        windows,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    with torch.inference_mode(), M51PostBlockCapture(model, block_index) as capture:
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        captured = capture.pop()
    activation_batch = build_m51_activation_batch(
        captured,
        input_ids,
        attention_mask,
        windows,
        block_index=block_index,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return output, activation_batch
