"""Correctness tests for m5.1 SAE activation extraction (issue #288)."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import marin_dna.model.sae as sae_module
from marin_dna.model.sae import (
    M51_HIDDEN_SIZE,
    M51_NUM_BLOCKS,
    M51_SEQUENCE_TOKENS,
    M51_WINDOW_BP,
    FrozenM51,
    M51GenomicWindow,
    build_m51_activation_batch,
    build_m51_next_base_batch,
    load_frozen_m51,
    reference_coordinates,
    run_m51_with_activations,
)

BOS_ID = 2
PAD_ID = 0


class _ToyBlock(nn.Module):
    def __init__(self, increment: float) -> None:
        super().__init__()
        self.increment = increment

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.increment


class _ToyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_ToyBlock(float(i + 1)) for i in range(M51_NUM_BLOCKS)]
        )


class _ToyM51(nn.Module):
    """Small-compute model with m5.1's real sequence/hidden dimensions."""

    def __init__(self) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            hidden_size=M51_HIDDEN_SIZE,
            num_hidden_layers=M51_NUM_BLOCKS,
        )
        self.model = _ToyBackbone()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        assert attention_mask.shape == input_ids.shape
        hidden = (
            input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, M51_HIDDEN_SIZE)
        )
        for layer in self.model.layers:
            hidden = layer(hidden)
        # A deterministic, small-vocabulary head is enough to prove that a
        # read-only block hook leaves the full forward bitwise unchanged.
        logits = hidden[..., :8]
        return SimpleNamespace(logits=logits)


class _ToyTokenizer:
    bos_token_id = BOS_ID
    pad_token_id = PAD_ID

    def __call__(
        self,
        sequence: str,
        *,
        add_special_tokens: bool,
        return_attention_mask: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert len(sequence) == M51_WINDOW_BP
        assert add_special_tokens
        assert return_attention_mask
        assert return_tensors == "pt"
        input_ids, attention_mask = _inputs(batch_size=1)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _inputs(batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.full((batch_size, M51_SEQUENCE_TOKENS), 4, dtype=torch.long)
    input_ids[:, 0] = BOS_ID
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def _windows() -> list[M51GenomicWindow]:
    return [
        M51GenomicWindow("chr1", 100, 100 + M51_WINDOW_BP, "+"),
        M51GenomicWindow("chr2", 1000, 1000 + M51_WINDOW_BP, "-"),
    ]


def _frozen(model: nn.Module) -> FrozenM51:
    model.requires_grad_(False)
    model.eval()
    tokenizer = SimpleNamespace(bos_token_id=BOS_ID, pad_token_id=PAD_ID)
    return FrozenM51(model=model, tokenizer=tokenizer)


def test_read_only_hook_preserves_logits_bitwise() -> None:
    model = _ToyM51().eval()
    input_ids, attention_mask = _inputs()
    with torch.inference_mode():
        expected_logits = model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits.clone()

    output, batch = run_m51_with_activations(
        _frozen(model),
        input_ids,
        attention_mask,
        _windows(),
        block_index=9,
    )

    assert torch.equal(output.logits, expected_logits)
    assert batch.activations.shape == (2, M51_WINDOW_BP, M51_HIDDEN_SIZE)
    assert batch.block_index == 9
    assert batch.report_block == 10
    # Block 10's output is input embedding + sum(1..10) = token + 55.
    assert torch.all(batch.activations == 59)


def test_load_frozen_m51_validates_and_freezes_local_checkpoint(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _ToyM51()
    tokenizer = _ToyTokenizer()
    model_calls: list[tuple[object, dict[str, object]]] = []

    monkeypatch.setattr(
        sae_module.AutoTokenizer,
        "from_pretrained",
        lambda path: tokenizer,
    )

    def _load_model(path, **kwargs):
        model_calls.append((path, kwargs))
        return model

    monkeypatch.setattr(sae_module.AutoModelForCausalLM, "from_pretrained", _load_model)

    frozen = load_frozen_m51(tmp_path, device="cpu", dtype=torch.float32)

    assert frozen.model is model
    assert frozen.tokenizer is tokenizer
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert model_calls == [
        (tmp_path, {"trust_remote_code": True, "torch_dtype": torch.float32})
    ]


def test_activation_coordinates_strip_bos_and_map_current_base_on_both_strands() -> (
    None
):
    captured = torch.zeros(2, M51_SEQUENCE_TOKENS, M51_HIDDEN_SIZE)
    captured[:, 0, :] = -1  # distinctive BOS vector must be removed
    captured[:, 1:, 0] = torch.arange(M51_WINDOW_BP)
    input_ids, attention_mask = _inputs()

    batch = build_m51_activation_batch(
        captured,
        input_ids,
        attention_mask,
        _windows(),
        block_index=4,
        bos_token_id=BOS_ID,
        pad_token_id=PAD_ID,
    )

    assert torch.equal(batch.activations[:, 0, 0], torch.tensor([0.0, 0.0]))
    assert torch.equal(batch.activations[:, -1, 0], torch.tensor([254.0, 254.0]))
    assert batch.coordinates.chroms == ("chr1", "chr2")
    assert batch.coordinates.strands == ("+", "-")
    assert torch.equal(batch.coordinates.starts[0, :3], torch.tensor([100, 101, 102]))
    assert torch.equal(batch.coordinates.starts[0, -3:], torch.tensor([352, 353, 354]))
    assert torch.equal(
        batch.coordinates.starts[1, :3], torch.tensor([1254, 1253, 1252])
    )
    assert torch.equal(
        batch.coordinates.starts[1, -3:], torch.tensor([1002, 1001, 1000])
    )
    assert torch.equal(batch.coordinates.ends, batch.coordinates.starts + 1)


def test_next_base_logits_map_to_following_base_on_both_strands() -> None:
    input_ids, attention_mask = _inputs()
    # Token identities make the causal one-token shift directly observable.
    input_ids[:, 1:] = torch.arange(10, 10 + M51_WINDOW_BP)
    logits = torch.zeros(2, M51_SEQUENCE_TOKENS, 3)
    logits[:, :, 0] = torch.arange(M51_SEQUENCE_TOKENS)

    batch = build_m51_next_base_batch(
        logits,
        input_ids,
        attention_mask,
        _windows(),
        bos_token_id=BOS_ID,
        pad_token_id=PAD_ID,
    )

    # Logit position 0 (the BOS position) predicts input token 1, at the first
    # model-order genomic base. Logit 1 predicts input token 2, and so on.
    assert torch.equal(batch.logits[0, :3, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(batch.target_ids[0, :3], torch.tensor([10, 11, 12]))
    assert torch.equal(batch.coordinates.starts[0, :3], torch.tensor([100, 101, 102]))
    assert torch.equal(
        batch.coordinates.starts[1, :3], torch.tensor([1254, 1253, 1252])
    )
    assert batch.logits.shape[1] == M51_WINDOW_BP
    assert batch.target_ids.shape[1] == M51_WINDOW_BP


def test_reference_coordinates_reject_wrong_window_length() -> None:
    with pytest.raises(AssertionError, match="must span 255 bp"):
        M51GenomicWindow("chr1", 0, 254, "+")


@pytest.mark.parametrize("corruption", ["padding", "bos", "nan"])
def test_activation_batch_fails_fast_on_silent_corruption(corruption: str) -> None:
    captured = torch.zeros(2, M51_SEQUENCE_TOKENS, M51_HIDDEN_SIZE)
    input_ids, attention_mask = _inputs()
    if corruption == "padding":
        attention_mask[0, -1] = 0
    elif corruption == "bos":
        input_ids[0, 1] = BOS_ID
    else:
        captured[0, 1, 0] = torch.nan

    with pytest.raises(AssertionError):
        build_m51_activation_batch(
            captured,
            input_ids,
            attention_mask,
            _windows(),
            block_index=0,
            bos_token_id=BOS_ID,
            pad_token_id=PAD_ID,
        )


def test_reference_coordinates_are_unique_and_in_bounds() -> None:
    coordinates = reference_coordinates(_windows())
    for starts, window in zip(coordinates.starts, _windows(), strict=True):
        assert starts.unique().numel() == M51_WINDOW_BP
        assert starts.min().item() == window.start
        assert starts.max().item() == window.end - 1
