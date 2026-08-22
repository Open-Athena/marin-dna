from __future__ import annotations

import pytest
import torch

from exp479_mntp.localized_attention_diagnostic import predictor_row_attention_mask


def _allowed(additive: torch.Tensor) -> torch.Tensor:
    return additive == 0


def test_predictor_row_attention_opens_only_selected_row_to_nonpadding_keys() -> None:
    token_mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ]
    )
    positions = torch.tensor([1, 2])

    observed = _allowed(
        predictor_row_attention_mask(
            token_mask,
            positions,
            dtype=torch.float32,
        )
    )

    expected = torch.tensor(
        [
            [
                [
                    [True, False, False, False],
                    [True, True, True, True],
                    [True, True, True, False],
                    [True, True, True, True],
                ]
            ],
            [
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [True, True, True, False],
                    [True, True, True, False],
                ]
            ],
        ]
    )
    assert torch.equal(observed, expected)


def test_predictor_row_attention_closed_control_is_causal_and_padding_aware() -> None:
    token_mask = torch.tensor([[1, 1, 1, 0]])
    observed = _allowed(
        predictor_row_attention_mask(
            token_mask,
            torch.tensor([1]),
            dtype=torch.bfloat16,
            open_predictor_row=False,
        )
    )
    expected = torch.tensor(
        [
            [
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [True, True, True, False],
                    [True, True, True, False],
                ]
            ]
        ]
    )
    assert torch.equal(observed, expected)


@pytest.mark.parametrize(
    ("attention_mask", "positions", "dtype", "message"),
    [
        (torch.ones(4), torch.tensor([1]), torch.float32, "token attention mask"),
        (torch.ones((1, 4)), torch.tensor([[1]]), torch.float32, "output positions"),
        (torch.ones((1, 4)), torch.tensor([4]), torch.float32, "outside"),
        (torch.ones((1, 4)), torch.tensor([1]), torch.int64, "floating"),
    ],
)
def test_predictor_row_attention_rejects_invalid_inputs(
    attention_mask: torch.Tensor,
    positions: torch.Tensor,
    dtype: torch.dtype,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        predictor_row_attention_mask(attention_mask, positions, dtype=dtype)
