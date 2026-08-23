"""Tests for issue #515 per-sequence current-loss selection."""

from __future__ import annotations

import torch

from glm_experiments.models.components.selection import (
    TokenSelector,
    select_token_mask,
    selected_count,
)


def test_selected_count_handles_empty_and_small_sets() -> None:
    assert selected_count(0, 0.5) == 0
    assert selected_count(1, 0.5) == 1
    assert selected_count(2, 0.5) == 1
    assert selected_count(5, 0.5) == 2


def test_ranked_selectors_use_each_sequence_and_position_ties() -> None:
    losses = torch.tensor(
        [
            [1.0, 1.0, 3.0, 2.0, 4.0, 0.0],
            [9.0, 4.0, 1.0, 3.0, 2.0, 8.0],
        ]
    )
    eligible = torch.tensor(
        [
            [True, True, True, True, True, False],
            [False, True, True, True, True, False],
        ]
    )

    low = select_token_mask(losses, eligible, mode="student_low", ratio=0.5)
    high = select_token_mask(losses, eligible, mode="student_high", ratio=0.5)
    middle = select_token_mask(losses, eligible, mode="student_middle", ratio=0.5)

    assert torch.equal(low[0], torch.tensor([True, True, False, False, False, False]))
    assert torch.equal(high[0], torch.tensor([False, False, True, False, True, False]))
    assert torch.equal(
        middle[0], torch.tensor([False, True, False, True, False, False])
    )
    assert torch.equal(low[1], torch.tensor([False, False, True, False, True, False]))
    assert torch.equal(high[1], torch.tensor([False, True, False, True, False, False]))
    assert torch.equal(
        middle[1], torch.tensor([False, False, False, True, True, False])
    )


def test_centered_selection_matches_registered_odd_and_even_ranks() -> None:
    losses = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    eligible = torch.tensor(
        [
            [True, True, True, True, True, False],
            [True, True, True, True, False, False],
        ]
    )
    selected = select_token_mask(
        losses,
        eligible,
        mode="student_middle",
        ratio=0.5,
    )
    assert torch.equal(
        selected[0], torch.tensor([False, True, True, False, False, False])
    )
    assert torch.equal(
        selected[1], torch.tensor([False, True, True, False, False, False])
    )


def test_ranked_selectors_handle_variable_and_empty_rows() -> None:
    losses = torch.tensor(
        [
            [7.0, 6.0, 5.0, 4.0, 3.0, 2.0],
            [3.0, 1.0, 2.0, 9.0, 8.0, 7.0],
            [1.0, 6.0, 2.0, 5.0, 3.0, 4.0],
        ]
    )
    eligible = torch.tensor(
        [
            [False, False, False, False, False, False],
            [False, True, False, False, False, False],
            [True, True, True, True, True, False],
        ]
    )
    for mode in ("student_low", "student_middle", "student_high"):
        selected = select_token_mask(losses, eligible, mode=mode, ratio=0.5)
        assert torch.equal(selected.sum(dim=1), torch.tensor([0, 1, 2]))
        assert not (selected & ~eligible).any()

    middle = select_token_mask(
        losses,
        eligible,
        mode="student_middle",
        ratio=0.5,
    )
    assert torch.equal(
        middle[2], torch.tensor([False, False, True, False, True, False])
    )


def test_random_selector_state_round_trips_through_state_dict() -> None:
    losses = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    eligible = torch.ones_like(losses, dtype=torch.bool)
    selector = TokenSelector(mode="random", ratio=0.5, seed=17)

    selector(losses, eligible)
    checkpoint = selector.state_dict()
    expected_next = selector(losses, eligible)

    resumed = TokenSelector(mode="random", ratio=0.5, seed=17)
    resumed.load_state_dict(checkpoint)
    observed_next = resumed(losses, eligible)

    assert torch.equal(observed_next, expected_next)


def test_uniform_selects_every_eligible_target() -> None:
    losses = torch.zeros((2, 3))
    eligible = torch.tensor([[True, False, True], [False, False, False]])
    assert torch.equal(
        select_token_mask(losses, eligible, mode="uniform", ratio=1.0),
        eligible,
    )
