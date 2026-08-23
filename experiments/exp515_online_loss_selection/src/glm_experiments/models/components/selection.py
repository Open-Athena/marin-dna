"""Per-sequence token selection for causal language-model objectives."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

SelectorMode = Literal[
    "uniform",
    "random",
    "student_low",
    "student_middle",
    "student_high",
]

VALID_SELECTOR_MODES: tuple[SelectorMode, ...] = (
    "uniform",
    "random",
    "student_low",
    "student_middle",
    "student_high",
)


def selected_count(eligible_count: int, ratio: float) -> int:
    """Return the registered selection count for one sequence."""

    if eligible_count < 0:
        raise ValueError("eligible_count must be non-negative")
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"selector ratio must be in (0, 1], got {ratio}")
    if eligible_count == 0:
        return 0
    return max(1, math.floor(ratio * eligible_count))


def select_token_mask(
    losses: torch.Tensor,
    eligible: torch.Tensor,
    *,
    mode: SelectorMode,
    ratio: float,
    random_generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Select targets within each sequence from detached current-model losses.

    The input order is token-position order.
    Stable ascending or descending sorts therefore break equal-loss ties by the
    lower token position in every loss-ranked arm.
    """

    if losses.ndim != 2 or eligible.shape != losses.shape:
        raise ValueError(
            "losses and eligible must have the same [batch, targets] shape"
        )
    if eligible.dtype is not torch.bool:
        raise TypeError("eligible must be a boolean tensor")
    if mode not in VALID_SELECTOR_MODES:
        raise ValueError(f"unknown selector mode {mode!r}")
    selected_count(1, ratio)
    if mode == "uniform":
        return eligible.clone()

    counts = eligible.sum(dim=1)
    selected_counts = torch.floor(counts * ratio).long()
    selected_counts = torch.where(
        counts > 0,
        selected_counts.clamp_min(1),
        torch.zeros_like(selected_counts),
    )
    if mode == "random":
        if random_generator is None:
            raise ValueError("random selection requires a separate generator")
        ranking_scores = torch.rand(
            losses.shape,
            generator=random_generator,
            device="cpu",
        ).to(device=losses.device)
    elif mode == "student_high":
        ranking_scores = -losses.detach()
    else:
        ranking_scores = losses.detach()
    ranking_scores = ranking_scores.masked_fill(~eligible, float("inf"))
    order = torch.argsort(ranking_scores, dim=1, stable=True)
    ranks = torch.empty_like(order)
    ordinal = torch.arange(order.shape[1], device=order.device).expand_as(order)
    ranks.scatter_(1, order, ordinal)
    starts = (
        (counts - selected_counts) // 2
        if mode == "student_middle"
        else torch.zeros_like(counts)
    )
    return (
        eligible
        & (ranks >= starts.unsqueeze(1))
        & (ranks < (starts + selected_counts).unsqueeze(1))
    )


class TokenSelector(nn.Module):
    """Stateful selector with a checkpointed RNG independent of global RNG."""

    def __init__(
        self,
        *,
        mode: SelectorMode = "uniform",
        ratio: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if mode not in VALID_SELECTOR_MODES:
            raise ValueError(f"unknown selector mode {mode!r}")
        selected_count(1, ratio)
        self.mode = mode
        self.ratio = ratio
        self.seed = seed
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)

    def forward(self, losses: torch.Tensor, eligible: torch.Tensor) -> torch.Tensor:
        """Return a per-sequence selection mask."""

        generator = self._generator if self.mode == "random" else None
        return select_token_mask(
            losses,
            eligible,
            mode=self.mode,
            ratio=self.ratio,
            random_generator=generator,
        )

    def get_extra_state(self) -> dict[str, object]:
        """Include the private RNG in model and Lightning checkpoints."""

        return {
            "mode": self.mode,
            "ratio": self.ratio,
            "seed": self.seed,
            "generator_state": self._generator.get_state(),
        }

    def set_extra_state(self, state: dict[str, object]) -> None:
        """Restore an arm resume, or reset the RNG when forking the bridge."""

        if state["mode"] != self.mode or float(state["ratio"]) != self.ratio:
            # Every arm loads the uniform bridge's model/optimizer/scheduler state.
            # The arm configuration is authoritative at that one fork boundary.
            self._generator.manual_seed(self.seed)
            return
        generator_state = state["generator_state"]
        if not isinstance(generator_state, torch.Tensor):
            raise TypeError("checkpoint selector generator state is not a tensor")
        self._generator.set_state(generator_state)
