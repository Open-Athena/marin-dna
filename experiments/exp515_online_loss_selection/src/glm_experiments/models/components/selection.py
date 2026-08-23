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
    selected = torch.zeros_like(eligible)
    detached = losses.detach()

    for row in range(losses.shape[0]):
        positions = torch.nonzero(eligible[row], as_tuple=False).flatten()
        count = int(positions.numel())
        k = selected_count(count, ratio)
        if k == 0:
            continue
        if mode == "uniform":
            chosen = positions
        elif mode == "random":
            if random_generator is None:
                raise ValueError("random selection requires a separate generator")
            random_scores = torch.rand(
                count,
                generator=random_generator,
                device="cpu",
            ).to(device=positions.device)
            order = torch.argsort(random_scores, stable=True)
            chosen = positions[order[:k]]
        else:
            scores = detached[row, positions]
            if mode == "student_low":
                order = torch.argsort(scores, descending=False, stable=True)
                chosen = positions[order[:k]]
            elif mode == "student_high":
                order = torch.argsort(scores, descending=True, stable=True)
                chosen = positions[order[:k]]
            else:
                order = torch.argsort(scores, descending=False, stable=True)
                start = (count - k) // 2
                chosen = positions[order[start : start + k]]
        selected[row, chosen] = True

    return selected


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
