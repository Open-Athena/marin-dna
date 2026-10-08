"""Exact token accounting and the Hero text-run learning-rate recipe."""

from dataclasses import dataclass, replace

import optax

from exp586_moe.config import (
    PEAK_CHECKPOINT_UPDATE,
    TOKENS_PER_UPDATE,
    TOTAL_UPDATES,
    TRAINING_TOKENS,
    WARMUP_UPDATES,
)
from experiments.grug.moe_hero_ep.heuristic import MoeHeuristic
from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig


@dataclass(frozen=True)
class HeroTokenSchedule:
    """One-percent warmup, then linear decay to 5%, matching Hero text runs."""

    total_updates: int = TOTAL_UPDATES
    warmup_updates: int = WARMUP_UPDATES
    tokens_per_update: int = TOKENS_PER_UPDATE
    min_lr_ratio: float = 0.05
    peak_update: int = PEAK_CHECKPOINT_UPDATE
    revision: str = "hero-linear-v1"

    def __post_init__(self) -> None:
        if not 0 < self.warmup_updates < self.total_updates:
            raise ValueError("Require 0 < warmup updates < total updates")
        if self.tokens_per_update <= 0:
            raise ValueError("Tokens per update must be positive")
        if self.peak_update != self.warmup_updates + 1:
            raise ValueError("Peak checkpoint must follow the exact peak-LR update")
        if not 0 <= self.min_lr_ratio < 1:
            raise ValueError("Minimum LR ratio must be in [0, 1)")

    def factor(self, tokens: int) -> float:
        """Return the LR factor at the start of the next update."""
        if tokens < 0 or tokens % self.tokens_per_update:
            raise ValueError("Token clock must be a nonnegative whole update")
        update = tokens // self.tokens_per_update
        if update < self.warmup_updates:
            return update / self.warmup_updates
        decay_updates = self.total_updates - self.warmup_updates
        decay_progress = min(1.0, (update - self.warmup_updates) / decay_updates)
        return 1.0 - (1.0 - self.min_lr_ratio) * decay_progress


@dataclass(frozen=True)
class TokenClock:
    """Exact Python integers stored in committed checkpoint metadata."""

    input_tokens: int = 0
    loss_targets: int = 0
    examples: int = 0
    updates: int = 0

    def advance(
        self, *, input_tokens: int, loss_targets: int, examples: int
    ) -> "TokenClock":
        if (
            examples <= 0
            or input_tokens - loss_targets != examples
            or loss_targets <= 0
        ):
            raise ValueError(
                "Each whole window must have exactly one fewer next-token target than input tokens"
            )
        return replace(
            self,
            input_tokens=self.input_tokens + input_tokens,
            loss_targets=self.loss_targets + loss_targets,
            examples=self.examples + examples,
            updates=self.updates + 1,
        )


@dataclass(frozen=True)
class TokenOptimizerConfig(GrugMoeMuonHConfig):
    """Inject the token-clock schedule into the existing Hero optimizer."""

    def lr_scheduler(
        self, num_train_steps: int, override_lr: float | None = None
    ) -> float:
        del num_train_steps
        return self.learning_rate if override_lr is None else override_lr


def optimizer_config(
    *, reference_tokens_per_update: int, multiplier: float = 1.0
) -> TokenOptimizerConfig:
    if reference_tokens_per_update <= 0 or multiplier <= 0:
        raise ValueError("Positive reference batch and LR multiplier required")
    heuristic = MoeHeuristic()
    return TokenOptimizerConfig(
        learning_rate=multiplier
        * heuristic._learning_rate(
            reference_tokens_per_update, TRAINING_TOKENS, hidden_dim=768
        ),
        adam_lr=multiplier
        * heuristic._adam_lr(
            reference_tokens_per_update, TRAINING_TOKENS, hidden_dim=768
        ),
        beta1=heuristic.beta1,
        beta2=heuristic._beta2(reference_tokens_per_update),
        epsilon=heuristic._epsilon(reference_tokens_per_update, TRAINING_TOKENS),
        min_lr_ratio=heuristic.min_lr_ratio,
        lr_schedule=heuristic.lr_schedule,
        weight_decay=0.0,
        gate_router_weight_decay=0.0,
        max_grad_norm=None,
    )


def set_token_learning_rates(
    opt_state: optax.InjectStatefulHyperparamsState,
    config: TokenOptimizerConfig,
    schedule: HeroTokenSchedule,
    clock: TokenClock,
) -> optax.InjectStatefulHyperparamsState:
    """Replace only the two LR scalars; preserve moments and optimizer counters."""
    factor = schedule.factor(clock.input_tokens)
    return opt_state._replace(
        hyperparams={
            **opt_state.hyperparams,
            "learning_rate": config.learning_rate * factor,
            "adam_lr": config.adam_lr * factor,
        }
    )
