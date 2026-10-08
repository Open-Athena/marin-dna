"""Host-side exact token accounting and Hero learning rates."""

from dataclasses import dataclass, replace

import optax

from exp582_moe.config import (
    CONTINUATION_TOKENS,
    HORIZON_TOKENS,
    SHORT_STAGE_TOKENS,
    STAGE_TOKENS,
    TOKENS_PER_UPDATE,
    WARMUP_UPDATES,
)
from experiments.grug.moe_hero_ep.heuristic import MoeHeuristic
from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig


@dataclass(frozen=True)
class TokenSchedule:
    """Original long schedule, retained for validating historical checkpoints."""

    horizon: int = HORIZON_TOKENS
    stage_end: int = STAGE_TOKENS
    cooldown_start: int = CONTINUATION_TOKENS
    cooldown: bool = True

    def __post_init__(self) -> None:
        if (
            not 0
            < self.horizon / 100
            < self.cooldown_start
            < self.stage_end
            <= self.horizon
        ):
            raise ValueError("Require warmup < cooldown start < stage end <= horizon")

    def factor(self, tokens: int) -> float:
        """LR at the start of the next update; padding never advances this clock."""
        if tokens < 0:
            raise ValueError("Token count must be nonnegative")
        warmup = self.horizon / 100
        if tokens < warmup:
            return tokens / warmup

        def long_factor(t: int) -> float:
            return 1 - 0.95 * min(1.0, (t - warmup) / (self.horizon - warmup))

        if self.cooldown and tokens >= self.cooldown_start:
            start = long_factor(self.cooldown_start)
            fraction = min(
                1.0,
                (tokens - self.cooldown_start) / (self.stage_end - self.cooldown_start),
            )
            return start + fraction * (0.05 - start)
        return long_factor(tokens)


@dataclass(frozen=True)
class ShortTokenSchedule:
    """Preserve warmup, save after a peak update, then linearly cool to 5%.

    LRs use the input clock before each update. The final warmup update uses
    exactly peak LR so the permanent boundary contains that update. Earlier
    warmup values remain identical to the original schedule.
    """

    peak_update: int = WARMUP_UPDATES
    stage_end: int = SHORT_STAGE_TOKENS
    revision: str = "short-linear-v2"

    def __post_init__(self) -> None:
        if not WARMUP_UPDATES <= self.peak_update < self.stage_end // TOKENS_PER_UPDATE:
            raise ValueError("Require original warmup <= peak update < stage end")
        if self.stage_end % TOKENS_PER_UPDATE:
            raise ValueError("Stage end must contain whole updates")

    def factor(self, tokens: int) -> float:
        if tokens < 0:
            raise ValueError("Token count must be nonnegative")
        peak_tokens = self.peak_update * TOKENS_PER_UPDATE
        if tokens < peak_tokens - TOKENS_PER_UPDATE:
            return min(1.0, tokens / (HORIZON_TOKENS / 100))
        if tokens < peak_tokens:
            return 1.0
        # The first update after the saved peak begins cooling; the actual final
        # update uses the floor, rather than reaching it only after training ends.
        fraction = min(
            1.0,
            (tokens - peak_tokens + TOKENS_PER_UPDATE) / (self.stage_end - peak_tokens),
        )
        return 1.0 - 0.95 * fraction


@dataclass(frozen=True)
class TokenClock:
    """Exact Python integers stored in the checkpoint's committed JSON metadata."""

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
    """Use the existing optimizer, with explicit LR injection before each update.

    Optax permits overriding numeric hyperparameters in its injected state.
    Keeping these numeric prevents an internal step schedule from overwriting the
    token schedule, and preserves MuonH's nonlinear, norm-preserving LR operation.
    """

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
        * heuristic._learning_rate(reference_tokens_per_update, HORIZON_TOKENS, 1536),
        adam_lr=multiplier
        * heuristic._adam_lr(reference_tokens_per_update, HORIZON_TOKENS, 1536),
        beta1=heuristic.beta1,
        beta2=heuristic._beta2(reference_tokens_per_update),
        epsilon=heuristic._epsilon(reference_tokens_per_update, HORIZON_TOKENS),
        weight_decay=0.0,
        gate_router_weight_decay=0.0,
        max_grad_norm=None,
    )


def set_token_learning_rates(
    opt_state: optax.InjectStatefulHyperparamsState,
    config: TokenOptimizerConfig,
    schedule: TokenSchedule | ShortTokenSchedule,
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
