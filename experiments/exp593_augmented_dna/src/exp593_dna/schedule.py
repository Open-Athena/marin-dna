"""Variable-length input-token scheduling and durable augmentation counters."""

from __future__ import annotations

from dataclasses import dataclass, replace

TRAINING_TOKENS = 216_158_699_520
WARMUP_TOKENS = 2_161_115_136
REFERENCE_TOKENS_PER_UPDATE = 524_288
EVAL_INTERVAL_TOKENS = 5_242_880_000


@dataclass(frozen=True)
class InputTokenSchedule:
    """Preserve the original Hero warmup and decay on an actual-input clock."""

    training_tokens: int = TRAINING_TOKENS
    warmup_tokens: int = WARMUP_TOKENS
    min_lr_ratio: float = 0.05

    def __post_init__(self) -> None:
        if not 0 < self.warmup_tokens < self.training_tokens:
            raise ValueError("Warmup must fit inside the input-token budget")
        if not 0 <= self.min_lr_ratio < 1:
            raise ValueError("Invalid minimum learning-rate ratio")

    def factor(self, input_tokens: int) -> float:
        """Return the next update's factor without rounding to full 8192 windows."""
        if input_tokens < 0:
            raise ValueError("Input-token clock must be nonnegative")
        if input_tokens < self.warmup_tokens:
            return input_tokens / self.warmup_tokens
        progress = min(
            1.0,
            (input_tokens - self.warmup_tokens)
            / (self.training_tokens - self.warmup_tokens),
        )
        return 1.0 - (1.0 - self.min_lr_ratio) * progress


@dataclass(frozen=True)
class TokenClock:
    """All counts are host Python integers; padding never advances the clock."""

    input_tokens: int = 0
    dna_tokens: int = 0
    text_tokens: int = 0
    loss_targets: int = 0
    examples: int = 0
    fim_examples: int = 0
    rc_examples: int = 0
    updates: int = 0

    def __post_init__(self) -> None:
        values = vars(self)
        if any(type(v) is not int or v < 0 for v in values.values()):
            raise ValueError("Token/data clocks require nonnegative Python integers")
        if self.input_tokens != self.dna_tokens + self.text_tokens:
            raise ValueError("Actual input must partition into DNA and text")
        if (
            self.loss_targets > self.dna_tokens
            or max(self.fim_examples, self.rc_examples) > self.examples
        ):
            raise ValueError("Target or augmentation count exceeds the source count")

    def advance(
        self,
        *,
        dna_tokens: int,
        text_tokens: int,
        loss_targets: int,
        examples: int,
        fim_examples: int,
        rc_examples: int,
    ) -> TokenClock:
        """Commit one optimizer update, retaining both input and target budgets."""
        increment = TokenClock(
            input_tokens=dna_tokens + text_tokens,
            dna_tokens=dna_tokens,
            text_tokens=text_tokens,
            loss_targets=loss_targets,
            examples=examples,
            fim_examples=fim_examples,
            rc_examples=rc_examples,
            updates=1,
        )
        if (
            examples < 1
            or loss_targets < examples
            or increment.input_tokens < examples * 2048
        ):
            raise ValueError(
                "Every consumed example must retain its DNA target and minimum length"
            )
        return replace(
            self, **{k: getattr(self, k) + getattr(increment, k) for k in vars(self)}
        )


def validation_events(previous_tokens: int, input_tokens: int) -> tuple[str, ...]:
    """Find milestones crossed by variable-size updates, with no rounded clocks."""
    if not 0 <= previous_tokens <= input_tokens:
        raise ValueError("Validation input clocks must be ordered and nonnegative")
    events = []
    if previous_tokens < WARMUP_TOKENS <= input_tokens:
        events.append("warmup_end")
    for index in range(
        previous_tokens // EVAL_INTERVAL_TOKENS + 1,
        min(input_tokens, TRAINING_TOKENS - 1) // EVAL_INTERVAL_TOKENS + 1,
    ):
        events.append(f"interval_{index}")
    if previous_tokens < TRAINING_TOKENS <= input_tokens:
        events.append("final")
    return tuple(events)
