"""The three approved exp593 full-state annealing recipes and stage clocks."""

from __future__ import annotations

from dataclasses import dataclass

SOURCE_RUN = "exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1"
SOURCE_ROOT = (
    "s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/checkpoints/"
    + SOURCE_RUN
    + "/2026.10.20/checkpoints"
)
SOURCE_BINDING_SHA256 = (
    "6a5273b56f94f5da995a882b44208678dbbb8baf8871a4c3ccc37c286f222899"
)
ORIGINAL_UPDATES = 1_907_349
ORIGINAL_WARMUP_UPDATES = 19_073
REFERENCE_TOKENS = 524_288
ORIGINAL_TOKENS = ORIGINAL_UPDATES * REFERENCE_TOKENS
CORPUS_TOKENS = 21_615_869_952
OWNER = "s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna"


@dataclass(frozen=True)
class AnnealTrial:
    """An immutable scientific recipe; placement and retries are independent."""

    key: str
    source_update: int
    source_metadata_sha256: str
    horizon: int

    @property
    def source_checkpoint(self) -> str:
        return f"{SOURCE_ROOT}/step-{self.source_update}"

    @property
    def source_tokens(self) -> int:
        return self.source_update * REFERENCE_TOKENS

    @property
    def source_examples(self) -> int:
        return self.source_update * 64

    @property
    def run_id(self) -> str:
        return f"dna-exp593-anneal-{self.key}-guidedfim30-seed0-v1"

    @property
    def ramp_tokens(self) -> int:
        return self.horizon // 5

    @property
    def inherited_factor(self) -> float:
        return 1 - 0.95 * (self.source_update - ORIGINAL_WARMUP_UPDATES) / (
            ORIGINAL_UPDATES - ORIGINAL_WARMUP_UPDATES
        )

    def factor(self, input_tokens: int) -> float:
        """Next-update LR relative to original peak; no new warmup or reset."""
        if not 0 <= input_tokens <= self.horizon:
            raise ValueError("Annealing token clock is outside its fixed horizon")
        if input_tokens <= self.ramp_tokens:
            return self.inherited_factor
        progress = (input_tokens - self.ramp_tokens) / (self.horizon - self.ramp_tokens)
        return self.inherited_factor + (0.05 - self.inherited_factor) * progress

    def fim_probability(self, input_tokens: int) -> float:
        if not 0 <= input_tokens <= self.horizon:
            raise ValueError("Annealing token clock is outside its fixed horizon")
        return 0.30 * min(1.0, input_tokens / self.ramp_tokens)

    def events(
        self, previous: int, current: int, *, tasks: bool = False
    ) -> tuple[str, ...]:
        """Fixed input-based cadence; scores cannot affect the schedule."""
        if not 0 <= previous <= current <= self.horizon:
            raise ValueError("Invalid stage evaluation interval")
        interval = 20_971_520_000 if tasks else 5_242_880_000
        events = [
            f"{'tasks' if tasks else 'eval'}_{i}"
            for i in range(previous // interval + 1, current // interval + 1)
        ]
        if previous < self.ramp_tokens <= current:
            events.append("curriculum_end")
        if previous < self.horizon <= current:
            events.append("final")
        return tuple(events)


TRIALS = {
    t.key: t
    for t in (
        AnnealTrial(
            "181b",
            346_791,
            "a6b7462d587727dc5631403395761c805c6e07eb7551a122dcbfcf492e54de1a",
            216_000_000_000,
        ),
        AnnealTrial(
            "364b",
            693_582,
            "ad4827f1160660c264712a99cfea4e6a9df9f699e3bc4d405a745bf86d2ddfe5",
            168_000_000_000,
        ),
        AnnealTrial(
            "545b",
            1_040_373,
            "62b1f09a0f5eb86b28012cdaa2a6ac095f408801ded533f7698b4ca70ef657cb",
            120_000_000_000,
        ),
    )
}


def terminal_lengths(remaining: int, size: int = 64) -> tuple[int, ...] | None:
    """Plan final valid examples to hit the cap exactly without partial loss masks.

    Ordinary batches are at most size*8192 inputs.
    Reserve a minimum-size last batch if a single maximum-size batch cannot
    consume the remainder; every example still contains 2048 or more inputs.
    """
    if remaining < size * 2048:
        raise ValueError("A complete minimum-size batch no longer fits")
    if remaining >= size * (8192 + 2048):
        return None
    consume = remaining if remaining <= size * 8192 else remaining - size * 2048
    base, extra = divmod(consume, size)
    result = tuple(base + (i < extra) for i in range(size))
    if min(result) < 2048 or max(result) > 8192:
        raise ValueError("Terminal batch violates the approved input bounds")
    return result
