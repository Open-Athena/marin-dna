import pytest

from exp593_dna.schedule import (
    EVAL_INTERVAL_TOKENS,
    TRAINING_TOKENS,
    WARMUP_TOKENS,
    InputTokenSchedule,
    TokenClock,
    validation_events,
)


def test_variable_input_clock_tracks_context_and_targets_independently():
    clock = TokenClock().advance(
        dna_tokens=4000,
        text_tokens=96,
        loss_targets=3,
        examples=1,
        fim_examples=1,
        rc_examples=1,
    )
    assert clock.input_tokens == 4096 and clock.loss_targets == 3
    assert clock.updates == 1 and clock.fim_examples == 1
    clock = clock.advance(
        dna_tokens=2048,
        text_tokens=0,
        loss_targets=2047,
        examples=1,
        fim_examples=0,
        rc_examples=0,
    )
    assert clock.input_tokens == 6144 and clock.examples == 2
    assert clock.dna_tokens + clock.text_tokens == clock.input_tokens
    assert clock.loss_targets == 2050 and clock.updates == 2
    assert clock == TokenClock(**vars(clock))
    with pytest.raises(ValueError, match="partition"):
        TokenClock(input_tokens=1)
    with pytest.raises(ValueError, match="minimum"):
        clock.advance(
            dna_tokens=1000,
            text_tokens=0,
            loss_targets=1,
            examples=1,
            fim_examples=1,
            rc_examples=0,
        )


def test_input_schedule_and_eval_events_do_not_round_variable_updates():
    schedule = InputTokenSchedule()
    assert schedule.factor(0) == 0
    assert schedule.factor(12345) == 12345 / WARMUP_TOKENS
    assert schedule.factor(WARMUP_TOKENS) == 1
    assert schedule.factor(TRAINING_TOKENS) == pytest.approx(0.05)
    assert validation_events(WARMUP_TOKENS - 57, WARMUP_TOKENS + 29) == ("warmup_end",)
    assert validation_events(EVAL_INTERVAL_TOKENS - 1, EVAL_INTERVAL_TOKENS + 40) == (
        "interval_1",
    )
    assert validation_events(EVAL_INTERVAL_TOKENS, EVAL_INTERVAL_TOKENS + 40) == ()
    assert validation_events(TRAINING_TOKENS - 1, TRAINING_TOKENS) == ("final",)
