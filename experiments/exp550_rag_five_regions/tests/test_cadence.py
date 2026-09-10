from types import SimpleNamespace

import pytest
from levanter.trainer import StepInfo, TrainerHooks
from marin_dna_exp550.cadence import MILESTONE_CALLBACKS, CompletedUpdateHooks


def test_completed_update_milestones_preserve_state_and_other_metrics():
    events = []
    upstream = TrainerHooks()

    def callback(identity):
        def record(info):
            events.append((identity, info.step, info.next_step, info.state))

        record.__module__, record.__qualname__ = identity
        return record

    for identity in MILESTONE_CALLBACKS:
        upstream.add_hook(callback(identity), every=10000)
    upstream.add_hook(callback(("metrics", "ordinary")), every=10000)
    hooks = CompletedUpdateHooks.from_existing(upstream)
    for completed in (9999, 10000, 10001, 100000):
        state = SimpleNamespace(step=completed)
        hooks.run_hooks(StepInfo(state, 1.0, 0.1))
        assert state.step == completed
    milestone_events = [row for row in events if row[0] in MILESTONE_CALLBACKS]
    assert len(milestone_events) == 6
    assert {row[1] for row in milestone_events} == {10000, 100000}
    assert all(row[1] == row[2] == row[3].step for row in milestone_events)
    assert [(row[1], row[2]) for row in events if row[0][0] == "metrics"] == [
        (10000, 10001)
    ]


def test_fail_closed_when_pinned_callback_registration_changes():
    with pytest.raises(ValueError, match="callbacks changed"):
        CompletedUpdateHooks.from_existing(TrainerHooks())
