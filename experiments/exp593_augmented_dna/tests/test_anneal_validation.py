import faulthandler

import pytest

from exp593_dna import anneal_train, anneal_validation


def smoke():
    return anneal_train.TrainingConfig(
        trial="181b",
        cluster="cw-rno2a",
        nodes=4,
        validation_updates=64,
        run_id="dna-exp593-smoke-watchdog-check",
    )


def test_production_rejected_before_native_run_or_dispatch(monkeypatch):
    def forbidden(_):
        pytest.fail("Production reached the diagnostic wrapper")

    monkeypatch.setattr(anneal_validation, "_NATIVE_RUN", forbidden)
    monkeypatch.setattr(anneal_train, "dispatch", forbidden)
    config = anneal_train.TrainingConfig(trial="181b", cluster="cw-rno2a")
    for operation in (anneal_validation.run, anneal_validation.dispatch):
        with pytest.raises(ValueError, match="bounded validation only"):
            operation(config)


def test_only_watchdog_suppressed_and_restored_on_failure(monkeypatch):
    original = faulthandler.dump_traceback_later
    cancellation = faulthandler.cancel_dump_traceback_later
    config = smoke()

    def native(received):
        assert received is config
        assert faulthandler.cancel_dump_traceback_later is cancellation
        assert faulthandler.dump_traceback_later is not original
        faulthandler.dump_traceback_later(120, repeat=True)
        raise RuntimeError("native validation failure")

    monkeypatch.setattr(anneal_validation, "_NATIVE_RUN", native)
    with pytest.raises(RuntimeError, match="native validation failure"):
        anneal_validation.run(config)
    assert faulthandler.dump_traceback_later is original


def test_dispatch_substitutes_worker_and_restores_on_failure(monkeypatch):
    original = anneal_train.run
    config = smoke()

    def native_dispatch(received):
        assert received is config
        assert anneal_train.run is anneal_validation.run
        raise RuntimeError("dispatch failure")

    monkeypatch.setattr(anneal_train, "dispatch", native_dispatch)
    with pytest.raises(RuntimeError, match="dispatch failure"):
        anneal_validation.dispatch(config)
    assert anneal_train.run is original
