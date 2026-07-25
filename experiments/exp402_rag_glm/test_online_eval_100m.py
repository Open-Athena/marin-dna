import subprocess
import sys

from online_eval_100m import (
    RAG_EVAL_BATCH_SIZE,
    build_config,
    finish_tracker_if_initialized,
)


def test_build_config_is_bounded_eval_only() -> None:
    from levanter.eval_harness import EvalHarnessMainConfig

    config = build_config(
        checkpoint_path="gs://example/checkpoints",
        max_examples=32,
        run_id="dna-exp402-online-parity-test",
    )

    assert isinstance(config, EvalHarnessMainConfig)
    assert config.checkpoint_path == "gs://example/checkpoints"
    assert config.checkpoint_is_hf is False
    assert config.eval_harness.max_examples == 32
    assert config.eval_harness.max_length == 2_048
    assert config.eval_harness.log_samples is False
    assert config.eval_harness.bootstrap_iters == 0
    assert len(config.eval_harness.task_spec) == 1
    assert config.eval_harness.task_spec[0].task == "mendelian_traits_rag_255"
    assert config.trainer.train_batch_size == RAG_EVAL_BATCH_SIZE
    assert config.trainer.per_device_eval_parallelism == 4
    assert config.trainer.require_accelerator is True
    assert config.model.max_seq_len == 2_048
    assert config.model.hidden_dim == 768
    assert config.model.num_layers == 11


def test_fresh_process_loads_real_lm_eval_evaluator() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import online_eval_100m; "
                "from levanter.eval_harness import evaluator; "
                "assert evaluator is not object; "
                "assert hasattr(evaluator, 'evaluate')"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_build_config_rejects_partial_strand_pair() -> None:
    try:
        build_config(
            checkpoint_path="gs://example/checkpoints",
            max_examples=31,
            run_id="dna-exp402-online-parity-test",
        )
    except AssertionError as error:
        assert "strand" in str(error)
    else:
        raise AssertionError("odd max_examples must fail")


def test_finish_tracker_tolerates_preinitialization_failure(monkeypatch) -> None:
    def no_tracker():
        raise RuntimeError("No global tracker set")

    monkeypatch.setattr("online_eval_100m.levanter.tracker.current_tracker", no_tracker)
    finish_tracker_if_initialized()
