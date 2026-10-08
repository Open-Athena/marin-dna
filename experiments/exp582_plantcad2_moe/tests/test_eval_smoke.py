import numpy as np
import pytest

from exp582_moe.eval_smoke import EvalSmokeConfig, native_loss_error, pool_features


def test_pooling_uses_base_spans_and_rejects_invalid_coverage():
    hidden = np.asarray([[1, 3], [5, 7], [9, 11]], dtype=np.float32)
    spans = [(0, 1), (1, 3), (3, 4)]
    mean, variant = pool_features(spans, hidden, 4, 2)
    np.testing.assert_array_equal(mean, np.mean(hidden[[0, 1, 1, 2]], axis=0))
    np.testing.assert_array_equal(variant, hidden[1])
    with pytest.raises(ValueError, match="overlap or leave gaps"):
        pool_features([(0, 1), (2, 3), (3, 4)], hidden, 4, 2)
    with pytest.raises(ValueError, match="exactly one"):
        pool_features(spans, hidden, 4, 4)


def test_evaluation_rejects_unpinned_and_non_smoke_checkpoints():
    with pytest.raises(ValueError, match="one eight-H100 node"):
        EvalSmokeConfig("random", "exp582-test-smoke", "cw-us-east-02a", nodes=8)
    with pytest.raises(ValueError, match="smoke checkpoint"):
        EvalSmokeConfig("random", "exp582-test-smoke", "cw-us-east-02a")
    prefix = "s3://marin-us-east-02a/tmp/ttl=7d/MarinDNA/exp582_plantcad2_moe/"
    with pytest.raises(ValueError, match="SHA-256"):
        EvalSmokeConfig(
            "random",
            "exp582-test-smoke",
            "cw-us-east-02a",
            checkpoint=prefix + "exp582-test-smoke/checkpoints/step-200",
            requests_uri=prefix + "requests.json",
        )


def test_native_loss_oracle_rejects_nonfinite_values():
    native = np.asarray([[2.0, 3.0, 0.0], [4.0, 5.0, 0.0]])
    logs = [np.asarray([-2.0, -3.0]), np.asarray([-4.0, -5.0])]
    assert native_loss_error(native, logs) == 0
    native[0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite native"):
        native_loss_error(native, logs)
    native[0, 0] = 2
    logs[0][0] = np.inf
    with pytest.raises(ValueError, match="Invalid adapter"):
        native_loss_error(native, logs)
