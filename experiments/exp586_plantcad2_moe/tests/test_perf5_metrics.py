"""Keep metric values exact while reducing small device-to-host transfers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exp586_moe.perf5_metrics import copy_metrics


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.bfloat16])
def test_float_packet_preserves_values_and_separate_integer_bits(dtype):
    metrics = {
        "loss": jnp.asarray(0.97631, dtype=dtype),
        "negative_zero": jnp.asarray(-0.0, dtype=dtype),
        "bias": jnp.asarray(-3.15, dtype=dtype),
        "counts": jnp.asarray([16_777_217, 2_147_483_647], dtype=jnp.int32),
        "scalar_counter": jnp.asarray(2_147_483_647, dtype=jnp.int32),
    }
    host = copy_metrics(metrics, pack_scalars=True)
    assert host.keys() == metrics.keys()
    for name in ("loss", "negative_zero", "bias"):
        assert float(host[name]) == float(metrics[name])
        assert np.asarray(host[name]).shape == ()
    assert np.signbit(host["negative_zero"])
    np.testing.assert_array_equal(host["counts"], metrics["counts"])
    assert host["counts"].dtype == np.int32
    assert host["scalar_counter"].dtype == np.int32
    assert int(host["scalar_counter"]) == 2_147_483_647


def test_wider_float_is_never_narrowed():
    with jax.enable_x64():
        value = jnp.asarray(1.000000000123, dtype=jnp.float64)
        host = copy_metrics(
            {"wide": value, "normal": jnp.asarray(1.0, dtype=jnp.float32)},
            pack_scalars=True,
        )
        assert host["wide"].dtype == np.float64
        assert float(host["wide"]) == float(value)


def test_empty_and_nonfloating_packets_need_no_float_stack():
    assert copy_metrics({}, pack_scalars=True) == {}
    host = copy_metrics({"count": jnp.asarray(123, dtype=jnp.int32)}, pack_scalars=True)
    assert host["count"].dtype == np.int32
