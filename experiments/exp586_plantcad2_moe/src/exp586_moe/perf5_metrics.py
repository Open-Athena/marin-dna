"""Copy consumed diagnostic fields without changing their logged values."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def _floating_scalars(values: tuple[jax.Array, ...]) -> jax.Array:
    # FP32 exactly represents every FP16/BF16 scalar. Wider floating types and
    # every integer stay outside this path, preserving all count bits.
    return jnp.stack([value.astype(jnp.float32) for value in values])


def copy_metrics(metrics: dict[str, Any], *, pack_scalars: bool) -> dict[str, Any]:
    """Copy selected fields together, optionally combining <=32-bit floats.

    Only a separate diagnostic JIT changes; the model/optimizer JIT is intact.
    """
    names = (
        [
            name
            for name, value in metrics.items()
            if isinstance(value, jax.Array)
            and value.ndim == 0
            and jnp.issubdtype(value.dtype, jnp.floating)
            and value.dtype.itemsize <= 4
        ]
        if pack_scalars
        else []
    )
    if not names:
        return jax.device_get(metrics)
    others = {name: value for name, value in metrics.items() if name not in names}
    host = jax.device_get(
        {
            "scalars": _floating_scalars(tuple(metrics[name] for name in names)),
            "others": others,
        }
    )
    values = np.asarray(host["scalars"])
    return host["others"] | {
        name: values[index : index + 1].reshape(()) for index, name in enumerate(names)
    }
