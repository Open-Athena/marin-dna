from types import SimpleNamespace

import numpy as np

from exp586_moe.inference.optimized_vllm import _write_captures


def test_write_captures_preserves_order_and_trims_padding(tmp_path) -> None:
    captures = [
        np.full((2, 3, 4), 1, dtype=np.float32),
        np.full((2, 3, 4), 2, dtype=np.float32),
    ]
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            model=SimpleNamespace(_exp586_captures=captures),
        )
    )
    output = tmp_path / "logits.npy"

    shape = _write_captures(worker, str(output), 3)

    values = np.load(output)
    assert shape == (3, 3, 4)
    assert np.array_equal(values[:2], np.full((2, 3, 4), 1, dtype=np.float32))
    assert np.array_equal(values[2:], np.full((1, 3, 4), 2, dtype=np.float32))
    assert captures == []
