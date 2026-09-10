import asyncio

import draccus
import haliax as hax
import numpy as np
from levanter.data.text.datasets import NamedLmDataset
from levanter.store.cache import CacheLedger, TreeCache
from levanter.store.tree_store import TreeStore
from marin_dna_exp550.formats import RagProcessor
from marin_dna_exp550.launch import training_config
from marin_dna_exp550.optimizer import FixedHorizonAdamH


def test_repeated_cache_draws_preserve_all_tokens_masks_and_order(tmp_path):
    rows = RagProcessor()(
        [{"sequence": base * 255} for base in "ACG"]
        + [{"sequence": "[SEQ]".join(["T" * 255] * 40)}]
    )
    root = str(tmp_path / "cache")
    store = TreeStore.open(rows[0], root, mode="w")
    store.extend(rows)
    cache = TreeCache(root, rows[0], CacheLedger(4, {"fixture": 4}, is_finished=True))
    config = training_config("unused", run_id="test", pilot=True, per_device=5)
    pos = hax.Axis("position", 10240)
    caches = {name: cache for name in config.data.components}
    datasets = config.data.build_token_datasets(caches, pos, split="train")
    # Repeated first rows reproduce the upstream bug through the real cache reader.
    indices = [0, 3, 1, 0, 2, 3, 0]
    named = NamedLmDataset(datasets["cds"], pos)
    actual = asyncio.run(named.get_batch(indices))
    for index, example in zip(indices, actual, strict=True):
        np.testing.assert_array_equal(example.tokens.array, rows[index]["input_ids"])
        np.testing.assert_array_equal(
            example.loss_weight.array, rows[index]["loss_weight"]
        )
    assert asyncio.run(named.async_len()) == 4
    assert asyncio.run(datasets["cds"].get_batch([])) == []


def test_worker_optimizer_identity_is_importable_and_config_serializes():
    import cloudpickle

    config = training_config("unused", run_id="test", pilot=True, per_device=5)
    restored = cloudpickle.loads(cloudpickle.dumps(config))
    assert type(restored.optimizer) is FixedHorizonAdamH
    assert FixedHorizonAdamH.__module__ == "marin_dna_exp550.optimizer"
    encoded = draccus.encode(restored)
    assert encoded["optimizer"]["type"] == "exp550_fixed_horizon_adamh"
    assert encoded["trainer"]["train_batch_size"] == 200
