from types import SimpleNamespace

import equinox as eqx
import jax
import jmp
import numpy as np
from exp586_moe.data_parallel import replicated_mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from tokenizers import Tokenizer, models, pre_tokenizers

from exp593_dna.development import ConservationSite, canonical_sequence_hash
from exp593_dna.development_evaluate import DevelopmentEvaluator
from exp593_dna.development_readout import (
    DevelopmentPanels,
    completion_target,
    conservation_examples,
)
from exp593_dna.pipeline import ALPHABET, Encoder


def test_compiled_readout_matches_fp32_reference_and_preserves_partial_batch_order(
    monkeypatch,
):
    import exp593_dna.evaluate as kernels

    monkeypatch.setattr(kernels, "apply_qb_betas", lambda params, pending: params)

    class TinyModel(eqx.Module):
        output_proj: jax.Array

        def __call__(self, tokens, *, mask):
            return jax.nn.one_hot(tokens % 8, 8, dtype=self.output_proj.dtype), {}

    tokenizer = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, **{c: i + 1 for i, c in enumerate(ALPHABET)}},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    encoder = Encoder(tokenizer)
    dna = "".join(np.random.default_rng(593).choice(list("ACGT"), 512))
    site = ConservationSite(
        0, "1", 1024, 768, 1280, dna, 1, canonical_sequence_hash(dna), "frozen"
    )
    views = conservation_examples(site, encoder)
    examples = [views[i % 3] for i in range(65)]
    projection = (
        np.random.default_rng(586)
        .normal(0, 0.1, (8, len(ALPHABET) + 1))
        .astype(np.float32)
    )
    mesh = replicated_mesh()
    expected = []
    for example in examples:
        prompt, target = completion_target(example)
        logits = projection[int(prompt[-1]) % 8]
        expected.append(float(np.log(np.exp(logits).sum()) - logits[int(target[0])]))
    with jax.set_mesh(mesh):
        model = TinyModel(
            jax.device_put(projection, NamedSharding(mesh, P(None, "model")))
        )
        for policy, tolerance in (("float32", 1e-6), ("bfloat16", 0.002)):
            engine = DevelopmentEvaluator(
                DevelopmentPanels((), ()),
                encoder,
                mesh,
                jmp.get_policy(f"params=float32,compute={policy},output=float32"),
            )
            actual = engine.readout(
                SimpleNamespace(params=model, pending_qb_betas=None), examples
            )
            assert len(actual) == 65
            np.testing.assert_allclose(
                [o[0][0] for o in actual], expected, atol=tolerance, rtol=tolerance
            )
            for example, (_, prediction) in zip(examples, actual, strict=True):
                prompt, _ = completion_target(example)
                assert prediction.tolist() == [
                    int(projection[int(prompt[-1]) % 8].argmax())
                ]
