from types import SimpleNamespace

import numpy as np
import torch
from marin_dna_carbon_conditioning_vep.scoring import (
    derive_score_atoms,
    masked_mean_causal_log_likelihood,
)


class FixedLogitModel:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    def __call__(self, **kwargs):
        del kwargs
        return SimpleNamespace(logits=self.logits)


def _pinned_carbon_reference(
    model: FixedLogitModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    targets = input_ids[:, 1:]
    mask = attention_mask[:, 1:].float()
    logp = torch.log_softmax(logits, dim=-1)
    token_logp = logp.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def test_untagged_scorer_matches_pinned_carbon_fixture() -> None:
    generator = torch.Generator().manual_seed(486)
    logits = torch.randn(2, 5, 7, generator=generator, dtype=torch.bfloat16)
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [1, 3, 2, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    model = FixedLogitModel(logits)
    observed = masked_mean_causal_log_likelihood(model, input_ids, attention_mask)
    expected = _pinned_carbon_reference(model, input_ids, attention_mask)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_fwd_rc_average_and_minus_llr_orientation() -> None:
    atoms = derive_score_atoms(
        np.array([-3.0, -2.0]),
        np.array([-4.0, -1.0]),
        np.array([-2.0, -5.0]),
        np.array([-4.0, -4.0]),
    )
    np.testing.assert_allclose(atoms["llr_fwd"], [-1.0, 1.0])
    np.testing.assert_allclose(atoms["llr_rc"], [-2.0, 1.0])
    np.testing.assert_allclose(atoms["llr"], [-1.5, 1.0])
    np.testing.assert_allclose(atoms["score"], [1.5, -1.0])
