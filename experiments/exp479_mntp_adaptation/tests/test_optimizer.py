from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from exp479_mntp.config import optimizer_hyperparameters
from exp479_mntp.optimizer import AdamH, build_optimizer, grouped_parameters


def test_adamh_matches_one_step_reference_and_preserves_norm() -> None:
    parameter = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64))
    gradient = torch.tensor([[0.5, -0.25], [0.75, -1.0]], dtype=torch.float64)
    parameter.grad = gradient.clone()
    before = parameter.detach().clone()
    optimizer = AdamH(
        [
            {
                "params": [parameter],
                "lr": 0.1,
                "algorithm": "adamh",
                "betas": (0.8, 0.9),
                "eps": 1e-8,
            }
        ]
    )
    update = gradient / (gradient.abs() + 1e-8)
    intermediate = before - 0.1 * update * torch.linalg.vector_norm(
        before
    ) / torch.linalg.vector_norm(update)
    expected = (
        intermediate * torch.linalg.vector_norm(before) / torch.linalg.vector_norm(intermediate)
    )
    optimizer.step()
    assert torch.allclose(parameter, expected)
    assert torch.linalg.vector_norm(parameter).item() == pytest.approx(
        torch.linalg.vector_norm(before).item()
    )


def test_parameter_grouping_matches_linear_embedding_norm_bias_contract() -> None:
    model = nn.Sequential(nn.Embedding(8, 4), nn.Linear(4, 6), nn.LayerNorm(6))
    adamh, adam = grouped_parameters(model, optimizer_hyperparameters(8))
    assert list(adamh["params"]) == [model[1].weight]
    assert {id(parameter) for parameter in adam["params"]} == {
        id(model[0].weight),
        id(model[1].bias),
        id(model[2].weight),
        id(model[2].bias),
    }


def test_optimizer_state_dict_round_trip() -> None:
    model = nn.Sequential(nn.Embedding(8, 4), nn.Linear(4, 6), nn.LayerNorm(6))
    clone = copy.deepcopy(model)
    optimizer = build_optimizer(model, optimizer_hyperparameters(8))
    clone_optimizer = build_optimizer(clone, optimizer_hyperparameters(8))
    loss = model(torch.tensor([1, 2])).square().sum()
    loss.backward()
    optimizer.step()
    clone_optimizer.load_state_dict(optimizer.state_dict())
    assert clone_optimizer.state_dict()["param_groups"] == optimizer.state_dict()["param_groups"]
    for left, right in zip(
        clone_optimizer.state_dict()["state"].values(),
        optimizer.state_dict()["state"].values(),
        strict=True,
    ):
        assert left["step"] == right["step"]
        assert torch.equal(left["exp_avg"], right["exp_avg"])
        assert torch.equal(left["exp_avg_sq"], right["exp_avg_sq"])
