"""PyTorch port of the pinned Levanter AdamH optimizer."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from exp479_mntp.config import OptimizerHyperparameters, wsd_multiplier


class AdamH(Optimizer):
    """Apply Levanter's norm-preserving AdamH update or ordinary Adam by group."""

    def __init__(self, param_groups: Iterable[dict[str, Any]]) -> None:
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> torch.Tensor | None:
        """Perform one AdamH/Adam update."""

        loss = None if closure is None else closure()
        for group in self.param_groups:
            algorithm = group["algorithm"]
            learning_rate = float(group["lr"])
            beta1, beta2 = group["betas"]
            epsilon = float(group["eps"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("AdamH does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                step = int(state["step"])
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                mu_hat = exp_avg / (1.0 - beta1**step)
                nu_hat = exp_avg_sq / (1.0 - beta2**step)
                update = mu_hat / (nu_hat.sqrt() + epsilon)

                if algorithm == "adam":
                    parameter.add_(update, alpha=-learning_rate)
                elif algorithm == "adamh":
                    parameter_norm = torch.linalg.vector_norm(parameter)
                    update_norm = torch.linalg.vector_norm(update).clamp_min(1e-10)
                    intermediate = parameter - learning_rate * update * parameter_norm / update_norm
                    intermediate_norm = torch.linalg.vector_norm(intermediate).clamp_min(1e-10)
                    parameter.copy_(intermediate * parameter_norm / intermediate_norm)
                else:
                    raise ValueError(f"unknown optimizer algorithm {algorithm!r}")
        return loss


def _embedding_parameter_ids(model: nn.Module) -> set[int]:
    parameter_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            parameter_ids.update(id(parameter) for parameter in module.parameters(recurse=False))
    return parameter_ids


def grouped_parameters(
    model: nn.Module,
    hyperparameters: OptimizerHyperparameters,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assign linear weights to AdamH and embeddings/norms/biases to Adam."""

    embedding_ids = _embedding_parameter_ids(model)
    input_embeddings = getattr(model, "get_input_embeddings", lambda: None)()
    output_embeddings = getattr(model, "get_output_embeddings", lambda: None)()
    if input_embeddings is not None and output_embeddings is not None:
        input_weight = getattr(input_embeddings, "weight", None)
        output_weight = getattr(output_embeddings, "weight", None)
        if input_weight is output_weight and input_weight is not None:
            embedding_ids.add(id(input_weight))

    adamh: list[nn.Parameter] = []
    adam: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if (
                isinstance(module, nn.Linear)
                and name == "weight"
                and id(parameter) not in embedding_ids
            ):
                adamh.append(parameter)
            else:
                adam.append(parameter)

    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if seen != expected:
        raise RuntimeError(
            f"optimizer grouping omitted {len(expected - seen)} trainable parameters"
        )
    if not adamh or not adam:
        raise RuntimeError("expected non-empty AdamH and Adam parameter groups")

    common = {
        "betas": (hyperparameters.beta1, hyperparameters.beta2),
        "eps": hyperparameters.epsilon,
    }
    return (
        {
            "params": adamh,
            "lr": hyperparameters.adamh_learning_rate,
            "algorithm": "adamh",
            "group_name": "linear_weights",
            **common,
        },
        {
            "params": adam,
            "lr": hyperparameters.adam_learning_rate,
            "algorithm": "adam",
            "group_name": "embeddings_norms_biases",
            **common,
        },
    )


def build_optimizer(model: nn.Module, hyperparameters: OptimizerHyperparameters) -> AdamH:
    """Construct the two-group optimizer."""

    return AdamH(grouped_parameters(model, hyperparameters))


def build_wsd_scheduler(
    optimizer: Optimizer,
    *,
    warmup_steps: int,
    cooldown_start_step: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Apply one WSD multiplier to both AdamH and Adam learning rates."""

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: wsd_multiplier(
            step,
            warmup_steps=warmup_steps,
            cooldown_start_step=cooldown_start_step,
            total_steps=total_steps,
        ),
    )
