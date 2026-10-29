from types import SimpleNamespace

import torch

from exp586_moe.inference.benchmark_vllm import _install_capture


class _FakeVllmModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm_head = torch.nn.Linear(3, 8, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


def test_capture_can_remove_unused_sampling_head() -> None:
    model = _FakeVllmModel()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))

    assert _install_capture(worker, [0, 1, 2, 3], "acgt_head", "float32", True)

    hidden_states = torch.randn(8192, 3)
    returned = model(hidden_states)

    assert returned is hidden_states
    assert model._exp586_captures[0].shape == (1, 8192, 4)
    sampling_logits = model.compute_logits(hidden_states[:2])
    assert sampling_logits.shape == (2, 1)
    assert torch.count_nonzero(sampling_logits) == 0
