"""Importable optimizer registration shared by the CLI and dispatched worker."""

from dataclasses import dataclass
from typing import Any

from levanter.optim.adamh import AdamHConfig
from levanter.optim.config import OptimizerConfig

from marin_dna_exp550.recipe import TRAIN_UPDATES


@OptimizerConfig.register_subclass("exp550_fixed_horizon_adamh")
@dataclass(frozen=True)
class FixedHorizonAdamH(AdamHConfig):
    def build(self, num_train_steps: int) -> Any:
        # A short pilot follows the identical first updates of the 100k schedule.
        del num_train_steps
        return super().build(TRAIN_UPDATES)
