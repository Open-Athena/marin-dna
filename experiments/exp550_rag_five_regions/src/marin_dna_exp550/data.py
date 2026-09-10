"""Preserve repeated draws around the pinned reader's repeated-row-zero bug."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

from haliax import Axis
from levanter.data.dataset import AsyncDataset
from levanter.data.text.datasets import LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.store.cache import TreeCache

T = TypeVar("T")


class DistinctReadDataset(AsyncDataset[T]):
    """Read each requested row once, then restore every draw in its original order.

    The pinned JaggedArrayStore repairs only the last repeated zero-index read.
    This adapter runs before shuffling/mixing and preserves sampling multiplicity.
    """

    def __init__(self, dataset: AsyncDataset[T]) -> None:
        self.dataset = dataset

    async def async_len(self) -> int:
        return await self.dataset.async_len()

    def is_finite(self) -> bool:
        return self.dataset.is_finite()

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T]:
        if len(indices) == 0:
            return []
        unique = list(dict.fromkeys(int(index) for index in indices))
        rows = await self.dataset.get_batch(unique)
        by_index = dict(zip(unique, rows, strict=True))
        return [by_index[int(index)] for index in indices]


@dataclass(frozen=True)
class RagDataConfig(LmDataConfig):
    def build_token_datasets(
        self, caches: Mapping[str, TreeCache[dict]], Pos: Axis, *, split: str
    ) -> dict[str, AsyncDataset[GrugLmExample]]:
        datasets = super().build_token_datasets(caches, Pos, split=split)
        return {
            name: DistinctReadDataset(dataset) for name, dataset in datasets.items()
        }
