from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import datasets
import pytest

from exp479_mntp import data
from exp479_mntp.config import DATA_COMPONENTS
from exp479_mntp.data import _stream_component, build_sequence_plan


class FakeDataset:
    def shuffle(self, *, seed: int, buffer_size: int) -> FakeDataset:
        assert seed == 7
        assert buffer_size == 11
        return self

    def __iter__(self) -> Iterator[dict[str, str]]:
        return iter([{"seq": "A" * 255}])


class TargetlessThenValidDataset(FakeDataset):
    def __iter__(self) -> Iterator[dict[str, str]]:
        return iter([{"seq": "N" * 255}, {"seq": "A" * 255}])


def test_stream_component_forwards_the_registered_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_load_dataset(
        repo: str,
        *,
        split: str,
        revision: str,
        streaming: bool,
    ) -> FakeDataset:
        observed.update(
            repo=repo,
            split=split,
            revision=revision,
            streaming=streaming,
        )
        return FakeDataset()

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    sequence = next(
        _stream_component(
            repo="org/repo",
            revision="abc",
            text_key="seq",
            split="validation",
            seed=7,
            shuffle_buffer_size=11,
        )
    )
    assert sequence == "A" * 255
    assert observed == {
        "repo": "org/repo",
        "split": "validation",
        "revision": "abc",
        "streaming": True,
    }


def test_stream_component_skips_sequences_without_eligible_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load_dataset(
        repo: str,
        *,
        split: str,
        revision: str,
        streaming: bool,
    ) -> TargetlessThenValidDataset:
        del repo, split, revision, streaming
        return TargetlessThenValidDataset()

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    sequence = next(
        _stream_component(
            repo="org/repo",
            revision="abc",
            text_key="seq",
            split="validation",
            seed=7,
            shuffle_buffer_size=11,
        )
    )
    assert sequence == "A" * 255


@pytest.mark.parametrize(("validation", "expected_split"), [(False, "train"), (True, "validation")])
def test_sequence_plan_selects_the_registered_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validation: bool,
    expected_split: str,
) -> None:
    observed: list[str] = []

    def fake_stream_component(**kwargs: object) -> Iterator[str]:
        observed.append(str(kwargs["split"]))
        return iter(["A" * 255])

    monkeypatch.setattr(data, "_stream_component", fake_stream_component)
    build_sequence_plan(
        tmp_path / f"{expected_split}.jsonl",
        samples_per_component=1,
        seed=0,
        validation=validation,
    )
    assert observed == [expected_split] * 5


def test_pinned_component_text_keys_match_observed_schemas() -> None:
    observed = {
        component.name: (component.train_text_key, component.validation_text_key)
        for component in DATA_COMPONENTS
    }
    assert observed == {
        "cds": ("seq", "seq"),
        "upstream": ("seq", "seq"),
        "downstream": ("seq", "seq"),
        "enhancer": ("sequence", "seq"),
        "ncrna": ("sequence", "seq"),
    }
