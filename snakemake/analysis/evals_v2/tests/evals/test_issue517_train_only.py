"""The order-control loader must not discover or prepare held-out data files."""

from types import SimpleNamespace

import pandas as pd
import pytest

from marin_dna_evals import issue517_train_only as loader


@pytest.mark.parametrize("repo_id,revision", list(loader.REVISIONS.items()))
def test_only_the_pinned_train_file_is_requested(monkeypatch, repo_id, revision):
    calls = []
    expected = pd.DataFrame({"chrom": ["chr1", "3", "chrX"]})

    def download(**kwargs):
        calls.append(kwargs)
        return "/fixture/train.parquet"

    def load(path, **kwargs):
        assert path == "parquet"
        assert kwargs == {
            "data_files": {"train": "/fixture/train.parquet"},
            "split": "train",
        }
        return SimpleNamespace(to_pandas=lambda: expected)

    monkeypatch.setattr(loader, "hf_hub_download", download)
    monkeypatch.setattr(loader, "load_dataset", load)
    result = loader.load_order_development_frame(repo_id, revision, "train")
    assert result is expected
    assert calls == [{
        "repo_id": repo_id,
        "repo_type": "dataset",
        "filename": "train.parquet",
        "revision": revision,
    }]


@pytest.mark.parametrize("split,revision", [("test", "pinned"), ("train", "main")])
def test_unapproved_scope_fails_before_network(monkeypatch, split, revision):
    def unexpected(**kwargs):
        pytest.fail("network access before scope validation")

    monkeypatch.setattr(loader, "hf_hub_download", unexpected)
    repo_id = next(iter(loader.REVISIONS))
    revision = loader.REVISIONS[repo_id] if revision == "pinned" else revision
    with pytest.raises(ValueError, match="pinned"):
        loader.load_order_development_frame(repo_id, revision, split)


def test_chromosome_boundary_is_checked(monkeypatch):
    monkeypatch.setattr(loader, "hf_hub_download", lambda **kwargs: "/train.parquet")
    monkeypatch.setattr(
        loader, "load_dataset",
        lambda *args, **kwargs: SimpleNamespace(
            to_pandas=lambda: pd.DataFrame({"chrom": ["chr2"]})
        ),
    )
    repo_id, revision = next(iter(loader.REVISIONS.items()))
    with pytest.raises(ValueError, match="non-development"):
        loader.load_order_development_frame(repo_id, revision, "train")
