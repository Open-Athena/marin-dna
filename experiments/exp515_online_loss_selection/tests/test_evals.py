"""Tests for evaluation dataset loading."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from biofoundation.model.adapters.hf import HFTokenizer
from datasets import Dataset
from transformers import AutoTokenizer

from glm_experiments.data.evals import (
    EVAL_FILTERS,
    download_genome,
    filter_traitgym_promoter,
    load_eval_dataset,
)


class TestHFTokenizer:
    """Tests for biofoundation HFTokenizer adapter."""

    @pytest.fixture
    def hf_tokenizer(self):
        """Load HuggingFace tokenizer for testing."""
        return AutoTokenizer.from_pretrained("gonzalobenegas/tokenizer-dna-mlm")  # nosec B615

    def test_encode_returns_list_of_ints(self, hf_tokenizer):
        """Test that encode returns a list of integers."""
        adapter = HFTokenizer(hf_tokenizer)
        result = adapter.encode("ATGC")

        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)

    def test_encode_produces_tokens(self, hf_tokenizer):
        """Test that encode produces tokens for DNA sequence."""
        adapter = HFTokenizer(hf_tokenizer)

        # Encode a simple sequence
        result = adapter.encode("ATGC")

        # Should produce tokens
        assert len(result) > 0

    def test_mask_token_id(self, hf_tokenizer):
        """Test that mask_token_id is accessible."""
        adapter = HFTokenizer(hf_tokenizer)

        assert hasattr(adapter, "mask_token_id")
        assert isinstance(adapter.mask_token_id, int)
        assert adapter.mask_token_id == hf_tokenizer.mask_token_id


class TestEvalFilters:
    """Tests for evaluation dataset filter functions."""

    def test_filter_registry_contains_expected_filters(self):
        """Test that EVAL_FILTERS registry contains expected filter names."""
        assert "traitgym_promoter" in EVAL_FILTERS
        assert "none" in EVAL_FILTERS
        assert callable(EVAL_FILTERS["traitgym_promoter"])
        assert callable(EVAL_FILTERS["none"])

    def test_none_filter_is_identity(self):
        """Test that 'none' filter returns dataset unchanged."""

        # Create a simple test dataset
        data = {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
        dataset = Dataset.from_dict(data)

        filtered = EVAL_FILTERS["none"](dataset)

        assert filtered is dataset  # Should be same object (identity)
        assert len(filtered) == len(dataset)


class TestDownloadGenome:
    """Tests for download_genome function."""

    def test_download_genome_auto_derives_path_from_url(self):
        """Test that download_genome auto-derives path from URL basename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock urlretrieve to create a dummy file
            with patch("urllib.request.urlretrieve") as mock_retrieve:

                def create_file(url, path):
                    Path(path).touch()

                mock_retrieve.side_effect = create_file

                result = download_genome("http://example.com/test_genome.fa.gz", tmpdir)

                # Check that path is auto-derived from URL
                expected_path = Path(tmpdir) / "test_genome.fa.gz"
                assert result == expected_path
                mock_retrieve.assert_called_once_with(
                    "http://example.com/test_genome.fa.gz", expected_path
                )

    def test_download_genome_skips_if_exists(self):
        """Test that download_genome skips download if file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing file
            existing_file = Path(tmpdir) / "existing_genome.fa.gz"
            existing_file.touch()

            with patch("urllib.request.urlretrieve") as mock_retrieve:
                result = download_genome("http://example.com/existing_genome.fa.gz", tmpdir)

                assert result == existing_file
                mock_retrieve.assert_not_called()

    def test_download_genome_creates_parent_dirs(self):
        """Test that download_genome creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "dir"

            with patch("urllib.request.urlretrieve") as mock_retrieve:

                def create_file(url, path):
                    Path(path).touch()

                mock_retrieve.side_effect = create_file

                result = download_genome("http://example.com/genome.fa.gz", nested_dir)

                assert nested_dir.exists()
                assert result == nested_dir / "genome.fa.gz"


class TestLoadTraitgymDataset:
    """Tests for loading TraitGym dataset with load_eval_dataset."""

    @pytest.fixture
    def tokenizer_adapter(self):
        """Create a tokenizer adapter for testing."""
        hf_tokenizer = AutoTokenizer.from_pretrained(
            "gonzalobenegas/tokenizer-dna-mlm"
        )  # nosec B615
        return HFTokenizer(hf_tokenizer)

    def test_load_traitgym_dataset_loads_raw_data(self):
        """Test that TraitGym dataset can be loaded from HuggingFace."""
        from datasets import load_dataset

        # Just test that the raw dataset loads correctly
        dataset = load_dataset("songlab/TraitGym", "mendelian_traits", split="test")  # nosec B615

        # Check expected columns exist
        assert "chrom" in dataset.column_names
        assert "pos" in dataset.column_names
        assert "ref" in dataset.column_names
        assert "alt" in dataset.column_names
        assert "label" in dataset.column_names

        # Check we have data
        assert len(dataset) > 0


class TestDNADataModuleWithTraitGymMendelianPromoter:
    """Tests for DNADataModule with TraitGym Mendelian Promoter evaluation."""

    def test_datamodule_accepts_evals_config(self):
        """Test that DNADataModule accepts evals config parameter."""
        from glm_experiments.data.lm_datamodule import MLMDataModule

        evals_config = {
            "traitgym_mendelian_promoter": {
                "dataset_name": "songlab/TraitGym",
                "dataset_config": "mendelian_traits",
                "genome_url": "http://example.com/genome.fa.gz",
                "genome_path": "data/genome.fa.gz",
                "window_size": 512,
                "batch_size": 128,
            }
        }

        dm = MLMDataModule(evals=evals_config)

        assert dm.hparams.evals == evals_config

    def test_datamodule_evals_none_by_default(self):
        """Test that evals is None by default."""
        from glm_experiments.data.lm_datamodule import MLMDataModule

        dm = MLMDataModule()

        assert dm.hparams.evals is None
        assert dm.eval_datasets == {}

    def test_val_dataloader_returns_single_loader_without_evals(self):
        """Test that val_dataloader returns single DataLoader without evals."""
        from torch.utils.data import DataLoader

        from glm_experiments.data.lm_datamodule import MLMDataModule

        dm = MLMDataModule(
            batch_size=32,
            num_workers=0,
            evals=None,
        )

        # Mock the data_val attribute
        dm.data_val = [{"input_ids": torch.zeros(10), "labels": torch.zeros(10)}]
        dm.eval_datasets = {}
        dm.batch_size_per_device = 32

        result = dm.val_dataloader()

        assert isinstance(result, DataLoader)

    def test_val_dataloader_returns_list_with_evals(self):
        """Test that val_dataloader returns list of DataLoaders with evals."""
        from torch.utils.data import DataLoader

        from glm_experiments.data.lm_datamodule import MLMDataModule

        evals_config = {
            "traitgym_mendelian_promoter": {
                "batch_size": 128,
            }
        }

        dm = MLMDataModule(
            batch_size=32,
            num_workers=0,
            evals=evals_config,
        )

        # Mock the data attributes
        dm.data_val = [{"input_ids": torch.zeros(10), "labels": torch.zeros(10)}]
        dm.eval_datasets = {
            "traitgym_mendelian_promoter": [
                {"input_ids": torch.zeros(512), "pos": 256, "ref": 1, "alt": 2, "label": 0}
            ]
        }
        dm.batch_size_per_device = 32

        result = dm.val_dataloader()

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(dl, DataLoader) for dl in result)

    def test_hydra_instantiation_with_evals(self):
        """Test that DataModule can be instantiated from Hydra config with evals."""
        from hydra import compose, initialize

        from glm_experiments.data.lm_datamodule import MLMDataModule

        with initialize(version_base="1.3", config_path="../configs"):
            cfg = compose(
                config_name="train.yaml",
                overrides=["data=default"],
            )

            # Instantiate datamodule
            import hydra

            dm = hydra.utils.instantiate(cfg.data)

            # Check that it's the right type
            assert isinstance(dm, MLMDataModule)

            # Check evals config is present
            assert dm.hparams.evals is not None
            assert "traitgym_mendelian_promoter" in dm.hparams.evals
            assert (
                dm.hparams.evals["traitgym_mendelian_promoter"]["dataset_name"]
                == "songlab/TraitGym"
            )
            assert dm.hparams.evals["traitgym_mendelian_promoter"]["window_size"] == 512
