from pathlib import Path

import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).parents[1]


def test_panel20_linclust_config_is_exact_full_panel() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "config/panel20_linclust.yaml").read_text())
    selection = pl.read_csv(PROJECT_ROOT / config["selection_path"], separator="\t")

    assert selection.height == 20
    assert selection["accession"].n_unique() == 20
    assert selection["order"].n_unique() == 20
    assert selection["selected"].all()
    assert config["smoke"]["candidates_per_assembly"] == "all"
    assert config["window"] == {
        "length": 255,
        "stride": 128,
        "max_repeat_fraction": 0.5,
    }


def test_panel20_linclust_uses_measured_best_scalable_recipe() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "config/panel20_linclust.yaml").read_text())
    recipe = config["panel_linclust"]

    assert config["mmseqs"]["min_sequence_identity"] == 0.4
    assert config["mmseqs"]["coverage"] == 0.7
    assert config["mmseqs"]["spaced_kmer_mode"] == 1
    assert recipe["kmers_per_sequence"] == 148
    assert recipe["kmer_per_sequence_scale"] == 0.0
    assert recipe["kmer_length"] == 0
    assert recipe["hash_shift"] == 1
    assert recipe["split_memory_limit"] == "1400G"
