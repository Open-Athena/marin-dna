from pathlib import Path

import polars as pl
import yaml
from marin_dna_linclust_conservation.provenance import configuration_sha256

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


def test_panel20_linclust_uses_no_split_scalable_recipe() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "config/panel20_linclust.yaml").read_text())
    recipe = config["panel_linclust"]

    assert config["mmseqs"]["min_sequence_identity"] == 0.5
    assert config["mmseqs"]["coverage"] == 0.8
    assert config["mmseqs"]["spaced_kmer_mode"] == 0
    assert recipe["min_sequence_identity"] == 0.4
    assert recipe["coverage"] == 0.7
    assert recipe["spaced_kmer_mode"] == 1
    assert recipe["kmers_per_sequence"] == 64
    assert recipe["kmer_per_sequence_scale"] == 0.0
    assert recipe["kmer_length"] == 0
    assert recipe["hash_shift"] == 1
    assert recipe["split_memory_limit"] == "400G"


def test_panel20_linclust_rule_uses_panel_recipe_without_release_gate() -> None:
    snakefile = (PROJECT_ROOT / "workflow/Snakefile").read_text()
    rule = snakefile.split("rule run_panel_linclust:", 1)[1].split(
        "rule panel_linclust:", 1
    )[0]

    for field in (
        "min_sequence_identity",
        "coverage",
        "coverage_mode",
        "evalue",
        "spaced_kmer_mode",
        "mask_lower_case",
        "low_complexity_masking",
        "cluster_mode",
    ):
        assert f'PANEL_LINCLUST_CONFIG["{field}"]' in rule
    assert "release_gate" not in rule


def test_panel20_sky_task_reuses_tiles_under_current_config_hash() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "config/panel20_linclust.yaml").read_text())
    sky_task = (PROJECT_ROOT / "sky/panel20_linclust.yaml").read_text()

    assert configuration_sha256(config) in sky_task
    assert "/runs/panel20-linclust-m64/" in sky_task
    assert "linclust-conservation-copy-s3-prefix" in sky_task
