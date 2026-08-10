"""Tests for the ``genomes-v*`` HF dataset-card generators."""

import re
from pathlib import Path

import pytest
import yaml

from marin_dna.pipelines.training_dataset.hf_readme import (
    GENOME_SET_BLURBS,
    GENOME_SET_TITLES,
    RECIPE_BLURBS,
    build_training_readme,
    build_validation_readme,
    count_parquet_rows,
)

_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "snakemake/training_dataset/dataset_creation/config/config.yaml"
)

SHA = "0123456789abcdef0123456789abcdef01234567"  # 40 chars
PREFIX = "bolinas-dna/genomes-v5"


def _training(**overrides: object) -> str:
    kwargs: dict[str, object] = {
        "genome_set": "animals",
        "recipe": "v5",
        "window": 255,
        "stride": 128,
        "hf_prefix": PREFIX,
        "commit_sha": SHA,
        "n_genomes": 499,
        "n_samples": 12_345_678,
    }
    kwargs.update(overrides)
    return build_training_readme(**kwargs)  # type: ignore[arg-type]


def _validation(**overrides: object) -> str:
    kwargs: dict[str, object] = {
        "recipe": "v1",
        "window": 255,
        "stride": 255,
        "hf_prefix": PREFIX,
        "commit_sha": SHA,
        "n_samples": 15_000,
        "phylop_threshold": 2.27,
        "max_samples": 16384,
        "seed": 42,
    }
    kwargs.update(overrides)
    return build_validation_readme(**kwargs)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------
# Training cards
# ----------------------------------------------------------------------------


def test_training_front_matter_minimal_tags() -> None:
    md = _training()
    head = md.split("---", 2)[1]
    assert "- biology" in head
    assert "- genomics" in head
    assert "- DNA" in head
    # Per project guidance: no fine-grained extra tags on bolinas-dna/* cards.
    assert "conservation" not in head
    assert "validation" not in head


def test_training_repo_name_and_title() -> None:
    md = _training(genome_set="mammals", recipe="v15", window=255, stride=128)
    assert "# `bolinas-dna/genomes-v5-genome_set-mammals-intervals-v15_255_128`" in md


def test_training_includes_genome_set_blurb_and_count() -> None:
    md = _training(genome_set="animals", n_genomes=499)
    assert "Metazoa" in md
    assert "499 genomes" in md


def test_training_includes_recipe_blurb() -> None:
    md = _training(recipe="v5")
    assert RECIPE_BLURBS["v5"] in md


def test_training_soft_mask_encoding_called_out() -> None:
    md = _training()
    # Training case == repeats, and explicitly *not* conservation.
    assert "soft-masking from the source assembly" in md
    assert "repeat-masked base" in md
    assert "not** a conservation encoding" in md


def test_training_reverse_complement_note() -> None:
    md = _training(add_rc=True)
    assert "_+" in md and "_-" in md
    assert "forward-vs-RC" in md
    no_rc = _training(add_rc=False)
    assert "were **not** added" in no_rc


def test_training_construction_numbering_is_contiguous() -> None:
    for add_rc in (True, False):
        md = _training(add_rc=add_rc)
        body = md.split("## Construction", 1)[1].split("This is a **train-only**", 1)[0]
        nums = [int(m) for m in re.findall(r"^(\d+)\.", body, flags=re.MULTILINE)]
        assert nums == list(range(1, len(nums) + 1)), (add_rc, nums)
    # add_rc adds exactly one extra step.
    assert _training(add_rc=True).count(". ") > 0


def test_training_singular_genome_word_for_humans() -> None:
    md = _training(genome_set="humans", recipe="v1", n_genomes=1)
    assert "1 genome)" in md
    assert "1 genomes" not in md


def test_training_matched_validation_link_present_when_recipe_has_one() -> None:
    md = _training(recipe="v5")
    assert "genomes-v5-validation-intervals-v5_255_255" in md
    assert "huggingface.co/datasets/" in md


def test_training_no_validation_link_for_recipe_without_one() -> None:
    md = _training(genome_set="mammals_seg20", recipe="v31")
    assert "No matched validation repo" in md


def test_training_commit_pinned_permalink() -> None:
    md = _training()
    assert f"/tree/{SHA}/snakemake/training_dataset" in md
    assert SHA[:12] in md


def test_training_unknown_recipe_raises() -> None:
    with pytest.raises(ValueError, match="unknown recipe"):
        _training(recipe="v999")


def test_training_unknown_genome_set_raises() -> None:
    with pytest.raises(ValueError, match="unknown genome_set"):
        _training(genome_set="plants")


# ----------------------------------------------------------------------------
# Validation cards
# ----------------------------------------------------------------------------


def test_validation_repo_name() -> None:
    md = _validation(recipe="v5", window=255, stride=255)
    assert "# `bolinas-dna/genomes-v5-validation-intervals-v5_255_255`" in md


def test_validation_conservation_encoding_called_out() -> None:
    md = _validation()
    assert "conservation-encoded" in md
    assert "phyloP-241way" in md
    assert "2.27" in md
    # opposite convention to the training sets
    assert "opposite" in md


def test_validation_includes_recipe_blurb_and_training_glob() -> None:
    md = _validation(recipe="v1")
    assert RECIPE_BLURBS["v1"] in md
    assert "genomes-v5-genome_set-*-intervals-v1_255_128" in md


def test_validation_reports_subsample_params() -> None:
    md = _validation(max_samples=16384, seed=42)
    assert "16384" in md
    assert "seed 42" in md


def test_validation_commit_pinned_permalink() -> None:
    md = _validation()
    assert f"/tree/{SHA}/snakemake/training_dataset" in md


def test_validation_unknown_recipe_raises() -> None:
    with pytest.raises(ValueError, match="unknown recipe"):
        _validation(recipe="nope")


# ----------------------------------------------------------------------------
# Sample count
# ----------------------------------------------------------------------------


def test_training_size_section_exact_count_and_hf_caveat() -> None:
    md = _training(n_samples=12_345_678)
    assert "## Size" in md
    assert "12,345,678 sequences" in md  # thousands-separated
    assert "auto-generated row count is frequently wrong" in md


def test_training_no_marindna_branding() -> None:
    # These datasets are general-purpose; the card must not imply they are
    # only for MarinDNA.
    assert "MarinDNA" not in _training()


def test_validation_size_section() -> None:
    md = _validation(n_samples=15_000, max_samples=16384)
    assert "## Size" in md
    assert "15,000 sequences" in md
    assert "16,384 human windows" in md


def test_count_parquet_rows(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import polars as pl

    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    pl.DataFrame({"id": ["x"] * 3, "seq": ["A"] * 3}).write_parquet(a)
    pl.DataFrame({"id": ["y"] * 5, "seq": ["C"] * 5}).write_parquet(b)
    assert count_parquet_rows([str(a)]) == 3
    assert count_parquet_rows([str(a), str(b)]) == 8


# ----------------------------------------------------------------------------
# Genome-set coverage (every config genome_set must have a card blurb)
# ----------------------------------------------------------------------------


def test_genome_set_titles_and_blurbs_have_matching_keys() -> None:
    assert set(GENOME_SET_TITLES) == set(GENOME_SET_BLURBS)


def test_all_config_genome_sets_have_blurbs() -> None:
    # Regression guard: `rule all` builds a training_readme target for *every*
    # genome_set in the config, and build_training_readme raises ValueError for
    # any without a blurb — so a config genome_set lacking one crashes the
    # pipeline (this is how `human_mouse` slipped through).
    cfg = yaml.safe_load(_CONFIG.read_text())
    names = [g["name"] for g in cfg["genome_sets"]]
    missing = [n for n in names if n not in GENOME_SET_BLURBS]
    assert not missing, f"config genome_sets missing a card blurb: {missing}"


def test_training_renders_for_human_mouse() -> None:
    md = _training(genome_set="human_mouse", recipe="v5", n_genomes=2)
    assert "Human + mouse" in md
    assert "GCF_000001635.27" in md  # mouse accession present in blurb
