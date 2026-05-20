"""Smoke tests for the HF dataset-card generator."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from marin_dna.pipelines.evals import hf_readme


SHA = "deadbeef1234567890abcdef1234567890abcdef"


def _matched_train_test(tmp_path: Path) -> tuple[Path, Path]:
    train = pl.DataFrame(
        {
            "chrom": ["1"] * 4,
            "pos": [10, 20, 30, 40],
            "ref": ["A"] * 4,
            "alt": ["T"] * 4,
            "label": [True, False, False, False],
            "subset": ["missense_variant"] * 4,
            "match_group": [0, 0, 0, 0],
        }
    )
    test = pl.DataFrame(
        {
            "chrom": ["2"] * 4,
            "pos": [10, 20, 30, 40],
            "ref": ["A"] * 4,
            "alt": ["T"] * 4,
            "label": [True, False, False, False],
            "subset": ["missense_variant"] * 4,
            "match_group": [0, 0, 0, 0],
        }
    )
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    train.write_parquet(train_path)
    test.write_parquet(test_path)
    return train_path, test_path


def _qc(tmp_path: Path, *, with_maf: bool = False) -> Path:
    cols = {
        "subset": ["missense_variant", "splicing"],
        "n_positives_input": [1500, 700],
        "n_positives_kept": [1000, 550],
        "n_dropped": [500, 150],
        "frac_dropped": [0.333, 0.214],
        "baseline_auprc": [0.1, 0.1],
        "distance_tss_pc_auprc": [0.101, 0.100],
        "distance_tss_pc_auprc_sign": [1, -1],
        "distance_tss_nc_auprc": [0.102, 0.110],
        "distance_tss_nc_auprc_sign": [1, 1],
        "distance_exon_pc_auprc": [0.103, 0.105],
        "distance_exon_pc_auprc_sign": [-1, -1],
        "distance_exon_nc_auprc": [0.104, 0.115],
        "distance_exon_nc_auprc_sign": [1, -1],
    }
    if with_maf:
        cols["MAF_auprc"] = [0.108, 0.122]
        cols["MAF_auprc_sign"] = [-1, 1]
    path = tmp_path / "qc.parquet"
    pl.DataFrame(cols).write_parquet(path)
    return path


class TestRender:
    def test_mendelian_renders_with_expected_sections(self, tmp_path: Path) -> None:
        train, test = _matched_train_test(tmp_path)
        qc = _qc(tmp_path)
        md = hf_readme.render("mendelian_traits", SHA, train, test, qc)
        # Frontmatter + minimal tag set.
        assert md.startswith("---\n")
        assert "biology" in md
        assert "genomics" in md
        assert "dna" in md
        # Section headers.
        for header in (
            "# evals_mendelian_traits",
            "## Description",
            "## Splits",
            "## Columns",
            "## Per-subset retention",
            "## Matching design",
            "## Matched-feature AUPRC diagnostic",
            "## Provenance",
            "## Citation",
        ):
            assert header in md, f"missing header: {header}"
        # SHA pin appears.
        assert SHA[:7] in md
        assert SHA in md  # Full SHA in the tree/blob links.
        # Retention table includes the input + kept counts.
        assert "1,500" in md
        assert "1,000" in md
        # Companion harness dataset link.
        assert "evals_mendelian_traits_harness_255" in md

    def test_complex_renders_with_expected_sections(self, tmp_path: Path) -> None:
        train, test = _matched_train_test(tmp_path)
        qc = _qc(tmp_path, with_maf=True)
        md = hf_readme.render("complex_traits", SHA, train, test, qc)
        assert md.startswith("---\n")
        for tag in ("biology", "genomics", "dna", "complex-traits", "fine-mapping"):
            assert tag in md, f"missing tag: {tag}"
        for header in (
            "# evals_complex_traits",
            "## Description",
            "## Splits",
            "## Matching design",
            "## Matched-feature AUPRC diagnostic",
            "## Provenance",
            "## License",
        ):
            assert header in md
        # MAF appears in the AUPRC table.
        assert "MAF" in md
        assert SHA[:7] in md

    def test_harness_renders_with_expected_sections(self, tmp_path: Path) -> None:
        # Harness has 2 rows per variant.
        df = pl.DataFrame(
            {
                "chrom": ["1"] * 8,
                "pos": [10, 10, 20, 20, 30, 30, 40, 40],
                "ref": ["A"] * 8,
                "alt": ["T"] * 8,
                "strand": ["+", "-"] * 4,
                "target": [True, True, False, False, False, False, False, False],
            }
        )
        train_path = tmp_path / "train.parquet"
        test_path = tmp_path / "test.parquet"
        df.write_parquet(train_path)
        df.write_parquet(test_path)
        md = hf_readme.render(
            "mendelian_traits_harness_255", SHA, train_path, test_path
        )
        assert md.startswith("---\n")
        for header in (
            "# evals_mendelian_traits_harness_255",
            "## Why 255 bp",
            "## Why two rows per variant",
            "## Splits",
            "## Eval-harness columns",
            "## Provenance",
        ):
            assert header in md
        # Variant count is rows // 2.
        assert "4" in md  # 8 rows / 2 = 4 variants per split
        # Window-size math: 127 / 128.
        assert "127" in md
        assert "128" in md

    def test_unknown_dataset_raises(self, tmp_path: Path) -> None:
        train, test = _matched_train_test(tmp_path)
        with pytest.raises(ValueError, match="no README template"):
            hf_readme.render("not_a_real_dataset", SHA, train, test)

    def test_retention_table_handles_zero_input(self, tmp_path: Path) -> None:
        path = tmp_path / "qc.parquet"
        pl.DataFrame(
            {
                "subset": ["s1", "s2"],
                "n_positives_input": [0, 100],
                "n_positives_kept": [0, 90],
                "n_dropped": [0, 10],
                "frac_dropped": [None, 0.1],
                "baseline_auprc": [None, 0.1],
                "distance_tss_pc_auprc": [None, 0.11],
                "distance_tss_pc_auprc_sign": [None, 1],
                "distance_tss_nc_auprc": [None, 0.11],
                "distance_tss_nc_auprc_sign": [None, 1],
                "distance_exon_pc_auprc": [None, 0.11],
                "distance_exon_pc_auprc_sign": [None, 1],
                "distance_exon_nc_auprc": [None, 0.11],
                "distance_exon_nc_auprc_sign": [None, 1],
            }
        ).write_parquet(path)
        train, test = _matched_train_test(tmp_path)
        # Should not crash on the zero-input subset.
        md = hf_readme.render("mendelian_traits", SHA, train, test, path)
        # Em-dash retention for the zero-input row.
        assert "—" in md
