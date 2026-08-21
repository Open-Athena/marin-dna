from pathlib import Path

import pandas as pd
import pytest
from exp473_center_seeded_projection.trace_flanks import run_analysis


def _psl(trace_id: str, blocks: str, q_starts: str, t_starts: str) -> str:
    count = len([part for part in blocks.split(",") if part])
    return (
        f"{trace_id}\t255\t0\t0\t0\t0\t0\t0\t0\t++\t"
        f"chr5\t1000\t100\t355\tchr7\t2000\t990\t1260\t{count}\t"
        f"{blocks}\t{q_starts}\t{t_starts}\n"
    )


def test_trace_flanks_are_oriented_and_anchor_bootstrapped(tmp_path: Path):
    sample = pd.DataFrame(
        [
            {
                "trace_id": f"trace_{policy}",
                "projection_policy": policy,
                "query_name": "anchor",
                "source_chrom": "chr7",
                "source_start": 1000,
                "source_end": 1255,
                "region_label": "cds",
                "species": "Mus musculus",
                "alignment_name": "Mus_musculus",
                "clade": "mammals",
                "t_chrom": "chr5",
                "t_start": 100,
                "t_end": 355,
                "t_strand": "+",
                "fragment_count": 1,
            }
            for policy in ("full_window", "center_1")
        ]
    )
    sample_path = tmp_path / "sample.tsv"
    sample.to_csv(sample_path, sep="\t", index=False)
    raw = tmp_path / "raw"
    for policy in ("full_window", "center_1"):
        (raw / policy).mkdir(parents=True)
    (raw / "full_window" / "Mus_musculus.psl").write_text(
        _psl("trace_full_window", "100,120,", "110,220,", "990,1140,")
    )
    (raw / "center_1" / "Mus_musculus.psl").write_text(
        _psl("trace_center_1", "255,", "100,", "1000,")
    )
    output = tmp_path / "output"
    run_analysis(
        sample_path,
        raw,
        output,
        analysis_commit="a" * 40,
        n_bootstrap=20,
        seed=473,
    )
    metrics = pd.read_parquet(output / "metrics.parquet").set_index(
        "projection_policy"
    )
    full = metrics.loc["full_window"]
    assert full["aligned_to_anchor_bases"] == 205
    assert full["left_flank"] == 20
    assert full["right_flank"] == 20
    assert full["internal_unaligned"] == 10
    assert bool(full["center_base_aligned"])
    center = metrics.loc["center_1"]
    assert center["left_flank"] == 0
    assert center["right_flank"] == 0
    deltas = pd.read_parquet(output / "paired_deltas.parquet").set_index("metric")
    assert deltas.loc["left_flank", "delta_center_minus_full"] == -20
    assert deltas.loc["right_flank", "delta_center_minus_full"] == -20
    assert deltas.loc["internal_unaligned", "delta_center_minus_full"] == -10
    assert deltas["n_anchors"].tolist() == [1] * 4
    assert "0-based half-open" in (output / "manifest.json").read_text()


def test_trace_flanks_reject_nonimmutable_identity(tmp_path: Path):
    with pytest.raises(AssertionError):
        run_analysis(
            tmp_path / "missing.tsv",
            tmp_path / "raw",
            tmp_path / "output",
            analysis_commit="short",
            n_bootstrap=20,
            seed=473,
        )


def test_trace_flank_launcher_is_pinned_and_additive():
    launcher = (Path(__file__).parents[1] / "sky" / "trace_flanks.yaml").read_text()
    assert "d0e5380a46cd66d4c42d763b3c42da1150c92073" in launcher
    assert "bf8367c285f955407cfb2dba6102661b2e528261b64fe52095d52a688cd6d039" in launcher
    assert "-eq 214" in launcher
    assert "trace_flanks_v1" in launcher
    assert "image_id: ami-0324f0ad73bdcd087" in launcher
    run = launcher.split("run: |", maxsplit=1)[1]
    assert " snakemake " not in run
    assert "--snakefile" not in run
