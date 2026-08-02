from __future__ import annotations

from pathlib import Path

import polars as pl

from analyze_feature1662_saturation import plot_profile


def test_plot_profile_emits_svg_and_png(tmp_path: Path) -> None:
    rows = []
    for orientation in ("forward", "reverse_complement"):
        for codon_position in (1, 2, 3):
            for offset in (-1, 0, 1):
                rows.append(
                    {
                        "orientation": orientation,
                        "focal_codon_position": codon_position,
                        "transcript_offset": offset,
                        "mean_abs_delta": 1.0 + codon_position / 10 + abs(offset),
                        "se_abs_delta": 0.1,
                        "mean_delta": codon_position / 10 + offset / 10,
                        "se_delta": 0.05,
                    }
                )
    profiles = pl.DataFrame(rows)
    outputs = plot_profile(profiles, output_dir=tmp_path, signed=False)
    outputs.extend(plot_profile(profiles, output_dir=tmp_path, signed=True))
    assert {path.suffix for path in outputs} == {".svg", ".png"}
    assert len(outputs) == 4
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
