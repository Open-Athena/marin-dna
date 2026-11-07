import numpy as np
import pytest

from exp593_dna.diagnostics import NormPolicy, collect_norms
from exp593_dna.validation import (
    loss_views,
    select_disjoint_validation,
    verify_disjoint_metadata,
)


def test_norms_include_all_leaves_once_and_selected_stacked_layers():
    arrays = {
        ".token_embed": np.array([[3.0, 4.0]], dtype=np.float32),
        ".stacked_blocks.stacked.attn.w_q": np.arange(
            8 * 2 * 3, dtype=np.float32
        ).reshape(8, 2, 3),
        ".stacked_blocks.stacked.mlp.router": np.ones((8, 2, 4), dtype=np.float32),
        ".stacked_blocks.stacked.mlp.expert_mlp.w_down": np.ones(
            (8, 3, 2, 2), dtype=np.float32
        ),
    }
    grads = {k: v * 0.25 for k, v in arrays.items()}
    result = collect_norms(arrays, grads)
    expected = np.linalg.norm(np.concatenate([v.ravel() for v in arrays.values()]))
    assert np.isclose(result["norms/param/overall"], expected)
    assert np.isclose(result["norms/grad/overall"], expected * 0.25)
    assert len(result) == 20
    assert np.isclose(
        result["norms/param/layer_4/attention_q"],
        np.linalg.norm(arrays[".stacked_blocks.stacked.attn.w_q"][4]),
    )
    p = NormPolicy()
    assert not p.due(0) and not p.due(1) and p.due(1000)
    with pytest.raises(ValueError):
        collect_norms(arrays, {})


def test_shared_prediction_loss_views_weight_sums_not_partition_means():
    sequence = "N" + "a" * 1000 + "C" * 7000 + "N" * 191
    losses = np.array([[2.0] * 1000 + [4.0] * 7000 + [6.0] * 191])
    result = loss_views([sequence], losses)
    assert result["eval/softmasked_targets"] == 1000
    assert result["eval/nonsoftmasked_targets"] == 7191
    assert result["eval/unknown_N_targets"] == 191
    assert result["eval/loss"] == losses.mean()
    assert result["eval/loss_softmasked"] == 2
    assert np.isclose(
        result["eval/loss_weighted_10to1"],
        (2000 + 10 * (28000 + 1146)) / (1000 + 10 * 7191),
    )


def test_holdout_selection_preserves_species_and_avoids_intersections():
    records = [
        {
            "split": "validation",
            "row": i,
            "assembly": "AB"[i % 2],
            "chrom": "chr1",
            "start": i * 100,
            "end": i * 100 + 150,
            "strand": "+",
        }
        for i in range(30)
    ]
    selected = select_disjoint_validation(records, count=8)
    assert selected == select_disjoint_validation(records, count=8)
    assert {r["assembly"] for r in selected} == {"A", "B"}
    for i, a in enumerate(selected):
        for b in selected[i + 1 :]:
            if a["assembly"] == b["assembly"]:
                assert a["end"] <= b["start"] or b["end"] <= a["start"]
    with pytest.raises(ValueError):
        select_disjoint_validation(records, count=100)


def test_full_holdout_verification_checks_interval_edges_and_source_strands(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    train = [{"assembly": "A", "chrom": "chr1", "start": 100, "end": 200}]
    pq.write_table(pa.Table.from_pylist(train), tmp_path / "train.parquet")
    holdout = [
        {"assembly": "A", "chrom": "chr1", "start": 201, "end": 300, "strand": "-"}
    ]
    assert verify_disjoint_metadata(holdout, tmp_path, expected_rows=1) == 1
    with pytest.raises(ValueError, match="training metadata row"):
        verify_disjoint_metadata(holdout, tmp_path)
    for start, end in [(199, 300), (200, 300), (0, 100)]:
        holdout[0].update(start=start, end=end)
        with pytest.raises(ValueError, match="intersects a training"):
            verify_disjoint_metadata(holdout, tmp_path, expected_rows=1)
