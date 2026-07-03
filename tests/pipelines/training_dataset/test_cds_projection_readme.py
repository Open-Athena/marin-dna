from marin_dna.pipelines.training_dataset.cds_projection import (
    build_cds_projection_readme,
)

SHA = "f45178fc5cac82757934c5dae542bb0760294186"


def test_all204_card_has_required_pieces():
    md = build_cds_projection_readme(
        cohort="all204",
        n_species=204,
        n_rows=25_127_998,
        commit_sha=SHA,
        is_vertebrate_subset=False,
    )
    # HF-upload convention: tags + a commit-pinned permalink to the pipeline.
    assert "- biology\n- genomics\n- dna" in md
    assert f"/blob/{SHA}/" in md and "cds_projection.smk" in md
    assert "animals_order204.tsv" in md
    assert "25,127,998 rows" in md and "12,563,999 projected windows" in md
    # all-204 keeps the vertebrate-weighted warning.
    assert "vertebrate-weighted" in md


def test_vert_subset_drops_warning_and_points_at_chordata_list():
    md = build_cds_projection_readme(
        cohort="vert125",
        n_species=125,
        n_rows=24_195_826,
        commit_sha=SHA,
        is_vertebrate_subset=True,
    )
    assert "vertebrate-weighted" not in md  # it IS the vertebrate subset
    assert "animals_order204_chordata.tsv" in md
    assert "125 species" in md
