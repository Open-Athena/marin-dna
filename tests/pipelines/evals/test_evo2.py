"""Tests for Evo2 variant scoring helpers.

The real inference path needs ``evo2`` + an H100/GH200 + ~40 GB of downloaded
weights, so this file only exercises the lightweight contract (signature,
import, aggregator math). The heavy lifting is checked at runtime on the
SkyPilot cluster.
"""

import numpy as np

from marin_dna.pipelines.evals.evo2 import aggregate_ll_gap, compute_evo2_ll


def test_evo2_helpers_importable():
    """Sanity-check that the public LL-gap functions exist and are importable
    without evo2 installed.

    Variant scoring lives in ``scripts/evo2_eval/_evo2_scoring.py``
    (intentionally not testable from here without an evo2 install)."""
    assert callable(compute_evo2_ll)
    assert callable(aggregate_ll_gap)


def test_aggregate_ll_gap_token_weighted_mean_and_sign():
    """Per-row sums and counts must aggregate to a token-weighted dataset
    mean (sum-then-divide), and the gap convention must be
    ``LL_upper - LL_lower``: positive when uppercase log-probs are
    closer to 0 (easier) than lowercase log-probs.
    """
    # Two rows, one all-upper and one all-lower — the edge case that
    # motivates returning sums+counts (per-row means would NaN here).
    # All-upper row: 4 target tokens, total log p = -2.0
    # All-lower row: 6 target tokens, total log p = -9.0
    pred = np.array(
        [
            [-2.0, 0.0, 4.0, 0.0],
            [0.0, -9.0, 0.0, 6.0],
        ],
        dtype=np.float32,
    )
    out = aggregate_ll_gap(pred)

    # token-weighted mean: -11.0 / 10 = -1.1
    assert out["LL_all"] == -1.1
    # mean LL on uppercase: -2 / 4 = -0.5
    assert out["LL_upper"] == -0.5
    # mean LL on lowercase: -9 / 6 = -1.5
    assert out["LL_lower"] == -1.5
    # gap = LL_upper - LL_lower = -0.5 - (-1.5) = +1.0
    # (positive ⇒ uppercase is easier to predict, the expected sign on CDS)
    assert out["gap"] == 1.0
    assert out["n_upper"] == 4
    assert out["n_lower"] == 6


def test_aggregate_ll_gap_uses_fp64():
    """Aggregator must cast to fp64 before the cross-row sum (PR #18 of
    biofoundation). We verify by feeding a fp32 input large enough that
    a naive fp32 ``sum`` would drop the bottom bits, then checking the
    result has fp64 precision relative to a fully-fp64 reference.
    """
    rng = np.random.default_rng(0)
    n_rows = 50_000
    upper_logp = rng.uniform(-3.0, -0.1, size=n_rows).astype(np.float32)
    lower_logp = rng.uniform(-3.0, -0.1, size=n_rows).astype(np.float32)
    pred = np.stack(
        [
            upper_logp,
            lower_logp,
            np.ones(n_rows, dtype=np.float32),
            np.ones(n_rows, dtype=np.float32),
        ],
        axis=1,
    )

    out = aggregate_ll_gap(pred)
    # All four returned floats are Python floats (fp64).
    for k in ("LL_all", "LL_upper", "LL_lower", "gap"):
        assert isinstance(out[k], float)
    # Reference: same arithmetic but fully fp64 from the start. Any miss
    # in the cast will show up as a >fp32-eps difference here on a 50k-row
    # sample where the per-token mean is ~-1.5.
    ref_upper = upper_logp.astype(np.float64).sum() / n_rows
    ref_lower = lower_logp.astype(np.float64).sum() / n_rows
    assert abs(out["LL_upper"] - ref_upper) < 1e-12
    assert abs(out["LL_lower"] - ref_lower) < 1e-12


def test_aggregate_ll_gap_rejects_bad_shape():
    import pytest

    with pytest.raises(AssertionError):
        aggregate_ll_gap(np.zeros((5, 3)))  # not [N, 4]


def test_aggregate_ll_gap_rejects_zero_count_buckets():
    import pytest

    # all upper, no lower at all
    pred = np.array([[-1.0, 0.0, 5.0, 0.0]], dtype=np.float32)
    with pytest.raises(AssertionError, match="non-functional"):
        aggregate_ll_gap(pred)

    # all lower, no upper at all
    pred = np.array([[0.0, -1.0, 0.0, 5.0]], dtype=np.float32)
    with pytest.raises(AssertionError, match="functional"):
        aggregate_ll_gap(pred)


# --------------------------------------------------------------------------- #
# scripts/evo2_eval/compute_auprc_metrics.py contract tests
#
# Light-touch: verify the COL_RENAME / SCORE_COLUMNS coupling and a single
# end-to-end pass on a synthetic 50-row predictions parquet, so a future
# typo in either constant (or a schema drift in `compute_auprc_metrics`)
# trips a fast unit test instead of a silent dashboard regression.
# --------------------------------------------------------------------------- #


def _load_driver():
    """Load the script as a module by path (it's not a package)."""
    import importlib.util
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    script = repo / "scripts" / "evo2_eval" / "compute_auprc_metrics.py"
    spec = importlib.util.spec_from_file_location("compute_auprc_metrics", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_score_columns_subset_of_col_rename_values():
    """SCORE_COLUMNS must reference columns that actually emerge from the
    rename — otherwise the driver would assert empty score_cols at runtime
    on every input.
    """
    mod = _load_driver()
    rename_targets = set(mod.COL_RENAME.values())
    missing = [c for c in mod.SCORE_COLUMNS if c not in rename_targets]
    assert not missing, (
        f"SCORE_COLUMNS contains entries not produced by COL_RENAME: {missing}; "
        f"available targets: {sorted(rename_targets)}"
    )


def test_driver_end_to_end_on_synthetic_predictions(tmp_path):
    """Round-trip: write a tiny synthetic predictions parquet in the evo2
    naming, invoke the driver, read back the metrics parquet, check schema
    and a couple of value invariants (AUPRC in [0,1], 6 score_types × 11
    subset rows for 9 subsets + global + macro).
    """
    import subprocess
    import sys

    import pandas as pd

    mod = _load_driver()

    # Synthetic predictions in evo2 naming. 30 match_groups across 2
    # subsets so both subsets clear n_min=30 separately? No — n_min=30 is
    # per-subset, so we use 30 groups in ONE subset (qualifying), and 5
    # in a smaller subset (not qualifying for macro). _global_ row still
    # contributes. Each group: 1 positive + 9 negatives = 10 rows.
    rng = np.random.default_rng(42)
    rows = []
    for gid in range(30):
        score_signal = rng.normal(loc=2.0)  # positives skew high
        rows.append(
            {
                "chrom": "1",
                "pos": gid,
                "ref": "A",
                "alt": "C",
                "label": 1,
                "subset": "big",
                "match_group": gid,
                "minus_llr": score_signal,
                "minus_llr_fwd": score_signal,
                "minus_llr_rev": score_signal,
                "next_token_jsd_mean": abs(score_signal) * 0.01,
                "next_token_jsd_mean_fwd": abs(score_signal) * 0.01,
                "next_token_jsd_mean_rev": abs(score_signal) * 0.01,
            }
        )
        for _ in range(9):
            noise = rng.normal(loc=0.0)
            rows.append(
                {
                    "chrom": "1",
                    "pos": gid,
                    "ref": "A",
                    "alt": "G",
                    "label": 0,
                    "subset": "big",
                    "match_group": gid,
                    "minus_llr": noise,
                    "minus_llr_fwd": noise,
                    "minus_llr_rev": noise,
                    "next_token_jsd_mean": abs(noise) * 0.01,
                    "next_token_jsd_mean_fwd": abs(noise) * 0.01,
                    "next_token_jsd_mean_rev": abs(noise) * 0.01,
                }
            )
    df = pd.DataFrame(rows)

    in_path = tmp_path / "synth_train.parquet"
    out_path = tmp_path / "synth_metrics.parquet"
    df.to_parquet(in_path, index=False)

    # Invoke the script as a subprocess so the argparse-driven `main()`
    # path is exercised end-to-end (the way Snakemake / a user would
    # invoke it). Small n_bootstrap so the test runs in <1s.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import importlib.util,sys; "
            f"spec=importlib.util.spec_from_file_location('m', "
            f"'{mod.__file__}'); m=importlib.util.module_from_spec(spec); "
            f"spec.loader.exec_module(m); sys.argv=['m', '--input', "
            f"'{in_path}', '--output', '{out_path}', '--model', 'evo2_test', "
            f"'--n-bootstrap', '50']; m.main()",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"driver failed: {result.stderr}"

    metrics = pd.read_parquet(out_path)
    expected_cols = {
        "score_type",
        "subset",
        "value",
        "se",
        "n_groups",
        "n_rows",
        "model",
        "dataset",
        "split",
    }
    assert set(metrics.columns) == expected_cols, (
        f"metrics columns: {sorted(metrics.columns)}"
    )

    score_types = set(metrics["score_type"].unique())
    assert score_types == set(mod.SCORE_COLUMNS), (
        f"missing score_types: {set(mod.SCORE_COLUMNS) - score_types}"
    )

    # 1 subset + _global_ + _macro_avg_ = 3 rows per score_type
    assert (metrics["score_type"] == mod.SCORE_COLUMNS[0]).sum() == 3

    # AUPRC bounded [0, 1]; SE non-negative.
    assert (metrics["value"] >= 0).all() and (metrics["value"] <= 1).all()
    assert (metrics["se"] >= 0).all()

    # Stamped metadata round-trips.
    assert (metrics["model"] == "evo2_test").all()
    assert (metrics["dataset"] == "mendelian_traits").all()  # default
    assert (metrics["split"] == "train").all()
