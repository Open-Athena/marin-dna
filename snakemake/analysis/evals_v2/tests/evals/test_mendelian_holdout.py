import numpy as np
import pandas as pd
import pytest
from marin_dna_evals.mendelian_holdout import (
    EXCLUDED,
    HEADLINE,
    IDENTITY,
    make_predictions,
    predict_batch,
)
from marin_dna_evals.mendelian_holdout_metrics import MODELS, align_predictions, compare
from marin_dna_evals.metrics import MACRO_AVG_SUBSET, per_chrom_weighted_ap
from sklearn.metrics import average_precision_score


class FrozenClassifier:
    """A predictable positive-class probability, with no fitting method."""

    classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probability = features[:, 1]
        return np.column_stack((1 - probability, probability))


def test_label_free_application_and_identity(tmp_path):
    classifiers = {
        subset: {"feature": "concat_ref_delta", "pipeline": FrozenClassifier()}
        for subset in HEADLINE
    }
    rows = pd.DataFrame(
        {
            "chrom": ["2", "2", "2"],
            "pos": [1, 2, 3],
            "ref": ["A"] * 3,
            "alt": ["T"] * 3,
            "subset": [HEADLINE[0], EXCLUDED, HEADLINE[1]],
            "emb_ref": [[0.25]] * 3,
            "emb_alt": [[0.5], [0.75], [0.875]],
            "llr_fwd": [2.0, 4.0, 6.0],
            "llr_rc": [0.0, 2.0, 4.0],
        }
    )
    before = predict_batch(rows, classifiers)
    np.testing.assert_equal(before, [0.25, np.nan, 0.625])
    np.testing.assert_equal(
        before, predict_batch(rows.assign(label=[0, 1, 0]), classifiers)
    )
    np.testing.assert_equal(
        before, predict_batch(rows.assign(label=[1, 0, 0]), classifiers)
    )
    path = tmp_path / "scores.parquet"
    rows.iloc[::-1].to_parquet(path)
    output = make_predictions(path, "marin", classifiers, rows[IDENTITY])
    np.testing.assert_equal(output.probe_score.to_numpy(), before)
    np.testing.assert_equal(output.minus_llr_avg.to_numpy(), [-1, -3, -5])
    bad_reference = rows[IDENTITY].copy()
    bad_reference.loc[0, "chrom"] = "1"
    with pytest.raises(ValueError, match="differs"):
        make_predictions(path, "marin", classifiers, bad_reference)
    with pytest.raises(ValueError, match="classifier"):
        predict_batch(rows, {})


def test_paired_metrics_keep_eight_classes_and_match_chromosome_weighting():
    data = []
    for k, subset in enumerate(HEADLINE):
        for group in range(3):
            for i in range(10):
                data.append(
                    {
                        "chrom": "2" if group == 0 else "4",
                        "match_group": k * 3 + group,
                        "label": int(i == 0),
                        "subset": subset,
                        "score": (i + 5 * group + k) % 8,
                    }
                )
    frame = pd.DataFrame(data)
    for model in MODELS:
        frame[f"{model}_minus_llr_avg"] = frame.score
        frame[f"{model}_probe_score"] = frame.score / 8
    result = compare(frame, draws=12).set_index(["readout", "subset"])
    assert len(result) == 18
    assert (
        (result[["delta_pp", "delta_lower_95_pp", "delta_upper_95_pp"]] == 0)
        .all()
        .all()
    )
    groups = list(frame.groupby("subset"))
    expected = np.mean([average_precision_score(g.label, g.score) for _, g in groups])
    assert result.loc[("minus_llr_avg", MACRO_AVG_SUBSET), "marin"] == pytest.approx(
        expected
    )
    probe = np.mean(
        [per_chrom_weighted_ap(g.label, g.score, g.chrom) for _, g in groups]
    )
    assert result.loc[("probe_score", MACRO_AVG_SUBSET), "marin"] == pytest.approx(
        probe
    )
    perfect = frame.assign(
        marin_minus_llr_avg=frame.label,
        marin_probe_score=frame.label,
        evo2_40b_minus_llr_avg=0.0,
        evo2_40b_probe_score=0.1,
    )
    compare_perfect = compare(perfect, draws=12)
    for col in ("delta_pp", "delta_lower_95_pp", "delta_upper_95_pp"):
        np.testing.assert_allclose(compare_perfect[col], 90.0)
    with pytest.raises(ValueError, match="1:9"):
        compare(frame.iloc[1:], draws=12)
    with pytest.raises(ValueError, match="eight"):
        compare(frame[frame["subset"] != "synonymous_variant"], draws=12)


def test_alignment_rejects_development_and_mismatched_rows():
    release = pd.DataFrame([["2", 1, "A", "G", HEADLINE[0]]], columns=IDENTITY)
    predictions = release.assign(
        model="marin", split="test", probe_score=0.6, minus_llr_avg=-0.2
    )
    result = align_predictions(release, predictions, "marin")
    assert result["marin_probe_score"].iloc[0] == 0.6
    with pytest.raises(ValueError, match="split"):
        align_predictions(release, predictions.assign(split="train"), "marin")
    with pytest.raises(ValueError, match="duplicate|Duplicate"):
        align_predictions(release, pd.concat([predictions, predictions]), "marin")
    with pytest.raises(ValueError, match="keys"):
        align_predictions(release, predictions.assign(pos=2), "marin")
