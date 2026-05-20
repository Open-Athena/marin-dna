"""Threshold calibration: match the *proportion* of non-NaN bases passing
between two phyloP tracks.

Use case: pick a threshold for ``phyloP_447m`` such that the fraction of
non-NaN bases with ``phyloP_447m >= T`` equals the fraction of non-NaN
bases with ``phyloP_241m >= 2.27``. Proportion-based (not count-based)
because the two tracks have slightly different coverage — matching
proportions controls for that and is more semantically faithful to "the
top X% of conserved bases".
"""

from .histogram import PhylopHistogram


def calibrate_to_match_proportion(
    target_hist: PhylopHistogram,
    ref_hist: PhylopHistogram,
    ref_threshold: float,
    *,
    target_name: str = "target",
    ref_name: str = "reference",
) -> dict:
    """Find ``T`` for ``target_hist`` such that ``target proportion ≈ ref proportion``.

    Where proportion is ``count_ge(threshold) / total()`` (non-NaN bases
    only). The returned threshold is linearly interpolated within the
    bracketing bin of ``target_hist``; precision is bounded by the bin
    width.

    Returns a JSON-serialisable dict with both tracks' thresholds, counts,
    proportions, the relative error, and the bin metadata. Asserts that
    the absolute relative error between the matched proportions is under
    1% — well below the bin-width-driven precision floor for sensible
    bin counts.
    """
    ref_count = ref_hist.count_ge(ref_threshold)
    ref_total = ref_hist.total()
    assert ref_total > 0, "reference histogram has no non-NaN bases"
    ref_proportion = ref_count / ref_total

    target_total = target_hist.total()
    assert target_total > 0, "target histogram has no non-NaN bases"
    target_count_target = int(round(ref_proportion * target_total))

    target_threshold = target_hist.threshold_for_count(target_count_target)
    target_count = target_hist.count_ge(target_threshold)
    target_proportion = target_count / target_total

    if ref_proportion > 0:
        rel_err = abs(target_proportion / ref_proportion - 1.0)
    else:
        rel_err = 0.0
    assert rel_err < 0.01, (
        f"calibration failed: {target_name} proportion {target_proportion:.6f} vs "
        f"{ref_name} proportion {ref_proportion:.6f} (rel err {rel_err:.4f}); "
        f"likely the histogram bin width is too coarse."
    )

    return {
        ref_name: {
            "threshold": float(ref_threshold),
            "count": int(ref_count),
            "proportion": float(ref_proportion),
            "total_bases": int(ref_total),
            "n_nan": int(ref_hist.n_nan),
        },
        target_name: {
            "threshold": float(target_threshold),
            "count": int(target_count),
            "count_target": int(target_count_target),
            "proportion": float(target_proportion),
            "abs_relative_error": float(rel_err),
            "total_bases": int(target_total),
            "n_nan": int(target_hist.n_nan),
        },
        "hist_meta": {
            "n_bins": int(target_hist.n_bins),
            "min": float(target_hist.edges[0]),
            "max": float(target_hist.edges[-1]),
            "bin_width": float(target_hist.edges[1] - target_hist.edges[0]),
        },
    }
