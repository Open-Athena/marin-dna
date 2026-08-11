"""Contact-prediction metrics for nucleotide dependency maps (issue #237).

Quantifies how well a (symmetric) dependency map recovers the known
base-pairing of a structured ncRNA — the nuc-dep paper's Fig 7b benchmark
(Tomaz da Silva et al., *Nat. Genet.* 2025; their
``manuscript_code/fig7_benchmark_trna.ipynb``). Each position pair is scored by
its dependency value and labelled by whether those two bases pair in the
reference secondary structure; we report AUROC / AUPRC / top-L precision / PPV /
MCC over the off-diagonal pairs.
"""

from __future__ import annotations

import numpy as np


def parse_trnascan_ss(ss: str) -> list[tuple[int, int]]:
    """Parse a tRNAscan-SE nested-bp secondary structure into 0-indexed pairs.

    The structure string uses ``>`` (open), ``<`` (close), ``.`` (unpaired);
    pairs nest, so a stack matches each ``>`` to its partner ``<``. Returns a
    sorted list of ``(i, j)`` with ``i < j``.
    """
    stack: list[int] = []
    pairs: list[tuple[int, int]] = []
    for k, c in enumerate(ss):
        if c == ">":
            stack.append(k)
        elif c == "<":
            assert stack, f"unbalanced structure: stray '<' at position {k}"
            pairs.append((stack.pop(), k))
        else:
            assert c == ".", f"unexpected structure char {c!r} at position {k}"
    assert not stack, f"unbalanced structure: {len(stack)} unclosed '>'"
    return sorted(pairs)


def base_pairs_to_contact_matrix(
    base_pairs: list[tuple[int, int]], n: int
) -> np.ndarray:
    """Symmetric ``[n, n]`` boolean contact matrix from a base-pair list."""
    contact = np.zeros((n, n), dtype=bool)
    for i, j in base_pairs:
        assert 0 <= i < n and 0 <= j < n, f"pair ({i},{j}) out of bounds for n={n}"
        contact[i, j] = contact[j, i] = True
    return contact


def contact_prediction_metrics(
    dep_map: np.ndarray,
    base_pairs: list[tuple[int, int]],
    *,
    min_sep: int = 1,
) -> dict[str, float]:
    """Fig-7b-style contact-prediction metrics for a dependency map.

    Candidate pairs are the strictly-lower triangle with separation
    ``|i - j| >= min_sep`` (``min_sep=1`` = all off-diagonal, matching the
    paper's active ``mask_diag=1``). Each candidate is scored by ``dep_map`` and
    labelled by base-pair membership.

    Args:
        dep_map: Symmetric ``[n, n]`` dependency map (the map is symmetrized via
            ``max(M, M.T)`` defensively, as in the reference).
        base_pairs: 0-indexed ``(i, j)`` reference base pairs.
        min_sep: Minimum sequence separation for a candidate pair.

    Returns:
        ``{auroc, auprc, top_L_precision, ppv, mcc, n_true_pairs, n_candidates}``.
        ``top_L_precision`` = precision among the top ``L = n`` scored pairs;
        ``ppv`` = precision among the top ``K = #true pairs``; ``mcc`` is at that
        top-``K`` operating point.
    """
    from sklearn.metrics import (
        auc,
        matthews_corrcoef,
        precision_recall_curve,
        roc_auc_score,
    )

    dep_map = np.asarray(dep_map, dtype=float)
    n = dep_map.shape[0]
    assert dep_map.shape == (n, n), f"expected square map, got {dep_map.shape}"
    dep_map = np.maximum(dep_map, dep_map.T)

    contact = base_pairs_to_contact_matrix(base_pairs, n)
    ii, jj = np.tril_indices(n, k=-min_sep)
    scores = dep_map[ii, jj]
    labels = contact[ii, jj]
    assert labels.any(), "no reference base pairs fall within the candidate set"

    prec, rec, _ = precision_recall_curve(labels, scores)
    order = np.argsort(scores)[::-1]
    n_true = int(labels.sum())
    pred_topk = np.zeros_like(labels)
    pred_topk[order[:n_true]] = True

    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(auc(rec, prec)),
        "top_L_precision": float(labels[order[:n]].mean()),
        "ppv": float(labels[order[:n_true]].mean()),
        "mcc": float(matthews_corrcoef(labels, pred_topk)),
        "n_true_pairs": n_true,
        "n_candidates": int(labels.size),
    }
