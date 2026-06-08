"""Cell-type linear-probe / kNN on DART task3 gLM embeddings (issue #298).

Reports balanced accuracy of predicting the 5-way cell-type label from the
embedding — the quantitative companion to the UMAP for detecting small-but-real
cell-type signal an intermixed scatter hides. Chance (balanced) = 1/5 = 0.20.

Two modes (both with a StandardScaler):
  - HELD-OUT (preferred): fit on ``train_parquet``, score on ``--eval <parquet>``
    — the dataset's chromosome-disjoint validation split, so no leakage.
  - CV fallback (no ``--eval``): ``GroupKFold`` **by chromosome**, keeping folds
    chromosome-disjoint. A plain random KFold would leak — peaks on the same
    chromosome share local sequence context, inflating accuracy, which is the very
    reason DART splits by chromosome — so it is deliberately not offered.

Usage:
    uv run --group genome-s3 python scripts/issue298_celltype_probe.py \\
        <train.parquet> --eval <validation.parquet>

Parquet paths may be local or ``s3://…`` (S3 needs the ``genome-s3`` group).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _load(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    emb = [c for c in df.columns if c.startswith("emb_")]
    assert emb, f"no emb_* columns in {path}"
    X = df[emb].to_numpy(dtype=np.float32)
    return X, df["label"].to_numpy(), df["chrom"].astype(str).to_numpy()


def _models() -> dict:
    return {
        "linear probe (logreg)": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)
        ),
        "kNN (k=15)": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=15)
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("train_parquet")
    ap.add_argument(
        "--eval",
        dest="eval_parquet",
        help="held-out eval parquet (chromosome-disjoint val split); preferred",
    )
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=2)  # shared 4-vCPU box etiquette
    args = ap.parse_args()

    Xtr, ytr, ctr = _load(args.train_parquet)
    classes = np.unique(ytr)
    print(f"train: {len(ytr)} windows x {Xtr.shape[1]} dims; {len(classes)} cell types")
    print(f"chance (balanced accuracy) = {1 / len(classes):.3f}\n")

    if args.eval_parquet:
        Xev, yev, _ = _load(args.eval_parquet)
        print(f"HELD-OUT eval (chromosome-disjoint): {len(yev)} windows\n")
        for name, clf in _models().items():
            clf.fit(Xtr, ytr)
            acc = balanced_accuracy_score(yev, clf.predict(Xev))
            print(f"{name:24s} held-out balanced_acc = {acc:.4f}")
        lp = _models()["linear probe (logreg)"]
        lp.fit(Xtr, ytr)
        print("\nlinear-probe per-class recall (held-out):")
        print(classification_report(yev, lp.predict(Xev), digits=3, zero_division=0))
    else:
        gkf = GroupKFold(n_splits=args.folds)
        print(f"GroupKFold-by-chromosome CV ({args.folds} folds, leakage-free):\n")
        for name, clf in _models().items():
            s = cross_val_score(
                clf,
                Xtr,
                ytr,
                groups=ctr,
                cv=gkf,
                scoring="balanced_accuracy",
                n_jobs=args.n_jobs,
            )
            print(
                f"{name:24s} balanced_acc = {s.mean():.4f} ± {s.std():.4f}  "
                f"folds={np.round(s, 3).tolist()}"
            )


if __name__ == "__main__":
    main()
