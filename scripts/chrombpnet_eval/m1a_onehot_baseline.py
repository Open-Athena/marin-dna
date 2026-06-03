"""M1a: validate the chrombpnet_eval harness against ARSENAL's precomputed
one-hot ChromBPNet (GM12878 DNase, ENCODE ENCSR000EMT) on our caqtl/dsqtl splits.

No training, no GPU. We pull ARSENAL's released ``variant_scores.tsv`` (signed
``logfc``), align it to our split parquets (caQTL hg38 direct; dsQTL hg19→hg38
lift), and compute the supervised metric set
(:func:`compute_supervised_qtl_metrics`):

- on **all** used variants (train+test) — cross-checks our harness against
  ARSENAL's published one-hot numbers (caQTL AUROC≈0.750/Pearson≈0.633;
  dsQTL AUROC≈0.883/Pearson≈0.720);
- on our **train** split — the dev baseline (test held out).

Run (needs ``SYNAPSE_AUTH_TOKEN`` in env; HF auth for the dataset repos):

    uv run python scripts/chrombpnet_eval/m1a_onehot_baseline.py

ARSENAL assets: https://www.synapse.org/Synapse:syn72351987 (ChromBPNet_Comparison
/ ChromBPNet_Models / GM12878 / run_1). Design: GitHub issue #236.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import requests
from huggingface_hub import hf_hub_download

from marin_dna.pipelines.chrombpnet_eval.metrics import compute_supervised_qtl_metrics
from marin_dna.pipelines.chrombpnet_eval.scores import (
    align_scores_to_variants,
    load_arsenal_scores,
)

# ARSENAL one-hot ChromBPNet GM12878 / run_1 variant_scores.tsv (Synapse).
# african_variants = caQTL (hg38); yoruba_variants = dsQTL (hg19).
DATASETS = {
    "caqtl": {
        "lift": False,
        "flip_logfc": False,
        "scores_syn": "syn73654973",
        "hf_repo": "bolinas-dna/evals_caqtl",
        "hf_rev": "9d004a21812c067b9ba1ebfe72f51b9095a5d0f8",
        "published": {"AUROC": 0.750, "pearson": 0.633, "spearman": 0.656},
    },
    "dsqtl": {
        "lift": True,
        # ARSENAL's released dsQTL logfc is sign-flipped vs the study effect /
        # DART-Eval convention our `effect` follows — see load_arsenal_scores.
        "flip_logfc": True,
        "scores_syn": "syn73655490",
        "hf_repo": "bolinas-dna/evals_dsqtl",
        "hf_rev": "b7e02a07beb831c7047286aacd3ddfd299d6f88f",
        "published": {"AUROC": 0.883, "pearson": 0.720, "spearman": 0.744},
    },
}
CACHE = Path(
    os.environ.get("CHROMBPNET_EVAL_CACHE", Path.home() / ".cache/chrombpnet_eval")
)


def synapse_download(syn_id: str, dest: Path) -> Path:
    """Download a Synapse FileEntity to ``dest`` (cached). Needs SYNAPSE_AUTH_TOKEN."""
    if dest.exists():
        return dest
    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    assert token, "set SYNAPSE_AUTH_TOKEN (a Synapse PAT with Download scope)"
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://repo-prod.prod.sagebase.org/repo/v1/entity/{syn_id}/file"
    # requests drops the auth header on the cross-host redirect to the presigned
    # S3 URL (which is exactly what we want).
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=120)
    resp.raise_for_status()
    # Write to a sibling .tmp then atomically rename, so an interrupted download
    # never leaves a truncated file that the dest.exists() cache silently reuses.
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(resp.content)
    tmp.replace(dest)
    return dest


def load_split(repo: str, rev: str, split: str) -> pl.DataFrame:
    path = hf_hub_download(repo, f"{split}.parquet", revision=rev, repo_type="dataset")
    return pl.read_parquet(path)


def main() -> None:
    rows: list[dict] = []
    for name, cfg in DATASETS.items():
        scores = load_arsenal_scores(
            str(synapse_download(cfg["scores_syn"], CACHE / f"onehot_{name}_run1.tsv")),
            flip_logfc=cfg["flip_logfc"],
        )
        train = load_split(cfg["hf_repo"], cfg["hf_rev"], "train")
        test = load_split(cfg["hf_repo"], cfg["hf_rev"], "test")
        splits = {"all (train+test)": pl.concat([train, test]), "train": train}
        for split_name, variants in splits.items():
            joined = align_scores_to_variants(
                variants, scores, lift=cfg["lift"], min_coverage=0.9
            )
            cov = joined.height / variants.height
            m = compute_supervised_qtl_metrics(joined.to_pandas(), score_col="score")
            for _, r in m.iterrows():
                rows.append(
                    {
                        "dataset": name,
                        "split": split_name,
                        "metric": r["metric"],
                        "value": r["value"],
                        "se": r["se"],
                        "n_rows": int(r["n_rows"]),
                        "n_pos": int(r["n_pos"]),
                        "coverage": round(cov, 4),
                        "published": cfg["published"].get(r["metric"]),
                    }
                )
    res = pl.DataFrame(rows)
    out = CACHE / "m1a_onehot_baseline.csv"
    res.write_csv(out)

    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4, tbl_width_chars=200):
        print(res)
    print(f"\nwrote {out}")
    # Cross-check: |ours_all − published| for the reproduced metrics. run_1 vs
    # ARSENAL's 3-run average → expect a small (<~0.05) gap, same sign.
    tol = 0.05
    chk = (
        res.filter(
            (pl.col("split") == "all (train+test)") & pl.col("published").is_not_null()
        )
        .with_columns(
            (pl.col("value") - pl.col("published")).abs().alias("abs_diff"),
        )
        .with_columns((pl.col("abs_diff") <= tol).alias("within_tol"))
    )
    print("\n=== cross-check vs ARSENAL published (all variants, run_1) ===")
    with pl.Config(tbl_rows=-1, float_precision=4):
        print(
            chk.select(
                ["dataset", "metric", "value", "published", "abs_diff", "within_tol"]
            )
        )
    n_ok = int(chk["within_tol"].sum())
    print(f"\n{n_ok}/{chk.height} reproduced within ±{tol} (same sign).")


if __name__ == "__main__":
    main()
