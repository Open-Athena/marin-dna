"""Model-agnostic caQTL/dsQTL official-metrics benchmark driver (issue #311).

Lightweight glue over the tested library logic in
``marin_dna.pipelines.evals.qtl_scoring`` — no logic of its own. For each dataset it:

1. loads the canonical HF dataset (``train`` + ``test``, tagged ``split_source``) and
   joins the genome-native AlphaGenome ``dnase_lfc`` predictions (corrected once by
   ``correct_ag_predictions.py``);
2. writes ``scores/{model}/{dataset}.parquet`` (``[chrom,pos,ref,alt,causality_score,
   direction_score]``) for each built-in model (AlphaGenome / ChromBPNet / Enformer) —
   the plug-in interface a future fine-tuned gLM (#243) also writes into;
3. computes ``metrics/{model}/{dataset}.parquet`` for **every** model whose scores
   parquet is present at the prefix (discovered by listing — adding a model is one new
   parquet, no code change), plus the static ``metrics_reference/{dataset}.parquet``.

``--reproduce`` prints the AG-test slice vs AlphaGenome Suppl Table 4 and asserts ≤0.005.

Outputs live under ``s3://oa-bolinas/qtl_benchmark/`` for the dashboard (#312). Runs
locally (small data; the bootstrap is the only cost). Run:

    uv run python scripts/qtl_benchmark.py            # build everything
    uv run python scripts/qtl_benchmark.py --reproduce
"""

from __future__ import annotations

import argparse
from urllib.parse import urlparse

import boto3
import pandas as pd
import polars as pl

from marin_dna.pipelines.evals.qtl_scoring import (
    MODEL_SCORE_COLUMNS,
    SUPPL_TABLE4_REFERENCE,
    compute_qtl_split_metrics,
    reference_metrics,
    to_score_interface,
)

KEY = ["chrom", "pos", "ref", "alt"]
DATASETS = ("caqtl", "dsqtl")
REGION = "us-east-2"

BENCH = "s3://oa-bolinas/qtl_benchmark"  # scores/{model}/{ds}, metrics/{model}/{ds}, metrics_reference/{ds}
AG_DNASE_LFC = (
    "s3://oa-bolinas/snakemake/alphagenome_eval/results/dnase_lfc"  # genome-native AG
)
AG_LFC_COL = "alphagenome_dnase_lfc"

HF_REVISION = {
    "caqtl": "27a24296f50ed55afdc412d1612df680d13138d6",
    "dsqtl": "4a3bf152cd7c28be290adde48a402ec40992cb62",
}

# Map built-in model keys to their AlphaGenome Suppl Table 4 reference name (Enformer is
# the ChromBPNet-paper baseline, not in Suppl Table 4 → no published reference).
REPRO_REF = {"alphagenome": "AlphaGenome", "chrombpnet": "ChromBPNet"}
REPRO_TOL = 0.005


def _split_s3(url: str) -> tuple[str, str]:
    p = urlparse(url)
    return p.netloc, p.path.lstrip("/")


def load_dataset(name: str) -> pl.DataFrame:
    """Canonical dataset (train ∪ test, ``split_source`` tagged) + genome-native AG LFC."""
    rev = HF_REVISION[name]
    parts = [
        pl.read_parquet(
            f"hf://datasets/bolinas-dna/evals_{name}@{rev}/{split}.parquet"
        ).with_columns(pl.lit(split).alias("split_source"))
        for split in ("train", "test")
    ]
    dataset = pl.concat(parts)
    ag = pl.read_parquet(f"{AG_DNASE_LFC}/{name}.parquet").select([*KEY, AG_LFC_COL])
    dataset = dataset.join(ag, on=KEY, how="left")
    n_null = int(dataset.get_column(AG_LFC_COL).is_null().sum())
    assert n_null == 0, f"{name}: {n_null} variants missing an AlphaGenome score"
    return dataset


def write_model_scores(name: str, dataset: pl.DataFrame) -> None:
    """Materialize the plug-in score parquet for each built-in model."""
    for model, (causality_col, direction_col) in MODEL_SCORE_COLUMNS.items():
        scores = to_score_interface(dataset, causality_col, direction_col)
        path = f"{BENCH}/scores/{model}/{name}.parquet"
        scores.write_parquet(path)
        print(f"[{name}] wrote scores/{model} ({scores.height} variants)")


def discover_models(name: str) -> list[str]:
    """Every model with a ``scores/{model}/{name}.parquet`` at the benchmark prefix —
    built-ins plus any externally dropped model (e.g. a fine-tuned gLM, #243)."""
    bucket, prefix = _split_s3(f"{BENCH}/scores/")
    s3 = boto3.client("s3", region_name=REGION)
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
    models = []
    for cp in resp.get("CommonPrefixes", []):
        model = cp["Prefix"].rstrip("/").split("/")[-1]
        try:
            s3.head_object(Bucket=bucket, Key=f"{prefix}{model}/{name}.parquet")
            models.append(model)
        except s3.exceptions.ClientError:
            continue  # this model has no scores for this dataset
    return sorted(models)


def compute_metrics(name: str, dataset: pl.DataFrame, n_bootstrap: int) -> pl.DataFrame:
    """Compute ``metrics/{model}/{name}`` for every discovered model + the reference."""
    rows = []
    for model in discover_models(name):
        scores = pl.read_parquet(f"{BENCH}/scores/{model}/{name}.parquet")
        metrics = compute_qtl_split_metrics(
            scores, dataset, dataset=name, model=model, n_bootstrap=n_bootstrap
        )
        pl.from_pandas(metrics).write_parquet(f"{BENCH}/metrics/{model}/{name}.parquet")
        print(f"[{name}] wrote metrics/{model}")
        rows.append(metrics)
    pl.from_pandas(reference_metrics(name)).write_parquet(
        f"{BENCH}/metrics_reference/{name}.parquet"
    )
    return pl.from_pandas(pd.concat(rows, ignore_index=True))


def reproduce(name: str, metrics: pl.DataFrame) -> bool:
    """Print the AG-test slice vs Suppl Table 4 for the reference models; return pass."""
    ag = metrics.filter(pl.col("split") == "ag_test")
    print(f"\n=== {name}: AG-test slice vs AlphaGenome Suppl Table 4 ===")
    ok = True
    for row in ag.iter_rows(named=True):
        model = row["model"]
        ref = SUPPL_TABLE4_REFERENCE[name].get(REPRO_REF.get(model, ""))
        line = (
            f"  {model:12s} auPRC={row['causality_auPRC']:.4f} "
            f"pearson={row['direction_pearson']:.4f}"
        )
        if ref is not None:
            d_auprc = abs(row["causality_auPRC"] - ref[0])
            d_pearson = abs(row["direction_pearson"] - ref[1])
            line += f"   ref {ref}  Δ=({d_auprc:.4f}, {d_pearson:.4f})"
            ok = ok and d_auprc <= REPRO_TOL and d_pearson <= REPRO_TOL
        print(line)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument(
        "--reproduce", action="store_true", help="assert AG-test vs Suppl Table 4"
    )
    args = ap.parse_args()

    all_ok = True
    for name in args.datasets:
        dataset = load_dataset(name)
        write_model_scores(name, dataset)
        metrics = compute_metrics(name, dataset, args.n_bootstrap)
        if args.reproduce:
            all_ok = reproduce(name, metrics) and all_ok
    if args.reproduce:
        assert all_ok, f"reproduction exceeded ±{REPRO_TOL} vs Suppl Table 4"
        print("\nReproduces AlphaGenome Suppl Table 4 (reference models) to ≤0.005.")


if __name__ == "__main__":
    main()
