"""Read only existing, small development metric artifacts; no VEP inference."""

import io
import json
import subprocess
import sys

import pyarrow.parquet as pq

MODELS = [
    ("phyloP order", "exp517-phylop-uniform-enhancer-order-step-4999", 15719320),
    ("phyloP family", "exp517-phylop-uniform-enhancer-arm-a-step-4999", 79725424),
    ("GPN family", "exp517-gpn-uniform-enhancer-arm-a-step-4999", 131467840),
    ("#232 uniform", "exp232-v4_ccre_non_promoter-step-4999", 88030162),
    ("#517 annotation-first", "exp517-enhancer-step-4999", 25364652),
]
for label, model, train_rows in MODELS:
    if len(sys.argv) > 1 and label not in sys.argv[1:]:
        continue
    for dataset, score_type in [
        ("mendelian_traits", "minus_llr_avg"),
        ("complex_traits", "abs_llr_avg"),
    ]:
        if label == "#232 uniform" and dataset == "complex_traits":
            continue  # No canonical Complex Traits artifact exists for this model.
        uri = (
            "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics/"
            f"{model}/{dataset}.parquet"
        )
        result = subprocess.run(
            ["aws", "s3", "cp", uri, "-", "--only-show-errors"],
            check=True, capture_output=True,
        )
        assert len(result.stdout) < 100_000, "unexpectedly large metric artifact"
        rows = pq.read_table(io.BytesIO(result.stdout)).to_pylist()
        chosen = [r for r in rows if r["subset"] == "distal" and r["score_type"] == score_type]
        assert len(chosen) == 1
        row = chosen[0]
        assert 0 <= row["value"] <= 1 and row["se"] >= 0
        assert row["n_groups"] >= 30
        print(json.dumps({
            "label": label, "model": model, "dataset": dataset,
            "effective_epochs": 40960000 / train_rows,
            "train_rows": train_rows, "source": uri, "bytes": len(result.stdout),
            **row,
        }), flush=True)
