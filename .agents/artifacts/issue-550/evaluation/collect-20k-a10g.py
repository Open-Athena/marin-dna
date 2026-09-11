"""Verify canonical output presence and retain compact metadata before stopping."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import boto3
import pyarrow.parquet as pq

WORK = Path("/opt/issue550")
MODEL = "dna-exp550-rag46m-five-regions-v1-step-20000"
COHORTS = {"mendelian_traits": 16140, "complex_traits": 11630, "sge": 23853}
PREFIX = "snakemake/analysis/evals_v2/"


def main() -> None:
    receipt = json.loads((WORK / "a10g-run-receipt.json").read_text())
    assert receipt["model"] == MODEL and receipt["completed"]
    assert receipt["exit_status"] == 0 and receipt["publication"] == "canonical_s3"
    assert receipt["cohort_sizes"] == COHORTS and receipt["split"] == "train"
    client = boto3.client("s3", region_name="us-east-2")
    files, metrics = {}, {}
    for kind in ("scores", "metrics"):
        for cohort, count in COHORTS.items():
            relative = f"results/{kind}/{MODEL}/{cohort}.parquet"
            path = WORK / "storage/s3/oa-bolinas" / PREFIX / relative
            parquet = pq.ParquetFile(path)
            size = path.stat().st_size
            assert 0 < size <= 256 * 1024**2
            if kind == "scores":
                assert parquet.metadata.num_rows == count
                assert {"emb_ref", "emb_alt", "llr_fwd", "llr_rc"} <= set(
                    parquet.schema_arrow.names
                )
            else:
                protocol = (
                    "abs_llr_avg" if cohort == "complex_traits" else "minus_llr_avg"
                )
                rows = [
                    row
                    for row in parquet.read().to_pylist()
                    if row["score_type"] == protocol
                    and row["subset"] == "_macro_avg_"
                    and row.get("metric", "AUPRC") == "AUPRC"
                    and row.get("accession", "_macro_avg_") == "_macro_avg_"
                ]
                assert len(rows) == 1
                metrics[cohort] = rows[0]
            with path.open("rb") as stream:
                sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            head = client.head_object(Bucket="oa-bolinas", Key=PREFIX + relative)
            assert head["ContentLength"] == size
            files[relative] = {"bytes": size, "sha256": sha256, "s3_etag": head["ETag"]}
    receipt.update(
        files=files, canonical_s3_sizes_verified=True, development_macro_auprc=metrics
    )
    output = WORK / "step-20000-completed.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print("DEVELOPMENT_VEP_OUTPUTS " + json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
