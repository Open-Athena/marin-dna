"""Verify every final VEP/probe output and print its receipt before shutdown."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import boto3
import pyarrow.parquet as pq

MODEL = "dna-exp550-rag46m-five-regions-v1-step-100000"
COHORTS = {"mendelian_traits": 16140, "complex_traits": 11630, "sge": 23853}
PREFIX = "snakemake/analysis/evals_v2/"
WORK = Path("/opt/issue550")


def main() -> None:
    receipt = json.loads((WORK / "a10g-run-receipt.json").read_text())
    assert receipt["model"] == MODEL and receipt["completed"] and receipt["with_probes"]
    assert receipt["exit_status"] == 0 and receipt["publication"] == "canonical_s3"
    assert receipt["split"] == "train" and receipt["cohort_sizes"] == COHORTS
    client = boto3.client("s3", region_name="us-east-2")
    files, metrics = {}, {}
    for kind, suffix in (
        ("scores", "parquet"),
        ("metrics", "parquet"),
        ("probe", "parquet"),
        ("probe", "joblib"),
        ("probe_metrics", "parquet"),
    ):
        for cohort, count in COHORTS.items():
            relative = f"results/{kind}/{MODEL}/{cohort}.{suffix}"
            path = WORK / "storage/s3/oa-bolinas" / PREFIX / relative
            size = path.stat().st_size
            assert 0 < size <= 256 * 1024**2
            if suffix == "parquet":
                parquet = pq.ParquetFile(path)
                if kind == "scores":
                    assert parquet.metadata.num_rows == count
                    assert {"emb_ref", "emb_alt", "llr_fwd", "llr_rc"} <= set(
                        parquet.schema_arrow.names
                    )
                elif kind == "probe":
                    assert 0 < parquet.metadata.num_rows <= count
                    assert "probe_score" in parquet.schema_arrow.names
                else:
                    protocol = (
                        "probe_score"
                        if kind == "probe_metrics"
                        else (
                            "abs_llr_avg"
                            if cohort == "complex_traits"
                            else "minus_llr_avg"
                        )
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
                    metrics[f"{kind}/{cohort}"] = rows[0]
            with path.open("rb") as stream:
                sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            remote = client.get_object(Bucket="oa-bolinas", Key=PREFIX + relative)
            assert remote["ContentLength"] == size
            with remote["Body"] as stream:
                remote_digest = hashlib.sha256()
                while chunk := stream.read(1024**2):
                    remote_digest.update(chunk)
            assert remote_digest.hexdigest() == sha256
            files[relative] = {"bytes": size, "sha256": sha256}
    assert len(files) == 15
    receipt.update(
        files=files,
        canonical_s3_content_sha256_verified=True,
        development_macro_auprc=metrics,
    )
    (WORK / "step-100000-completed.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print("DEVELOPMENT_VEP_OUTPUTS " + json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
