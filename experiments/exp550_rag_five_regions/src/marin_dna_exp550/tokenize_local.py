"""Run standard Marin tokenization using only threads on the allocated TPU host."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from marin.processing.tokenize.tokenize import (
    HfTokenizeConfig,
    TokenizeConfig,
    tokenize,
)

from marin_dna_exp550.formats import RagFormat
from marin_dna_exp550.recipe import REGIONS


def main() -> None:
    if "IRIS_TASK_ID" in os.environ or os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("tokenization must run in an isolated local-only child")
    request = json.loads(Path(sys.argv[1]).read_text())
    client = LocalClient(max_threads=2)
    try:
        with (
            set_current_client(client),
            TemporaryDirectory(prefix="exp550-pilot-data-") as temporary,
        ):
            for region in REGIONS:
                common = {
                    "cache_path": f"{request['cache_root']}/{region}",
                    "tokenizer": str(Path(__file__).with_name("tokenizer")),
                    "format": RagFormat(),
                    "max_workers": 2,
                    "num_shards": 16,
                    "levanter_batch_size": 128,
                    "worker_resources": ResourceConfig.with_cpu(cpu=1, ram="2g"),
                    "tags": ["dna-exp550", f"region={region}"],
                }
                if request["pilot"]:
                    import pyarrow as pa
                    import pyarrow.parquet as pq

                    train = Path(temporary) / f"{region}-train.parquet"
                    validation = Path(temporary) / f"{region}-validation.parquet"
                    rows = [
                        {
                            "sequence": "[SEQ]".join(
                                [("ACGTN" * 51)[offset:] + ("ACGTN" * 51)[:offset]]
                                * count
                            )
                        }
                        for count in (1, 20, 40, 40)
                        for offset in range(16)
                    ]
                    pq.write_table(pa.Table.from_pylist(rows), train)
                    pq.write_table(pa.Table.from_pylist(rows[:8]), validation)
                    config = TokenizeConfig(
                        train_paths=[str(train)],
                        validation_paths=[str(validation)],
                        **common,
                    )
                else:
                    spec = request["datasets"][region]
                    config = HfTokenizeConfig(
                        id=spec["repo_id"], revision=spec["revision"], **common
                    )
                tokenize(config)
    finally:
        client.shutdown(wait=True)


if __name__ == "__main__":
    main()
