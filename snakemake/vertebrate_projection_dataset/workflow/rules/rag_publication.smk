"""Publish RAG documents while retaining row provenance in the owning workflow."""

from marin_dna_vertebrate_projection.provenance import validate_producer_manifest
from marin_dna_vertebrate_projection.rag.publication import (
    prepare_release,
    publish_release,
)

if config.get("rag"):
    RAG_PUBLICATION_SOURCE = config.get(
        "rag_publication_source",
        {
            "root": RAG_RESULTS,
            "pipeline_commit": PIPELINE_COMMIT,
            "config_sha256": PIPELINE_CONFIG_SHA256,
            "pipeline_version": PIPELINE_VERSION,
            "tier": TIER,
        },
    )
    RAG_SOURCE_ROOT = RAG_PUBLICATION_SOURCE["root"].rstrip("/")
    RAG_PUBLISH_SHARDS = int(RAG.get("publication_shards", 16))
    assert RAG_PUBLISH_SHARDS > 0

    def rag_publication_input(path):
        return storage.s3(path) if path.startswith("s3://") else path

    rule rag_prepare_release:
        input:
            train=rag_publication_input(
                f"{RAG_SOURCE_ROOT}/datasets/{{region}}/train.parquet"
            ),
            validation=rag_publication_input(
                f"{RAG_SOURCE_ROOT}/datasets/{{region}}/validation.parquet"
            ),
            producer=rag_publication_input(
                f"{RAG_SOURCE_ROOT.rsplit('/',1)[0]}/metadata/producer.json"
            ),
        output:
            train=expand(
                f"{RAG_RESULTS}/publication/{{region}}/train/{{shard:05d}}-of-{RAG_PUBLISH_SHARDS:05d}.parquet",
                region="{region}",
                shard=range(RAG_PUBLISH_SHARDS),
            ),
            validation=f"{RAG_RESULTS}/publication/{{region}}/validation/00000-of-00001.parquet",
            card=f"{RAG_RESULTS}/publication/{{region}}/README.md",
            mapping=f"{RAG_RESULTS}/publication_provenance/{{region}}/rows.parquet",
            manifest=f"{RAG_RESULTS}/publication_provenance/{{region}}/release.json",
        wildcard_constraints:
            region="|".join(RAG_REGIONS),
        resources:
            mem_mb=4000,
        run:
            validate_producer_manifest(
                input.producer,
                **{
                    key: value
                    for key, value in RAG_PUBLICATION_SOURCE.items()
                    if key != "root"
                },
            )
            prepare_release(
                {"train": input.train, "validation": input.validation},
                str(Path(output.card).parent),
                output.mapping,
                output.manifest,
                region=wildcards.region,
                repo_id=f"{config['hf_owner']}/{config.get('rag_repo_prefix', 'rag-five-regions-v1')}-{wildcards.region}",
                producer=RAG_PUBLICATION_SOURCE,
                train_shards=RAG_PUBLISH_SHARDS,
            )

    rule rag_publish_release:
        input:
            train=expand(
                f"{RAG_RESULTS}/publication/{{region}}/train/{{shard:05d}}-of-{RAG_PUBLISH_SHARDS:05d}.parquet",
                region="{region}",
                shard=range(RAG_PUBLISH_SHARDS),
            ),
            validation=f"{RAG_RESULTS}/publication/{{region}}/validation/00000-of-00001.parquet",
            card=f"{RAG_RESULTS}/publication/{{region}}/README.md",
            manifest=f"{RAG_RESULTS}/publication_provenance/{{region}}/release.json",
        output:
            f"{RAG_RESULTS}/publication_provenance/{{region}}/hub_receipt.json",
        resources:
            hf_uploads=1,
            mem_mb=4000,
        run:
            publish_release(str(Path(input.card).parent), input.manifest, output[0])

    rule rag_all_publication_files:
        input:
            expand(
                f"{RAG_RESULTS}/publication_provenance/{{region}}/release.json",
                region=RAG_REGIONS,
            ),

    rule rag_publish:
        input:
            expand(
                f"{RAG_RESULTS}/publication_provenance/{{region}}/hub_receipt.json",
                region=RAG_REGIONS,
            ),
