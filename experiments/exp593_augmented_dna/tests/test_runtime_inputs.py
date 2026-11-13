import asyncio
import hashlib
import io
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from botocore.response import StreamingBody
from exp586_moe.corpus import RawWindows

from exp593_dna.corpus import AnnotatedWindows
from exp593_dna.inputs import verified_download


def test_runtime_join_preserves_native_case_and_source_indices(tmp_path):
    sequences = ["ACGTacgt" * 1024, "NNacgtAC" * 1024]
    path = tmp_path / "raw.parquet"
    pq.write_table(pa.table({"seq": sequences}), path, row_group_size=1)
    inventory = [
        {
            "uri": str(path),
            "bytes": path.stat().st_size,
            "etag": None,
            "rows": 2,
            "row_groups": [1, 1],
        }
    ]
    metadata = pa.Table.from_pylist(
        [
            {
                "row": i,
                "split": "train",
                "assembly": "Zmays",
                "chrom": "1",
                "start": 100,
                "end": 12100,
                "strand": strand,
            }
            for i, strand in enumerate(("+", "-"))
        ]
    )
    source = AnnotatedWindows(RawWindows(inventory), metadata, "train", expected_rows=2)
    windows = asyncio.run(source.get_batch([1, 0, 1]))
    assert [w.row for w in windows] == [1, 0, 1]
    assert [w.seq for w in windows] == [sequences[1], sequences[0], sequences[1]]
    assert windows[0].strand == "-" and windows[1].strand == "+"
    assert source.as_sync_dataset().get_batch([0])[0] == windows[1]
    with pytest.raises(ValueError, match="row/split"):
        AnnotatedWindows(
            RawWindows(inventory), metadata.take([1, 0]), "train", expected_rows=2
        )
    with pytest.raises(ValueError, match="row/split"):
        AnnotatedWindows(RawWindows(inventory), metadata, "validation", expected_rows=2)


def test_input_download_rejects_corruption_and_reuses_only_verified_cache(tmp_path):
    class Client:
        payload = b"native prep input"
        reads = 0

        def get_object(self, **options):
            assert options["IfMatch"] == '"source-etag"'
            self.reads += 1
            return {"Body": StreamingBody(io.BytesIO(self.payload), len(self.payload))}

    client = Client()
    path = tmp_path / "cached.json"
    digest = hashlib.sha256(client.payload).hexdigest()
    options = {"size": len(client.payload), "etag": '"source-etag"'}
    verified_download(client, "s3://owner/input", path, digest, **options)
    verified_download(client, "s3://owner/input", path, digest, **options)
    assert client.reads == 1
    path.write_bytes(b"corrupt local file")
    verified_download(client, "s3://owner/input", path, digest, **options)
    assert client.reads == 2 and path.read_bytes() == client.payload
    path.unlink()
    client.payload = b"corrupt remote bytes"
    with pytest.raises(ValueError, match="identity"):
        verified_download(client, "s3://owner/input", path, digest, **options)
    assert not path.exists() and not path.with_suffix(".json.part").exists()


def test_distributed_batch_counts_actual_input_and_unit_scored_base_loss():
    import jax
    import numpy as np
    from exp586_moe.data_parallel import replicated_mesh
    from tokenizers import Tokenizer, models, pre_tokenizers

    from exp593_dna.batches import distributed_batch
    from exp593_dna.pipeline import (
        ALPHABET,
        AugmentationConfig,
        Augmenter,
        Encoder,
        RawWindow,
    )
    from exp593_dna.schedule import TokenClock

    tokenizer = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, **{c: i + 1 for i, c in enumerate(ALPHABET)}},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    config = replace(
        AugmentationConfig(), length_buckets=(2048,), bucket_probabilities=(1.0,)
    )
    augmenter = Augmenter(Encoder(tokenizer), config)

    class Dataset:
        dna = "ACGTacgt" * 1024

        def get_batch(self, occurrences):
            return [
                RawWindow("Zmays", "1", 0, 12000, "+", self.dna, occurrence % 17)
                for occurrence in occurrences
            ]

    mesh = replicated_mesh()
    with jax.set_mesh(mesh):
        result = distributed_batch(Dataset(), augmenter, mesh, 0)
        replay = distributed_batch(Dataset(), augmenter, mesh, 0)
        assert result.input_fingerprint == replay.input_fingerprint
        assert result.counts["examples"] == 64
        assert (
            result.counts["fim_examples"]
            == result.mixture["short_fim"] + result.mixture["long_fim"]
        )
        assert result.model_batch.tokens.shape == (64, 2048)
        weights = np.asarray(result.model_batch.loss_weight)
        assert np.array_equal(weights, (weights > 0).astype(np.float32))
        assert int(weights.sum()) == result.counts["loss_targets"]
        clock = TokenClock().advance(**result.counts)
        assert clock.input_tokens == 64 * 2048 and clock.updates == 1
        assert 0 < clock.loss_targets < clock.input_tokens - 64
        different = Dataset()
        different.dna = "TGCAacgt" * 1024
        assert (
            distributed_batch(different, augmenter, mesh, 0).input_fingerprint
            != result.input_fingerprint
        )
