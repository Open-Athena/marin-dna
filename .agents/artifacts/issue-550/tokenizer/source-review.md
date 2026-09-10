# Production tokenizer review, 2026-09-10

Reviewed the source wheels locked by experiment commit [57c4124e24b313215f9d7662cf3eb9faac3f9200](https://github.com/Open-Athena/marin-dna/blob/57c4124e24b313215f9d7662cf3eb9faac3f9200/experiments/exp550_rag_five_regions/uv.lock).
All five Marin packages are version `0.2.106.dev34338714012`; each downloaded wheel's byte count and SHA-256 matched that lockfile before source extraction.
Package URLs, hashes, sizes, and upload times are retained in [pinned-wheels.json](pinned-wheels.json).
Extracted source is under `/tmp/issue550-pinned-tokenizer-review`.
No cloud resources or repository files were changed.

## Confirmed latent writer failure

In `zephyr/writers.py:523–528`, `ThreadedBatchWriter.close()` calls blocking `queue.put(_SENTINEL)` before checking `_error`.
If the writer fails after the last successful producer submission while the queue is full, close cannot enqueue the sentinel, so it never reports the recorded write failure.
The foreground `submit()` path checks `_error` with timed queue insertion; the closure path lacks that protection.
This is a failure-handling bug, not evidence that the current production run deadlocked.

Immutable source: [locked Zephyr wheel](https://files.pythonhosted.org/packages/4e/b7/0eed31bc8bc623c77915d16cc790344c446fcc3ce02eea1550faa0715929/marin_zephyr-0.2.106.dev34338714012-py3-none-any.whl).
Its SHA-256 is `a099d5ff4d00f61c7095078d61e1e237c800d3674a6abe8928488f1baa779bba`.
Reproducer: [threaded-writer-close-repro.py](threaded-writer-close-repro.py) loads only the exact writer class and queue iterator with Python's standard library.
Invoke it with the extracted `zephyr/writers.py` path as its only argument.
Observed result: `{'writer_failed': True, 'queued_batches': 1, 'close_blocked_in_sentinel_put': True}`.
The closer is a daemon thread and the reproducer exits immediately after the bounded check, leaving no process behind.
A fix should use timed sentinel insertion while checking the background error, with a regression test for failure during closure while the queue is full.

## Ordinary backpressure and throughput

`ThreadedBatchWriter` defaults to 128 queued batches (`zephyr/writers.py:495`).
The experiment writes 128 records per batch; each has two 10,240-element arrays, int32 and float32.
Each batch contains 10 MiB of array payload, allowing 1.25 GiB per queue and 2.5 GiB across the two active shard writers, before Python, Arrow, compression, and TensorStore overhead.
LocalClient does not enforce an operating-system RAM limit from the `2g` resource declaration.
The measured approximately 4.9 GB RSS is compatible with this implementation and does not by itself establish an OOM on the 48 GiB allocation.
Increasing batch size without reducing queue depth increases memory proportionally: batch size 512 would permit 10 GiB of queued arrays across two writers.

`levanter/store/cache.py:1044–1052` increments the logged record count on submission, before its write finishes.
A shard can have approximately 16,384 queued records plus its current write outstanding.
Completion requires draining the queue, writing the shard ledger, moving the temporary GCS sibling prefix, and creating `.success` (`cache.py:1037–1057`, `rigging/filesystem/atomic.py:38–50`).
The split ledger is only consolidated after all shards return (`marin/processing/tokenize/store_builder.py:183–202`).
Thus queued counts and absent top-level ledgers are insufficient evidence of a stall.
The parent observed two CDS shards durably complete and the next two advancing at approximately 16:43 UTC.

Immutable writer callsite: [locked Levanter wheel](https://files.pythonhosted.org/packages/0f/b5/49183dd9348fbfc03f8072d38d6e00242d15ef026f24532ed99c201869ab/marin_levanter-0.2.106.dev34338714012-py3-none-any.whl), SHA-256 `56be114d144da80c19ba5a45d43a8c81882f8ae72cac51514969c9c09daf2911`.
`jagged_array.py:438–449` waits for data and offset writes to commit, then separately commits row count, for each field on each batch.
The source defines 512 MiB outer Zarr shards and 1 MiB inner chunks for the four-byte data fields (`jagged_array.py:27–28, 723–731`).
Frequent small synchronous cloud commits are a plausible throughput bottleneck; their actual network cost was not profiled here.
The roughly 1.11 million fixed-length rows expand to approximately 91 GB of array data before compression, even though the published Parquet input is approximately 1.95 GB.
Reader memory is bounded by a Parquet row group, which is converted to a Python list before downstream windowing (`zephyr/readers.py:38–88, 276–346`); a 64-record tokenization window does not impose a 64-record input-reader memory bound.

The earlier Iris worker reconcile failures cannot be attributed to tokenizer OOM or this closure bug from the available observations.
