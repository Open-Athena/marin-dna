# Issue 515: online loss-based token selection

This independent locked project vendors `Open-Athena/glm-experiments` at commit `b46cf87c2926201473797f9b00c13e1781c16403` and adds the experiment registered in MarinDNA issue #515.
The original upstream layout is retained, while maintained package code lives under `src/glm_experiments/`.

CSV files under each run directory are authoritative.
W&B is optional and an initialization failure falls back to CSV without stopping training.

## No-GPU preflight

Run the bounded source-case audit before any paid launch:

```bash
uv sync --locked --group dev
uv run --locked exp515-audit \
  --output ../../.agents/artifacts/515-online-loss-selection/case-distribution-audit.json
uv run --locked pytest tests/test_selection.py tests/test_exp515.py
```

The audit streams 128 fixed-seed examples from each of the 108 represented species.
It records lowercase-base fraction, completely lowercase sequence fraction, eligible-target quantiles, and uppercase/lowercase run-length quantiles.
The paid runner refuses to start training if this committed audit requests the preregistered RefSeq fallback.

## Evaluation endpoint

The primary endpoint is the `tss_proximal` subset of `marin-dna/evals_mendelian_traits` at revision `4aed58e50c5dea0b878a665007af2ef9e5108e9f`.
The pinned train split contains 2,050 TSS-proximal variants in 205 matched groups, with 205 positive variants, on odd autosomes and chromosome X.
The source `pos` field is VCF-like and 1-based; reference-window extraction converts it to 0-based, half-open coordinates at the FASTA boundary.
Per-variant scores and summary JSON are written for the bridge, every midpoint, and every endpoint.

## Paid launch

The launch must reference a clean, pushed, full commit SHA:

```bash
uv run --locked python launch.py --commit "$COMMIT" --execute --retry-until-up
```

The Sky task provisions exactly one Lambda A100, forwards Google and AWS credentials as secrets, runs the targeted remote tests, executes the canary and gated matrix, uploads curated immutable artifacts below `s3://oa-bolinas/issues/515/online-loss-selection/v1/`, and shuts the instance down.
The runner stops GPU compute at an estimated $28 and preserves the $30 all-in cap.

## Resume and data contract

The training plan stores 255-byte sequences and uint16 species IDs in fixed-width checksumed files.
Every arm consumes the same row interval.
Each full Lightning checkpoint records the plan checksum, exact next sample ID, effective batch, model state, optimizer state, scheduler state, selector RNG, and Python, NumPy, PyTorch, and CUDA RNG states.
An intentional bridge-to-arm fork resets only the arm selector stream; a same-arm resume restores it exactly.

All genomic coordinates are 0-based and half-open internally.
The newer Mendelian dataset's VCF-like 1-based `pos` field is converted only when extracting the 255-bp reference window.
