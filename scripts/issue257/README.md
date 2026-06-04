# issue #257 — online↔offline VEP AUPRC divergence: reproduction scripts

Investigation of [Open-Athena/marin-dna#257](https://github.com/Open-Athena/marin-dna/issues/257):
the online (in-training `lm_eval`) and offline (`evals_v2`) VEP eval disagree on
`distal/fwd` AUPRC for `exp232 v4_ccre_non_promoter @ step-4999` (online 0.251 vs
offline 0.159).

**Conclusion: it's a missing `[BOS]` token.** The online (in-training
`lm_eval`) harness forwards each variant sequence **without** the leading
`[BOS]` token that the model was trained with and that offline `evals_v2`
correctly prepends. The online sequence is 255 tokens (no BOS); the offline one
is 256 (`[BOS]` + 255 nt). This no-BOS distribution shift changes every variant's
LLR — e.g. `chr7:156791472 C>A`: **−3.2 with BOS → −8.0 without** — and on the
`distal/+` subset shifts AUPRC from **0.159** (offline, with-BOS, correct) to
**0.2501** (online, no-BOS).

The effect is **framework-independent**: torch reproduces −8.06 without BOS and
−3.16 with BOS, matching the online and offline references respectively; levanter
agrees (−7.90 / −3.04). Earlier hypotheses are all refuted — softmax space,
packing, prefix-sharing, precision, RoPE-scaling, and (crucially) the
"transformers-vs-levanter forward divergence" of #261: the two frameworks agree
to <0.2 nats once given the **same** BOS condition, and the native checkpoint and
its HF export are **bit-identical** (135/135 tensors, max|Δ|=0). See the issue
comments for the full chain.

## Setup

```bash
# checkpoint (the exact one in the issue) -> scratch/ (gitignored data dir)
mkdir -p scratch/issue257/ckpt-ccre-4999
gcloud storage cp -r \
  "gs://marin-us-east5/checkpoints/dna-exp232-zoonomia-v1-0p25b-v4_ccre_non_promoter-v0.1-feca83/hf/step-4999/*" \
  scratch/issue257/ckpt-ccre-4999/
export AWS_DEFAULT_REGION=us-east-2   # S3 (scores parquet, genome)
```

Run from the repo root with `uv run --group genome-s3 python scripts/issue257/<script>.py`
(the `genome-s3` group lets pyfaidx read GRCh38 from S3). Scripts read/write
ephemeral data under `scratch/issue257/`.

## Scripts → what each shows

**The root-cause scripts (read these first):**

| script | shows |
| --- | --- |
| `bos_test.py` | **The smoking gun.** Single most-divergent variant, both stacks, with-BOS vs no-BOS: torch −3.16/−8.06, levanter −3.04/−7.90. no-BOS matches online (−8.05), with-BOS matches offline (−3.21). Gap is entirely the BOS prefix, framework-independent. |
| `bos_auprc.py` | End-to-end: distal/+ AUPRC over all 580 variants, with-BOS (≈0.159, reproduces offline) vs no-BOS (≈0.2501, reproduces online). |
| `native_vs_export.py` | Native levanter checkpoint vs its HF export, loaded into the *same* levanter class: **bit-identical** (135/135 tensors, max\|Δ\|=0). Refutes "lossy export". Both give the same LLR (−3.04) under a plain forward. |
| `compare_stacks.py` | torch vs levanter, single sequence, fp32+bf16, per-layer residual diffs. A plain levanter forward gives the **offline** value (−3.04), not the online (−8.05) → the online value is not a property of the levanter *framework* (it's the harness input). |

**The earlier elimination scripts:**

| script | shows |
| --- | --- |
| `verify_softmax_space.py` | Dual kernel: per-variant LLR in **4-nuc** (offline) vs **full-vocab** (online) softmax from *identical logits*. 4-nuc reproduces the offline parquet (corr 0.9997); full-vocab gives the **same** AUPRC (Δ=0.0000) → softmax space is not the cause. |
| `diag_leakage.py` | Special-token probability mass at scored positions ≈ 1e-5 (max 4e-4) → the "differential leakage" premise is false; 4-nuc ≈ full-vocab. |
| `compare_inputs.py` | The online harness dataset (`evals_mendelian_traits_harness_255`) vs offline on-the-fly extraction: all 580 distal inputs byte-identical *as nucleotide content* (the BOS difference is in tokenization, not the stored strings). |
| `compare_offline_online.py` | Offline `evals_v2` metrics vs the in-training online metric for the two *clean* finished arms (utr3, bg): agree to ~0.01 across ~50 cells → offline==online is the norm; ccre/distal/fwd is the lone outlier. |
| `full_forward_test.py` | torch full-forward (no KV-cache) == prefix-share == 0.158 → prefix-sharing/full-forward is not the cause. |
| `diff_per_variant.py` | Per-variant diff of online (levanter, GCS dump) vs offline (kernel) LLRs. ~6 distal positives at `chr7:156791470–480` drive the gap (these are the variants the BOS shift moves most). |
| `rope_test.py` | torch RoPE-scaling A/B: disabling llama3 `rope_scaling` → 0.1585 ≈ as-is 0.1589 → RoPE config is not the cause. |
| `wandb_check.py` | Helper: compact status of an `exp257` online-repro wandb run. |

The **online reproduction** (real levanter `lm_eval` on the checkpoint, which
gives 0.2501) is the eval-only harness at
[`experiments/parity/exp257_ccre_eval_only.py`](../../experiments/parity/exp257_ccre_eval_only.py)
— launched on iris (`v6e-4`), with `MAX_PACKED_SEGMENTS` / `DUMP_GCS` knobs for
the packing and per-variant-dump experiments. `diff_per_variant.py` consumes its
`DUMP_GCS` JSONL.
