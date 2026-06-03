# chrombpnet_eval — supervised ChromBPNet VEP eval (issue #236)

Train a ChromBPNet-style accessibility head and score caQTL/dsQTL variant
effects, following the [ARSENAL](https://www.biorxiv.org/content/10.64898/2026.02.05.703637v1.full)
recipe. Library code lives in `src/marin_dna/pipelines/chrombpnet_eval/`
(`_vendor/` is the pinned `arsenal-chrombpnet` fork — see its `PROVENANCE.md`);
these scripts are thin drivers.

The `chrombpnet` dependency extra is opt-in: `uv sync --extra chrombpnet`
(PyTorch Lightning etc.; eval contributors who don't train ChromBPNet skip it).

## Scripts

| Script | What it does |
|---|---|
| `m1a_onehot_baseline.py` | **M1a** (no training/GPU): validate the metric+scoring harness against ARSENAL's *precomputed* one-hot ChromBPNet scores on our caqtl/dsqtl splits. |
| `train_onehot.py` | **M1b** (#241): train our own one-hot ChromBPNet on GM12878 DNase and validate the whole training pipeline + W&B instrumentation. |
| `sky/run_onehot.yaml` | SkyPilot launch for `train_onehot.py` (stages the GM12878 data + hg38 fasta). |

## One-hot ChromBPNet training (`train_onehot.py`)

Faithful one-hot ChromBPNet (vendored `ChromBPNet`, 4-channel one-hot input, a
frozen pretrained Tn5/DNase bias model), fp32. ChromBPNet trains on its **regular
DART-Eval chrom splits** (`val = chr6, chr21`; `test = chr5, 10, 14, 18, 20, 22`),
which are orthogonal to the caqtl/dsqtl variant splits.

Loss = `beta * multinomial_nll(profile) + alpha * mse(log_counts)` (the vendored
ChromBPNet objective), `alpha = median_count / 10`. Early-stops + checkpoints on
`val_count_pearson`.

**Logged to W&B** (`chrombpnet-eval` project): per-step `train_loss` /
`train_profile_loss` / `train_count_loss` / `grad_norm` / `lr`; per validation
pass `val_count_pearson` + `val_count_spearman` (predicted vs observed log counts,
the ChromBPNet accessibility metrics) + the val losses.

With `--qtl-eval`, also logs the **online QTL metric** each validation —
`qtl_caqtl_pearson`/`_spearman` and `qtl_dsqtl_pearson`/`_spearman`: the signed
correlation of the model's predicted `log2 FC` of counts vs the observed QTL
effect, over the train-split **positives** (caqtl 3,173 + dsqtl 309). This is the
eval *target* (does it rank QTL effects?), distinct from `val_count_pearson` (the
accessibility-count fit). It reuses the staged fasta via `--qtl-chrom-prefix chr`.

### Data (ARSENAL Synapse `syn72513540` + a chr-prefixed hg38 fasta)

`filtered.peaks.bed` (`syn73665410`), `filtered.nonpeaks.bed` (`syn73665411`),
`GM12878_unstranded.bw` (`syn73665418`), `bias_model_scaled.h5` (`syn73665413`),
hg38 fasta (`syn60756064`) + `.fai` (`syn62284997`), `hg38.chrom.sizes` (UCSC).
Needs `SYNAPSE_AUTH_TOKEN`. The sky launch stages all of these.

### Run on SkyPilot (recommended — real data is large)

```bash
# Smoke: logs every step, validates often, online W&B — confirm the pipeline works
sky launch -c cbp-onehot scripts/chrombpnet_eval/sky/run_onehot.yaml \
    --env SYNAPSE_AUTH_TOKEN --env WANDB_API_KEY -i 60 --down
# Full early-stopped run on the same cluster
sky exec cbp-onehot scripts/chrombpnet_eval/sky/run_onehot.yaml \
    --env SYNAPSE_AUTH_TOKEN --env WANDB_API_KEY --env SMOKE=0
```

Stability + tuning knobs: full fp32 by default (`--matmul-precision highest`),
with `--seed`, `--warmup-steps` (linear LR warmup, default 100) and `--grad-clip`
(global-norm, default 1000) taming an early gradient spike that can diverge to
NaN (see #247). Plus `--lr` (default 1e-3, official), `--lr-scheduler
{none,plateau}`, `--precision {32,bf16-mixed}` (the forward is bf16-safe),
`--batch-size`, `--patience`, `--val-check-interval`, `--log-every-n-steps`,
`--compile`.

### Local (synthetic) smoke

The training loop + instrumentation are covered on CPU/synthetic data by
`tests/pipelines/chrombpnet_eval/test_lit.py` and `test_onehot.py`
(`uv run --extra chrombpnet pytest tests/pipelines/chrombpnet_eval/`). A real
data run needs a GPU + the staged GM12878 data, so use the sky launch above.
