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
frozen pretrained Tn5/DNase bias model). With `--all-chroms` (#259) it trains on
**all** chromosomes (1-22,X,Y) — not leakage, since accessibility is trained on
the reference genome only and VEP is zero-shot.

Loss = `beta * multinomial_nll(profile) + alpha * mse(log_counts)` (the vendored
ChromBPNet objective), `alpha = median_count / 10`. **No accessibility validation
loop** (#259): it trains a **fixed budget** (`--max-steps`) with a **WSD** LR
schedule (`--lr-scheduler wsd`, warmup `--warmup-frac` 0.01 / decay `--decay-frac`
0.2) and saves the final checkpoint — the decay lands the endpoint, so there's no
early-stopping and no monitored selection.

**Logged to W&B** (`chrombpnet-eval` project): per-step `train_loss` /
`train_profile_loss` / `train_count_loss` / `grad_norm` / `lr` (the in-training
health signals — there is no accessibility val metric).

With `--qtl-eval`, also logs the **eval target** every `--qtl-every-steps`
**global** optimizer steps (decoupled from epoch boundaries) — `qtl_caqtl_pearson`,
`qtl_dsqtl_pearson`, and their mean `qtl_avg_pearson`: the signed Pearson of the
model's predicted `log2 FC` of counts vs the observed QTL effect, over the
train-split **positives** (caqtl 3,173 + dsqtl 309). It reuses the staged fasta
via `--qtl-chrom-prefix chr`. (The full AUROC/AUPRC benchmark over all variants is
a separate offline pass on the final checkpoint.)

### Data (ARSENAL Synapse `syn72513540` + a chr-prefixed hg38 fasta)

`filtered.peaks.bed` (`syn73665410`), `filtered.nonpeaks.bed` (`syn73665411`),
`GM12878_unstranded.bw` (`syn73665418`), `bias_model_scaled.h5` (`syn73665413`),
hg38 fasta (`syn60756064`) + `.fai` (`syn62284997`), `hg38.chrom.sizes` (UCSC).
Needs `SYNAPSE_AUTH_TOKEN`. The sky launch stages all of these.

### Run on SkyPilot (recommended — real data is large)

```bash
# Smoke (~60 steps, logs every step, online W&B) — confirm the pipeline works
sky launch -c cbp-onehot scripts/chrombpnet_eval/sky/run_onehot.yaml \
    --env SYNAPSE_AUTH_TOKEN --env WANDB_API_KEY --env HF_TOKEN -i 60 --down
# #259 all-chroms fixed-budget WSD baseline (needs a 32 GB+ host)
sky launch -c cbp-259 scripts/chrombpnet_eval/sky/run_onehot.yaml --memory 32+ \
    --env SYNAPSE_AUTH_TOKEN --env WANDB_API_KEY --env HF_TOKEN \
    --env SMOKE=0 --env ALL_CHROMS=1 -i 30 --down
```

Stability + tuning knobs: `--matmul-precision` (default `highest` = full fp32
matmuls), `--grad-clip` (global-norm, default 1000) and the WSD warmup taming an
early gradient spike that can diverge to NaN (see #247). Plus `--lr` (default
1e-3), `--precision {32,bf16-mixed}` (the forward is bf16-safe), `--batch-size`,
`--seed`, `--log-every-n-steps`, `--compile`.

**#259 (strengthen + simplify).** `--all-chroms` trains on all of 1-22,X,Y (not
leakage — VEP is zero-shot on the reference; needs a 32 GB+ host). `--max-steps`
sets the fixed step budget; `--lr-scheduler wsd` runs Warmup-Stable-Decay
(`--warmup-frac` 0.01 / `--decay-frac` 0.2) over it; `--qtl-every-steps` sets the
global-step cadence of the QTL eval. Pick the budget from where the online
`qtl_avg_pearson` curve plateaus.

### Local (synthetic) smoke

The training loop + instrumentation are covered on CPU/synthetic data by
`tests/pipelines/chrombpnet_eval/test_lit.py` and `test_onehot.py`
(`uv run --extra chrombpnet pytest tests/pipelines/chrombpnet_eval/`). A real
data run needs a GPU + the staged GM12878 data, so use the sky launch above.
