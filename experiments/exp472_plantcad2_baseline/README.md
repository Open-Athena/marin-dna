# exp472 PlantCAD2 baseline

Baseline Qwen3 language-model training on
[`plantcad/Angiosperm_65_genomes_8192bp`](https://huggingface.co/datasets/plantcad/Angiosperm_65_genomes_8192bp)
with the `kuleshov-group/PlantCAD2-Small-l24-d0768` tokenizer. Every example is
exactly 8,192 tokens. Training uses the train split, validation uses the
validation split, and test remains held out. The default training target is
206,145 steps, or 10 epochs at global batch size 128.

Each training occurrence is independently reverse-complemented with 50%
probability. The deterministic choice is keyed by its absolute training-stream
offset, so repeated epochs receive fresh choices while checkpoint resume and
retries reproduce the same stream. Complements map `a` to `t`, `c` to `g`, and
vice versa; `[PAD]`, `[MASK]`, and `[UNK]` map to themselves. Validation and
test are not augmented.

Learning rate follows a linear warmup-stable-decay schedule: 10% warmup, 70%
at the trial's peak learning rate, and 20% linear decay to zero. AdamW uses
Levanter's default bad-step skipping (128-step history and six-sigma threshold).

Temporary recovery checkpoints are saved every 15 minutes. Production runs
retain ten permanent checkpoints: nine approximately evenly spaced checkpoints
plus the forced final checkpoint. Validation runs twenty times per production
trial (twice the permanent-checkpoint count) and covers the full validation split
each time. At the default 206,145 steps, evaluation runs every 10,308 steps plus
the forced final evaluation. Corrected production runs use the default `v2`
run/checkpoint suffix; the short-lived pre-correction `v1` runs are not resumed.
Short smoke runs retain each completed step.

Selected models are continued in numbered stages. Stage 2 resumes the two
selected Stage-1 runs from retained pre-cooldown checkpoint `step-329840`,
restores the full trainer and optimizer state, and resets the WSD cycle for the
next stage. Its production run IDs use the suffix `train-s02-v1`.

The sweep contains 14 trials over learning rates `1e-4`, `2e-4`, `5e-4`, and
`1e-3`, and weight decays `0.1`, `0.2`, `0.8`, and `1.6`. It omits the
low-LR/low-WD corner (`1e-4`, `0.1`) and high-LR/high-WD corner (`1e-3`,
`1.6`).

## Dataset size

| Split | Examples | Tokens |
| --- | ---: | ---: |
| Train | 2,638,656 | 21,615,869,952 |
| Validation | 329,832 | 2,701,983,744 |
| Test | 329,832 | 2,701,983,744 |
| **Total** | **3,298,320** | **27,019,837,440** |

## Token frequencies

These counts were computed directly from every `input_ids` value in the
completed CoreWeave cache.

| ID | Token | Train | Validation | Test | Total | Frequency |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | `[PAD]` | 0 | 0 | 0 | 0 | 0.000000% |
| 1 | `[MASK]` | 0 | 0 | 0 | 0 | 0.000000% |
| 2 | `[UNK]` | 35,133,353 | 4,390,651 | 4,335,504 | 43,859,508 | 0.162323% |
| 3 | `a` | 6,524,411,566 | 815,618,836 | 815,612,853 | 8,155,643,255 | 30.183909% |
| 4 | `c` | 4,022,660,407 | 502,676,023 | 502,911,954 | 5,028,248,384 | 18.609469% |
| 5 | `g` | 4,148,055,652 | 518,501,778 | 518,449,933 | 5,185,007,363 | 19.189632% |
| 6 | `t` | 6,885,608,974 | 860,796,456 | 860,673,500 | 8,607,078,930 | 31.854666% |

## Storage

CoreWeave:

- Token cache: `s3://marin-us-east-02a/MarinDNA/tokenized/plantcad/Angiosperm_65_genomes_8192bp`
- Checkpoints: `s3://marin-us-east-02a/MarinDNA/exp472_plantcad2_baseline/checkpoints/<run>/`

TRC uses a separate in-region copy in every TPU region:

- Token cache: `gs://<regional-marin-bucket>/MarinDNA/tokenized/plantcad/Angiosperm_65_genomes_8192bp`
- Checkpoints: `gs://<regional-marin-bucket>/MarinDNA/exp472_plantcad2_baseline/checkpoints/<run>/`

TRC sweep placement is restricted to v5e, v5p, and v6e slices with 32–512
physical chips according to Marin's `TPU_TOPOLOGIES` table. v5p labels encode
twice their physical chip count, so the allowed v5p labels are `v5p-64` through
`v5p-1024`. Short smoke tests may opt into a smaller slice with
`EXP472_ALLOW_SMALL_TPU_SMOKE=1`; the entry point requires both shortened
`EXP472_STEPS` and a `smoke-` run suffix before honoring that exception.

For example, `us-east5` uses `gs://marin-us-east5`, while `europe-west4` uses
`gs://marin-eu-west4`. GPU and TPU runs share the same path structure; device
type is not part of the checkpoint path.
