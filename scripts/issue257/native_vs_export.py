"""Issue #257: does the levanter native checkpoint differ from its HF export?

The reframing. Earlier work assumed offline (transformers) and online (levanter)
evaluate the *same* weights and chased a forward-pass discrepancy. They do not.

  * Offline evals_v2 loads the **HF export** (transformers). AUPRC 0.159.
  * A levanter eval of that **same HF export** also scores ~0.150
    (iris run exp257-…-hfckpt: macro_avg_auprc 0.1499).
  * The in-training / online eval loads the **native levanter checkpoint**
    (Orbax). AUPRC 0.250.

So both frameworks agree the HF export is a 0.15 model, while the native
checkpoint is a 0.25 model. The discrepancy is not softmax-space, precision,
packing, prefix-sharing or RoPE — offline and online simply evaluate
**different weights**, because the levanter→HF export does not round-trip.

This script proves that locally on CPU, two ways:
  (1) Per-parameter max-abs-diff between the native checkpoint and the HF export,
      both loaded into the *same* levanter model class (so any diff is a real
      weight difference, not a framework artifact). Localizes WHICH tensors the
      export changed.
  (2) The single most-divergent distal variant's 4-nuc LLR under each set of
      weights (plain causal bf16 forward): native should reproduce the online
      -8.05, the export the offline -3.04.

Usage (CPU box with gcloud creds for the gs:// native checkpoint):
  AWS_DEFAULT_REGION=us-east-2 JAX_PLATFORMS=cpu \
    uv run --group genome-s3 --extra marin python scripts/issue257/native_vs_export.py
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jmp
import numpy as np
import polars as pl

jax.config.update("jax_platform_name", "cpu")

import haliax as hax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from haliax.partitioning import set_mesh  # noqa: E402
from haliax.state_dict import to_torch_compatible_state_dict  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from levanter.checkpoint import latest_checkpoint_path, load_checkpoint  # noqa: E402
from levanter.layers.attention import AttentionMask  # noqa: E402
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig  # noqa: E402
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel  # noqa: E402
from levanter.utils.jax_utils import use_cpu_device  # noqa: E402
from levanter.utils.tree_utils import inference_mode  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from marin_dna.data.dna import NUCLEOTIDES  # noqa: E402
from marin_dna.data.genome import Genome  # noqa: E402
from marin_dna.data.transforms import (  # noqa: E402
    _get_nucleotide_token_ids,
    _get_special_token_counts,
    in_seq_var_pos,
    transform_llr_clm,
)

NATIVE = (
    "gs://marin-us-east5/checkpoints/"
    "dna-exp232-zoonomia-v1-0p25b-v4_ccre_non_promoter-v0.1-feca83/checkpoints"
)
EXPORT = "scratch/issue257/ckpt-ccre-4999"
GENOME = (
    "s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/ensembl-release-115/"
    "Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"
)
SCORES = (
    "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores/"
    "exp232-v4_ccre_non_promoter-step-4999/mendelian_traits.parquet"
)
WINDOW = 255
SEQ_LEN = 256
TARGET = dict(chrom="7", pos=156791472, ref="C", alt="A")


def _native_config() -> Qwen3Config:
    # Exactly experiments/parity/exp257_ccre_eval_only.py:_build_model_config.
    return Qwen3Config(
        hidden_dim=1152,
        intermediate_dim=4608,
        num_layers=12,
        num_heads=9,
        num_kv_heads=9,
        max_seq_len=SEQ_LEN,
        rope=Llama3RotaryEmbeddingsConfig(),
        initializer_range=0.02,
    )


def load_native() -> Qwen3LMHeadModel:
    cfg = _native_config()
    cp = latest_checkpoint_path(NATIVE)
    print(f"[native] latest checkpoint: {cp}")
    # The checkpoint was saved on TPU with the vocab padded for partitioning;
    # try the plausible padded sizes until the Orbax shapes line up.
    for vsize in (8, 7, 16, 64, 128, 256):
        try:
            with use_cpu_device():
                Vocab = hax.Axis("vocab", vsize)
                template = eqx.filter_eval_shape(cfg.build, Vocab, key=jax.random.PRNGKey(0))
                model = load_checkpoint(template, cp, subpath="model", axis_mapping={})
            print(f"[native] loaded with Vocab={vsize}")
            return inference_mode(model, True)
        except Exception as e:  # noqa: BLE001
            print(f"[native] Vocab={vsize} failed: {type(e).__name__}: {str(e)[:120]}")
    raise RuntimeError("could not load native checkpoint at any candidate vocab size")


def load_export() -> Qwen3LMHeadModel:
    cfg0 = Qwen3Config()
    conv = cfg0.hf_checkpoint_converter(ref_checkpoint=EXPORT)
    cfg = conv.config_from_hf_config(conv.hf_config_from_hf_checkpoint(EXPORT))
    model = conv.load_pretrained(Qwen3LMHeadModel, ref=EXPORT, config=cfg, dtype=jnp.float32)
    return inference_mode(model, True)


def diff_weights(native: Qwen3LMHeadModel, export: Qwen3LMHeadModel) -> None:
    sd_n = to_torch_compatible_state_dict(native)
    sd_e = to_torch_compatible_state_dict(export)
    keys = sorted(set(sd_n) | set(sd_e))
    print(f"\n=== per-parameter native-vs-export diff ({len(keys)} tensors) ===")
    rows = []
    for k in keys:
        if k not in sd_n or k not in sd_e:
            print(f"  {k:60s} MISSING in {'native' if k not in sd_n else 'export'}")
            continue
        a = np.asarray(sd_n[k]).astype(np.float64)
        b = np.asarray(sd_e[k]).astype(np.float64)
        if a.shape != b.shape:  # embed/lm_head: native padded vocab vs export trimmed
            m = min(a.shape[0], b.shape[0])
            note = f"  [shape {a.shape} vs {b.shape}; compare [:{m}]]"
            a, b = a[:m], b[:m]
        else:
            note = ""
        md = float(np.abs(a - b).max())
        rel = md / (float(np.abs(a).max()) + 1e-9)
        rows.append((md, rel, k, note))
    rows.sort(reverse=True)
    print(f"  {'max|Δ|':>11} | {'rel':>9} | tensor")
    for md, rel, k, note in rows:
        print(f"  {md:>11.4e} | {rel:>9.2e} | {k}{note}")
    allbig = [k for md, rel, k, _ in rows if md > 1e-3]
    print(f"\n  tensors with max|Δ| > 1e-3: {len(allbig)} / {len(rows)}")


def variant_llr(model: Qwen3LMHeadModel, inp: dict) -> float:
    mp = jmp.get_policy("p=f32,c=bfloat16")
    model = mp.cast_to_compute(model)
    Batch = hax.Axis("batch", 1)
    Pos = hax.Axis("position", SEQ_LEN)
    mask = AttentionMask.causal()

    def logits(ids_np):
        ids = hax.named(jnp.asarray(ids_np[None, :], dtype=jnp.int32), (Batch, Pos))
        lg = model(ids, attn_mask=mask)
        return np.asarray(lg.astype(jnp.float32).array)[0]

    nuc_ids = inp["nuc_ids"]
    n2i = {int(t): i for i, t in enumerate(nuc_ids)}

    def lsm(rowl):
        z = rowl[nuc_ids].astype(np.float64)
        return z - (np.log(np.exp(z - z.max()).sum()) + z.max())

    lr, la = logits(inp["ref_ids"]), logits(inp["alt_ids"])
    p, L = inp["var_pos"], SEQ_LEN
    tot = 0.0
    for i in range(p - 1, L - 1):
        tot += lsm(la[i])[n2i[int(inp["alt_ids"][i + 1])]] - lsm(lr[i])[n2i[int(inp["ref_ids"][i + 1])]]
    return float(tot)


def build_input(tok) -> dict:
    genome = Genome(GENOME)
    row = pl.read_parquet(SCORES).filter(
        (pl.col("subset") == "distal")
        & (pl.col("chrom").cast(pl.Utf8) == TARGET["chrom"])
        & (pl.col("pos") == TARGET["pos"])
        & (pl.col("ref") == TARGET["ref"])
        & (pl.col("alt") == TARGET["alt"])
    ).to_pandas().to_dict("records")[0]
    rec = transform_llr_clm(row, tok, genome, WINDOW, "+")
    var_pos = in_seq_var_pos(WINDOW, "+") + _get_special_token_counts(tok)[0]
    nuc_map = _get_nucleotide_token_ids(tok)
    nuc_ids = np.array([nuc_map[n] for n in NUCLEOTIDES])
    ref_ids = rec["input_ids"].numpy().astype(np.int64)
    alt_ids = ref_ids.copy()
    alt_ids[var_pos] = int(rec["alt_token_id"])
    return dict(ref_ids=ref_ids, alt_ids=alt_ids, var_pos=var_pos, nuc_ids=nuc_ids)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(EXPORT)
    inp = build_input(tok)
    print(f"[target] {TARGET['chrom']}:{TARGET['pos']} {TARGET['ref']}>{TARGET['alt']}  "
          "(offline/torch -3.21 ; online/levanter -8.05)")

    mesh = Mesh(np.asarray(jax.devices()).reshape(1, 1, 1), ("replica", "data", "model"))
    with set_mesh(mesh):
        native = load_native()
        export = load_export()
        diff_weights(native, export)
        llr_native = variant_llr(native, inp)
        llr_export = variant_llr(export, inp)

    print("\n=== this variant's 4-nuc LLR (plain causal bf16 forward) ===")
    print(f"  native checkpoint : {llr_native:+.4f}   (online reference -8.05)")
    print(f"  HF export         : {llr_export:+.4f}   (offline reference -3.21)")


if __name__ == "__main__":
    main()
