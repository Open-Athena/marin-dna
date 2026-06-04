"""Issue #257: localize the torch-HF (0.158) vs JAX-levanter (0.251) distal/fwd
divergence on a SINGLE sequence, entirely on CPU — no iris/TPU needed.

Background. The online (levanter) and offline (transformers) VEP evals run the
*same* weights on *byte-identical* inputs, yet the distal/FWD AUPRC differs
(0.159 offline vs 0.251 online). We already ruled out softmax-space, packing,
prefix-sharing, the checkpoint, HF-conversion and RoPE-scaling. What's left is
the forward pass itself: transformers (torch) vs levanter (JAX) Qwen3.

This script runs ONE forward of ONE sequence (the single most-divergent distal
variant, chr7:156791472 C>A — online -8.05 vs offline -3.21) through BOTH stacks
at fp32 and bf16, and reports:

  (1) the per-variant 4-nucleotide LLR in every config, and
  (2) per-layer residual-stream max-abs-diff, to localize the first op that
      diverges.

The fork it settles:
  * levanter-fp32 LLR ~= torch (-3.21)  -> the divergence is bf16 *precision*,
    not math; per-layer bf16 diffs localize the unstable op.
  * levanter-fp32 LLR ~= online (-8.05) -> a genuine *math* difference; per-layer
    fp32 diffs localize the diverging op.

Leading hypothesis going in: levanter attention runs scores+softmax in the
compute dtype (bf16) because AttentionConfig.upcast_attn defaults to False
(layers/attention.py:1749), whereas transformers Qwen3 upcasts the softmax to
fp32. The `--upcast` A/B below tests exactly that: loading levanter with
upcast_attn=True should recover torch's value if attention precision is the cause.

Usage:
  uv run --group genome-s3 --extra marin python scripts/issue257/compare_stacks.py
"""

from __future__ import annotations

import dataclasses

import jax
import jmp
import numpy as np
import polars as pl
import torch

jax.config.update("jax_platform_name", "cpu")

import haliax as hax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from haliax.partitioning import set_mesh  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from levanter.layers.attention import AttentionMask  # noqa: E402
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel  # noqa: E402
from levanter.utils.tree_utils import inference_mode  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from marin_dna.data.dna import NUCLEOTIDES  # noqa: E402
from marin_dna.data.genome import Genome  # noqa: E402
from marin_dna.data.transforms import (  # noqa: E402
    _get_nucleotide_token_ids,
    _get_special_token_counts,
    in_seq_var_pos,
    transform_llr_clm,
)

CKPT = "scratch/issue257/ckpt-ccre-4999"
GENOME = (
    "s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/ensembl-release-115/"
    "Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"
)
SCORES = (
    "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores/"
    "exp232-v4_ccre_non_promoter-step-4999/mendelian_traits.parquet"
)
WINDOW = 255
# The single most-divergent distal/+ positive (online -8.05, offline -3.21).
TARGET = dict(chrom="7", pos=156791472, ref="C", alt="A")


def _logsoftmax_nuc(logits_row: np.ndarray, nuc_ids: np.ndarray) -> np.ndarray:
    """4-nucleotide log-softmax of a single logits row, in fp64."""
    z = logits_row[nuc_ids].astype(np.float64)
    return z - (np.log(np.exp(z - z.max()).sum()) + z.max())


def _seq_llr(
    logits_ref: np.ndarray,
    logits_alt: np.ndarray,
    ref_ids: np.ndarray,
    alt_ids: np.ndarray,
    p: int,
    nuc_ids: np.ndarray,
) -> float:
    """Full-sequence 4-nuc LLR = logP(alt_seq) - logP(ref_seq), matching the
    offline kernel: variant-position term (logits[p-1]) + downstream
    autoregressive terms (logits[p..L-2]); the swapped token at p changes the
    context for every later prediction, so ref/alt suffixes diverge."""
    nuc_to_idx = {int(t): i for i, t in enumerate(nuc_ids)}
    L = len(ref_ids)
    total = 0.0
    for i in range(p - 1, L - 1):  # logits index i predicts token i+1
        za = _logsoftmax_nuc(logits_alt[i], nuc_ids)
        zr = _logsoftmax_nuc(logits_ref[i], nuc_ids)
        total += za[nuc_to_idx[int(alt_ids[i + 1])]] - zr[nuc_to_idx[int(ref_ids[i + 1])]]
    return float(total)


def build_input(tok) -> dict:
    genome = Genome(GENOME)
    df = pl.read_parquet(SCORES).filter(
        (pl.col("subset") == "distal")
        & (pl.col("chrom").cast(pl.Utf8) == TARGET["chrom"])
        & (pl.col("pos") == TARGET["pos"])
        & (pl.col("ref") == TARGET["ref"])
        & (pl.col("alt") == TARGET["alt"])
    )
    assert df.height == 1, f"expected 1 target row, got {df.height}"
    row = df.to_pandas().to_dict("records")[0]
    print(f"[target] {row['chrom']}:{row['pos']} {row['ref']}>{row['alt']} "
          f"label={row['label']}  offline llr_fwd={row['llr_fwd']:.4f}")
    rec = transform_llr_clm(row, tok, genome, WINDOW, "+")
    n_prefix, _ = _get_special_token_counts(tok)
    var_pos = in_seq_var_pos(WINDOW, "+") + n_prefix
    nuc_map = _get_nucleotide_token_ids(tok)
    nuc_ids = np.array([nuc_map[n] for n in NUCLEOTIDES])
    alt_id = int(rec["alt_token_id"])
    ref_ids = rec["input_ids"].numpy().astype(np.int64)
    ref_id = int(ref_ids[var_pos])  # ref token at var_pos by construction
    assert ref_id == nuc_map[TARGET["ref"]], (
        f"ref token {ref_id} != nuc id for {TARGET['ref']} ({nuc_map[TARGET['ref']]})"
    )
    assert alt_id == nuc_map[TARGET["alt"]], (
        f"alt token {alt_id} != nuc id for {TARGET['alt']} ({nuc_map[TARGET['alt']]})"
    )
    alt_ids = ref_ids.copy()
    alt_ids[var_pos] = alt_id
    assert 0 < var_pos < len(ref_ids) - 1
    return dict(
        ref_ids=ref_ids, alt_ids=alt_ids, var_pos=var_pos, nuc_ids=nuc_ids,
        ref_id=ref_id, alt_id=alt_id,
    )


# ---------------------------------------------------------------------------
# torch / transformers
# ---------------------------------------------------------------------------
def run_torch(inp: dict, dtype: torch.dtype) -> dict:
    model = AutoModelForCausalLM.from_pretrained(
        CKPT, trust_remote_code=True, torch_dtype=dtype
    ).eval()

    def fwd(ids_np):
        ids = torch.tensor(ids_np)[None, :]
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        logits = out.logits[0].float().numpy()
        hs = [h[0].float().numpy() for h in out.hidden_states]
        return logits, hs

    logits_ref, hs = fwd(inp["ref_ids"])
    logits_alt, _ = fwd(inp["alt_ids"])
    llr = _seq_llr(logits_ref, logits_alt, inp["ref_ids"], inp["alt_ids"],
                   inp["var_pos"], inp["nuc_ids"])
    del model
    return dict(llr=llr, logits=logits_ref, hiddens=hs)


# ---------------------------------------------------------------------------
# levanter / JAX
# ---------------------------------------------------------------------------
def _load_levanter(upcast_attn: bool) -> Qwen3LMHeadModel:
    cfg0 = Qwen3Config()  # only used to get a converter bound to the right HF class
    conv = cfg0.hf_checkpoint_converter(ref_checkpoint=CKPT)
    hf_cfg = conv.hf_config_from_hf_checkpoint(CKPT)
    cfg = conv.config_from_hf_config(hf_cfg)
    cfg = dataclasses.replace(cfg, upcast_attn=upcast_attn)
    model = conv.load_pretrained(Qwen3LMHeadModel, ref=CKPT, config=cfg, dtype=jnp.float32)
    return inference_mode(model, True)


def run_levanter(inp: dict, model: Qwen3LMHeadModel, compute_dtype: jnp.dtype) -> dict:
    mp = jmp.get_policy(f"p=f32,c={'bfloat16' if compute_dtype == jnp.bfloat16 else 'float32'}")
    model = mp.cast_to_compute(model)

    seq_len = inp["ref_ids"].shape[0]
    Batch = hax.Axis("batch", 1)
    Pos = hax.Axis("position", seq_len)
    mask = AttentionMask.causal()

    def fwd(ids_np):
        ids = hax.named(jnp.asarray(ids_np[None, :], dtype=jnp.int32), (Batch, Pos))
        x = model.embeddings.embed(ids)
        hs = [np.asarray(x.astype(jnp.float32).array)[0]]  # [seq, hidden]
        for layer in model.transformer.layers.unstacked():
            x = layer(x, mask, key=None, pos_ids=None)
            hs.append(np.asarray(x.astype(jnp.float32).array)[0])
        x_normed = model.transformer.norm(x)
        # Online upcasts activations + lm_head to fp32 before the final projection.
        logits = model.embeddings.unembed(x_normed.astype(jnp.float32))
        return np.asarray(logits.astype(jnp.float32).array)[0], hs

    logits_ref, hs = fwd(inp["ref_ids"])
    logits_alt, _ = fwd(inp["alt_ids"])
    llr = _seq_llr(logits_ref, logits_alt, inp["ref_ids"], inp["alt_ids"],
                   inp["var_pos"], inp["nuc_ids"])
    return dict(llr=llr, logits=logits_ref, hiddens=hs)


def _maxdiff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def main() -> None:
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(CKPT)
    inp = build_input(tok)

    print("\n=== loading + forward (CPU) ===")
    t_f32 = run_torch(inp, torch.float32)
    t_bf16 = run_torch(inp, torch.bfloat16)

    # Single-device mesh with a `data` axis — mirrors the harness's
    # `with config.trainer.use_device_mesh()` so the HF loader's best-effort
    # sharding finds a `data` axis (size 1 ⇒ no real sharding).
    mesh = Mesh(np.asarray(jax.devices()).reshape(1, 1, 1), ("replica", "data", "model"))
    with set_mesh(mesh):
        lev = _load_levanter(upcast_attn=False)
        lev_up = _load_levanter(upcast_attn=True)
        l_f32 = run_levanter(inp, lev, jnp.float32)
        l_bf16 = run_levanter(inp, lev, jnp.bfloat16)
        l_bf16_up = run_levanter(inp, lev_up, jnp.bfloat16)

    print("\n=== per-variant 4-nuc LLR (target chr7:156791472 C>A) ===")
    print(f"  offline reference (parquet llr_fwd)         : -3.2072")
    print(f"  online  reference (levanter in-training)    : -8.0487")
    print("  ---")
    print(f"  torch    fp32                               : {t_f32['llr']:+.4f}")
    print(f"  torch    bf16                               : {t_bf16['llr']:+.4f}")
    print(f"  levanter fp32  (upcast_attn=F)              : {l_f32['llr']:+.4f}")
    print(f"  levanter bf16  (upcast_attn=F, == online)   : {l_bf16['llr']:+.4f}")
    print(f"  levanter bf16  (upcast_attn=T)              : {l_bf16_up['llr']:+.4f}")

    print("\n=== final-logit max-abs-diff (torch vs levanter) ===")
    print(f"  fp32:  {_maxdiff(t_f32['logits'], l_f32['logits']):.4e}")
    print(f"  bf16:  {_maxdiff(t_bf16['logits'], l_bf16['logits']):.4e}")
    print(f"  bf16 (levanter upcast_attn=T): {_maxdiff(t_bf16['logits'], l_bf16_up['logits']):.4e}")

    print("\n=== per-layer residual-stream max-abs-diff (torch hidden_states[i] vs levanter) ===")
    n = min(len(t_f32["hiddens"]), len(l_f32["hiddens"]))
    print(f"  (torch has {len(t_f32['hiddens'])} hidden_states, levanter captured {len(l_f32['hiddens'])})")
    print(f"  {'idx':>3} | {'fp32 diff':>12} | {'bf16 diff':>12} | {'bf16 upcast':>12}")
    for i in range(n):
        d_f32 = _maxdiff(t_f32["hiddens"][i], l_f32["hiddens"][i])
        d_bf16 = _maxdiff(t_bf16["hiddens"][i], l_bf16["hiddens"][i])
        d_up = _maxdiff(t_bf16["hiddens"][i], l_bf16_up["hiddens"][i])
        tag = "  <- embed" if i == 0 else ""
        print(f"  {i:>3} | {d_f32:>12.4e} | {d_bf16:>12.4e} | {d_up:>12.4e}{tag}")


if __name__ == "__main__":
    main()
