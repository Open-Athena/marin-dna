"""Issue #257: is the online↔offline LLR gap a BOS-prefix mismatch?

The harness dataset (bolinas-dna/evals_mendelian_traits_harness_255) stores each
variant as raw nucleotide strings: context (127 nt) + ref/alt_completion (128 nt)
= 255 nt, NO BOS. The offline evals_v2 kernel builds a 256-token input = [BOS] +
255 nt (n_prefix=1). If the levanter lm_eval harness tokenizes the raw strings
*without* prepending BOS, the online model scores a 255-token, no-BOS sequence
while offline scores a 256-token, BOS sequence — different context for the
variant ⇒ different LLR.

This tests it directly on the single most-divergent distal variant
(chr7:156791472 C>A, offline -3.21 / online -8.05) under identical weights:
compute the full-sequence 4-nuc LLR with-BOS vs no-BOS, in BOTH torch and
levanter. If with-BOS ≈ -3.2 and no-BOS ≈ -8.0 in *both* stacks, the gap is the
BOS prefix (a framework-independent input difference), not a forward-pass bug.

Usage:
  AWS_DEFAULT_REGION=us-east-2 JAX_PLATFORMS=cpu \
    uv run --group genome-s3 --extra marin python scripts/issue257/bos_test.py
"""

from __future__ import annotations


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
TARGET = dict(chrom="7", pos=156791472, ref="C", alt="A")


def seq_llr(logits_ref, logits_alt, ref_ids, alt_ids, p, nuc_ids):
    """Full-sequence 4-nuc LLR over [p-1, L-2] (variant + downstream)."""
    n2i = {int(t): i for i, t in enumerate(nuc_ids)}
    L = len(ref_ids)

    def lsm(rowl):
        z = rowl[nuc_ids].astype(np.float64)
        return z - (np.log(np.exp(z - z.max()).sum()) + z.max())

    tot = 0.0
    for i in range(p - 1, L - 1):
        tot += (
            lsm(logits_alt[i])[n2i[int(alt_ids[i + 1])]]
            - lsm(logits_ref[i])[n2i[int(ref_ids[i + 1])]]
        )
    return float(tot)


def build_inputs(tok):
    genome = Genome(GENOME)
    row = (
        pl.read_parquet(SCORES)
        .filter(
            (pl.col("subset") == "distal")
            & (pl.col("chrom").cast(pl.Utf8) == TARGET["chrom"])
            & (pl.col("pos") == TARGET["pos"])
            & (pl.col("ref") == TARGET["ref"])
            & (pl.col("alt") == TARGET["alt"])
        )
        .to_pandas()
        .to_dict("records")[0]
    )
    rec = transform_llr_clm(row, tok, genome, WINDOW, "+")
    n_prefix, _ = _get_special_token_counts(tok)
    var_pos = in_seq_var_pos(WINDOW, "+") + n_prefix  # 128 (= 127 + 1 BOS)
    nuc_map = _get_nucleotide_token_ids(tok)
    nuc_ids = np.array([nuc_map[n] for n in NUCLEOTIDES])

    # WITH BOS: the offline 256-token input (BOS + 255 nt), variant at index 128.
    ref_bos = rec["input_ids"].numpy().astype(np.int64)
    alt_bos = ref_bos.copy()
    alt_bos[var_pos] = int(rec["alt_token_id"])
    assert n_prefix == 1, f"expected 1 BOS, got n_prefix={n_prefix}"

    # NO BOS: drop the BOS prefix → 255 nt, variant now at index 127.
    ref_nobos = ref_bos[n_prefix:].copy()
    alt_nobos = alt_bos[n_prefix:].copy()

    return dict(
        nuc_ids=nuc_ids,
        bos=dict(ref=ref_bos, alt=alt_bos, p=var_pos, L=len(ref_bos)),
        nobos=dict(
            ref=ref_nobos, alt=alt_nobos, p=var_pos - n_prefix, L=len(ref_nobos)
        ),
    )


def torch_logits(model, ids_np):
    with torch.no_grad():
        out = model(torch.tensor(ids_np)[None, :])
    return out.logits[0].float().numpy()


def lev_logits(model, ids_np):
    L = len(ids_np)
    Batch, Pos = hax.Axis("batch", 1), hax.Axis("position", L)
    ids = hax.named(jnp.asarray(ids_np[None, :], dtype=jnp.int32), (Batch, Pos))
    lg = model(ids, attn_mask=AttentionMask.causal())
    return np.asarray(lg.astype(jnp.float32).array)[0]


def main():
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(CKPT)
    inp = build_inputs(tok)
    nuc_ids = inp["nuc_ids"]

    print(
        f"[target] {TARGET['chrom']}:{TARGET['pos']} {TARGET['ref']}>{TARGET['alt']}  "
        "(offline -3.21 ; online -8.05)"
    )
    print(f"  with-BOS: L={inp['bos']['L']} variant@{inp['bos']['p']}")
    print(f"  no-BOS  : L={inp['nobos']['L']} variant@{inp['nobos']['p']}")

    # torch
    tmodel = AutoModelForCausalLM.from_pretrained(CKPT, trust_remote_code=True).eval()
    t_bos = seq_llr(
        torch_logits(tmodel, inp["bos"]["ref"]),
        torch_logits(tmodel, inp["bos"]["alt"]),
        inp["bos"]["ref"],
        inp["bos"]["alt"],
        inp["bos"]["p"],
        nuc_ids,
    )
    t_nobos = seq_llr(
        torch_logits(tmodel, inp["nobos"]["ref"]),
        torch_logits(tmodel, inp["nobos"]["alt"]),
        inp["nobos"]["ref"],
        inp["nobos"]["alt"],
        inp["nobos"]["p"],
        nuc_ids,
    )
    del tmodel

    # levanter
    mesh = Mesh(
        np.asarray(jax.devices()).reshape(1, 1, 1), ("replica", "data", "model")
    )
    with set_mesh(mesh):
        cfg0 = Qwen3Config()
        conv = cfg0.hf_checkpoint_converter(ref_checkpoint=CKPT)
        cfg = conv.config_from_hf_config(conv.hf_config_from_hf_checkpoint(CKPT))
        lmodel = inference_mode(
            conv.load_pretrained(
                Qwen3LMHeadModel, ref=CKPT, config=cfg, dtype=jnp.float32
            ),
            True,
        )
        lmodel = jmp.get_policy("p=f32,c=bfloat16").cast_to_compute(lmodel)
        l_bos = seq_llr(
            lev_logits(lmodel, inp["bos"]["ref"]),
            lev_logits(lmodel, inp["bos"]["alt"]),
            inp["bos"]["ref"],
            inp["bos"]["alt"],
            inp["bos"]["p"],
            nuc_ids,
        )
        l_nobos = seq_llr(
            lev_logits(lmodel, inp["nobos"]["ref"]),
            lev_logits(lmodel, inp["nobos"]["alt"]),
            inp["nobos"]["ref"],
            inp["nobos"]["alt"],
            inp["nobos"]["p"],
            nuc_ids,
        )

    print("\n=== full-sequence 4-nuc LLR (chr7:156791472 C>A) ===")
    print(f"  {'':10} | {'with-BOS':>10} | {'no-BOS':>10}")
    print(f"  {'torch':10} | {t_bos:>10.4f} | {t_nobos:>10.4f}")
    print(f"  {'levanter':10} | {l_bos:>10.4f} | {l_nobos:>10.4f}")
    print("\n  offline reference -3.21 ; online reference -8.05")


if __name__ == "__main__":
    main()
