"""Does the offline kernel's prefix-sharing (KV-cache) vs a plain full forward
(what lm_eval does) explain the distal/fwd gap (offline 0.158 vs online 0.251)?

Computes distal/FWD LLR three ways on the same checkpoint+inputs:
  ps_full   = prefix-shared full-vocab (the offline kernel path)         -> expect 0.158
  ff_full   = full-forward full-vocab, fp32 (lm_eval math, no KV-cache)
  ff_full16 = full-forward full-vocab, bf16 (closest to the online run)
All score the same 128 completion positions with the same full-vocab softmax.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin_dna.data.genome import Genome
from marin_dna.data.transforms import (
    _get_nucleotide_token_ids,
    _get_special_token_counts,
    in_seq_var_pos,
    transform_llr_clm,
)
from marin_dna.model.scoring import _repeat_interleave_kv_cache

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


@torch.no_grad()
def full_forward_llr(model, input_ids, alt_tok, var_pos, dtype):
    """lm_eval-style: forward whole ref & alt seqs, sum full-vocab logprob over
    completion positions [var_pos:L], LLR = alt - ref. No KV-cache."""
    B, L = input_ids.shape
    alt_ids = input_ids.clone()
    alt_ids[:, var_pos] = alt_tok
    both = torch.cat([input_ids, alt_ids], 0)  # [2B, L]
    m = model.to(dtype) if dtype != torch.float32 else model
    logits = m(both).logits.float()  # [2B, L, V]
    lp = F.log_softmax(logits, dim=-1)
    # predict token t (t in [var_pos, L-1]) from position t-1
    idx = both[:, var_pos:].unsqueeze(-1)  # [2B, L-var_pos, 1] targets
    pred = lp[:, var_pos - 1 : L - 1]  # [2B, L-var_pos, V]
    tok_lp = (
        pred.gather(-1, idx).squeeze(-1).sum(-1)
    )  # [2B] seq logprob over completion
    ref_lp, alt_lp = tok_lp[:B], tok_lp[B:]
    if dtype != torch.float32:
        model.to(torch.float32)
    return (alt_lp - ref_lp).numpy()


@torch.no_grad()
def prefix_share_full_llr(model, input_ids, alt_tok, var_pos):
    """Prefix-shared full-vocab LLR (offline kernel structure), fp32."""
    from einops import rearrange

    B, L = input_ids.shape
    p = var_pos
    prefix = input_ids[:, :p].contiguous()
    ref_suf = input_ids[:, p:].contiguous()
    alt_suf = torch.cat([alt_tok.unsqueeze(-1), ref_suf[:, 1:]], -1)
    suf = rearrange(torch.stack([ref_suf, alt_suf], 1), "B V L -> (B V) L").contiguous()
    out = model(prefix, use_cache=True, logits_to_keep=1)
    plast = out.logits[:, -1]
    past = _repeat_interleave_kv_cache(out.past_key_values, 2)
    slog = model(suf, past_key_values=past, use_cache=False).logits
    lp = F.log_softmax(slog.float(), -1)
    lp = rearrange(lp, "(B V) L W -> B V L W", B=B)
    lpr, lpa = lp[:, 0, :-1], lp[:, 1, :-1]
    plp = F.log_softmax(plast.float(), -1)
    at_var = plp.gather(-1, alt_tok[:, None]).squeeze(-1) - plp.gather(
        -1, input_ids[:, p][:, None]
    ).squeeze(-1)
    tgt = input_ids[:, p + 1 :].unsqueeze(-1)
    down = (lpa.gather(-1, tgt).squeeze(-1) - lpr.gather(-1, tgt).squeeze(-1)).sum(-1)
    return (at_var + down).numpy()


def main():
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForCausalLM.from_pretrained(CKPT, trust_remote_code=True).eval()
    genome = Genome(GENOME)
    df = pl.read_parquet(SCORES).filter(pl.col("subset") == "distal").to_pandas()
    lab = df["label"].to_numpy()
    n_prefix, _ = _get_special_token_counts(tok)
    nuc = _get_nucleotide_token_ids(tok)
    var_pos = in_seq_var_pos(WINDOW, "+") + n_prefix
    rows = [
        transform_llr_clm(r, tok, genome, WINDOW, "+") for r in df.to_dict("records")
    ]
    input_ids = torch.stack([r["input_ids"] for r in rows])
    alt_tok = torch.tensor([r["alt_token_id"] for r in rows])

    res = {}
    for name, fn in [
        ("ps_full_fp32", lambda b, a: prefix_share_full_llr(model, b, a, var_pos)),
        (
            "ff_full_fp32",
            lambda b, a: full_forward_llr(model, b, a, var_pos, torch.float32),
        ),
        (
            "ff_full_bf16",
            lambda b, a: full_forward_llr(model, b, a, var_pos, torch.bfloat16),
        ),
    ]:
        out = np.zeros(len(rows))
        for i in range(0, len(rows), 64):
            out[i : i + 64] = fn(input_ids[i : i + 64], alt_tok[i : i + 64])
        res[name] = out
        print(f"{name:<16} distal/fwd AUPRC = {average_precision_score(lab, -out):.4f}")

    print(
        "\noffline parquet (bf16 prefix-share) = 0.1589 ; online (levanter bf16) = 0.2507"
    )
    print(
        f"corr(ps_full_fp32, ff_full_fp32) = {np.corrcoef(res['ps_full_fp32'], res['ff_full_fp32'])[0, 1]:.6f}"
    )
    print(
        f"max|ps_fp32 - ff_fp32| = {np.abs(res['ps_full_fp32'] - res['ff_full_fp32']).max():.2e}"
    )
    print(
        f"max|ff_fp32 - ff_bf16| = {np.abs(res['ff_full_fp32'] - res['ff_full_bf16']).max():.3f}"
    )


if __name__ == "__main__":
    main()
