"""Sanity-check the refutation: is the 4-nuc vs full-vocab equivalence real
(model puts ~all mass on ACGT) or a bug in my full-vocab branch?

Measures, on the actual distal sequences, the special-token probability mass
P([PAD],[UNK],[BOS]) = 1 - sum(P_ACGT) at every scored (downstream) position,
and the per-variant |llr_4nuc - llr_fullvocab|. If leakage ~ 0 and the LLR
diff ~ 0, then full-vocab == 4-nuc is correct → softmax space can't explain the
0.158 -> 0.251 gap.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin_dna.data.dna import NUCLEOTIDES
from marin_dna.data.genome import Genome
from marin_dna.data.transforms import (
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
N = 64


@torch.no_grad()
def main() -> None:
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForCausalLM.from_pretrained(CKPT, trust_remote_code=True).eval()
    genome = Genome(GENOME)
    df = pl.read_parquet(SCORES).filter(pl.col("subset") == "distal").head(N).to_pandas()

    n_prefix, _ = _get_special_token_counts(tok)
    nuc_ids = torch.tensor(
        [_get_nucleotide_token_ids(tok)[n] for n in NUCLEOTIDES], dtype=torch.long
    )
    var_pos = in_seq_var_pos(WINDOW, "+") + n_prefix

    rows = [transform_llr_clm(r, tok, genome, WINDOW, "+") for r in df.to_dict("records")]
    input_ids = torch.stack([r["input_ids"] for r in rows])  # [N, L]
    alt_tok = torch.tensor([r["alt_token_id"] for r in rows], dtype=torch.long)

    # Forward the full (ref) sequences; look at downstream scored positions.
    logits = model(input_ids).logits.float()  # [N, L, 7]
    # scored next-token positions for the suffix: predict tokens var_pos..L-1,
    # i.e. logits at positions var_pos-1 .. L-2.
    L = input_ids.shape[1]
    pos = slice(var_pos - 1, L - 1)
    lp = F.log_softmax(logits[:, pos], dim=-1)  # [N, n_scored, 7]
    p_acgt = lp[..., nuc_ids].exp().sum(-1)  # [N, n_scored] — prob mass on ACGT
    leak = 1.0 - p_acgt  # special-token mass
    logZ = torch.log(p_acgt.clamp_min(1e-30))  # the term that differs 4nuc vs full

    print(f"n_seq={N}  scored_positions/seq={lp.shape[1]}  vocab={logits.shape[-1]}")
    print("\n=== special-token probability mass (leakage) at scored positions ===")
    print(f"  mean leakage      : {leak.mean().item():.3e}")
    print(f"  median leakage    : {leak.median().item():.3e}")
    print(f"  max leakage       : {leak.max().item():.3e}")
    print(f"  99th pct leakage  : {torch.quantile(leak.flatten(), 0.99).item():.3e}")
    print(f"  frac positions >1e-3 leakage: {(leak > 1e-3).float().mean().item():.3e}")
    print("\n=== logZ = log(sum P_ACGT) — the per-position 4nuc-vs-full correction ===")
    print(f"  mean |logZ| : {logZ.abs().mean().item():.3e}")
    print(f"  max  |logZ| : {logZ.abs().max().item():.3e}")
    print(
        "\nInterpretation: llr_full - llr_4nuc = sum over scored positions of "
        "[logZ(ctx_alt) - logZ(ctx_ref)].\n"
        "If |logZ| ~ 0 everywhere, the two softmax spaces give the same LLR "
        "(confirmed: full-vocab AUPRC == 4-nuc AUPRC), so the softmax convention "
        "cannot be the source of the online 0.251."
    )


if __name__ == "__main__":
    main()
