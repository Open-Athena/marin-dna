"""Issue #257: does the BOS-prefix mismatch reproduce the whole distal/fwd
AUPRC gap (offline 0.159 with-BOS vs online 0.2501 no-BOS), end-to-end?

bos_test.py proved it for the single most-divergent variant. This confirms it
over the *entire* distal/+ subset (580 variants) under identical weights (the HF
export, torch + offline kernel): score every variant with-BOS (the offline
256-token input) and no-BOS (drop the leading [BOS] → 255 tokens, the online
harness input), and report distal/fwd AUPRC for each.

Expectation: with-BOS ≈ 0.159 (reproduces offline), no-BOS ≈ 0.2501
(reproduces online) — i.e. the online↔offline gap is entirely the BOS token.

Usage:
  AWS_DEFAULT_REGION=us-east-2 uv run --group genome-s3 python scripts/issue257/bos_auprc.py
"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
from sklearn.metrics import average_precision_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin_dna.data.dna import NUCLEOTIDES
from marin_dna.data.genome import Genome
from marin_dna.data.transforms import (
    _get_nucleotide_token_ids,
    _get_special_token_counts,
    in_seq_var_pos,
    transform_llr_clm,
)
from marin_dna.model.scoring import compute_variant_score_bundle

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


def main() -> None:
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForCausalLM.from_pretrained(CKPT, trust_remote_code=True).eval()
    genome = Genome(GENOME)

    df = pl.read_parquet(SCORES).filter(pl.col("subset") == "distal").to_pandas()
    lab = df["label"].to_numpy()
    n_prefix, _ = _get_special_token_counts(tok)
    assert n_prefix == 1
    nuc = torch.tensor([_get_nucleotide_token_ids(tok)[n] for n in NUCLEOTIDES])
    vp_bos = in_seq_var_pos(WINDOW, "+") + n_prefix  # 128
    vp_nobos = in_seq_var_pos(WINDOW, "+")  # 127

    rows = [
        transform_llr_clm(r, tok, genome, WINDOW, "+") for r in df.to_dict("records")
    ]
    llr_bos = np.zeros(len(rows))
    llr_nobos = np.zeros(len(rows))
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            ch = rows[i : i + 64]
            ids = torch.stack([c["input_ids"] for c in ch])  # [B, 256] with BOS
            alt = torch.tensor([c["alt_token_id"] for c in ch])
            out_bos = compute_variant_score_bundle(
                model, ids, alt, var_pos=vp_bos, nuc_token_ids=nuc
            )
            out_nobos = compute_variant_score_bundle(
                model,
                ids[:, n_prefix:].contiguous(),
                alt,
                var_pos=vp_nobos,
                nuc_token_ids=nuc,
            )
            llr_bos[i : i + len(ch)] = out_bos[:, 0].numpy()
            llr_nobos[i : i + len(ch)] = out_nobos[:, 0].numpy()

    ap_bos = average_precision_score(lab, -llr_bos)
    ap_nobos = average_precision_score(lab, -llr_nobos)
    print("\n=== distal/fwd AUPRC, n=%d (torch, HF export) ===" % len(rows))
    print(
        f"  with-BOS (256-tok, offline input) : {ap_bos:.4f}   (offline reference 0.1589)"
    )
    print(
        f"  no-BOS   (255-tok, online input)  : {ap_nobos:.4f}   (online  reference 0.2501)"
    )
    print(
        f"\n  mean LLR pos (label=1): BOS {llr_bos[lab == 1].mean():+.3f}  noBOS {llr_nobos[lab == 1].mean():+.3f}"
    )
    print(
        f"  mean LLR neg (label=0): BOS {llr_bos[lab == 0].mean():+.3f}  noBOS {llr_nobos[lab == 0].mean():+.3f}"
    )


if __name__ == "__main__":
    main()
