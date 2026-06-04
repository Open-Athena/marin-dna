"""Issue #257: is transformers' llama3 `rope_scaling` the source of the
torch-HF (0.158) vs JAX-levanter (0.251) distal/fwd discrepancy?

The HF config has rope_scaling={rope_type:llama3, factor:8, original_max:8192}
with max_position_embeddings=256 (transformers warns the config is inverted).
Score distal/fwd in torch with the rope config swapped, to see which one
matches levanter's 0.251:

  asis    -> the exported config (llama3, original_max 8192)   [expect 0.158]
  none    -> rope_scaling disabled (standard RoPE, theta only)
  fixed   -> llama3 but original_max=256 (matches the real context)

Usage: uv run --group genome-s3 python scripts/issue257/rope_test.py <asis|none|fixed>
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl
import torch
from sklearn.metrics import average_precision_score
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
GENOME = ("s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/ensembl-release-115/"
          "Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz")
SCORES = ("s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores/"
          "exp232-v4_ccre_non_promoter-step-4999/mendelian_traits.parquet")
WINDOW = 255


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "none"
    torch.set_num_threads(4)
    cfg = AutoConfig.from_pretrained(CKPT)
    print(f"[rope] original rope_scaling = {cfg.rope_scaling}")
    if mode == "none":
        cfg.rope_scaling = None
    elif mode == "fixed":
        cfg.rope_scaling = dict(cfg.rope_scaling)
        cfg.rope_scaling["original_max_position_embeddings"] = WINDOW + 1
    print(f"[rope] mode={mode} -> rope_scaling = {cfg.rope_scaling}")

    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForCausalLM.from_pretrained(
        CKPT, config=cfg, trust_remote_code=True
    ).eval()
    genome = Genome(GENOME)
    df = pl.read_parquet(SCORES).filter(pl.col("subset") == "distal").to_pandas()
    lab = df["label"].to_numpy()

    n_prefix, _ = _get_special_token_counts(tok)
    nuc = torch.tensor([_get_nucleotide_token_ids(tok)[n] for n in NUCLEOTIDES])
    var_pos = in_seq_var_pos(WINDOW, "+") + n_prefix
    rows = [transform_llr_clm(r, tok, genome, WINDOW, "+") for r in df.to_dict("records")]
    llr = np.zeros(len(rows))
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            ch = rows[i:i + 64]
            ids = torch.stack([c["input_ids"] for c in ch])
            alt = torch.tensor([c["alt_token_id"] for c in ch])
            out = compute_variant_score_bundle(model, ids, alt, var_pos=var_pos, nuc_token_ids=nuc)
            llr[i:i + len(ch)] = out[:, 0].numpy()
    ap = average_precision_score(lab, -llr)
    print(f"\n[rope] mode={mode}  distal/fwd 4-nuc AUPRC = {ap:.4f}")
    print("  reference: transformers as-is (offline) 0.1589 ; levanter (JAX) 0.2501")


if __name__ == "__main__":
    main()
