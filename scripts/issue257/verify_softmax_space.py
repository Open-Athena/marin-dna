"""Issue #257 verification — is the online↔offline distal-FWD AUPRC gap caused
*solely* by the softmax-normalization space?

Offline `compute_variant_score_bundle` renormalizes next-token log-probs over
the 4 nucleotides {A,C,G,T}; online lm_eval `loglikelihood` uses the full
7-token vocab ([PAD],[UNK],[BOS],a,c,g,t). This runs the exp232
v4_ccre_non_promoter step-4999 checkpoint ONCE on the mendelian `distal`
variants and computes the per-variant LLR BOTH ways *from the same logits*, so
the only thing that varies is the normalizer.

Expected if the proposed root cause is right (offline numbers from the S3
scores parquet; online from the in-training wandb metric reported in #257):

    4-nuc      FWD 0.159, RC 0.112   (reproduces the offline parquet)
    full-vocab FWD ~0.251 (jumps to the online value), RC ~0.112 (control: stays)

The RC strand is a built-in negative control: online and offline already agree
there (~0.112), so flipping the normalizer should NOT move RC.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from einops import rearrange
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
from marin_dna.model.scoring import _repeat_interleave_kv_cache, _token_id_to_nuc_idx

CKPT = "scratch/issue257/ckpt-ccre-4999"
GENOME = (
    "s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/ensembl-release-115/"
    "Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"
)
SCORES = (
    "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores/"
    "exp232-v4_ccre_non_promoter-step-4999/mendelian_traits.parquet"
)
WINDOW = 255  # 255 bp DNA + BOS = 256 tokens


@torch.no_grad()
def compute_llr_dual(
    model,
    input_ids: torch.Tensor,
    alt_token_id: torch.Tensor,
    *,
    var_pos: int,
    nuc_token_ids: torch.Tensor,
) -> torch.Tensor:
    """[B, 2] = (llr_4nuc, llr_fullvocab) from a single shared forward.

    The 4-nuc branch is a verbatim copy of `compute_variant_score_bundle`'s LLR
    (minus the JSD); the full-vocab branch reuses the *same* `prefix_last_logits`
    / `suffix_logits`, only swapping the slice-then-log_softmax for a
    log_softmax over the whole vocab and gathering at the actual token ids.
    """
    B, L = input_ids.shape
    p = var_pos
    assert 0 < p < L - 1

    prefix = input_ids[:, :p].contiguous()
    ref_suffix = input_ids[:, p:].contiguous()
    alt_suffix = torch.cat([alt_token_id.unsqueeze(-1), ref_suffix[:, 1:]], dim=-1)
    suffixes = torch.stack([ref_suffix, alt_suffix], dim=1)  # [B, 2, L-p]
    suffixes_flat = rearrange(suffixes, "B V L -> (B V) L").contiguous()

    prefix_out = model(prefix, use_cache=True, logits_to_keep=1)
    prefix_last_logits = prefix_out.logits[:, -1]  # [B, V]
    past_kv = _repeat_interleave_kv_cache(prefix_out.past_key_values, 2)
    suffix_logits = model(
        suffixes_flat, past_key_values=past_kv, use_cache=False
    ).logits  # [B*2, L-p, V]

    nuc_ids = nuc_token_ids.to(suffix_logits.device)
    suffix_targets = input_ids[:, p + 1 :]  # [B, L-p-1] — shared ref/alt downstream

    # ---------- 4-nuc (offline) ----------
    log_p_nuc = F.log_softmax(suffix_logits[..., nuc_ids].float(), dim=-1)
    log_p_nuc = rearrange(log_p_nuc, "(B V) L C -> B V L C", B=B)
    log_p_ref4 = log_p_nuc[:, 0, :-1]
    log_p_alt4 = log_p_nuc[:, 1, :-1]
    prefix_log_p4 = F.log_softmax(prefix_last_logits[..., nuc_ids].float(), dim=-1)
    ref_var_idx = _token_id_to_nuc_idx(input_ids[:, p], nuc_ids)
    alt_var_idx = _token_id_to_nuc_idx(alt_token_id, nuc_ids)
    llr_at_var4 = prefix_log_p4.gather(-1, alt_var_idx.unsqueeze(-1)).squeeze(
        -1
    ) - prefix_log_p4.gather(-1, ref_var_idx.unsqueeze(-1)).squeeze(-1)
    tgt4 = _token_id_to_nuc_idx(suffix_targets, nuc_ids).unsqueeze(-1)
    llr_down4 = (
        log_p_alt4.gather(-1, tgt4).squeeze(-1)
        - log_p_ref4.gather(-1, tgt4).squeeze(-1)
    ).sum(dim=-1)
    llr_4nuc = llr_at_var4 + llr_down4

    # ---------- full-vocab (online lm_eval loglikelihood) ----------
    log_p_full = F.log_softmax(suffix_logits.float(), dim=-1)
    log_p_full = rearrange(log_p_full, "(B V) L W -> B V L W", B=B)
    log_p_reff = log_p_full[:, 0, :-1]
    log_p_altf = log_p_full[:, 1, :-1]
    prefix_log_pf = F.log_softmax(prefix_last_logits.float(), dim=-1)  # [B, V]
    ref_var_tok = input_ids[:, p].unsqueeze(-1)
    alt_var_tok = alt_token_id.unsqueeze(-1)
    llr_at_varf = prefix_log_pf.gather(-1, alt_var_tok).squeeze(
        -1
    ) - prefix_log_pf.gather(-1, ref_var_tok).squeeze(-1)
    tgtf = suffix_targets.unsqueeze(-1)  # actual token ids
    llr_downf = (
        log_p_altf.gather(-1, tgtf).squeeze(-1)
        - log_p_reff.gather(-1, tgtf).squeeze(-1)
    ).sum(dim=-1)
    llr_full = llr_at_varf + llr_downf

    return torch.stack([llr_4nuc, llr_full], dim=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="distal")
    ap.add_argument("--strands", default="+,-")
    ap.add_argument("--limit", type=int, default=0, help="0 = all (probe with small N)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    print(f"[load] tokenizer + model from {CKPT}")
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForCausalLM.from_pretrained(CKPT, trust_remote_code=True).eval()
    genome = Genome(GENOME)

    df = pl.read_parquet(SCORES).filter(pl.col("subset") == args.subset)
    if args.limit:
        df = df.head(args.limit)
    pdf = df.to_pandas()
    label = pdf["label"].to_numpy()
    print(
        f"[data] subset={args.subset} n={len(pdf)} positives={int(label.sum())} "
        f"(offline llr_fwd/llr_rc carried for cross-check)"
    )

    n_prefix, _ = _get_special_token_counts(tok)
    nuc_ids_dict = _get_nucleotide_token_ids(tok)
    nuc_token_ids = torch.tensor(
        [nuc_ids_dict[n] for n in NUCLEOTIDES], dtype=torch.long
    )
    print(f"[tok] n_prefix(BOS)={n_prefix} nuc_token_ids={nuc_ids_dict}")

    strands = [s for s in args.strands.split(",") if s]
    out: dict[str, np.ndarray] = {}
    for strand in strands:
        var_pos = in_seq_var_pos(WINDOW, strand) + n_prefix
        rows = [
            transform_llr_clm(rec, tok, genome, WINDOW, strand)
            for rec in pdf.to_dict("records")
        ]
        llrs = np.zeros((len(rows), 2), dtype=np.float64)
        t0 = time.time()
        for i in range(0, len(rows), args.batch_size):
            chunk = rows[i : i + args.batch_size]
            input_ids = torch.stack([c["input_ids"] for c in chunk])
            alt_tok = torch.tensor([c["alt_token_id"] for c in chunk], dtype=torch.long)
            o = compute_llr_dual(
                model, input_ids, alt_tok, var_pos=var_pos, nuc_token_ids=nuc_token_ids
            )
            llrs[i : i + len(chunk)] = o.numpy()
        dt = time.time() - t0
        out[strand] = llrs
        print(
            f"[run] strand={strand} var_pos={var_pos} n={len(rows)} "
            f"wall={dt:.1f}s ({dt / max(len(rows), 1) * 1000:.0f} ms/variant)"
        )

    if args.limit:
        print("\n[probe] timing only; AUPRC on a truncated set is not meaningful.")
        return

    # ---- cross-check + AUPRC (minus_llr protocol: score = -llr) ----
    strand_key = {"+": "fwd", "-": "rc"}
    print("\n=== per-variant cross-check: my 4-nuc llr vs offline parquet ===")
    for s in strands:
        col = f"llr_{strand_key[s]}"
        mine = out[s][:, 0]
        offline = pdf[col].to_numpy()
        mad = np.mean(np.abs(mine - offline))
        r = np.corrcoef(mine, offline)[0, 1]
        print(f"  strand={s}: MAD(mine_4nuc, {col})={mad:.4f}  corr={r:.5f}")

    print("\n=== AUPRC(label, -llr) ===")
    print(f"{'strand':<8}{'4-nuc (offline)':<20}{'full-vocab (online)':<22}{'Δ':<8}")
    for s in strands:
        ap4 = average_precision_score(label, -out[s][:, 0])
        apf = average_precision_score(label, -out[s][:, 1])
        print(f"{s:<8}{ap4:<20.4f}{apf:<22.4f}{apf - ap4:+.4f}")
    # AVG = mean raw llr across strands, then negate
    if set(strands) == {"+", "-"}:
        llr4_avg = (out["+"][:, 0] + out["-"][:, 0]) / 2
        llrf_avg = (out["+"][:, 1] + out["-"][:, 1]) / 2
        ap4 = average_precision_score(label, -llr4_avg)
        apf = average_precision_score(label, -llrf_avg)
        print(f"{'avg':<8}{ap4:<20.4f}{apf:<22.4f}{apf - ap4:+.4f}")

    print(
        "\nReference: offline parquet distal FWD 0.1589 / RC 0.1118 ; "
        "online (wandb, #257) distal FWD 0.251."
    )


if __name__ == "__main__":
    main()
