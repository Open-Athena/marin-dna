"""Compare the ONLINE harness dataset vs OFFLINE source for the distal subset:
variant set, labels, match_groups, AND the materialized (context, completion)
sequences vs an on-the-fly offline extraction. No model — pure data check.

If the online↔offline AUPRC gap were an input/sequence/tokenization difference,
it would show up here. If everything matches, the gap is in the scoring path
(or is an in-training-eval artifact), not the inputs.
"""

from __future__ import annotations

import datasets
import pandas as pd
from transformers import AutoTokenizer

from marin_dna.data.dna import complement_base
from marin_dna.data.genome import Genome
from marin_dna.data.transforms import _get_variant_window, in_seq_var_pos

CKPT = "scratch/issue257/ckpt-ccre-4999"
GENOME = (
    "s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/ensembl-release-115/"
    "Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"
)
OFFLINE = ("bolinas-dna/evals_mendelian_traits", "4aed58e50c5dea0b878a665007af2ef9e5108e9f")
HARNESS = ("bolinas-dna/evals_mendelian_traits_harness_255", "7b92f047f9a36f90e9ac47886afa2a99264ee35c")
WINDOW = 255


def offline_materialize(rec: dict, genome: Genome, strand: str) -> dict:
    """Reproduce materialize._add_eval_harness_fields from the offline variant."""
    window, var_pos = _get_variant_window(rec, genome, WINDOW, strand=strand)
    alt = str(rec["alt"]).upper()
    alt_in_strand = alt if strand == "+" else complement_base(alt)
    right = window[var_pos + 1 :]
    return {
        "context": window[:var_pos],
        "ref_completion": window[var_pos:],
        "alt_completion": alt_in_strand + right,
    }


def main() -> None:
    off = datasets.load_dataset(OFFLINE[0], revision=OFFLINE[1], split="train").to_pandas()
    off = off[off["subset"] == "distal"].reset_index(drop=True)
    har = datasets.load_dataset(
        HARNESS[0], name="default", revision=HARNESS[1], split="train"
    ).to_pandas()
    har = har[har["subset"] == "distal"].reset_index(drop=True)
    print(f"offline distal rows: {len(off)} | harness distal rows: {len(har)} "
          f"(harness = 2 strands/variant → expect 2x)")
    print("harness columns:", list(har.columns))
    print("harness strand counts:", har["strand"].value_counts().to_dict() if "strand" in har else "NO strand col")

    har_fwd = har[har["strand"] == "+"].reset_index(drop=True) if "strand" in har else har
    print(f"\nharness distal '+' rows: {len(har_fwd)}")

    key = ["chrom", "pos", "ref", "alt"]
    off_k = off.set_index([*key])
    # target vs label name
    label_col = "label" if "label" in off.columns else "target"
    tgt_col = "target" if "target" in har_fwd.columns else "label"

    merged = har_fwd.merge(off, on=key, suffixes=("_har", "_off"), how="outer", indicator=True)
    print("\nmerge indicator:", merged["_merge"].value_counts().to_dict())

    # label / subset / match_group agreement
    both = merged[merged["_merge"] == "both"].copy()
    lab_off = both[f"{label_col}_off"] if f"{label_col}_off" in both else both[label_col]
    lab_har = both[f"{tgt_col}_har"] if f"{tgt_col}_har" in both else both[tgt_col]
    print(f"label/target mismatches: {(lab_off.values != lab_har.values).sum()} / {len(both)}")
    if "match_group_off" in both and "match_group_har" in both:
        print(f"match_group mismatches: {(both['match_group_off'].values != both['match_group_har'].values).sum()}")

    # ---- sequence comparison: harness vs offline on-the-fly ----
    genome = Genome(GENOME)
    tok = AutoTokenizer.from_pretrained(CKPT)
    n_ctx_mismatch = n_ref_mismatch = n_alt_mismatch = 0
    n_tok_mismatch = 0
    examples = []
    for _, row in har_fwd.iterrows():
        rec = {"chrom": str(row["chrom"]), "pos": int(row["pos"]),
               "ref": str(row["ref"]), "alt": str(row["alt"])}
        mine = offline_materialize(rec, genome, "+")
        cm = mine["context"] != row["context"]
        rm = mine["ref_completion"] != row["ref_completion"]
        am = mine["alt_completion"] != row["alt_completion"]
        n_ctx_mismatch += cm
        n_ref_mismatch += rm
        n_alt_mismatch += am
        # tokenization: lm_eval-style whole vs offline-style full window
        whole_off = tok.encode(mine["context"] + mine["ref_completion"])
        whole_har = tok.encode(str(row["context"]) + str(row["ref_completion"]))
        if whole_off != whole_har:
            n_tok_mismatch += 1
        if (cm or rm or am) and len(examples) < 3:
            examples.append((rec, mine, dict(row[["context", "ref_completion", "alt_completion"]])))

    n = len(har_fwd)
    print(f"\n=== sequence mismatches (harness '+' vs offline extraction), n={n} ===")
    print(f"  context mismatches:        {n_ctx_mismatch}")
    print(f"  ref_completion mismatches: {n_ref_mismatch}")
    print(f"  alt_completion mismatches: {n_alt_mismatch}")
    print(f"  whole-encoding mismatches: {n_tok_mismatch}")
    for rec, mine, har_row in examples:
        print("\n  MISMATCH", rec)
        for fld in ["context", "ref_completion", "alt_completion"]:
            m = mine[fld]; h = har_row[fld]
            print(f"    {fld}: offline_len={len(m)} har_len={len(h)} equal={m==h}")
            if m != h:
                print(f"      offline: ...{m[-30:]}")
                print(f"      harness: ...{h[-30:]}")


if __name__ == "__main__":
    main()
