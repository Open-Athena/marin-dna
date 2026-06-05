"""Issue #263: verify the BOS monkeypatch fixes the *real* harness tokenization.

Imports `marin_dna.pipelines.evals.lm_eval` (which installs `_install_bos_fix`),
then runs levanter's actual loglikelihood tokenizer
(`eval_harness._iterate_tokenized_requests`) on the real checkpoint tokenizer and
the real dataset row for chr7:156791472 C>A — the variant whose missing-BOS LLR
drove #257. Asserts the forwarded sequence now starts with `[BOS]`, is 256 tokens
(`[BOS]` + 127 context + 128 completion), and that the prompt/completion boundary
lands at 128 (so the scored completion tokens are unchanged — only BOS is added).

Usage:
  uv run --extra marin python scripts/issue257/verify_bos_fix.py
"""

from __future__ import annotations

import datasets
from lm_eval.api.instance import Instance
from transformers import AutoTokenizer

import marin_dna.pipelines.evals.lm_eval  # noqa: F401  installs the BOS monkeypatch
from levanter import eval_harness

CKPT = "scratch/issue257/ckpt-ccre-4999"
HARNESS = ("bolinas-dna/evals_mendelian_traits_harness_255", "7b92f047f9a36f90e9ac47886afa2a99264ee35c")


def main() -> None:
    assert getattr(eval_harness, "_marin_dna_bos_patched", False), "BOS patch not installed"
    tok = AutoTokenizer.from_pretrained(CKPT)
    bos = tok.bos_token_id
    assert bos is not None

    df = datasets.load_dataset(HARNESS[0], name="default", revision=HARNESS[1], split="train").to_pandas()
    for c in ("chrom", "pos", "ref", "alt", "strand"):
        df[c] = df[c].astype(str)
    row = df[(df.chrom == "7") & (df.pos == "156791472") & (df.ref == "C")
             & (df.alt == "A") & (df.strand == "+")].iloc[0]
    ctx, refc, altc = row["context"], row["ref_completion"], row["alt_completion"]
    print(f"[data] len(context)={len(ctx)} len(ref_completion)={len(refc)} (raw strings, no BOS)")

    reqs = [
        Instance(request_type="loglikelihood", doc={}, arguments=(ctx, refc), idx=0, metadata=(None, None, None)),
        Instance(request_type="loglikelihood", doc={}, arguments=(ctx, altc), idx=1, metadata=(None, None, None)),
    ]
    pcs = list(eval_harness._iterate_tokenized_requests(reqs, tok, max_length=4096, batch_size=8))
    assert len(pcs) == 2
    for pc, name in zip(pcs, ("ref", "alt")):
        ids = list(pc.ids)
        print(f"[{name}] len={len(ids)} ids[0]={ids[0]} prompt_length={pc.prompt_length} "
              f"first_scored_token_idx={pc.prompt_length}")
        assert ids[0] == bos, f"{name}: first token {ids[0]} != BOS {bos} — patch did not apply"
        assert len(ids) == 1 + len(ctx) + len(refc), f"{name}: unexpected length {len(ids)}"
        # boundary = 1 BOS + 127 context tokens (1 token/nt for this tokenizer)
        assert pc.prompt_length == 1 + len(ctx), f"{name}: boundary {pc.prompt_length} != {1 + len(ctx)}"

    print("\n[OK] real harness tokenization now forwards [BOS] + 255 nt (256 tokens), "
          "completion boundary at 128 — matches the offline with-BOS input.")


if __name__ == "__main__":
    main()
