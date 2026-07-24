"""Publish the immutable issue #402 BOS/[SEQ] DNA tokenizer to Hugging Face."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from marin_dna.pipelines.rag_glm.tokenizer import create_rag_char_tokenizer

REPO_ID = "bolinas-dna/tokenizer-char-bos-seq-v1"


def tokenizer_readme(commit_sha: str) -> str:
    """Return the small tokenizer card with commit-pinned provenance."""
    script_url = (
        "https://github.com/Open-Athena/marin-dna/blob/"
        f"{commit_sha}/scripts/issue402_upload_rag_tokenizer.py"
    )
    return f"""---
tags:
- biology
- genomics
- dna
---

# Character DNA tokenizer with BOS and `[SEQ]`

This immutable tokenizer is the eight-token vocabulary used by MarinDNA issue
[#402](https://github.com/Open-Athena/marin-dna/issues/402): `[PAD]`, `[UNK]`,
`[BOS]`, atomic `[SEQ]`, and the four DNA bases. It has BOS but no EOS; BOS is
also registered as CLS. Raw `N` maps to `[UNK]`.

Produced by the commit-pinned [upload script]({script_url}).
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--repo-id", default=REPO_ID)
    args = parser.parse_args()
    assert len(args.commit_sha) == 40

    tokenizer = create_rag_char_tokenizer()
    assert tokenizer.vocab_size == 8
    assert tokenizer.bos_token_id == tokenizer.cls_token_id == 2
    assert tokenizer.eos_token_id is None
    assert tokenizer.convert_tokens_to_ids("[SEQ]") == 3

    with tempfile.TemporaryDirectory(prefix="dna-exp402-tokenizer-") as tmpdir:
        output = Path(tmpdir)
        tokenizer.save_pretrained(output)
        (output / "README.md").write_text(tokenizer_readme(args.commit_sha))
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="model", exist_ok=False)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=output,
            commit_message="Publish issue #402 fixed-layout DNA tokenizer",
        )


if __name__ == "__main__":
    main()
