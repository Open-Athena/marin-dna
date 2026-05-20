"""DNA k-mer tokenizer creation utility.

Creates a HuggingFace-compatible tokenizer that can be saved and loaded
via ``AutoTokenizer.from_pretrained`` with no custom code required.
"""

import itertools

from tokenizers import Regex, Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast

DNA_BASES = "acgt"
SPECIAL_TOKENS = ["[PAD]", "[UNK]"]


def _build_kmer_vocab(k: int) -> dict[str, int]:
    """Build vocabulary mapping all k-mers to integer IDs.

    IDs: [PAD]=0, [UNK]=1, then all 4^k k-mers in lexicographic order
    starting at 2.  All lowercase to match the Lowercase() normalizer.
    """
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    offset = len(SPECIAL_TOKENS)
    for i, bases in enumerate(itertools.product(DNA_BASES, repeat=k)):
        vocab["".join(bases)] = offset + i
    return vocab


def _build_backend_tokenizer(k: int) -> Tokenizer:
    """Assemble a Rust-backed tokenizer with serializable components."""
    vocab = _build_kmer_vocab(k)
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.normalizer = Lowercase()
    backend.pre_tokenizer = Split(pattern=Regex(f".{{{k}}}"), behavior="isolated")
    backend.decoder = Fuse()
    return backend


def create_kmer_tokenizer(k: int) -> PreTrainedTokenizerFast:
    """Create a DNA k-mer tokenizer backed by HuggingFace's Rust engine.

    The returned ``PreTrainedTokenizerFast`` can be saved with
    ``tok.save_pretrained(path)`` and reloaded anywhere via
    ``AutoTokenizer.from_pretrained(path)`` — no ``marin_dna`` dependency
    or ``trust_remote_code`` needed.
    """
    backend = _build_backend_tokenizer(k)
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
