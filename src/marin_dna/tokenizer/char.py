"""DNA character-level tokenizer with configurable BOS/EOS tokens.

Creates a HuggingFace-compatible tokenizer that can be saved and loaded
via ``AutoTokenizer.from_pretrained`` with no custom code required.

By default both BOS and EOS tokens are included. Each can be independently
disabled, which excludes them from the vocabulary entirely (affecting vocab
size and token IDs).
"""

from typing import Any

from tokenizers import Regex, Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

DNA_BASES = "acgt"
SEQUENCE_BOUNDARY_TOKEN = "[SEQ]"


def _build_char_vocab(
    *, bos: bool, eos: bool, sequence_boundary: bool = False
) -> dict[str, int]:
    """Build vocabulary mapping special tokens and DNA bases to integer IDs.

    Always includes [PAD] and [UNK], then conditionally [BOS] and [EOS],
    followed by DNA bases. All lowercase to match the Lowercase() normalizer.
    """
    special_tokens = ["[PAD]", "[UNK]"]
    if bos:
        special_tokens.append("[BOS]")
    if eos:
        special_tokens.append("[EOS]")
    if sequence_boundary:
        special_tokens.append(SEQUENCE_BOUNDARY_TOKEN)

    vocab = {tok: i for i, tok in enumerate(special_tokens)}
    offset = len(special_tokens)
    for i, base in enumerate(DNA_BASES):
        vocab[base] = offset + i
    return vocab


def _build_backend_tokenizer(
    *, bos: bool, eos: bool, sequence_boundary: bool = False
) -> Tokenizer:
    """Assemble a Rust-backed tokenizer with serializable components."""
    vocab = _build_char_vocab(bos=bos, eos=eos, sequence_boundary=sequence_boundary)
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.normalizer = Lowercase()
    backend.pre_tokenizer = Split(pattern=Regex(".{1}"), behavior="isolated")
    backend.decoder = Fuse()

    if bos and eos:
        backend.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[("[BOS]", vocab["[BOS]"]), ("[EOS]", vocab["[EOS]"])],
        )
    elif bos:
        backend.post_processor = TemplateProcessing(
            single="[BOS] $A",
            special_tokens=[("[BOS]", vocab["[BOS]"])],
        )
    elif eos:
        backend.post_processor = TemplateProcessing(
            single="$A [EOS]",
            special_tokens=[("[EOS]", vocab["[EOS]"])],
        )

    return backend


def create_char_tokenizer(
    *, bos: bool = True, eos: bool = True, sequence_boundary: bool = False
) -> PreTrainedTokenizerFast:
    """Create a DNA character-level tokenizer backed by HuggingFace's Rust engine.

    Parameters
    ----------
    bos : bool
        Include a [BOS] token in the vocabulary and prepend it to every
        encoded sequence. Default ``True``.
    eos : bool
        Include an [EOS] token in the vocabulary and append it to every
        encoded sequence. Default ``True``.
    sequence_boundary : bool
        Include ``[SEQ]`` as a dedicated atomic token. When enabled with BOS,
        the BOS token is also registered as CLS so ``cls_token_id ==
        bos_token_id``. Default ``False``.

    The returned ``PreTrainedTokenizerFast`` can be saved with
    ``tok.save_pretrained(path)`` and reloaded anywhere via
    ``AutoTokenizer.from_pretrained(path)`` — no ``marin_dna`` dependency
    or ``trust_remote_code`` needed.
    """
    backend = _build_backend_tokenizer(
        bos=bos, eos=eos, sequence_boundary=sequence_boundary
    )

    kwargs: dict[str, Any] = {
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
    }
    if bos:
        kwargs["bos_token"] = "[BOS]"
    if eos:
        kwargs["eos_token"] = "[EOS]"
    if sequence_boundary:
        kwargs["additional_special_tokens"] = [SEQUENCE_BOUNDARY_TOKEN]
        if bos:
            kwargs["cls_token"] = "[BOS]"

    return PreTrainedTokenizerFast(tokenizer_object=backend, **kwargs)
