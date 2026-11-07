"""Deterministic, case-truthful, single-document DNA/FIM training examples."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib.resources import files
from typing import Any, Literal

import numpy as np
from tokenizers import Tokenizer

from exp593_dna.annotations import GapChoice, RowAnnotations, choose_guided_gap
from exp593_dna.templates import phrase_catalog, pick, render_metadata

DATASET_REVISION = "4a444fff5520b992aa978d92a5af509a81977098"
TOKENIZER_REPOSITORY = "marin-community/marin-tokenizer"
TOKENIZER_REVISION = "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"
ALPHABET = "ACGTRYSWKMBDHVNacgtryswkmbdhvn"
COMPLEMENT = str.maketrans(ALPHABET, "TGCAYRSWMKVHDBNtgcayrswmkvhdbn")
Kind = Literal["bare_causal", "annotated_causal", "short_fim", "long_fim"]
KINDS: tuple[Kind, ...] = ("bare_causal", "annotated_causal", "short_fim", "long_fim")
CASE_MODES = ("native_softmask", "unmasked_lower", "unmasked_upper")


@dataclass(frozen=True)
class AugmentationConfig:
    """Sample mixture and execution shapes; actual input tokens drive scheduling."""

    seed: int = 472
    kind_probabilities: tuple[float, ...] = (0.40, 0.10, 0.375, 0.125)
    case_probabilities: tuple[float, ...] = (0.50, 0.25, 0.25)
    length_buckets: tuple[int, ...] = (2048, 4096, 6144, 8192)
    bucket_probabilities: tuple[float, ...] = (0.15, 0.25, 0.25, 0.35)
    global_batch_size: int = 64
    minimum_fill_fraction: float = 0.875
    species_probability: float = 0.70
    taxonomy_probability: float = 0.35
    chrom_probability: float = 0.25
    assembly_probability: float = 0.15
    source_strand_probability: float = 0.15
    orientation_probability: float = 0.20
    mask_description_probability: float = 0.85
    description_probability: float = 0.80
    center_gap_probability: float = 0.20
    minimal_fim_probability: float = 0.25
    gap_length_probability: float = 0.80
    guided_probability: float = 0.75

    def __post_init__(self) -> None:
        for values, count in [
            (self.kind_probabilities, len(KINDS)),
            (self.case_probabilities, len(CASE_MODES)),
            (self.bucket_probabilities, len(self.length_buckets)),
        ]:
            if (
                len(values) != count
                or any(p < 0 for p in values)
                or not np.isclose(sum(values), 1)
            ):
                raise ValueError("Invalid categorical probabilities")
        if self.seed < 0 or self.global_batch_size < 1:
            raise ValueError("Seed must be nonnegative and batch size positive")
        if not self.length_buckets or any(
            b < 2048 or b > 8192 for b in self.length_buckets
        ):
            raise ValueError("Input buckets must lie within [2048, 8192]")
        if not 0 < self.minimum_fill_fraction <= 1:
            raise ValueError("Invalid bucket fill fraction")
        for name, p in asdict(self).items():
            if name.endswith("_probability") and not 0 <= p <= 1:
                raise ValueError(f"Invalid probability: {name}")


@dataclass(frozen=True)
class RawWindow:
    """One original window; source coordinates are opaque and never rendered."""

    assembly: str
    chrom: str
    start: int
    end: int
    strand: str
    seq: str
    row: int
    split: str = "train"

    def __post_init__(self) -> None:
        if len(self.seq) != 8192 or not set(self.seq) <= set(ALPHABET):
            raise ValueError("Expected an 8192-base IUPAC DNA window")
        if not self.assembly or not self.chrom or self.strand not in ("+", "-"):
            raise ValueError("Missing or invalid source metadata")
        if self.start < 0 or self.end <= self.start or self.row < 0:
            raise ValueError("Invalid recorded source interval or row")

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> RawWindow:
        """Ignore prep-only hashes, but require every raw data field."""
        return cls(**{k: record[k] for k in cls.__dataclass_fields__ if k in record})


@dataclass(frozen=True)
class Span:
    """Text uses BPE; DNA is character-forced, with explicit scored targets."""

    text: str
    role: str
    score: bool = False


@dataclass(frozen=True)
class Example:
    """Loss weights at position i supervise token i+1, normalized per example."""

    token_ids: np.ndarray
    loss_weights: np.ndarray
    spans: tuple[Span, ...]
    bucket: int
    attributes: dict[str, Any]

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans)

    @property
    def scored_tokens(self) -> int:
        return int(np.count_nonzero(self.loss_weights))


def reverse_complement(sequence: str) -> str:
    """Preserve native mask case and complement the full IUPAC alphabet."""
    if not set(sequence) <= set(ALPHABET):
        raise ValueError("Unexpected DNA characters")
    return sequence.translate(COMPLEMENT)[::-1]


def rc_selected(occurrence: int, seed: int = 472) -> bool:
    """Retain exp586's 50% Bernoulli draw, independent of format and case."""
    if occurrence < 0:
        raise ValueError("Occurrence must be nonnegative")
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, occurrence & 0xFFFFFFFF, occurrence >> 32])
    )
    return bool(rng.random() < 0.5)


def rng_for(seed: int, occurrence: int, stream: str) -> np.random.Generator:
    """Separate random streams so metadata edits cannot change RC or lengths."""
    if occurrence < 0:
        raise ValueError("Occurrence must be nonnegative")
    words = np.frombuffer(
        hashlib.sha256(stream.encode()).digest()[:16], dtype="<u4"
    ).tolist()
    return np.random.default_rng(
        np.random.SeedSequence(
            [seed, occurrence & 0xFFFFFFFF, occurrence >> 32, *words]
        )
    )


class Encoder:
    """Combine untouched language BPE with one original token ID per DNA base."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.dna_lookup = np.full(128, -1, dtype=np.int32)
        for base in ALPHABET:
            ids = tokenizer.encode(base, add_special_tokens=False).ids
            if len(ids) != 1:
                raise ValueError("Every DNA character must have one existing token")
            self.dna_lookup[ord(base)] = ids[0]
        self.digest = hashlib.sha256(tokenizer.to_str().encode()).hexdigest()
        self._text_cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()

    def text_ids(self, text: str) -> tuple[int, ...]:
        """Bound the cache even with varied metadata values."""
        if text not in self._text_cache:
            self._text_cache[text] = tuple(
                self.tokenizer.encode(text, add_special_tokens=False).ids
            )
            if len(self._text_cache) > 8192:
                self._text_cache.popitem(last=False)
        self._text_cache.move_to_end(text)
        return self._text_cache[text]

    def encode(self, spans: tuple[Span, ...]) -> tuple[np.ndarray, np.ndarray]:
        """Return tokens and normalized prediction-position loss weights."""
        tokens, scored = [], []
        for span in spans:
            if span.role == "text":
                ids = np.asarray(self.text_ids(span.text), dtype=np.int32)
            else:
                ids = self.dna_lookup[
                    np.frombuffer(span.text.encode("ascii"), dtype=np.uint8)
                ]
                if np.any(ids < 0):
                    raise ValueError("Unexpected DNA token")
            tokens.append(ids)
            scored.append(np.full(len(ids), span.score, dtype=bool))
        token_ids = np.concatenate(tokens)
        targets = np.concatenate(scored)
        weights = np.zeros(len(token_ids), dtype=np.float32)
        weights[:-1] = targets[1:]
        count = int(weights.sum())
        if count == 0:
            raise ValueError("Every example needs at least one prediction target")
        weights /= count
        return token_ids, weights


def load_encoder() -> Encoder:
    """Read the pinned tokenizer; this loads no model or training framework."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        TOKENIZER_REPOSITORY, "tokenizer.json", revision=TOKENIZER_REVISION
    )
    return Encoder(Tokenizer.from_file(path))


def load_registry() -> dict[str, dict[str, Any]]:
    """Read the provenance-bearing, complete 65-species registry."""
    records = json.loads(
        files("exp593_dna").joinpath("assets/species_registry.json").read_text()
    )
    if len(records) != 65 or len({r["assembly_code"] for r in records}) != 65:
        raise ValueError("Species registry must cover all 65 raw codes")
    return {r["assembly_code"]: r for r in records}


class Augmenter:
    """Build each occurrence independently; no packing, stateful RNG or training."""

    def __init__(
        self,
        encoder: Encoder,
        config: AugmentationConfig | None = None,
        registry: dict[str, dict[str, Any]] | None = None,
        annotations: Mapping[int, RowAnnotations] | None = None,
    ) -> None:
        self.encoder = encoder
        self.config = config or AugmentationConfig()
        self.registry = registry if registry is not None else load_registry()
        self.phrases = phrase_catalog()
        self.annotations = annotations or {}

    def _fields(
        self, raw: RawWindow, rng: np.random.Generator, rc: bool
    ) -> dict[str, str]:
        species = self.registry[raw.assembly]
        c = self.config
        fields = {}
        if rng.random() < c.species_probability:
            fields["species"] = pick(rng, species["identity_names"])
        if rng.random() < c.taxonomy_probability:
            line = (
                species["taxonomy_path"]
                if rng.random() < 0.25
                else species["compact_taxonomy_path"]
            )
            fields["taxonomy"] = line.replace(
                " > ", pick(rng, [" > ", " / ", " -> ", " | "])
            )
        for name, value, probability in [
            ("chrom", raw.chrom, c.chrom_probability),
            ("assembly", raw.assembly, c.assembly_probability),
            ("source_strand", raw.strand, c.source_strand_probability),
            (
                "orientation",
                "reverse complement of the stored window"
                if rc
                else "same orientation as the stored window",
                c.orientation_probability,
            ),
        ]:
            if rng.random() < probability:
                fields[name] = value
        return fields

    def build(self, raw: RawWindow, occurrence: int) -> Example:
        """Build training data; a held-out row cannot pass this entry point."""
        if raw.split != "train":
            raise ValueError("Training augmentation must not consume held-out data")
        return self._build(raw, occurrence)

    def build_validation(self, raw: RawWindow, occurrence: int) -> Example:
        """Render a held-out presentation without admitting it to training."""
        if raw.split != "validation":
            raise ValueError("FIM validation must use validation-split rows")
        return self._build(raw, occurrence)

    def batch_bucket(self, occurrence: int) -> int:
        """Determine the common distributed input shape before loading local rows."""
        c = self.config
        rng = rng_for(c.seed, occurrence // c.global_batch_size, "batch_bucket")
        return c.length_buckets[
            int(rng.choice(len(c.length_buckets), p=c.bucket_probabilities))
        ]

    def _build(self, raw: RawWindow, occurrence: int) -> Example:
        """Apply RC, length/crop, presentation and gap transforms to one row."""
        if raw.assembly not in self.registry:
            raise ValueError(f"Unmapped assembly code {raw.assembly}")
        c = self.config
        kind_rng = rng_for(c.seed, occurrence, "kind")
        kind = KINDS[int(kind_rng.choice(len(KINDS), p=c.kind_probabilities))]
        case_rng = rng_for(c.seed, occurrence, "case")
        case = CASE_MODES[int(case_rng.choice(len(CASE_MODES), p=c.case_probabilities))]
        rc = rc_selected(occurrence, c.seed)
        # All examples in a global batch share one compilation shape.
        bucket = self.batch_bucket(occurrence)
        length_rng = rng_for(c.seed, occurrence, "length")
        length = int(
            length_rng.integers(
                max(2048, int(bucket * c.minimum_fill_fraction)), bucket + 1
            )
        )
        rng = rng_for(c.seed, occurrence, "presentation")
        gap_rng = rng_for(c.seed, occurrence, "gap")
        is_fim = kind.endswith("fim")
        gap_length = (
            int(gap_rng.choice([1, 2, 3]))
            if kind == "short_fim"
            else int(gap_rng.choice([16, 32, 64, 128, 256, 512]))
            if is_fim
            else 0
        )
        minimal_fim = is_fim and bool(
            rng_for(c.seed, occurrence, "minimal_fim").random()
            < c.minimal_fim_probability
        )
        gap_length_visible = is_fim and bool(
            rng_for(c.seed, occurrence, "gap_length_hint").random()
            < c.gap_length_probability
        )
        formatted = kind != "bare_causal" and not minimal_fim
        fields = self._fields(raw, rng, rc) if formatted else {}
        metadata, structure = (
            render_metadata(fields, rng) if formatted else ("", "none")
        )
        mask_description = formatted and rng.random() < c.mask_description_probability
        header = metadata
        if mask_description:
            header += pick(rng, self.phrases[case]) + "\n"
        description = formatted and rng.random() < c.description_probability
        if description:
            header += pick(rng, self.phrases["causal_description"]) + "\n"
        layout = (
            pick(rng, ("instruction_first", "instruction_after_flanks", "compact"))
            if is_fim
            else "causal"
        )
        instruction = (
            pick(rng, self.phrases["fim_instruction"]) + "\n"
            if is_fim and formatted
            else ""
        )
        left_label = pick(rng, self.phrases["left_label"]) if is_fim else ""
        right_label = pick(rng, self.phrases["right_label"]) if is_fim else ""
        answer_label = pick(rng, self.phrases["answer_label"]) if is_fim else ""
        if minimal_fim:
            layout = "compact"
            left_label = pick(rng, ("left", "prefix", "L"))
            right_label = pick(rng, ("right", "suffix", "R"))
            answer_label = pick(rng, ("fill", "answer", "middle"))
        gap_hint = ""
        if gap_length_visible:
            gap_hint = (
                f"[gap:{gap_length}]\n"
                if minimal_fim or rng.random() < 0.25
                else pick(rng, self.phrases["gap_length"]).format(
                    length=gap_length, unit="base" if gap_length == 1 else "bases"
                )
                + "\n"
            )
        separator = pick(rng, [":\n", " = ", ": "])
        reverse_flanks = is_fim and bool(rng.random() < 0.25)

        def render(left: str, right: str, middle: str, dna: str) -> tuple[Span, ...]:
            if not is_fim:
                lead = header + pick_label if kind != "bare_causal" else ""
                return (Span(lead, "text"), Span(dna, "dna", True))
            spans = [
                Span(
                    header + (instruction if layout == "instruction_first" else ""),
                    "text",
                )
            ]
            flanks = [(left_label, left, "left"), (right_label, right, "right")]
            if reverse_flanks:
                flanks.reverse()
            for label, seq, role in flanks:
                spans.extend(
                    [
                        Span(
                            f"[{label}] " if layout == "compact" else label + separator,
                            "text",
                        ),
                        Span(seq, role),
                        Span("\n", "text"),
                    ]
                )
            if layout != "instruction_first":
                spans.append(Span(instruction, "text"))
            if gap_hint:
                spans.append(Span(gap_hint, "text"))
            spans.extend(
                [
                    Span(
                        f"[{answer_label}] "
                        if layout == "compact"
                        else answer_label + separator,
                        "text",
                    ),
                    Span(middle, "middle", True),
                ]
            )
            return tuple(spans)

        pick_label = (
            pick(rng, self.phrases["dna_label"]) + separator
            if kind == "annotated_causal"
            else ""
        )
        empty_spans = render("", "", "", "")
        text_tokens = sum(
            len(self.encoder.text_ids(s.text)) for s in empty_spans if s.role == "text"
        )
        dna_length = length - text_tokens
        if dna_length < 1024 or dna_length < gap_length + 256:
            raise ValueError("Metadata leaves insufficient DNA context")
        oriented = reverse_complement(raw.seq) if rc else raw.seq
        crop_rng = rng_for(c.seed, occurrence, "crop")
        crop_start = int(crop_rng.integers(8192 - dna_length + 1))
        native = oriented[crop_start : crop_start + dna_length]
        dna = (
            native
            if case == "native_softmask"
            else native.lower()
            if case == "unmasked_lower"
            else native.upper()
        )
        centered = is_fim and bool(gap_rng.random() < c.center_gap_probability)
        gap_start = (
            (dna_length - gap_length) // 2
            if centered
            else int(gap_rng.integers(128, dna_length - gap_length - 128 + 1))
            if is_fim
            else 0
        )
        guide = GapChoice(reason="not_short_fim")
        if kind == "short_fim":
            annotation = self.annotations.get(raw.row)
            if annotation is not None:
                annotation.verify(raw.seq, raw.split)
            guide = choose_guided_gap(
                annotation.landmarks if annotation else (),
                rng_for(c.seed, occurrence, "guided_gap"),
                reverse_complemented=rc,
                crop_start=crop_start,
                dna_length=dna_length,
                gap_length=gap_length,
                guided_probability=c.guided_probability,
            )
            if guide.guided:
                assert guide.start is not None
                gap_start = guide.start
                centered = False
        left, middle, right = (
            dna[:gap_start],
            dna[gap_start : gap_start + gap_length],
            dna[gap_start + gap_length :],
        )
        spans = render(left, right, middle, dna)
        tokens, weights = self.encoder.encode(spans)
        if len(tokens) != length or not 2048 <= len(tokens) <= bucket:
            raise ValueError("Encoded input violates its length contract")
        attributes = {
            "occurrence": occurrence,
            "raw_row": raw.row,
            "assembly": raw.assembly,
            "kind": kind,
            "case": case,
            "rc": rc,
            "bucket": bucket,
            "input_tokens": len(tokens),
            "text_tokens": text_tokens,
            "dna_tokens": dna_length,
            "mask_description": bool(mask_description),
            "description": bool(description),
            "metadata_fields": sorted(fields),
            "metadata_structure": structure,
            "layout": layout,
            "right_flank_first": bool(reverse_flanks),
            "gap_length": gap_length,
            "gap_start": gap_start,
            "center_gap": bool(centered),
            "minimal_fim": bool(minimal_fim),
            "gap_length_visible": bool(gap_length_visible),
            "annotation_eligible": guide.eligible,
            "annotation_guided": guide.guided,
            "landmark_class": guide.landmark_class,
            "landmark_mode": guide.mode,
            "landmark_offset": guide.offset,
            "annotation_source": guide.source,
            "annotation_fallback": guide.reason,
            "crop_start_in_oriented_window": crop_start,
            "native_repeat_bases": sum(ch.islower() for ch in native),
            "scored_tokens": int(np.count_nonzero(weights)),
        }
        return Example(tokens, weights, spans, bucket, attributes)


def collate(examples: list[Example], pad_id: int = 0) -> dict[str, np.ndarray]:
    """Pad separate windows to their shared bucket; never concatenate documents."""
    if not examples or len({e.bucket for e in examples}) != 1:
        raise ValueError("A batch requires nonempty examples from one length bucket")
    shape = (len(examples), examples[0].bucket)
    tokens = np.full(shape, pad_id, dtype=np.int32)
    weights = np.zeros(shape, dtype=np.float32)
    segments = np.full(shape, -1, dtype=np.int32)
    lengths = np.asarray([len(e.token_ids) for e in examples], dtype=np.int32)
    for i, e in enumerate(examples):
        tokens[i, : lengths[i]] = e.token_ids
        weights[i, : lengths[i]] = e.loss_weights
        segments[i, : lengths[i]] = 0
    return {
        "token_ids": tokens,
        "loss_weights": weights,
        "segment_ids": segments,
        "lengths": lengths,
    }
