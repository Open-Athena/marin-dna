"""Generic streaming DataModules for DNA language modeling."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from Bio.Seq import Seq
from biofoundation.model.adapters.hf import HFTokenizer
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from lightning import LightningDataModule
from torch.utils.data import DataLoader, default_collate
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from glm_experiments.data.evals import download_genome, load_eval_dataset
from glm_experiments.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def apply_reverse_complement(sequences: list[str]) -> list[str]:
    """Randomly reverse-complement sequences using PyTorch's worker RNG."""

    reverse_mask = torch.randint(0, 2, (len(sequences),))
    return [
        str(Seq(sequence).reverse_complement()) if reverse_mask[index] else sequence
        for index, sequence in enumerate(sequences)
    ]


def has_eligible_target(sequence: str) -> bool:
    """Return whether a sequence has at least one non-lowercase target."""

    return any(base.isupper() for base in sequence)


def build_soft_mask(
    sequences: list[str],
    input_ids: torch.Tensor,
    *,
    bos_token_id: int | None = None,
    require_leading_bos: bool = False,
) -> torch.Tensor:
    """Align source-case repeat flags to zero- or one-prefix-token coordinates."""

    if input_ids.ndim != 2 or input_ids.shape[0] != len(sequences):
        raise ValueError("input IDs and sequences must share the batch dimension")
    soft_masked = torch.zeros(input_ids.shape, dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        offset = input_ids.shape[1] - len(sequence)
        if require_leading_bos and (
            offset != 1
            or bos_token_id is None
            or int(input_ids[row, 0]) != bos_token_id
        ):
            raise ValueError(
                "issue #515 requires exactly one leading BOS and one token per base; "
                f"observed offset={offset}"
            )
        if offset not in {0, 1}:
            raise ValueError(
                "cannot align source-case repeat flags to token coordinates; "
                f"observed offset={offset}"
            )
        soft_masked[row, offset:] = torch.tensor(
            [base.islower() for base in sequence],
            dtype=torch.bool,
        )
    return soft_masked


def apply_mlm_masking(
    input_ids: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    mlm_probability: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply standard 80/10/10 BERT masking."""

    input_ids = input_ids.clone().to(torch.int8)
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    input_ids[indices_replaced] = mask_token_id
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.int8)
    input_ids[indices_random] = random_words[indices_random]
    return input_ids, labels


def apply_clm_labels(input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Use token IDs as CLM labels; the model performs causal alignment."""

    input_ids = input_ids.clone().to(torch.int8)
    return input_ids, input_ids.clone()


def apply_dlm_masking(
    input_ids: torch.Tensor,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask each sequence at an independently sampled diffusion rate."""

    input_ids = input_ids.clone().to(torch.int8)
    labels = input_ids.clone()
    batch_size, seq_len = input_ids.shape
    masking_ratios = torch.rand(batch_size, 1)
    masked_indices = torch.bernoulli(masking_ratios.expand(batch_size, seq_len)).bool()
    labels[~masked_indices] = -100
    input_ids[masked_indices] = mask_token_id
    return input_ids, labels


class LMDataModule(LightningDataModule):
    """Stream a pinned HF DNA dataset with configurable schema and evaluation."""

    def __init__(
        self,
        dataset_name: str = "songlab/gpn-animal-promoter-dataset",
        dataset_revision: str | None = None,
        text_key: str = "seq",
        species_key: str | None = None,
        tokenizer_name: str = "gonzalobenegas/tokenizer-dna-mlm",
        tokenizer_revision: str | None = None,
        sequence_length: int | None = None,
        batch_size: int = 2048,
        per_device_batch_size: int = 256,
        num_workers: int = 8,
        pin_memory: bool = True,
        soft_masked_weight: float = 0.01,
        data_augmentation: bool = True,
        filter_all_lowercase: bool = False,
        include_lm_validation: bool = True,
        max_val_lm_samples: int | None = None,
        shuffle_buffer_size: int = 10_000,
        seed: int = 42,
        evals: Mapping[str, Mapping[str, Any]] | list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.batch_size_per_device = per_device_batch_size
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.data_train: Any | None = None
        self.data_val: Any | None = None
        self.eval_datasets: dict[str, Any] = {}

    def _evaluation_configs(self) -> list[dict[str, Any]]:
        """Normalize legacy named mappings and current config lists."""

        configured = self.hparams.get("evals")
        if configured is None:
            return []
        if isinstance(configured, Mapping):
            return [
                {"name": str(name), **dict(values)}
                for name, values in configured.items()
            ]
        return [dict(values) for values in configured]

    def apply_labels(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create labels for the objective."""

        raise NotImplementedError

    def get_objective(self) -> str:
        """Return the objective name."""

        raise NotImplementedError

    def prepare_data(self) -> None:
        """Download tokenizer and registered evaluation genomes on rank zero."""

        AutoTokenizer.from_pretrained(
            self.hparams.tokenizer_name,
            revision=self.hparams.tokenizer_revision,
        )
        for eval_cfg in self._evaluation_configs():
            download_genome(
                url=eval_cfg["genome_url"],
                data_dir=eval_cfg.get("data_dir", "data"),
            )

    def _configure_batching(self) -> None:
        if self.trainer is None:
            return
        world_size = self.trainer.world_size
        per_device = self.hparams.per_device_batch_size
        total = self.hparams.batch_size
        denominator = per_device * world_size
        if total % denominator != 0:
            raise RuntimeError(
                f"Total batch size ({total}) must be divisible by "
                f"per_device_batch_size * world_size ({denominator})"
            )
        accumulation = total // denominator
        self.trainer.accumulate_grad_batches = accumulation
        if accumulation > 1 and self.trainer.val_check_interval is not None:
            self.trainer.val_check_interval *= accumulation
        log.info(
            "Batch size: per_device=%d world_size=%d accumulation=%d effective=%d",
            per_device,
            world_size,
            accumulation,
            denominator * accumulation,
        )

    def _transform_batch(
        self,
        examples: dict[str, list[Any]],
        *,
        data_augmentation: bool,
    ) -> dict[str, Any]:
        if self.tokenizer is None:
            raise RuntimeError("setup() must load the tokenizer first")
        sequences = [str(value) for value in examples[self.hparams.text_key]]
        if data_augmentation:
            sequences = apply_reverse_complement(sequences)
        tokenized = self.tokenizer(
            sequences,
            padding=(
                "max_length" if self.hparams.sequence_length is not None else False
            ),
            truncation=self.hparams.sequence_length is not None,
            max_length=self.hparams.sequence_length,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_special_tokens_mask=False,
        )
        input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.int8)
        attention_mask = torch.tensor(
            tokenized["attention_mask"],
            dtype=torch.bool,
        )
        soft_masked = build_soft_mask(
            sequences,
            input_ids,
            bos_token_id=self.tokenizer.bos_token_id,
        )
        input_ids, labels = self.apply_labels(input_ids)
        result: dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
            "soft_masked": soft_masked,
            "attention_mask": attention_mask,
        }
        if self.hparams.species_key is not None:
            result["species"] = examples[self.hparams.species_key]
        return result

    def _map_split(self, dataset: Any, *, data_augmentation: bool) -> Any:
        sample = next(iter(dataset.take(1)))
        return dataset.map(
            lambda examples: self._transform_batch(
                examples,
                data_augmentation=data_augmentation,
            ),
            batched=True,
            batch_size=min(4096, self.hparams.batch_size),
            remove_columns=list(sample),
        )

    def setup(self, stage: str | None = None) -> None:
        """Build deterministic streaming train and requested validation loaders."""

        torch.manual_seed(self.hparams.seed)
        self._configure_batching()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.tokenizer_name,
            revision=self.hparams.tokenizer_revision,
        )
        raw_datasets = load_dataset(
            self.hparams.dataset_name,
            revision=self.hparams.dataset_revision,
            streaming=True,
        )

        if stage in {"fit", None}:
            train_dataset = raw_datasets["train"].shuffle(
                seed=self.hparams.seed,
                buffer_size=self.hparams.shuffle_buffer_size,
            )
            if self.hparams.filter_all_lowercase:
                text_key = self.hparams.text_key
                train_dataset = train_dataset.filter(
                    lambda row: has_eligible_target(str(row[text_key]))
                )
            train_dataset = self._map_split(
                train_dataset,
                data_augmentation=self.hparams.data_augmentation,
            )

            val_dataset = None
            if self.hparams.include_lm_validation:
                if "validation" not in raw_datasets:
                    raise ValueError(
                        "include_lm_validation=true but the dataset has no validation split"
                    )
                val_dataset = raw_datasets["validation"]
                if self.hparams.max_val_lm_samples is not None:
                    val_dataset = val_dataset.take(self.hparams.max_val_lm_samples)
                val_dataset = self._map_split(
                    val_dataset,
                    data_augmentation=False,
                )

            if self.trainer is not None and self.trainer.world_size > 1:
                train_dataset = split_dataset_by_node(
                    train_dataset,
                    rank=self.trainer.global_rank,
                    world_size=self.trainer.world_size,
                )
                if val_dataset is not None:
                    val_dataset = split_dataset_by_node(
                        val_dataset,
                        rank=self.trainer.global_rank,
                        world_size=self.trainer.world_size,
                    )
            self.data_train = train_dataset
            self.data_val = val_dataset

            for eval_cfg in self._evaluation_configs():
                eval_name = eval_cfg["name"]
                self.eval_datasets[eval_name] = load_eval_dataset(
                    tokenizer=HFTokenizer(self.tokenizer),
                    dataset_name=eval_cfg["dataset_name"],
                    dataset_revision=eval_cfg.get("dataset_revision"),
                    genome_url=eval_cfg["genome_url"],
                    filter_name=eval_cfg.get("filter_name", "none"),
                    dataset_config=eval_cfg.get("dataset_config"),
                    split=eval_cfg.get("split", "train"),
                    window_size=eval_cfg.get("window_size", 255),
                    objective=self.get_objective(),
                    data_dir=eval_cfg.get("data_dir", "data"),
                    label_column=eval_cfg.get("label_column", "label"),
                )

    def train_dataloader(self) -> DataLoader:
        """Return the streaming training loader."""

        if self.data_train is None:
            raise RuntimeError("setup() must run before train_dataloader()")
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=default_collate,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        """Return only the validation loaders explicitly enabled by config."""

        loaders: list[DataLoader] = []
        if self.hparams.include_lm_validation:
            if self.data_val is None:
                raise RuntimeError("LM validation was requested but not initialized")
            loaders.append(
                DataLoader(
                    dataset=self.data_val,
                    batch_size=self.batch_size_per_device,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    shuffle=False,
                    collate_fn=default_collate,
                )
            )
        eval_by_name = {item["name"]: item for item in self._evaluation_configs()}
        for name, dataset in self.eval_datasets.items():
            config = eval_by_name[name]
            loaders.append(
                DataLoader(
                    dataset=dataset,
                    batch_size=config.get("batch_size", 128),
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    shuffle=False,
                    collate_fn=default_collate,
                )
            )
        if not loaders:
            raise RuntimeError("no validation loader is enabled")
        return loaders[0] if len(loaders) == 1 else loaders


class MLMDataModule(LMDataModule):
    """Masked-language-model data module."""

    def __init__(self, mlm_probability: float = 0.15, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.mlm_probability = mlm_probability

    def get_objective(self) -> str:
        return "mlm"

    def apply_labels(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer is None or self.tokenizer.mask_token_id is None:
            raise RuntimeError("MLM tokenizer must define a mask token")
        return apply_mlm_masking(
            input_ids,
            mask_token_id=int(self.tokenizer.mask_token_id),
            vocab_size=self.tokenizer.vocab_size,
            mlm_probability=self.mlm_probability,
        )


class DLMDataModule(LMDataModule):
    """Diffusion-language-model data module."""

    def get_objective(self) -> str:
        return "dlm"

    def apply_labels(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer is None or self.tokenizer.mask_token_id is None:
            raise RuntimeError("DLM tokenizer must define a mask token")
        return apply_dlm_masking(
            input_ids,
            mask_token_id=int(self.tokenizer.mask_token_id),
        )


class CLMDataModule(LMDataModule):
    """Causal-language-model data module."""

    def get_objective(self) -> str:
        return "clm"

    def apply_labels(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_clm_labels(input_ids)
