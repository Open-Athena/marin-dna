"""Lightning modules for language modeling (MLM and CLM)."""

from typing import Any

import torch
import torch.nn as nn
from biofoundation.model.scoring import compute_llr_clm, compute_llr_mlm
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from sklearn.metrics import average_precision_score
from torchmetrics.aggregation import CatMetric

from glm_experiments.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class MaskedLMAdapter(nn.Module):
    """Adapter to make MLM compatible with biofoundation's MaskedLM protocol.

    biofoundation's compute_llr_mlm expects a model with forward(input_ids) -> logits,
    but our LM model has get_logits(input_ids) -> logits.
    """

    def __init__(self, lm: nn.Module):
        super().__init__()
        self.lm = lm

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits (MaskedLM protocol)."""
        return self.lm.get_logits(input_ids)


class CausalLMAdapter(nn.Module):
    """Adapter to make CLM compatible with biofoundation's CausalLM protocol.

    biofoundation's compute_llr_clm expects a model with forward(input_ids) -> logits.
    """

    def __init__(self, lm: nn.Module):
        super().__init__()
        self.lm = lm

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits (CausalLM protocol)."""
        return self.lm.get_logits(input_ids)


class LMLitModule(LightningModule):
    """Base Lightning module for language modeling.

    Subclasses override create_adapter() and get_loss_name() to implement
    MLM vs CLM-specific adapters and metric names.

    Args:
        net: Language model (MLM or CLM)
        optimizer: Optimizer partial function (from Hydra with _partial_: true)
        scheduler: Scheduler partial function (from Hydra with _partial_: true)
        soft_masked_weight: Weight for soft-masked positions in training loss
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        soft_masked_weight: float = 0.01,
        evals: dict[str, dict] | None = None,
        lm_validation_enabled: bool = True,
        effective_batch_size: int = 2048,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        self.soft_masked_weight = soft_masked_weight
        self.lm_validation_enabled = lm_validation_enabled
        self.effective_batch_size = effective_batch_size
        self.last_selector_diagnostics: dict[str, torch.Tensor] | None = None

        # Create objective-specific adapter
        self.adapter = self.create_adapter(net)
        self.loss_name = self.get_loss_name()

        # Initialize dynamic eval metrics from config
        self._eval_names: list[str] = []
        self._eval_metrics: dict[str, dict] = {}

        if evals:
            self._setup_eval_metrics(evals)

    def _setup_eval_metrics(self, evals: list[dict]) -> None:
        """Initialize eval metrics from configuration.

        Args:
            evals: List of eval configurations (each with 'name' field)
        """
        from glm_experiments.utils.metrics import get_metric, get_transform

        # Store eval names in list order (guaranteed consistent)
        self._eval_names = [e["name"] for e in evals]

        # Create CatMetrics and config for each eval
        for eval_config in evals:
            eval_name = eval_config["name"]

            # Validate transform exists
            transform_name = eval_config.get("transform", "identity")
            get_transform(transform_name)  # Raises ValueError if invalid

            # Validate metrics exist
            metrics = eval_config.get("metrics", ["auprc"])
            for metric_name in metrics:
                get_metric(metric_name)  # Raises ValueError if invalid

            # Store config with CatMetrics
            self._eval_metrics[eval_name] = {
                "scores": CatMetric(),
                "labels": CatMetric(),
                "transform": transform_name,
                "metrics": metrics,
                "label_column": eval_config.get("label_column", "label"),
            }

    def create_adapter(self, net: nn.Module) -> nn.Module:
        """Create biofoundation adapter (override in subclasses).

        Args:
            net: Language model

        Returns:
            Adapter module for biofoundation scoring
        """
        raise NotImplementedError("Subclasses must implement create_adapter")

    def get_loss_name(self) -> str:
        """Get loss metric name (override in subclasses).

        Returns:
            Loss metric name (e.g., "mlm_loss" or "clm_loss")
        """
        raise NotImplementedError("Subclasses must implement get_loss_name")

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, soft_masked: torch.Tensor):
        """Forward pass through model."""
        return self.net(input_ids, labels, soft_masked, self.soft_masked_weight)

    def model_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Perform a single model step (shared by train/val).

        Args:
            batch: Batch dict with keys: input_ids, labels, soft_masked

        Returns:
            Dictionary with loss components
        """
        return self.forward(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            soft_masked=batch["soft_masked"],
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        loss_dict = self.model_step(batch)

        # Log all three loss variants
        self.log(
            f"train/{self.loss_name}",
            loss_dict["loss"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log(
            f"train/{self.loss_name}_full",
            loss_dict["loss_full"],
            on_step=True,
            on_epoch=False,
        )
        self.log(
            f"train/{self.loss_name}_non_soft_masked",
            loss_dict["loss_non_soft_masked"],
            on_step=True,
            on_epoch=False,
        )

        return loss_dict["loss"]  # Return main loss for backprop

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Validation step for LM and eval dataloaders.

        Args:
            batch: Batch dict (keys depend on dataloader)
            batch_idx: Batch index
            dataloader_idx: 0 for LM validation, 1+ for eval datasets
        """
        if dataloader_idx == 0:
            # LM validation
            loss_dict = self.model_step(batch)

            # Log all three loss variants
            self.log(
                f"val/{self.loss_name}",
                loss_dict["loss"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
                sync_dist=True,
            )
            self.log(
                f"val/{self.loss_name}_full",
                loss_dict["loss_full"],
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
                sync_dist=True,
            )
            self.log(
                f"val/{self.loss_name}_non_soft_masked",
                loss_dict["loss_non_soft_masked"],
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
                sync_dist=True,
            )
        else:
            # Dynamic eval dataset handling
            # dataloader_idx 1+ map to eval datasets in config order
            eval_idx = dataloader_idx - 1

            if eval_idx >= len(self._eval_names):
                raise IndexError(
                    f"dataloader_idx={dataloader_idx} exceeds configured evals. "
                    f"Expected indices 0-{len(self._eval_names)} for {len(self._eval_names)} eval(s)."
                )

            eval_name = self._eval_names[eval_idx]
            eval_config = self._eval_metrics[eval_name]

            # Compute raw LLR scores using objective-specific method
            raw_scores = self._compute_raw_llr(batch)

            # Apply transform
            from glm_experiments.utils.metrics import get_transform

            transform_fn = get_transform(eval_config["transform"])
            transformed_scores = transform_fn(raw_scores)

            # Update CatMetrics
            label_col = eval_config["label_column"]
            eval_config["scores"].update(transformed_scores)
            eval_config["labels"].update(batch[label_col])

    def _compute_raw_llr(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute raw LLR scores without transformation.

        This is the inverse of the old compute_vep_scores() - it returns raw LLR
        without the hardcoded negation.

        Args:
            batch: Batch dict from eval dataset (fields vary by objective)

        Returns:
            Raw LLR scores (higher = more likely under model)
        """
        raise NotImplementedError("Subclasses must implement _compute_raw_llr")

    def on_validation_epoch_end(self) -> None:
        """Compute and log metrics for all eval datasets."""
        from glm_experiments.utils.metrics import get_metric

        for eval_name, eval_config in self._eval_metrics.items():
            # Skip if no data collected
            if eval_config["scores"].update_count == 0:
                continue

            # Compute accumulated scores and labels
            scores = eval_config["scores"].compute().cpu().numpy()
            labels = eval_config["labels"].compute().cpu().numpy()
            sample_size = len(labels)

            # Compute and log each metric
            for metric_name in eval_config["metrics"]:
                metric_fn = get_metric(metric_name)
                try:
                    metric_value = metric_fn(labels, scores)

                    # Log to wandb/logger
                    log_key = f"val/{eval_name}_{metric_name}"
                    self.log(
                        log_key,
                        metric_value,
                        prog_bar=(metric_name == eval_config["metrics"][0]),
                    )

                    # Log to console
                    log.info(
                        f"{eval_name}: sample_size={sample_size}, "
                        f"{metric_name.upper()}={metric_value:.4f}"
                    )
                except Exception as e:
                    log.error(f"Failed to compute {metric_name} for {eval_name}: {e}")

            # Reset metrics for next epoch
            eval_config["scores"].reset()
            eval_config["labels"].reset()

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = self.hparams.optimizer(params=self.parameters())
        scheduler = self.hparams.scheduler(optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Log gradient norm before optimizer step."""
        norms = grad_norm(self, norm_type=2)
        self.log("train/grad_norm", norms["grad_2.0_norm_total"])


class MLMLitModule(LMLitModule):
    """Lightning module for Masked Language Modeling.

    Args:
        net: MLM model
        optimizer: Optimizer partial function
        scheduler: Scheduler partial function
    """

    def create_adapter(self, net: nn.Module) -> nn.Module:
        """Create MaskedLMAdapter for biofoundation scoring."""
        return MaskedLMAdapter(net)

    def get_loss_name(self) -> str:
        """Return MLM loss metric name."""
        return "mlm_loss"

    def _compute_raw_llr(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute raw MLM LLR scores (no transformation).

        Args:
            batch: Batch with keys {input_ids, pos, ref, alt, label}

        Returns:
            Raw LLR scores (higher = more likely under model)
        """
        return compute_llr_mlm(
            model=self.adapter,
            input_ids=batch["input_ids"],
            pos=batch["pos"],
            ref=batch["ref"],
            alt=batch["alt"],
        )


class DLMLitModule(LMLitModule):
    """Lightning module for Diffusion Language Modeling.

    Uses same adapter and VEP scoring as MLM since both are bidirectional
    masked models. Only difference is the masking strategy during training
    (handled by DLMDataModule).

    Args:
        net: DLM model
        optimizer: Optimizer partial function
        scheduler: Scheduler partial function
    """

    def create_adapter(self, net: nn.Module) -> nn.Module:
        """Create MaskedLMAdapter for biofoundation scoring."""
        return MaskedLMAdapter(net)

    def get_loss_name(self) -> str:
        """Return DLM loss metric name."""
        return "dlm_loss"

    def _compute_raw_llr(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute raw DLM LLR scores (no transformation).

        Uses same MLM scoring since DLM is also a bidirectional masked model.

        Args:
            batch: Batch with keys {input_ids, pos, ref, alt, label}

        Returns:
            Raw LLR scores (higher = more likely under model)
        """
        return compute_llr_mlm(
            model=self.adapter,
            input_ids=batch["input_ids"],
            pos=batch["pos"],
            ref=batch["ref"],
            alt=batch["alt"],
        )


class CLMLitModule(LMLitModule):
    """Lightning module for Causal Language Modeling.

    Args:
        net: CLM model
        optimizer: Optimizer partial function
        scheduler: Scheduler partial function
    """

    def create_adapter(self, net: nn.Module) -> nn.Module:
        """Create CausalLMAdapter for biofoundation scoring."""
        return CausalLMAdapter(net)

    def get_loss_name(self) -> str:
        """Return CLM loss metric name."""
        return "clm_loss"

    def _compute_raw_llr(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute raw CLM LLR scores (no transformation).

        Args:
            batch: Batch with keys {input_ids, label} where input_ids is shape [B, 2, L]

        Returns:
            Raw LLR scores (higher = more likely under model)
        """
        return compute_llr_clm(
            model=self.adapter,
            input_ids=batch["input_ids"],
        )
