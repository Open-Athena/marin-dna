# Author: Lei Xiong <jsxlei@gmail.com>

"""
Model-specific wrappers for different architectures.

This module provides specialized wrappers for BPNet, ChromBPNet, and RegNet models,
extending the base ModelWrapper class with model-specific functionality.
"""

from typing import Dict, Any, Union
import torch
import torch.nn.functional as F
import lightning as L
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
import numpy as np
import argparse
from .chrombpnet import BPNet, ChromBPNet, ArsenalChromBPNet, ArsenalChromBPNetNoBias
from .model_config import ChromBPNetConfig, ArsenalChromBPNetConfig
import os
import sys

arsenal_dir = os.environ.get("ARSENAL_MODEL_DIR", "")
sys.path.append(f"{arsenal_dir}/src/regulatory_lm/")
# [vendored] removed ARSENAL `from modeling.model import *` (replaced by HF adapter)


# def multinomial_nll(logits, true_counts):
# 	dist = Multinomial(logits=logits, validate_args=False)
# 	log_probs = dist.log_prob(true_counts.long())
# 	return -log_probs.mean()


def multinomial_nll(logits, true_counts):
    """Compute the multinomial negative log-likelihood in PyTorch.

    Args:
      true_counts: Tensor of observed counts (batch_size, num_classes) (integer counts)
      logits: Tensor of predicted logits (batch_size, num_classes)

    Returns:
      Mean negative log-likelihood across the batch.
    """
    # Ensure true_counts is an integer tensor
    true_counts = true_counts.to(
        torch.float
    )  # Keep as float to prevent conversion issues

    # Compute total counts per example (should already be integer-like)
    counts_per_example = true_counts.sum(dim=-1, keepdim=True)

    # Convert logits to log probabilities (Softmax + Log)
    log_probs = F.log_softmax(logits, dim=-1)

    # Compute log-probability of the observed counts
    log_likelihood = (true_counts * log_probs).sum(dim=-1)

    # Compute multinomial coefficient (log factorial term)
    log_factorial_counts = torch.lgamma(counts_per_example + 1) - torch.lgamma(
        true_counts + 1
    ).sum(dim=-1)

    # Compute final NLL
    nll = -(log_factorial_counts + log_likelihood).mean()

    return nll


def pearson_corr(x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute the Pearson correlation coefficient along a given dimension for multi-dimensional tensors.

    Args:
            x (torch.Tensor): Input tensor of shape (..., N).
            y (torch.Tensor): Input tensor of shape (..., N).
            dim (int): The dimension along which to compute the Pearson correlation. Default is -1 (last dimension).

    Returns:
            torch.Tensor: Pearson correlation coefficients along the specified dimension.
    """
    # Ensure x and y have the same shape
    assert x.shape == y.shape, "Input tensors must have the same shape"

    # Step 1: Center the data (subtract the mean along the given dimension)
    x_centered = x - torch.mean(x, dim=dim, keepdim=True)
    y_centered = y - torch.mean(y, dim=dim, keepdim=True)

    # Step 2: Compute covariance (sum of element-wise products of centered tensors)
    cov = torch.sum(x_centered * y_centered, dim=dim)

    # Step 3: Compute standard deviations for each tensor along the specified dimension
    std_x = torch.sqrt(torch.sum(x_centered**2, dim=dim))
    std_y = torch.sqrt(torch.sum(y_centered**2, dim=dim))

    # Step 4: Compute Pearson correlation (with numerical stability)
    eps = 1e-8  # Small constant to prevent division by zero
    corr = cov / (std_x * std_y + eps)

    return corr


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array.

    Args:
            tensor: Input tensor

    Returns:
            Numpy array
    """
    return tensor.detach().cpu().numpy()


def adjust_bias_model_logcounts(bias_model, dataloader, verbose=False, device=1):
    """
    Given a bias model, sequences and associated counts, the function adds a
    constant to the output of the bias_model's logcounts that minimises squared
    error between predicted logcounts and observed logcounts (infered from
    cts). This simply reduces to adding the average difference between observed
    and predicted to the "bias" (constant additive term) of the Dense layer.
    Typically the seqs and counts would correspond to training nonpeak regions.
    ASSUMES model_bias's last layer is a dense layer that outputs logcounts.
    This would change if you change the model.
    """

    print("Predicting within adjust counts")
    bias_model.eval()
    with torch.no_grad():
        output = L.Trainer(logger=False, devices=device).predict(bias_model, dataloader)
        parsed_output = {
            key: np.concatenate([batch[key] for batch in output]) for key in output[0]
        }
        delta = parsed_output["true_count"].mean(-1) - parsed_output["pred_count"].mean(
            -1
        )

        bias_model.linear.bias += torch.Tensor(delta).to(bias_model.linear.bias.device)

    if verbose:
        print("### delta", delta.mean(), flush=True)
    return bias_model


def init_bias(bias, dataloader=None, verbose=False, device=1):
    print(f"Loading bias model from {bias}")
    bias_model = BPNet.from_keras(bias, name="bias")
    bias_model.eval()  # Freeze the sub-model
    for param in bias_model.parameters():
        param.requires_grad = False

    if dataloader is not None:
        bias_model = adjust_bias_model_logcounts(
            bias_model, dataloader, verbose=verbose, device=device
        )
    return bias_model


def load_bpnet_checkpoint(filename):
    model = torch.load(filename, weights_only=False)
    model["state_dict"] = {
        x: model["state_dict"][x]
        for x in model["state_dict"].keys()
        if "model.model" in x
    }
    model["state_dict"] = {
        x[12:]: model["state_dict"][x] for x in model["state_dict"].keys()
    }
    bpnet = BPNet(n_filters=512, n_layers=8)
    bpnet.load_state_dict(model["state_dict"])
    return bpnet


def init_chrombpnet_wo_bias(chrombpnet_wo_bias, freeze=True):
    print(f"Loading chrombpnet_wo_bias model from {chrombpnet_wo_bias}")
    if chrombpnet_wo_bias.endswith(".h5"):
        model = BPNet.from_keras(chrombpnet_wo_bias)
    elif chrombpnet_wo_bias.endswith(".pt"):
        model = BPNet(n_filters=512, n_layers=8)
        model.load_state_dict(torch.load(chrombpnet_wo_bias, map_location="cpu"))
    elif chrombpnet_wo_bias.endswith(".ckpt"):
        # model = BPNet.load_from_checkpoint(chrombpnet_wo_bias)
        model = load_bpnet_checkpoint(chrombpnet_wo_bias)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model


def init_arsenal_wo_bias(model_path, freeze=True):
    print(f"Loading arsenal_wo_bias model from {model_path}")
    model = ArsenalChromBPNetNoBiasWrapper.load_from_checkpoint(model_path)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model


class ControlWrapper(torch.nn.Module):
    """This wrapper automatically creates a control track of all zeroes.

    This wrapper will check to see whether the model is expecting a control
    track (e.g., most BPNet-style models) and will create one with the expected
    shape. If no control track is expected then it will provide the normal
    output from the model.
    """

    def __init__(self, model):
        super(ControlWrapper, self).__init__()
        self.model = model

    def forward(self, X, X_ctl=None):
        if X_ctl is not None:
            return self.model(X, X_ctl)

        if self.model.n_control_tracks == 0:
            return self.model(X)

        X_ctl = torch.zeros(
            X.shape[0],
            self.model.n_control_tracks,
            X.shape[-1],
            dtype=X.dtype,
            device=X.device,
        )
        return self.model(X, X_ctl)


class _ProfileLogitScaling(torch.nn.Module):
    """This ugly class is necessary because of Captum.

    Captum internally registers classes as linear or non-linear. Because the
    profile wrapper performs some non-linear operations, those operations must
    be registered as such. However, the inputs to the wrapper are not the
    logits that are being modified in a non-linear manner but rather the
    original sequence that is subsequently run through the model. Hence, this
    object will contain all of the operations performed on the logits and
    can be registered.


    Parameters
    ----------
    logits: torch.Tensor, shape=(-1, -1)
            The logits as they come out of a Chrom/BPNet model.
    """

    def __init__(self):
        super(_ProfileLogitScaling, self).__init__()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, logits):
        y_softmax = self.softmax(logits)
        y = logits * y_softmax
        return y
        # print("a")
        # y_lsm = torch.nn.functional.log_softmax(logits, dim=-1)
        # return torch.sign(logits) * torch.exp(torch.log(abs(logits)) + y_lsm)


class _Exp(torch.nn.Module):
    def __init__(self):
        super(_Exp, self).__init__()

    def forward(self, X):
        return torch.exp(X)


class _Log(torch.nn.Module):
    def __init__(self):
        super(_Log, self).__init__()

    def forward(self, X):
        return torch.log(X)


class ProfileWrapper(torch.nn.Module):
    """A wrapper class that returns transformed profiles.

    This class takes in a trained model and returns the weighted softmaxed
    outputs of the first dimension. Specifically, it takes the predicted
    "logits" and takes the dot product between them and the softmaxed versi
    ons
    of those logits. This is for convenience when using captum to calculate
    attribution scores.

    Parameters
    ----------
    model: torch.nn.Module
            A torch model to be wrapped.
    """

    def __init__(self, model):
        super(ProfileWrapper, self).__init__()
        self.model = model
        self.flatten = torch.nn.Flatten()
        self.scaling = _ProfileLogitScaling()

    def forward(self, x, x_ctl=None, **kwargs):
        logits = self.model(x, x_ctl=x_ctl, **kwargs)[0]
        logits = self.flatten(logits)
        logits = logits - torch.mean(logits, dim=-1, keepdims=True)
        return self.scaling(logits).sum(dim=-1, keepdims=True)


class CountWrapper(torch.nn.Module):
    """A wrapper class that only returns the predicted counts.

    This class takes in a trained model and returns only the second output.
    For BPNet models, this means that it is only returning the count
    predictions. This is for convenience when using captum to calculate
    attribution scores.

    Parameters
    ----------
    model: torch.nn.Module
            A torch model to be wrapped.
    """

    def __init__(self, model):
        super(CountWrapper, self).__init__()
        self.model = model

    def forward(self, x, x_ctl=None, **kwargs):
        return self.model(x, x_ctl=x_ctl, **kwargs)[1]


class ModelWrapper(LightningModule):
    """A generic wrapper for different model architectures to be used with PyTorch Lightning.

    This wrapper provides a flexible interface for training, validation, and testing different model architectures
    while maintaining consistent logging and optimization strategies.

    Attributes:
            model (nn.Module): The underlying model architecture
            learning_rate (float): Learning rate for optimization
            weight_decay (float): Weight decay for optimization
            warmup_steps (int): Number of warmup steps for learning rate scheduling
            finetune (bool): Whether to use fine-tuning mode
            alpha (float): Weight for count loss
            beta (float): Weight for profile loss
            metrics (Dict[str, List[float]]): Dictionary to store metrics during training
    """

    # Assigned by subclasses (BPNetWrapper, ChromBPNetWrapper, ...). Declared here so
    # ``self.model(...)`` type-checks — otherwise nn.Module.__getattr__ resolves it to
    # ``Tensor | Module`` and the ``Tensor`` branch is not callable.
    model: torch.nn.Module

    def __init__(self, args, **kwargs):
        """Initialize the model wrapper.

        Args:
                model: The underlying model architecture
                learning_rate: Learning rate for optimization
                weight_decay: Weight decay for optimization
                warmup_steps: Number of warmup steps for learning rate scheduling
                finetune: Whether to use fine-tuning mode
                alpha: Weight for count loss
                beta: Weight for profile loss
                **kwargs: Additional arguments to be passed to the model
        """

        super().__init__()
        self.alpha = args.alpha
        self.beta = args.beta
        self.verbose = args.verbose

        self.metrics = {
            "train": {"preds": [], "targets": []},
            "val": {"preds": [], "targets": []},
            "test": {"preds": [], "targets": []},
            "predict": {"preds": [], "targets": []},
        }

        for k, v in kwargs.items():
            setattr(self, k, v)

        # Save hyperparameters
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through the model.

        Args:
                x: Input tensor
                **kwargs: Additional arguments to be passed to the model's forward method

        Returns:
                Model output
        """
        return self.model(x, **kwargs)

    def _step(self, batch, batch_idx, mode="train"):
        raise NotImplementedError("Subclasses must implement this method")

    def init_bias(self, bias, dataloader=None, verbose=False, device=1):
        # print(f"Loading bias model from {bias}")
        return init_bias(bias, dataloader=dataloader, verbose=verbose, device=device)

    def init_chrombpnet_wo_bias(self, chrombpnet_wo_bias, freeze=True):
        # print(f"Initializing chrombpnet_wo_bias model from {chrombpnet_wo_bias}")
        return init_chrombpnet_wo_bias(chrombpnet_wo_bias, freeze=freeze)

    def _predict_on_dataloader(self, dataloader, func, **kwargs):
        outs = []
        for batch in dataloader:
            out = func(self, batch, **kwargs)
            outs.append(out.detach().cpu())
        return torch.cat(outs, dim=0)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        """Training step.

        Args:
                batch: Dictionary containing batch data
                batch_idx: Index of the current batch

        Returns:
                Loss value
        """
        return self._step(batch, batch_idx, "train")

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        """Validation step.

        Args:
                batch: Dictionary containing batch data
                batch_idx: Index of the current batch

        Returns:
                Loss value
        """
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Test step.

        Args:
                batch: Dictionary containing batch data
                batch_idx: Index of the current batch

        Returns:
                Loss value
        """
        return self._step(batch, batch_idx, "test")

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, np.ndarray]:
        """Prediction step.

        Args:
                batch: Dictionary containing batch data
                batch_idx: Index of the current batch

        Returns:
                Dictionary containing predictions and true values
        """
        return self._step(batch, batch_idx, "predict")

    def _epoch_end(self, mode: str) -> None:
        """Handle end of epoch operations.

        Args:
                mode: Mode of operation ('train', 'val', or 'test')
        """
        # Concatenate predictions and targets
        all_preds = torch.cat(self.metrics[mode]["preds"]).reshape(-1)
        all_targets = torch.cat(self.metrics[mode]["targets"]).reshape(-1)

        # Calculate and log correlation
        pr = self._pearson_corr(all_preds, all_targets)
        self.log(
            f"{mode}_count_pearson", pr, prog_bar=True, logger=True, sync_dist=True
        )

        # Reset metrics storage
        self.metrics[mode]["preds"] = []
        self.metrics[mode]["targets"] = []

    def on_train_epoch_end(self) -> None:
        """Handle end of training epoch."""
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        """Handle end of validation epoch."""
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        """Handle end of test epoch."""
        self._epoch_end("test")

    def configure_optimizers(  # type: ignore[override]  # Lightning's return union is broader; runtime-compatible
        self,
    ) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array.

        Args:
                tensor: Input tensor

        Returns:
                Numpy array
        """
        return tensor.detach().cpu().numpy()

    @staticmethod
    def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate Pearson correlation coefficient.

        Args:
                x: First tensor
                y: Second tensor

        Returns:
                Pearson correlation coefficient
        """
        x_centered = x - x.mean(dim=-1, keepdim=True)
        y_centered = y - y.mean(dim=-1, keepdim=True)
        numerator = (x_centered * y_centered).sum(dim=-1)
        denominator = torch.sqrt(
            (x_centered**2).sum(dim=-1) * (y_centered**2).sum(dim=-1)
        )
        return numerator / denominator

    @staticmethod
    def _multinomial_nll(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate multinomial negative log likelihood loss.

        Args:
                pred: Predicted probabilities
                target: Target probabilities

        Returns:
                Loss value
        """
        return -torch.sum(target * torch.log(pred + 1e-10), dim=-1).mean()


class BPNetWrapper(ModelWrapper):
    """Wrapper for BPNet model with specific configurations and loss functions.

    This wrapper extends the base ModelWrapper to handle BPNet-specific features
    such as profile and count predictions, and appropriate loss calculations.
    """

    def __init__(self, args):
        super().__init__(args)
        self.model = BPNet(
            out_dim=args.out_dim,
            n_filters=args.n_filters,
            n_layers=args.n_layers,
            conv1_kernel_size=args.conv1_kernel_size,
            profile_kernel_size=args.profile_kernel_size,
            n_outputs=args.n_outputs,
            n_control_tracks=args.n_control_tracks,
            profile_output_bias=args.profile_output_bias,
            count_output_bias=args.count_output_bias,
        )

    def _step(self, batch, batch_idx, mode="train"):
        x = batch["onehot_seq"]  # batch_size x 4 x seq_length
        true_profile = batch["profile"]  # batch_size x seq_length
        true_counts = torch.log1p(true_profile.sum(dim=-1))

        y_profile, y_count = self(x)
        y_count = y_count.squeeze(-1)  # batch_size x 1

        if mode == "predict":
            return {
                "pred_count": _to_numpy(y_count),
                "true_count": _to_numpy(true_counts),
                "pred_profile": _to_numpy(y_profile),  # .softmax(-1)),
                "true_profile": _to_numpy(true_profile),
            }

        self.metrics[mode]["preds"].append(y_count)
        self.metrics[mode]["targets"].append(true_counts)
        with torch.no_grad():
            # count_pearson = pearson_corr(y_count, true_counts).mean()
            profile_pearson = pearson_corr(y_profile.softmax(-1), true_profile).mean()
            self.log_dict(
                {f"{mode}_profile_pearson": profile_pearson},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        profile_loss = multinomial_nll(y_profile, true_profile)
        count_loss = F.mse_loss(y_count, true_counts)
        loss = self.beta * profile_loss + self.alpha * count_loss

        dict_show = {
            f"{mode}_loss": loss,
            f"{mode}_profile_loss": profile_loss,
            f"{mode}_count_loss": count_loss,
        }

        self.log_dict(
            dict_show,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, eps=1e-7)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.4, patience=3, min_lr=1e-8)
        # schedule_info = {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch", "frequency": 1}
        # return {"optimizer": optimizer, "scheduler": schedule_info}
        return optimizer

    def predict(self, x, forward_only=True):
        y_profile, y_count = self(x)
        y_count = torch.exp(y_count)

        if not forward_only:
            y_profile_revcomp, y_count_revcomp = self(x[:, ::-1, ::-1])
            y_count_revcomp = torch.exp(y_count_revcomp)
            y_profile = (y_profile + y_profile_revcomp) / 2
            y_count = (y_count + y_count_revcomp) / 2

        return y_profile.cpu().numpy(), y_count.cpu().numpy()


class ChromBPNetWrapper(BPNetWrapper):
    """Wrapper for ChromBPNet model with specific configurations and loss functions.

    This wrapper extends the base ModelWrapper to handle ChromBPNet-specific features
    such as chromatin accessibility predictions and appropriate loss calculations.
    """

    def __init__(
        self,
        args,
    ):
        """Initialize ChromBPNet wrapper.

        Args:
                model: ChromBPNet model instance
                alpha: Weight for count loss
                beta: Weight for profile loss
                bias_scaled: Path to bias model if using scaled bias
                **kwargs: Additional arguments to be passed to the model
        """
        super().__init__(args)

        config = ChromBPNetConfig.from_argparse_args(args)
        self.model = ChromBPNet(config)


class ArsenalChromBPNetWrapper(BPNetWrapper):
    """Wrapper for ArsenalChromBPNet model with specific configurations and loss functions.

    This wrapper extends the base ModelWrapper to handle ChromBPNet-specific features
    such as chromatin accessibility predictions and appropriate loss calculations.
    """

    def __init__(
        self,
        args,
    ):
        """Initialize ChromBPNet wrapper.

        Args:
                model: ChromBPNet model instance
                alpha: Weight for count loss
                beta: Weight for profile loss
                bias_scaled: Path to bias model if using scaled bias
                **kwargs: Additional arguments to be passed to the model
        """
        super().__init__(args)

        config = ArsenalChromBPNetConfig.from_argparse_args(args)
        arsenal_model = torch.load(args.arsenal_model, weights_only=False)
        self.model = ArsenalChromBPNet(config, arsenal_model)
        self.arsenal_output_type = args.arsenal_output_type

    def configure_optimizers(self):
        if self.arsenal_output_type == "likelihood":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, eps=1e-7)
        elif self.arsenal_output_type == "embedding":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, eps=1e-7)
        return optimizer

    def _step(self, batch, batch_idx, mode="train"):
        if (mode != "train") or (not self.model.finetune):
            self.model.arsenal_model.eval()
        x = batch["onehot_seq"]  # batch_size x 4 x seq_length
        true_profile = batch["profile"]  # batch_size x seq_length
        true_counts = torch.log1p(true_profile.sum(dim=-1))

        y_profile, y_count = self(x)
        y_count = y_count.squeeze(-1)  # batch_size x 1

        if mode == "predict":
            return {
                "pred_count": _to_numpy(y_count),
                "true_count": _to_numpy(true_counts),
                "pred_profile": _to_numpy(y_profile),  # .softmax(-1)),
                "true_profile": _to_numpy(true_profile),
            }

        self.metrics[mode]["preds"].append(y_count)
        self.metrics[mode]["targets"].append(true_counts)
        with torch.no_grad():
            # count_pearson = pearson_corr(y_count, true_counts).mean()
            profile_pearson = pearson_corr(y_profile.softmax(-1), true_profile).mean()
            self.log_dict(
                {f"{mode}_profile_pearson": profile_pearson},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        profile_loss = multinomial_nll(y_profile, true_profile)
        count_loss = F.mse_loss(y_count, true_counts)
        loss = self.beta * profile_loss + self.alpha * count_loss

        dict_show = {
            f"{mode}_loss": loss,
            f"{mode}_profile_loss": profile_loss,
            f"{mode}_count_loss": count_loss,
        }

        self.log_dict(
            dict_show,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        return loss


class ArsenalChromBPNetNoBiasWrapper(ArsenalChromBPNetWrapper):
    """Wrapper for ArsenalChromBPNet model with specific configurations and loss functions.

    This wrapper extends the base ModelWrapper to handle ChromBPNet-specific features
    such as chromatin accessibility predictions and appropriate loss calculations.
    """

    def __init__(
        self,
        args,
    ):

        super().__init__(args)

        config = ArsenalChromBPNetConfig.from_argparse_args(args)
        arsenal_model = torch.load(args.arsenal_model, weights_only=False)
        self.model = ArsenalChromBPNetNoBias(config, arsenal_model)


def create_model_wrapper(args, **kwargs) -> ModelWrapper:
    """Factory function to create appropriate model wrapper.

    Args:
            model_type: Type of model ('bpnet', 'chrombpnet')
            config: Model configuration
            **kwargs: Additional arguments to be passed to the wrapper

    Returns:
            Appropriate model wrapper instance

    Raises:
            ValueError: If model_type is not recognized
    """
    model_type = args.model_type.lower()
    if model_type == "bpnet":
        return BPNetWrapper(args)
    elif model_type == "chrombpnet":
        # annotate with the common base so the `arsenal-chrombpnet` branch below
        # (a sibling BPNetWrapper subclass) is a compatible reassignment
        model_wrapper: BPNetWrapper = ChromBPNetWrapper(args)
        if args.bias_scaled:
            model_wrapper.model.bias = model_wrapper.init_bias(args.bias_scaled)
        if args.chrombpnet_wo_bias:
            model_wrapper.model.model = model_wrapper.init_chrombpnet_wo_bias(
                args.chrombpnet_wo_bias, freeze=False
            )
        return model_wrapper
    elif model_type == "arsenal-chrombpnet":
        model_wrapper = ArsenalChromBPNetWrapper(args)
        if args.bias_scaled:
            model_wrapper.model.bias = model_wrapper.init_bias(args.bias_scaled)
        if args.chrombpnet_wo_bias:
            model_wrapper.model.model = model_wrapper.init_chrombpnet_wo_bias(
                args.chrombpnet_wo_bias, freeze=False
            )
        return model_wrapper

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_pretrained_model(args, checkpoint):
    import os

    wrapper_class = (
        ArsenalChromBPNetWrapper
        if args.model_type == "arsenal-chrombpnet"
        else ChromBPNetWrapper
    )
    if args.bias_scaled is None and os.path.exists(
        os.path.join(args.data_dir, "bias_scaled.h5")
    ):
        args.bias_scaled = os.path.join(args.data_dir, "bias_scaled.h5")
    if checkpoint is not None:
        if checkpoint.endswith(".ckpt"):
            model_wrapper = wrapper_class.load_from_checkpoint(
                checkpoint, map_location="cpu"
            )
            if args.bias_scaled:
                model_wrapper.model.bias = model_wrapper.init_bias(args.bias_scaled)
            else:
                print("No bias model found")
            return model_wrapper

        elif checkpoint.endswith(".pt"):
            model_wrapper = wrapper_class(args)
            model_wrapper.model.model.load_state_dict(
                torch.load(checkpoint, map_location="cpu")
            )
            if args.bias_scaled:
                model_wrapper.model.bias = model_wrapper.init_bias(args.bias_scaled)
            else:
                print("No bias model found")
            return model_wrapper
        elif checkpoint.endswith(".h5"):
            model_wrapper = wrapper_class(args)
            # For Keras H5 files, load using the from_keras method
            print(f"Loading chrombpnet_wo_bias model from {checkpoint}")
            model_wrapper.model.model = BPNet.from_keras(checkpoint)
            if args.bias_scaled:
                print(f"Loading bias model from {args.bias_scaled}")
                model_wrapper.model.bias = model_wrapper.init_bias(args.bias_scaled)
            else:
                print("No bias model found")
            return model_wrapper
    else:
        model_wrapper = wrapper_class(args)

    return model_wrapper


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="chrombpnet")
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    model_wrapper = create_model_wrapper(args)
    x = torch.randn(1, 4, 2114)
    batch = {
        "onehot_seq": x,
        "profile": torch.randn(1, 1000),
    }
    loss = model_wrapper._step(batch, 0, mode="train")
    print(loss)
