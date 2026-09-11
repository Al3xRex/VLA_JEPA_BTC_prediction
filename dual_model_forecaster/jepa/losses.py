from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as torch_functional


def latent_prediction_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str = "mse",
) -> torch.Tensor:
    if mode == "cosine":
        return 1.0 - torch_functional.cosine_similarity(predicted, target, dim=-1).mean()
    return torch_functional.mse_loss(predicted, target)


def variance_regularization_loss(latents: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    if latents.ndim != 2 or latents.shape[0] < 2:
        return latents.new_tensor(0.0)
    std = torch.sqrt(latents.var(dim=0, unbiased=False) + eps)
    return torch.relu(1.0 - std).mean()


def covariance_regularization_loss(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 2 or latents.shape[0] < 2 or latents.shape[1] < 2:
        return latents.new_tensor(0.0)
    centered = latents - latents.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(latents.shape[0] - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag.pow(2).sum() / latents.shape[1]).to(dtype=latents.dtype)


def temporal_smoothness_loss(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 2 or latents.shape[0] < 2:
        return latents.new_tensor(0.0)
    return (latents[1:] - latents[:-1]).pow(2).mean()


def total_jepa_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    context_latent: torch.Tensor | None = None,
    lambda_var: float = 1.0,
    lambda_cov: float = 0.04,
    lambda_smooth: float = 0.02,
    prediction_mode: str = "mse",
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_loss = latent_prediction_loss(predicted, target, mode=prediction_mode)
    regularization_latents = target if context_latent is None else torch.cat([context_latent, target], dim=0)
    var_loss = variance_regularization_loss(regularization_latents)
    cov_loss = covariance_regularization_loss(regularization_latents)
    smooth_loss = temporal_smoothness_loss(predicted)
    total = (
        pred_loss
        + float(lambda_var) * var_loss
        + float(lambda_cov) * cov_loss
        + float(lambda_smooth) * smooth_loss
    )
    metrics = {
        "loss": float(total.detach().cpu()),
        "latent_prediction_loss": float(pred_loss.detach().cpu()),
        "variance_regularization_loss": float(var_loss.detach().cpu()),
        "covariance_regularization_loss": float(cov_loss.detach().cpu()),
        "temporal_smoothness_loss": float(smooth_loss.detach().cpu()),
    }
    return total, metrics
