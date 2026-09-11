from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from dual_model_forecaster.world_jepa.world_encoder import WorldJEPAOutput


def _valid_rows(latents: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    if latents.ndim == 2:
        rows = latents
        if valid_mask is not None:
            mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=latents.device).reshape(-1)
            if mask.shape[0] != rows.shape[0]:
                raise ValueError("valid_mask does not match latent rows.")
            rows = rows[mask]
        return rows
    if latents.ndim != 3:
        raise ValueError("latents must have shape [N, D] or [B, H, D].")
    if valid_mask is None:
        return latents.reshape(-1, latents.shape[-1])
    mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=latents.device)
    if mask.shape != latents.shape[:2]:
        raise ValueError(f"Expected valid_mask {tuple(latents.shape[:2])}, got {tuple(mask.shape)}.")
    return latents[mask]


def variance_regularization_loss(
    latents: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """VICReg-style variance floor that is positive for collapsed embeddings."""

    rows = _valid_rows(latents, valid_mask)
    if rows.shape[0] < 2:
        return latents.new_tensor(float(target_std))
    std = torch.sqrt(rows.var(dim=0, unbiased=False) + float(eps))
    return torch.relu(float(target_std) - std).mean()


def covariance_regularization_loss(
    latents: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize redundant latent axes without penalizing their variances."""

    rows = _valid_rows(latents, valid_mask)
    if rows.shape[0] < 2 or rows.shape[1] < 2:
        return latents.new_tensor(0.0)
    centered = rows - rows.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(rows.shape[0] - 1, 1)
    off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
    return off_diagonal.pow(2).sum() / rows.shape[1]


def anti_collapse_components(
    context_summary: torch.Tensor,
    predicted_target: torch.Tensor,
    target_valid_mask: torch.Tensor | None = None,
    *,
    target_std: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return branch- and horizon-aware VICReg-style penalties.

    The EMA target is deliberately excluded because it is stop-gradient. Its
    diversity must be monitored separately. Context and prediction branches
    are not concatenated: otherwise a high-variance context can conceal a
    collapsed predictor (or vice versa). Prediction horizons are regularized
    separately before their penalties are averaged.
    """

    if context_summary.ndim != 2:
        raise ValueError("context_summary must have shape [B, D].")
    if predicted_target.ndim != 3:
        raise ValueError("predicted_target must have shape [B, H, D].")
    if predicted_target.shape[0] != context_summary.shape[0]:
        raise ValueError("context_summary and predicted_target batch dimensions must match.")
    if predicted_target.shape[-1] != context_summary.shape[-1]:
        raise ValueError("context_summary and predicted_target latent dimensions must match.")

    if target_valid_mask is None:
        mask = torch.ones(
            predicted_target.shape[:2],
            dtype=torch.bool,
            device=predicted_target.device,
        )
    else:
        mask = torch.as_tensor(
            target_valid_mask,
            dtype=torch.bool,
            device=predicted_target.device,
        )
        if mask.shape != predicted_target.shape[:2]:
            raise ValueError("target_valid_mask must have shape [B, H].")

    context_variance = variance_regularization_loss(
        context_summary,
        target_std=target_std,
    )
    context_covariance = covariance_regularization_loss(context_summary)
    horizon_variances: list[torch.Tensor] = []
    horizon_covariances: list[torch.Tensor] = []
    for horizon_index in range(predicted_target.shape[1]):
        horizon_mask = mask[:, horizon_index]
        if not horizon_mask.any():
            continue
        horizon_rows = predicted_target[horizon_mask, horizon_index, :]
        horizon_variances.append(
            variance_regularization_loss(horizon_rows, target_std=target_std)
        )
        horizon_covariances.append(covariance_regularization_loss(horizon_rows))

    if horizon_variances:
        prediction_variance = torch.stack(horizon_variances).mean()
        prediction_covariance = torch.stack(horizon_covariances).mean()
        variance = 0.5 * (context_variance + prediction_variance)
        covariance = 0.5 * (context_covariance + prediction_covariance)
    else:
        prediction_variance = predicted_target.new_tensor(0.0)
        prediction_covariance = predicted_target.new_tensor(0.0)
        variance = context_variance
        covariance = context_covariance

    return {
        "variance": variance,
        "covariance": covariance,
        "context_variance": context_variance,
        "context_covariance": context_covariance,
        "prediction_variance": prediction_variance,
        "prediction_covariance": prediction_covariance,
    }


def anti_collapse_loss(
    context_summary: torch.Tensor,
    predicted_target: torch.Tensor,
    target_valid_mask: torch.Tensor | None = None,
    *,
    target_std: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible aggregate anti-collapse loss."""

    components = anti_collapse_components(
        context_summary,
        predicted_target,
        target_valid_mask,
        target_std=target_std,
    )
    return components["variance"], components["covariance"]


def temporal_consistency_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    adjacency_pairs: torch.Tensor | None,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match latent changes only across explicitly adjacent samples.

    There is intentionally no fallback to ``batch[i]``/``batch[i + 1]``. A
    shuffled batch has temporal meaning only when the dataset supplies actual
    adjacency pairs.
    """

    if adjacency_pairs is None:
        return predicted.new_tensor(0.0)
    pairs = torch.as_tensor(adjacency_pairs, dtype=torch.long, device=predicted.device)
    if pairs.numel() == 0:
        return predicted.new_tensor(0.0)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("adjacency_pairs must have shape [P, 2].")
    if (pairs < 0).any() or (pairs >= predicted.shape[0]).any():
        raise ValueError("adjacency_pairs contains a batch index outside the latent tensor.")
    left, right = pairs[:, 0], pairs[:, 1]
    predicted_delta = predicted[right] - predicted[left]
    target_delta = target.detach()[right] - target.detach()[left]
    per_horizon = (predicted_delta - target_delta).pow(2).mean(dim=-1)
    if valid_mask is None:
        return per_horizon.mean()
    mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=predicted.device)
    if mask.shape != predicted.shape[:2]:
        raise ValueError("valid_mask must have shape [B, H].")
    pair_mask = mask[left] & mask[right]
    if not pair_mask.any():
        return predicted.new_tensor(0.0)
    return per_horizon[pair_mask].mean()


def uncertainty_nll_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    uncertainty: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Heteroscedastic Gaussian NLL used to make uncertainty operational."""

    if uncertainty.shape != predicted.shape[:2]:
        raise ValueError("uncertainty must have shape [B, H].")
    variance = uncertainty.pow(2).clamp_min(float(eps))
    squared_error = (predicted - target.detach()).pow(2).mean(dim=-1)
    nll = 0.5 * (squared_error / variance + variance.log())
    if valid_mask is None:
        return nll.mean()
    mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=predicted.device)
    if mask.shape != nll.shape:
        raise ValueError("valid_mask must have shape [B, H].")
    if not mask.any():
        return predicted.new_tensor(0.0)
    return nll[mask].mean()


def disagreement_diagnostics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    cosine_disagreement = 1.0 - F.cosine_similarity(predicted, target.detach(), dim=-1, eps=1e-8)
    rmse = torch.sqrt((predicted - target.detach()).pow(2).mean(dim=-1) + 1e-8)
    if valid_mask is not None:
        mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=predicted.device)
        cosine_disagreement = torch.where(mask, cosine_disagreement, torch.zeros_like(cosine_disagreement))
        rmse = torch.where(mask, rmse, torch.zeros_like(rmse))
    return {"cosine_disagreement": cosine_disagreement, "latent_rmse": rmse}


def world_jepa_loss(
    output: WorldJEPAOutput,
    *,
    target_valid_mask: torch.Tensor | None = None,
    adjacency_pairs: torch.Tensor | None = None,
    regime_targets: torch.Tensor | None = None,
    prediction_mode: str = "smooth_l1",
    lambda_variance: float = 1.0,
    lambda_covariance: float = 0.04,
    lambda_temporal: float = 0.10,
    lambda_uncertainty: float = 0.05,
    lambda_regime: float = 0.10,
    target_std: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if output.target_latent is None:
        raise ValueError("world_jepa_loss requires target_x in the model forward pass.")
    target = output.target_latent.detach()
    mask = target_valid_mask if target_valid_mask is not None else output.target_valid_mask
    if mask is None:
        mask = torch.ones(output.predicted_target.shape[:2], dtype=torch.bool, device=output.predicted_target.device)
    else:
        mask = torch.as_tensor(mask, dtype=torch.bool, device=output.predicted_target.device)
    if mask.shape != output.predicted_target.shape[:2]:
        raise ValueError("target_valid_mask must have shape [B, H].")
    if not mask.any():
        raise ValueError("world_jepa_loss received no valid targets.")

    if prediction_mode == "mse":
        elementwise_prediction = (output.predicted_target - target).pow(2).mean(dim=-1)
    elif prediction_mode == "cosine":
        elementwise_prediction = 1.0 - F.cosine_similarity(
            output.predicted_target,
            target,
            dim=-1,
            eps=1e-8,
        )
    elif prediction_mode == "smooth_l1":
        elementwise_prediction = F.smooth_l1_loss(
            output.predicted_target,
            target,
            reduction="none",
        ).mean(dim=-1)
    else:
        raise ValueError(f"Unsupported prediction_mode={prediction_mode!r}.")
    prediction_loss = elementwise_prediction[mask].mean()
    anti_collapse = anti_collapse_components(
        output.context_summary,
        output.predicted_target,
        mask,
        target_std=float(target_std),
    )
    variance_loss = anti_collapse["variance"]
    covariance_loss = anti_collapse["covariance"]
    temporal_loss = temporal_consistency_loss(
        output.predicted_target,
        target,
        adjacency_pairs,
        mask,
    )
    uncertainty_loss = uncertainty_nll_loss(
        output.predicted_target,
        target,
        output.uncertainty,
        mask,
    )
    regime_loss = output.predicted_target.new_tensor(0.0)
    if regime_targets is not None and float(lambda_regime) != 0.0:
        labels = torch.as_tensor(regime_targets, dtype=torch.long, device=output.regime_logits.device)
        if labels.shape != (output.regime_logits.shape[0],):
            raise ValueError("regime_targets must have shape [B].")
        valid_labels = (labels >= 0) & (labels < output.regime_logits.shape[1])
        if valid_labels.any():
            regime_loss = F.cross_entropy(output.regime_logits[valid_labels], labels[valid_labels])

    total = (
        prediction_loss
        + float(lambda_variance) * variance_loss
        + float(lambda_covariance) * covariance_loss
        + float(lambda_temporal) * temporal_loss
        + float(lambda_uncertainty) * uncertainty_loss
        + float(lambda_regime) * regime_loss
    )
    diagnostics = disagreement_diagnostics(output.predicted_target, target, mask)
    metrics = {
        "loss": float(total.detach().cpu()),
        "prediction_loss": float(prediction_loss.detach().cpu()),
        "variance_regularization_loss": float(variance_loss.detach().cpu()),
        "covariance_regularization_loss": float(covariance_loss.detach().cpu()),
        "context_variance_regularization_loss": float(
            anti_collapse["context_variance"].detach().cpu()
        ),
        "context_covariance_regularization_loss": float(
            anti_collapse["context_covariance"].detach().cpu()
        ),
        "prediction_variance_regularization_loss": float(
            anti_collapse["prediction_variance"].detach().cpu()
        ),
        "prediction_covariance_regularization_loss": float(
            anti_collapse["prediction_covariance"].detach().cpu()
        ),
        "temporal_consistency_loss": float(temporal_loss.detach().cpu()),
        "uncertainty_nll_loss": float(uncertainty_loss.detach().cpu()),
        "regime_loss": float(regime_loss.detach().cpu()),
        "mean_disagreement": float(diagnostics["cosine_disagreement"][mask].mean().detach().cpu()),
        "mean_latent_rmse": float(diagnostics["latent_rmse"][mask].mean().detach().cpu()),
        "valid_target_count": float(mask.sum().detach().cpu()),
    }
    return total, metrics


class WorldJEPALoss(nn.Module):
    """Module wrapper around :func:`world_jepa_loss` for trainer configs."""

    def __init__(self, **loss_kwargs: Any) -> None:
        super().__init__()
        self.loss_kwargs = dict(loss_kwargs)

    def forward(self, output: WorldJEPAOutput, **batch_kwargs: Any) -> tuple[torch.Tensor, dict[str, float]]:
        return world_jepa_loss(output, **self.loss_kwargs, **batch_kwargs)


total_world_jepa_loss = world_jepa_loss
