from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as torch_functional


def _patchify(x: torch.Tensor, patch_length: int) -> torch.Tensor:
    batch, length, features = x.shape
    patch_length = max(int(patch_length), 1)
    remainder = length % patch_length
    if remainder:
        pad = patch_length - remainder
        x = torch_functional.pad(x, (0, 0, pad, 0))
        length = x.shape[1]
    patch_count = max(length // patch_length, 1)
    return x.reshape(batch, patch_count, patch_length * features)


class JEPAPatchEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        patch_length: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.patch_length = int(patch_length)
        self.patch_proj = nn.Linear(self.patch_length * self.input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=hidden_dim * 2,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, self.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = _patchify(x, self.patch_length)
        encoded = self.encoder(self.patch_proj(patches))
        pooled = self.norm(encoded.mean(dim=1))
        return self.head(pooled)


class JEPATargetEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        patch_length: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.encoder = JEPAPatchEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            patch_length=patch_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        return self.encoder(x)


class JEPAPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        horizon_embedding_dim: int = 24,
        hidden_dim: int = 128,
        max_horizon: int = 64,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.max_horizon = int(max_horizon)
        self.horizon_embedding = nn.Embedding(self.max_horizon + 1, int(horizon_embedding_dim))
        self.net = nn.Sequential(
            nn.Linear(int(latent_dim) + int(horizon_embedding_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(latent_dim)),
        )

    def forward(self, context_latent: torch.Tensor, horizon: torch.Tensor) -> torch.Tensor:
        horizon = horizon.to(dtype=torch.long, device=context_latent.device).clamp(0, self.max_horizon)
        embedded = self.horizon_embedding(horizon)
        return self.net(torch.cat([context_latent, embedded], dim=-1))


class SpecialistJEPA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        patch_length: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.10,
        max_horizon: int = 64,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.context_encoder = JEPAPatchEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            patch_length=patch_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.target_encoder = JEPATargetEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            patch_length=max(1, min(int(patch_length), 4)),
            hidden_dim=hidden_dim,
            num_layers=max(1, int(num_layers) - 1),
            num_heads=num_heads,
            dropout=dropout,
        )
        self.predictor = JEPAPredictor(
            latent_dim=latent_dim,
            horizon_embedding_dim=24,
            hidden_dim=hidden_dim,
            max_horizon=max_horizon,
            dropout=dropout,
        )

    def forward(
        self,
        context_features: torch.Tensor,
        horizon: torch.Tensor,
        target_features: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        context_latent = self.context_encoder(context_features)
        predicted_target_latent = self.predictor(context_latent, horizon)
        output = {
            "context_latent": context_latent,
            "predicted_target_latent": predicted_target_latent,
        }
        if target_features is not None:
            output["target_latent"] = self.target_encoder(target_features, mask=target_mask)
        return output
