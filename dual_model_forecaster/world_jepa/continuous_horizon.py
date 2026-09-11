"""Continuous horizon representations for JEPA-space forecasting.

The production model must be able to train on a dense grid of integer-day
labels and still query non-integer horizons at inference time.  A learned
lookup table cannot provide that behaviour.  This module therefore represents
positive time with log-time features and fixed Fourier bands, followed by a
small smooth MLP.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


class ContinuousHorizonEmbedding(nn.Module):
    """Embed arbitrary positive horizons without a discrete lookup table.

    Parameters
    ----------
    embedding_dim:
        Size of the returned representation.
    hidden_dim:
        Width of the log-time/Fourier projection MLP.  Defaults to twice the
        embedding size (with a small minimum for narrow embeddings).
    num_frequencies:
        Number of fixed Fourier bands applied to ``log1p(horizon / time_scale)``.
    min_frequency, max_frequency:
        Inclusive geometric range for the Fourier bands.
    time_scale:
        Unit conversion applied before taking log-time.  With the default,
        horizons are interpreted directly in days.

    Notes
    -----
    The output is continuous in the horizon.  Integer training horizons and
    float inference horizons take exactly the same path through the module.
    """

    def __init__(
        self,
        embedding_dim: int,
        *,
        hidden_dim: int | None = None,
        num_frequencies: int = 8,
        min_frequency: float = 0.25,
        max_frequency: float = 4.0,
        time_scale: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if num_frequencies <= 0:
            raise ValueError("num_frequencies must be positive")
        if not 0.0 < min_frequency <= max_frequency:
            raise ValueError("Fourier frequencies must satisfy 0 < min <= max")
        if time_scale <= 0.0:
            raise ValueError("time_scale must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.embedding_dim = int(embedding_dim)
        self.time_scale = float(time_scale)
        hidden_dim = int(hidden_dim or max(32, 2 * embedding_dim))

        bands = torch.logspace(
            math.log10(float(min_frequency)),
            math.log10(float(max_frequency)),
            steps=int(num_frequencies),
            dtype=torch.float32,
        )
        self.register_buffer("fourier_bands", bands)

        # log(t), a bounded log(t), and a reciprocal-time feature complement
        # the periodic coordinates.  All are smooth for strictly positive t.
        feature_dim = 3 + 2 * int(num_frequencies)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
        )

    def forward(self, horizons: Tensor | Sequence[float] | float) -> Tensor:
        """Return an embedding with shape ``horizons.shape + (embedding_dim,)``.

        ``horizons`` may contain dense integer days or arbitrary floating-point
        values, but every value must be finite and strictly positive.
        """

        values = torch.as_tensor(
            horizons,
            dtype=self.fourier_bands.dtype,
            device=self.fourier_bands.device,
        )
        if values.numel() == 0:
            raise ValueError("horizons must contain at least one value")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("horizons must be finite")
        if not bool((values > 0).all()):
            raise ValueError("horizons must be strictly positive")

        original_shape = values.shape
        flat = values.reshape(-1, 1)
        log_time = torch.log1p(flat / self.time_scale)
        bounded_log_time = log_time / (1.0 + log_time)
        reciprocal_time = 1.0 / (1.0 + flat / self.time_scale)
        phases = math.pi * log_time * self.fourier_bands.view(1, -1)
        features = torch.cat(
            (
                log_time,
                bounded_log_time,
                reciprocal_time,
                torch.sin(phases),
                torch.cos(phases),
            ),
            dim=-1,
        )
        embedded = self.projection(features)
        return embedded.reshape(*original_shape, self.embedding_dim)

    def extra_repr(self) -> str:
        return (
            f"embedding_dim={self.embedding_dim}, "
            f"num_frequencies={self.fourier_bands.numel()}, "
            f"time_scale={self.time_scale:g}"
        )

