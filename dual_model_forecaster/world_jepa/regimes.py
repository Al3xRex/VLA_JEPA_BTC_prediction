"""Causal latent prototype regimes for JEPA world states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import KMeans


def causal_smooth_regime_probabilities(
    probabilities: np.ndarray,
    *,
    alpha: float = 0.25,
) -> np.ndarray:
    """Exponentially smooth regime states using present and past rows only."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or not len(values):
        raise ValueError("probabilities must have non-empty shape [N, R]")
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    normalized = values / values.sum(axis=1, keepdims=True).clip(min=1e-12)
    out = np.empty_like(normalized)
    out[0] = normalized[0]
    for position in range(1, len(normalized)):
        out[position] = (1.0 - float(alpha)) * out[position - 1] + float(alpha) * normalized[position]
        out[position] /= max(out[position].sum(), 1e-12)
    return out.astype(np.float32)


@dataclass(frozen=True)
class CausalRegimeCodebook:
    """Soft semi-state assignments to prototypes fitted on training latents.

    The codebook does not claim that a prototype is an economic regime. It is
    a stable latent-state coordinate whose centroids and distance scale are fit
    strictly before selection/test timestamps.
    """

    centers: np.ndarray
    distance_scale: float
    novelty_scale: float
    temperature: float
    seed: int
    fitted_rows: int

    @classmethod
    def fit(
        cls,
        latents: np.ndarray,
        regime_count: int,
        *,
        seed: int = 7,
        temperature: float = 0.35,
    ) -> "CausalRegimeCodebook":
        values = np.asarray(latents, dtype=float)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
            raise ValueError("latents must have shape [N, D] with at least two rows")
        finite_rows = np.isfinite(values).all(axis=1)
        values = values[finite_rows]
        if len(values) < 2:
            raise ValueError("not enough finite latent rows to fit regimes")
        clusters = min(max(int(regime_count), 2), len(values))
        if not 0.0 < float(temperature):
            raise ValueError("temperature must be positive")
        model = KMeans(n_clusters=clusters, n_init=10, random_state=int(seed))
        model.fit(values)
        centers = np.asarray(model.cluster_centers_, dtype=float)
        squared_distances = ((values[:, None, :] - centers[None, :, :]) ** 2).mean(axis=-1)
        nearest = squared_distances.min(axis=1)
        positive = nearest[np.isfinite(nearest) & (nearest > 1e-10)]
        distance_scale = float(np.median(positive)) if len(positive) else 1.0
        novelty_scale = float(np.quantile(positive, 0.95)) if len(positive) else distance_scale
        return cls(
            centers=centers,
            distance_scale=max(distance_scale, 1e-6),
            novelty_scale=max(novelty_scale, 1e-6),
            temperature=float(temperature),
            seed=int(seed),
            fitted_rows=int(len(values)),
        )

    def predict_proba(self, latents: np.ndarray) -> np.ndarray:
        values = np.asarray(latents, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.centers.shape[1]:
            raise ValueError(
                f"latents must have shape [N, {self.centers.shape[1]}] for this codebook"
            )
        if not np.isfinite(values).all():
            raise ValueError("latent rows must be finite")
        squared_distances = ((values[:, None, :] - self.centers[None, :, :]) ** 2).mean(axis=-1)
        logits = -squared_distances / (self.distance_scale * self.temperature)
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(np.clip(logits, -60.0, 0.0))
        probabilities = weights / weights.sum(axis=1, keepdims=True).clip(min=1e-12)
        return probabilities.astype(np.float32)

    def diagnostics(self, latents: np.ndarray) -> dict[str, float | int]:
        probabilities = self.predict_proba(latents).astype(float)
        marginal = probabilities.mean(axis=0)
        entropy = -float(np.sum(marginal * np.log(np.clip(marginal, 1e-12, None))))
        return {
            "fitted_rows": self.fitted_rows,
            "regime_count": int(self.centers.shape[0]),
            "mean_assignment_confidence": float(probabilities.max(axis=1).mean()),
            "maximum_marginal_regime_share": float(marginal.max()),
            "effective_regime_count": float(np.exp(entropy)),
            "distance_scale": float(self.distance_scale),
            "novelty_scale": float(self.novelty_scale),
            "temperature": float(self.temperature),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "centers": self.centers.tolist(),
            "distance_scale": float(self.distance_scale),
            "novelty_scale": float(self.novelty_scale),
            "temperature": float(self.temperature),
            "seed": int(self.seed),
            "fitted_rows": int(self.fitted_rows),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CausalRegimeCodebook":
        return cls(
            centers=np.asarray(payload["centers"], dtype=float),
            distance_scale=float(payload["distance_scale"]),
            novelty_scale=float(payload.get("novelty_scale", payload["distance_scale"])),
            temperature=float(payload["temperature"]),
            seed=int(payload["seed"]),
            fitted_rows=int(payload["fitted_rows"]),
        )

    def novelty_score(self, latents: np.ndarray) -> np.ndarray:
        """Return nearest-prototype distance relative to the train-only p95."""

        values = np.asarray(latents, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.centers.shape[1]:
            raise ValueError(
                f"latents must have shape [N, {self.centers.shape[1]}] for this codebook"
            )
        squared_distances = ((values[:, None, :] - self.centers[None, :, :]) ** 2).mean(axis=-1)
        return (squared_distances.min(axis=1) / max(self.novelty_scale, 1e-6)).astype(np.float32)


__all__ = ["CausalRegimeCodebook", "causal_smooth_regime_probabilities"]
