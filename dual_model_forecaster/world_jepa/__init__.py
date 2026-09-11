"""Five-world JEPA representation learning and attention forecasting."""

from .config import WORLD_NAMES, WorldJEPAConfig, WorldSpec, load_world_jepa_config
from .continuous_horizon import ContinuousHorizonEmbedding
from .pipeline import RuntimeOverrides, build_dense_return_targets, run_world_jepa_pipeline
from .regimes import CausalRegimeCodebook, causal_smooth_regime_probabilities
from .router import ContextRelevantWorldRouter, WorldAttentionOutput
from .world_encoder import EMAMomentumSchedule, WorldJEPAEncoder, WorldJEPAOutput

__all__ = [
    "WORLD_NAMES",
    "WorldSpec",
    "WorldJEPAConfig",
    "load_world_jepa_config",
    "ContinuousHorizonEmbedding",
    "WorldJEPAEncoder",
    "WorldJEPAOutput",
    "EMAMomentumSchedule",
    "ContextRelevantWorldRouter",
    "WorldAttentionOutput",
    "CausalRegimeCodebook",
    "causal_smooth_regime_probabilities",
    "RuntimeOverrides",
    "build_dense_return_targets",
    "run_world_jepa_pipeline",
]
