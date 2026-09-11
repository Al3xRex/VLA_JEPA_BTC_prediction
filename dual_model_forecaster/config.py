from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_name": "dual_model_multiscale",
    "seed": 7,
    "device": "cpu",
    "paths": {
        "category_db_path": "database/data_classes.duckdb",
        "price_db_path": "database/ohlcv.duckdb",
        "price_table_name": "ohlcv",
        "reports_dir": "reports",
        "results_dir": "results",
        "features_dir": "features",
        "models_dir": "models",
        "logs_dir": "logs",
        "data_dir": "data",
    },
    "data": {
        "symbol": "BTCUSDT",
        "interval": "1d",
        "bucket_tables": {
            "structure": "structure_set",
            "environment": "environment_set",
            "edges": "edges_set",
            "movement": "movement_set",
            "liquidation": "liquidation_set",
        },
        "target_horizons": [1, 3, 7, 14, 21],
        "return_type": "log",
        "freshness_half_life_days": {
            "structure": 90.0,
            "environment": 90.0,
            "edges": 14.0,
            "movement": 10.0,
            "liquidation": 7.0,
        },
    },
    "splits": {
        "min_train_days": 1_460,
        "outer_test_days": 180,
        "outer_step_days": 180,
        "max_outer_folds": 4,
        "outer_fold_index": -1,
        "inner_min_train_days": 1_000,
        "inner_segment_days": 90,
        "specialist_validation_days": 120,
    },
    "specialists": {
        "communication_channels": [
            "hazard",
            "freshness",
            "validity",
            "internal_disagreement",
            "scale_entropy",
            "state_velocity",
            "state_acceleration",
        ],
        "grouping": {
            "minimum_groups": 1,
            "buckets": {
                "structure": {
                    "scale_groups": [
                        {"name": "medium", "min_lookback": 72, "max_lookback": 144},
                        {"name": "long", "min_lookback": 145, "max_lookback": 224},
                        {"name": "ultra", "min_lookback": 225, "max_lookback": 512},
                    ],
                },
                "environment": {
                    "scale_groups": [
                        {"name": "medium_long", "min_lookback": 96, "max_lookback": 192},
                        {"name": "long", "min_lookback": 193, "max_lookback": 256},
                        {"name": "ultra", "min_lookback": 257, "max_lookback": 512},
                    ],
                },
                "edges": {
                    "scale_groups": [
                        {"name": "short", "min_lookback": 8, "max_lookback": 36},
                        {"name": "medium", "min_lookback": 37, "max_lookback": 96},
                    ],
                },
                "movement": {
                    "scale_groups": [
                        {"name": "short", "min_lookback": 8, "max_lookback": 36},
                        {"name": "medium", "min_lookback": 37, "max_lookback": 96},
                    ],
                },
                "liquidation": {
                    "scale_groups": [
                        {"name": "short", "min_lookback": 8, "max_lookback": 36},
                        {"name": "medium", "min_lookback": 37, "max_lookback": 75},
                        {"name": "attention", "min_lookback": 76, "max_lookback": 128},
                    ],
                },
            },
        },
        "training": {
            "batch_size": 128,
            "max_epochs": 40,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "patience": 6,
        },
        "buckets": {
            "structure": {
                "post_smooth_alpha": 0.15,
                "candidates": [
                    {"name": "linear_lb96", "architecture": "linear", "lookback": 96},
                    {"name": "linear_lb144", "architecture": "linear", "lookback": 144},
                    {
                        "name": "mlp_lb144",
                        "architecture": "mlp",
                        "lookback": 144,
                        "hidden_dim": 96,
                        "dropout": 0.10,
                    },
                    {
                        "name": "gru_lb192",
                        "architecture": "gru",
                        "lookback": 192,
                        "hidden_dim": 64,
                        "num_layers": 1,
                        "dropout": 0.10,
                    },
                    {
                        "name": "tcn_lb192",
                        "architecture": "tcn",
                        "lookback": 192,
                        "hidden_dim": 48,
                        "kernel_size": 3,
                        "num_layers": 3,
                        "dropout": 0.10,
                    },
                    {
                        "name": "patchtst_lb256",
                        "architecture": "patchtst_like",
                        "lookback": 256,
                        "patch_length": 16,
                        "hidden_dim": 64,
                        "num_layers": 2,
                        "num_heads": 4,
                        "dropout": 0.10,
                    },
                ],
            },
            "environment": {
                "post_smooth_alpha": 0.18,
                "candidates": [
                    {"name": "linear_lb128", "architecture": "linear", "lookback": 128},
                    {"name": "linear_lb192", "architecture": "linear", "lookback": 192},
                    {
                        "name": "mlp_lb160",
                        "architecture": "mlp",
                        "lookback": 160,
                        "hidden_dim": 96,
                        "dropout": 0.10,
                    },
                    {
                        "name": "gru_lb224",
                        "architecture": "gru",
                        "lookback": 224,
                        "hidden_dim": 64,
                        "num_layers": 1,
                        "dropout": 0.10,
                    },
                    {
                        "name": "tcn_lb224",
                        "architecture": "tcn",
                        "lookback": 224,
                        "hidden_dim": 48,
                        "kernel_size": 3,
                        "num_layers": 3,
                        "dropout": 0.10,
                    },
                    {
                        "name": "patchtst_lb256",
                        "architecture": "patchtst_like",
                        "lookback": 256,
                        "patch_length": 16,
                        "hidden_dim": 64,
                        "num_layers": 2,
                        "num_heads": 4,
                        "dropout": 0.10,
                    },
                ],
            },
            "edges": {
                "post_smooth_alpha": 0.35,
                "candidates": [
                    {"name": "linear_lb24", "architecture": "linear", "lookback": 24},
                    {"name": "linear_lb45", "architecture": "linear", "lookback": 45},
                    {
                        "name": "mlp_lb36",
                        "architecture": "mlp",
                        "lookback": 36,
                        "hidden_dim": 64,
                        "dropout": 0.10,
                    },
                    {
                        "name": "mlp_lb60",
                        "architecture": "mlp",
                        "lookback": 60,
                        "hidden_dim": 80,
                        "dropout": 0.10,
                    },
                    {
                        "name": "gru_lb75",
                        "architecture": "gru",
                        "lookback": 75,
                        "hidden_dim": 48,
                        "num_layers": 1,
                        "dropout": 0.10,
                    },
                    {
                        "name": "tcn_lb75",
                        "architecture": "tcn",
                        "lookback": 75,
                        "hidden_dim": 40,
                        "kernel_size": 3,
                        "num_layers": 3,
                        "dropout": 0.10,
                    },
                    {
                        "name": "patchtst_lb96",
                        "architecture": "patchtst_like",
                        "lookback": 96,
                        "patch_length": 8,
                        "hidden_dim": 48,
                        "num_layers": 2,
                        "num_heads": 4,
                        "dropout": 0.10,
                    },
                ],
            },
            "movement": {
                "post_smooth_alpha": 0.30,
                "candidates": [
                    {"name": "linear_lb24", "architecture": "linear", "lookback": 24},
                    {"name": "linear_lb45", "architecture": "linear", "lookback": 45},
                    {
                        "name": "mlp_lb36",
                        "architecture": "mlp",
                        "lookback": 36,
                        "hidden_dim": 64,
                        "dropout": 0.10,
                    },
                    {
                        "name": "mlp_lb60",
                        "architecture": "mlp",
                        "lookback": 60,
                        "hidden_dim": 80,
                        "dropout": 0.10,
                    },
                    {
                        "name": "gru_lb75",
                        "architecture": "gru",
                        "lookback": 75,
                        "hidden_dim": 48,
                        "num_layers": 1,
                        "dropout": 0.10,
                    },
                    {
                        "name": "tcn_lb75",
                        "architecture": "tcn",
                        "lookback": 75,
                        "hidden_dim": 40,
                        "kernel_size": 3,
                        "num_layers": 3,
                        "dropout": 0.10,
                    },
                    {
                        "name": "patchtst_lb96",
                        "architecture": "patchtst_like",
                        "lookback": 96,
                        "patch_length": 8,
                        "hidden_dim": 48,
                        "num_layers": 2,
                        "num_heads": 4,
                        "dropout": 0.10,
                    },
                ],
            },
            "liquidation": {
                "post_smooth_alpha": 0.38,
                "candidates": [
                    {"name": "linear_lb24", "architecture": "linear", "lookback": 24},
                    {
                        "name": "mlp_lb36",
                        "architecture": "mlp",
                        "lookback": 36,
                        "hidden_dim": 64,
                        "dropout": 0.10,
                    },
                    {
                        "name": "gru_lb75",
                        "architecture": "gru",
                        "lookback": 75,
                        "hidden_dim": 48,
                        "num_layers": 1,
                        "dropout": 0.10,
                    },
                    {
                        "name": "patchtst_lb96",
                        "architecture": "patchtst_like",
                        "lookback": 96,
                        "patch_length": 8,
                        "hidden_dim": 48,
                        "num_layers": 2,
                        "num_heads": 4,
                        "dropout": 0.10,
                    },
                    {
                        "name": "category_attention_lb96",
                        "architecture": "category_attention",
                        "lookback": 96,
                        "hidden_dim": 64,
                        "num_heads": 4,
                        "dropout": 0.10,
                    },
                ],
            },
        },
    },
    "fusion": {
        "quantiles": [0.05, 0.25, 0.50, 0.75, 0.95],
        "state_window": 64,
        "variants": ["A", "C", "D", "E", "F", "G"],
        "horizon_sets": {
            "set_a": [1, 3, 7],
            "set_b": [3, 7, 14, 21],
            "set_c": [1, 3, 7, 14, 21],
        },
        "communication_variants": [
            [],
            ["hazard", "freshness", "validity"],
            ["internal_disagreement", "scale_entropy", "state_velocity", "state_acceleration"],
            [
                "hazard",
                "freshness",
                "validity",
                "internal_disagreement",
                "scale_entropy",
                "state_velocity",
                "state_acceleration",
            ],
        ],
        "output_styles": ["direct_quantiles", "structured_cwt"],
        "calibration_methods": ["additive_quantile", "conformal", "disagreement_conformal"],
        "training": {
            "batch_size": 128,
            "max_epochs": 50,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "patience": 7,
            "monotonicity_penalty": 2.0,
            "lower_tail_weight": 1.25,
            "horizon_weights": {"1": 1.35, "3": 1.25, "7": 1.0, "14": 0.90, "15": 0.90, "21": 0.85},
            "coverage_slack_penalty": 0.10,
            "width_floor_penalty": 0.05,
        },
        "architectures": [
            {"name": "quantile_mlp", "architecture": "quantile_mlp", "hidden_dim": 128, "dropout": 0.10},
            {
                "name": "horizon_gated_mlp",
                "architecture": "horizon_gated_mlp",
                "hidden_dim": 128,
                "dropout": 0.10,
                "horizon_embedding_dim": 24,
            },
            {"name": "dlinear", "architecture": "dlinear", "hidden_dim": 96, "moving_average": 7},
            {"name": "tide_like", "architecture": "tide_like", "hidden_dim": 128, "dropout": 0.10},
            {"name": "nhits_like", "architecture": "nhits_like", "hidden_dim": 128, "dropout": 0.10},
        ],
    },
}

REPLACE_DICT_KEYS = {
    "horizon_sets",
    "freshness_half_life_days",
}


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if key in REPLACE_DICT_KEYS and isinstance(value, dict):
            merged[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    resolved = copy.deepcopy(DEFAULT_CONFIG)
    if config_path is None:
        return finalize_config(resolved)

    path = Path(config_path)
    payload = json.loads(path.read_text())
    return finalize_config(_deep_merge(resolved, payload))


def finalize_config(config: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    fusion_cfg = resolved.get("fusion", {})
    data_cfg = resolved.get("data", {})

    if "meta_variants" in fusion_cfg and "communication_variants" not in fusion_cfg:
        fusion_cfg["communication_variants"] = fusion_cfg["meta_variants"]

    if "horizon_sets" not in fusion_cfg or not fusion_cfg["horizon_sets"]:
        target_horizons = list(data_cfg.get("target_horizons", [1, 3, 7]))
        fusion_cfg["horizon_sets"] = {"default": target_horizons}

    all_horizons = sorted({int(h) for horizons in fusion_cfg["horizon_sets"].values() for h in horizons})
    data_cfg["target_horizons"] = all_horizons

    resolved["fusion"] = fusion_cfg
    resolved["data"] = data_cfg
    return resolved
