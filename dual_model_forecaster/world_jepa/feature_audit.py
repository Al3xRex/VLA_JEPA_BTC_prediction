from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


WORLD_NAMES: tuple[str, ...] = (
    "structure",
    "environment",
    "edges",
    "movement",
    "liquidation",
)


@dataclass(frozen=True)
class WorldFeatureAuditConfig:
    """Configuration for causal feature audit and selection.

    Sanitation, redundancy detection, and temporal utility are fit on the
    training segment.  The selection segment may choose candidates and the
    number of features, while the test segment is opened only after the
    selected column list is frozen.
    """

    target_horizon: int = 7
    world_target_horizons: tuple[int, ...] = (1, 3, 7, 15)
    downstream_utility_weight: float = 0.60
    future_world_predictability_weight: float = 0.40
    require_target_horizon_in_name: bool = True
    target_leak_correlation_threshold: float = 0.9995
    target_leak_max_lead_rows: int | None = None
    feature_budget: int = 12
    per_world_feature_budget: Mapping[str, int] = field(default_factory=dict)
    minimum_world_features: int = 4
    per_world_minimum_features: Mapping[str, int] = field(default_factory=dict)
    train_fraction: float = 0.60
    selection_fraction: float = 0.20
    selection_rows: int | None = None
    test_rows: int | None = None
    minimum_train_rows: int | None = None
    purge_gap_rows: int = 7
    n_temporal_folds: int = 4
    min_fold_train_rows: int = 120
    min_fold_validation_rows: int = 30
    min_aligned_rows: int = 180
    min_non_null_train: int = 24
    min_numeric_ratio: float = 0.90
    max_missing_ratio: float = 0.40
    max_infinite_ratio: float = 0.02
    constant_tolerance: float = 1e-12
    redundancy_threshold: float = 0.95
    ridge_alpha: float = 1.0
    stability_penalty: float = 0.50
    sign_instability_penalty: float = 0.02
    min_utility_folds: int = 2
    min_train_utility: float = -0.05
    min_selection_utility: float = -0.02
    selection_weight: float = 0.35
    selection_complexity_penalty: float = 0.002
    candidate_pool_multiplier: int = 3
    allow_empty_selection: bool = True
    require_all_worlds: bool = True
    forbidden_feature_name_tokens: tuple[str, ...] = (
        "target",
        "future_return",
        "forward_return",
        "realized_return",
        "actual_return",
        "label",
    )

    def __post_init__(self) -> None:
        if int(self.target_horizon) < 1:
            raise ValueError("target_horizon must be positive.")
        if not self.world_target_horizons or any(int(value) < 1 for value in self.world_target_horizons):
            raise ValueError("world_target_horizons must contain positive horizons.")
        if len(set(int(value) for value in self.world_target_horizons)) != len(self.world_target_horizons):
            raise ValueError("world_target_horizons must be unique.")
        if float(self.downstream_utility_weight) < 0.0 or float(self.future_world_predictability_weight) < 0.0:
            raise ValueError("Utility weights cannot be negative.")
        if float(self.downstream_utility_weight) + float(self.future_world_predictability_weight) <= 0.0:
            raise ValueError("At least one utility weight must be positive.")
        if not 0.0 < float(self.target_leak_correlation_threshold) <= 1.0:
            raise ValueError("target_leak_correlation_threshold must be in (0, 1].")
        if self.target_leak_max_lead_rows is not None and int(self.target_leak_max_lead_rows) < 0:
            raise ValueError("target_leak_max_lead_rows cannot be negative.")
        if int(self.feature_budget) < 1:
            raise ValueError("feature_budget must be positive.")
        if int(self.minimum_world_features) < 0:
            raise ValueError("minimum_world_features cannot be negative.")
        if any(int(value) < 0 for value in self.per_world_minimum_features.values()):
            raise ValueError("per_world_minimum_features values cannot be negative.")
        if not 0.0 < float(self.train_fraction) < 1.0:
            raise ValueError("train_fraction must be between zero and one.")
        if not 0.0 < float(self.selection_fraction) < 1.0:
            raise ValueError("selection_fraction must be between zero and one.")
        if float(self.train_fraction) + float(self.selection_fraction) >= 1.0:
            raise ValueError("train_fraction + selection_fraction must be below one.")
        if (self.selection_rows is None) != (self.test_rows is None):
            raise ValueError("selection_rows and test_rows must be provided together.")
        if self.selection_rows is not None and int(self.selection_rows) < 1:
            raise ValueError("selection_rows must be positive when provided.")
        if self.test_rows is not None and int(self.test_rows) < 1:
            raise ValueError("test_rows must be positive when provided.")
        if self.minimum_train_rows is not None and int(self.minimum_train_rows) < 1:
            raise ValueError("minimum_train_rows must be positive when provided.")
        if int(self.n_temporal_folds) < 1:
            raise ValueError("n_temporal_folds must be positive.")
        if not 0.0 <= float(self.max_missing_ratio) < 1.0:
            raise ValueError("max_missing_ratio must be in [0, 1).")
        if not 0.0 <= float(self.max_infinite_ratio) < 1.0:
            raise ValueError("max_infinite_ratio must be in [0, 1).")
        if not 0.0 < float(self.redundancy_threshold) <= 1.0:
            raise ValueError("redundancy_threshold must be in (0, 1].")

    def budget_for(self, world_name: str) -> int:
        return max(1, int(self.per_world_feature_budget.get(world_name, self.feature_budget)))

    def minimum_for(self, world_name: str) -> int:
        requested = int(
            self.per_world_minimum_features.get(world_name, self.minimum_world_features)
        )
        return min(self.budget_for(world_name), max(0, requested))

    @property
    def effective_purge_gap(self) -> int:
        return max(
            int(self.purge_gap_rows),
            int(self.target_horizon),
            max(int(value) for value in self.world_target_horizons),
        )

    @property
    def normalized_utility_weights(self) -> tuple[float, float]:
        total = float(self.downstream_utility_weight) + float(
            self.future_world_predictability_weight
        )
        return (
            float(self.downstream_utility_weight) / total,
            float(self.future_world_predictability_weight) / total,
        )

    @property
    def effective_target_leak_max_lead(self) -> int:
        if self.target_leak_max_lead_rows is not None:
            return int(self.target_leak_max_lead_rows)
        return max(
            int(self.target_horizon),
            max(int(value) for value in self.world_target_horizons),
        )


@dataclass(frozen=True)
class _PurgedFold:
    fold: int
    train_positions: np.ndarray
    validation_positions: np.ndarray
    purge_positions: np.ndarray


def _index_value(value: Any) -> str | int | float | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return _index_value(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if value is pd.NA:
        return None
    return value


def _pandas_content_hash(value: pd.DataFrame | pd.Series) -> str:
    digest = hashlib.sha256()
    if isinstance(value, pd.DataFrame):
        digest.update(json.dumps([str(column) for column in value.columns]).encode("utf-8"))
    else:
        digest.update(str(value.name).encode("utf-8"))
    try:
        hashes = pd.util.hash_pandas_object(value, index=True).to_numpy(dtype=np.uint64)
    except TypeError:
        hashes = pd.util.hash_pandas_object(value.astype(str), index=True).to_numpy(dtype=np.uint64)
    digest.update(hashes.tobytes())
    return digest.hexdigest()


def _file_content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_ordered_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "timestamp" in out.columns and not isinstance(out.index, pd.DatetimeIndex):
        timestamp = pd.to_datetime(out.pop("timestamp"), errors="coerce")
        out.index = timestamp
    if out.index.has_duplicates:
        out = out.loc[~out.index.duplicated(keep="last")]
    try:
        out = out.sort_index()
    except TypeError as exc:
        raise ValueError("Feature index must be sortable for chronological auditing.") from exc
    out.columns = [str(column) for column in out.columns]
    if out.columns.duplicated().any():
        duplicates = sorted(set(out.columns[out.columns.duplicated()].tolist()))
        raise ValueError(f"Duplicate feature names are not supported: {duplicates}")
    return out


def _as_ordered_target(target: pd.Series) -> pd.Series:
    out = pd.Series(target).copy()
    if out.index.has_duplicates:
        out = out.loc[~out.index.duplicated(keep="last")]
    try:
        out = out.sort_index()
    except TypeError as exc:
        raise ValueError("Target index must be sortable for chronological auditing.") from exc
    out = pd.to_numeric(out, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out.name = str(target.name or "downstream_target")
    return out


def _validate_target_horizon(
    target: pd.Series,
    config: WorldFeatureAuditConfig,
) -> dict[str, Any]:
    target_name = str(target.name or "")
    matches = {
        int(value)
        for value in re.findall(r"(?:target|return).*?(\d+)d", target_name.lower())
    }
    if not matches:
        if config.require_target_horizon_in_name:
            raise ValueError(
                f"Target name {target_name!r} does not encode its label horizon; "
                f"expected a name such as 'target_{config.target_horizon}d'."
            )
        return {
            "target_name": target_name,
            "configured_horizon_rows": int(config.target_horizon),
            "parsed_horizon_rows": None,
            "validated": False,
            "validation_bypassed_by_config": True,
        }
    if len(matches) != 1:
        raise ValueError(f"Target name {target_name!r} encodes ambiguous horizons: {sorted(matches)}")
    parsed_horizon = next(iter(matches))
    if parsed_horizon != int(config.target_horizon):
        raise ValueError(
            f"Target name {target_name!r} encodes horizon {parsed_horizon}d but "
            f"config.target_horizon={config.target_horizon}; refusing an under-purged audit."
        )
    return {
        "target_name": target_name,
        "configured_horizon_rows": int(config.target_horizon),
        "parsed_horizon_rows": int(parsed_horizon),
        "validated": True,
        "validation_bypassed_by_config": False,
    }


def _align_frame_target(frame: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    ordered_frame = _as_ordered_frame(frame)
    ordered_target = _as_ordered_target(target)
    common_index = ordered_frame.index.intersection(ordered_target.dropna().index).sort_values()
    if common_index.empty:
        raise ValueError("Features and target have no finite, aligned timestamps.")
    return ordered_frame.reindex(common_index), ordered_target.reindex(common_index)


def _positions(start: int, stop: int) -> np.ndarray:
    return np.arange(max(0, int(start)), max(0, int(stop)), dtype=int)


def _segment(index: pd.Index, positions: np.ndarray) -> dict[str, Any]:
    if len(positions) == 0:
        return {
            "rows": 0,
            "start_position": None,
            "end_position": None,
            "start_timestamp": None,
            "end_timestamp": None,
        }
    start = int(positions[0])
    end = int(positions[-1])
    return {
        "rows": int(len(positions)),
        "start_position": start,
        "end_position": end,
        "start_timestamp": _index_value(index[start]),
        "end_timestamp": _index_value(index[end]),
    }


def _outer_split(index: pd.Index, config: WorldFeatureAuditConfig) -> dict[str, Any]:
    n_rows = len(index)
    purge = config.effective_purge_gap
    if config.selection_rows is not None and config.test_rows is not None:
        test_start = n_rows - int(config.test_rows)
        selection_stop = test_start - purge
        selection_start = selection_stop - int(config.selection_rows)
        train_stop = selection_start - purge
        split_mode = "fixed_selection_test_rows"
    else:
        train_stop = int(math.floor(n_rows * float(config.train_fraction)))
        selection_stop = int(
            math.floor(n_rows * (float(config.train_fraction) + float(config.selection_fraction)))
        )
        selection_start = train_stop + purge
        test_start = selection_stop + purge
        split_mode = "fractional"
    train = _positions(0, train_stop)
    train_selection_purge = _positions(train_stop, selection_start)
    selection = _positions(selection_start, selection_stop)
    selection_test_purge = _positions(selection_stop, test_start)
    test = _positions(test_start, n_rows)
    if min(len(train), len(selection), len(test)) == 0:
        raise ValueError(
            "Not enough rows for non-empty train/selection/test segments after purging: "
            f"rows={n_rows}, purge={purge}."
        )
    if config.minimum_train_rows is not None and len(train) < int(config.minimum_train_rows):
        raise ValueError(
            f"Training segment has {len(train)} rows, below minimum_train_rows="
            f"{config.minimum_train_rows}."
        )
    return {
        "aligned_rows": int(n_rows),
        "split_mode": split_mode,
        "effective_purge_gap_rows": int(purge),
        "train": _segment(index, train),
        "train_selection_purge": _segment(index, train_selection_purge),
        "selection": _segment(index, selection),
        "selection_test_purge": _segment(index, selection_test_purge),
        "test": _segment(index, test),
        "_positions": {
            "train": train,
            "train_selection_purge": train_selection_purge,
            "selection": selection,
            "selection_test_purge": selection_test_purge,
            "test": test,
        },
    }


def _purged_folds(train_rows: int, config: WorldFeatureAuditConfig) -> list[_PurgedFold]:
    purge = config.effective_purge_gap
    min_train = int(config.min_fold_train_rows)
    available = train_rows - min_train - purge
    minimum_validation = int(config.min_fold_validation_rows)
    if available < minimum_validation:
        return []
    desired_folds = int(config.n_temporal_folds)
    fold_count = min(desired_folds, max(1, available // max(minimum_validation, 1)))
    validation_rows = available // max(fold_count, 1)
    folds: list[_PurgedFold] = []
    for fold_number in range(fold_count):
        validation_start = min_train + purge + fold_number * validation_rows
        validation_stop = (
            train_rows
            if fold_number == fold_count - 1
            else min(train_rows, validation_start + validation_rows)
        )
        train_stop = validation_start - purge
        if train_stop < 8 or validation_stop - validation_start < 4:
            continue
        folds.append(
            _PurgedFold(
                fold=fold_number,
                train_positions=_positions(0, train_stop),
                validation_positions=_positions(validation_start, validation_stop),
                purge_positions=_positions(train_stop, validation_start),
            )
        )
    return folds


def _target_content_leak_check(
    feature: pd.Series,
    target: pd.Series,
    train_positions: np.ndarray,
    config: WorldFeatureAuditConfig,
) -> dict[str, Any]:
    feature_numeric = pd.to_numeric(feature, errors="coerce").replace([np.inf, -np.inf], np.nan)
    correlations: list[dict[str, Any]] = []
    suspicious_leads: list[int] = []
    minimum_overlap = max(12, int(config.min_non_null_train))
    final_train_origin = int(train_positions[-1])
    for lead in range(config.effective_target_leak_max_lead + 1):
        future_target = target.shift(-lead)
        eligible_positions = train_positions[train_positions + lead <= final_train_origin]
        feature_train = feature_numeric.iloc[eligible_positions]
        target_train = future_target.iloc[eligible_positions]
        valid = feature_train.notna() & target_train.notna()
        overlap = int(valid.sum())
        correlation = float("nan")
        if overlap >= minimum_overlap:
            feature_values = feature_train.loc[valid].to_numpy(dtype=float)
            target_values = target_train.loc[valid].to_numpy(dtype=float)
            if float(np.std(feature_values)) > 0.0 and float(np.std(target_values)) > 0.0:
                correlation = float(np.corrcoef(feature_values, target_values)[0, 1])
                if abs(correlation) >= float(config.target_leak_correlation_threshold):
                    suspicious_leads.append(int(lead))
        correlations.append(
            {
                "future_target_lead_rows": int(lead),
                "eligible_train_origin_rows": int(len(eligible_positions)),
                "overlap_rows": overlap,
                "correlation": correlation,
            }
        )
    finite_correlations = [abs(float(row["correlation"])) for row in correlations if math.isfinite(float(row["correlation"]))]
    return {
        "max_abs_future_target_correlation_train": max(finite_correlations) if finite_correlations else None,
        "suspicious_future_target_lead_rows": suspicious_leads,
        "lead_correlations": correlations,
        "fit_segment": "train_only",
        "shifted_target_origins_restricted_to_train": True,
    }


def _sanitation_audit(
    frame: pd.DataFrame,
    target: pd.Series,
    train_positions: np.ndarray,
    config: WorldFeatureAuditConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, list[str]]]:
    train = frame.iloc[train_positions]
    numeric_columns: dict[str, pd.Series] = {}
    report: list[dict[str, Any]] = []
    rejected: dict[str, list[str]] = {}
    forbidden_tokens = tuple(token.lower() for token in config.forbidden_feature_name_tokens)

    for column in frame.columns:
        raw_all = frame[column]
        raw_train = train[column]
        numeric_all = pd.to_numeric(raw_all, errors="coerce")
        numeric_train = pd.to_numeric(raw_train, errors="coerce")
        infinite_train = np.isinf(numeric_train.to_numpy(dtype=float, na_value=np.nan))
        infinite_all = np.isinf(numeric_all.to_numpy(dtype=float, na_value=np.nan))
        finite_train = numeric_train.replace([np.inf, -np.inf], np.nan)
        finite_all = numeric_all.replace([np.inf, -np.inf], np.nan)
        original_non_null = int(raw_train.notna().sum())
        numeric_non_null_before_inf = int(numeric_train.notna().sum())
        finite_non_null = int(finite_train.notna().sum())
        numeric_ratio = float(numeric_non_null_before_inf / max(original_non_null, 1))
        missing_ratio = float(finite_train.isna().mean())
        infinite_ratio = float(infinite_train.mean()) if len(infinite_train) else 0.0
        unique_values = int(finite_train.nunique(dropna=True))
        if finite_non_null:
            spread = float(finite_train.max() - finite_train.min())
        else:
            spread = float("nan")

        reasons: list[str] = []
        warnings: list[str] = []
        target_leak_check = _target_content_leak_check(
            finite_all,
            target,
            train_positions,
            config,
        )
        lower_name = column.lower()
        if any(token in lower_name for token in forbidden_tokens):
            reasons.append("forbidden_target_like_name")
        for suspicious_lead in target_leak_check["suspicious_future_target_lead_rows"]:
            if int(suspicious_lead) == 0:
                reasons.append("near_exact_target_content_on_training_segment")
            else:
                reasons.append(f"suspicious_future_target_lead:{int(suspicious_lead)}")
        if original_non_null and numeric_ratio < float(config.min_numeric_ratio):
            reasons.append("insufficient_numeric_values")
        if infinite_ratio > float(config.max_infinite_ratio):
            reasons.append("excessive_infinite_values")
        elif int(infinite_train.sum()) > 0:
            warnings.append("infinite_values_replaced_with_missing")
        if missing_ratio > float(config.max_missing_ratio):
            reasons.append("excessive_missing_values")
        if finite_non_null < min(int(config.min_non_null_train), max(3, len(train_positions) // 4)):
            reasons.append("insufficient_finite_training_values")
        if unique_values <= 1 or (math.isfinite(spread) and abs(spread) <= float(config.constant_tolerance)):
            reasons.append("constant_on_training_segment")

        status = "rejected" if reasons else "kept"
        if reasons:
            rejected[column] = list(dict.fromkeys(reasons))
        else:
            numeric_columns[column] = finite_all.astype(float)
        report.append(
            {
                "feature": column,
                "status": status,
                "reasons": list(dict.fromkeys(reasons)),
                "warnings": warnings,
                "train_rows": int(len(train_positions)),
                "train_finite_rows": finite_non_null,
                "train_missing_ratio": missing_ratio,
                "train_infinite_count": int(infinite_train.sum()),
                "train_infinite_ratio": infinite_ratio,
                "all_infinite_count": int(infinite_all.sum()),
                "train_numeric_ratio": numeric_ratio,
                "train_unique_values": unique_values,
                "train_spread": spread,
                "target_content_leak_check": target_leak_check,
                "decision_fit_segment": "train_only",
            }
        )
    numeric_frame = pd.DataFrame(numeric_columns, index=frame.index)
    return numeric_frame, report, rejected


def _prepare_matrix(
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    medians = train_frame.median(axis=0, skipna=True).fillna(0.0)
    train_values = train_frame.fillna(medians).to_numpy(dtype=float)
    evaluation_values = evaluation_frame.fillna(medians).to_numpy(dtype=float)
    scaler = StandardScaler()
    return scaler.fit_transform(train_values), scaler.transform(evaluation_values)


def _metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    residual = y - pred
    mse = float(np.mean(np.square(residual)))
    mae = float(np.mean(np.abs(residual)))
    variance = float(np.mean(np.square(y - float(np.mean(y)))))
    r2 = float(1.0 - mse / variance) if variance > 1e-15 else float("nan")
    if len(y) > 1 and float(np.std(y)) > 0.0 and float(np.std(pred)) > 0.0:
        correlation = float(np.corrcoef(y, pred)[0, 1])
    else:
        correlation = float("nan")
    directional_accuracy = float(np.mean(np.sign(y) == np.sign(pred)))
    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "correlation": correlation,
        "directional_accuracy": directional_accuracy,
    }


def _fit_and_score(
    frame: pd.DataFrame,
    target: pd.Series,
    train_positions: np.ndarray,
    evaluation_positions: np.ndarray,
    features: Sequence[str],
    alpha: float,
) -> dict[str, Any]:
    y_train = target.iloc[train_positions].to_numpy(dtype=float)
    y_evaluation = target.iloc[evaluation_positions].to_numpy(dtype=float)
    baseline_prediction = np.full(len(evaluation_positions), float(np.mean(y_train)), dtype=float)
    baseline = _metrics(y_evaluation, baseline_prediction)
    if not features:
        model_metrics = dict(baseline)
        prediction = baseline_prediction
        coefficients: dict[str, float] = {}
    else:
        train_x, evaluation_x = _prepare_matrix(
            frame.iloc[train_positions].loc[:, list(features)],
            frame.iloc[evaluation_positions].loc[:, list(features)],
        )
        model = Ridge(alpha=float(alpha))
        model.fit(train_x, y_train)
        prediction = np.asarray(model.predict(evaluation_x), dtype=float)
        model_metrics = _metrics(y_evaluation, prediction)
        coefficients = {
            feature: float(coefficient)
            for feature, coefficient in zip(features, np.ravel(model.coef_), strict=True)
        }
    baseline_mse = float(baseline["mse"])
    utility = (
        float(1.0 - float(model_metrics["mse"]) / baseline_mse)
        if baseline_mse > 1e-15
        else 0.0
    )
    return {
        "features": list(features),
        "rows": int(len(evaluation_positions)),
        "baseline": baseline,
        "model": model_metrics,
        "relative_mse_utility": utility,
        "coefficients": coefficients,
        "prediction_mean": float(np.mean(prediction)) if len(prediction) else float("nan"),
    }


def _future_world_feature_score(
    frame: pd.DataFrame,
    feature: str,
    train_positions: np.ndarray,
    evaluation_positions: np.ndarray,
    config: WorldFeatureAuditConfig,
) -> dict[str, Any]:
    horizon_rows: list[dict[str, Any]] = []
    if len(train_positions) == 0 or len(evaluation_positions) == 0:
        return {"mean_relative_mse_utility": 0.0, "horizons": horizon_rows}
    final_train_origin = int(train_positions[-1])
    final_evaluation_origin = int(evaluation_positions[-1])
    for horizon in sorted(int(value) for value in config.world_target_horizons):
        future_target = frame[feature].shift(-horizon).rename(
            f"{feature}__future_{horizon}d"
        )
        fit_positions = train_positions[train_positions + horizon <= final_train_origin]
        score_positions = evaluation_positions[
            evaluation_positions + horizon <= final_evaluation_origin
        ]
        fit_positions = fit_positions[
            future_target.iloc[fit_positions].notna().to_numpy(dtype=bool)
        ]
        score_positions = score_positions[
            future_target.iloc[score_positions].notna().to_numpy(dtype=bool)
        ]
        if len(fit_positions) < 12 or len(score_positions) < 4:
            horizon_rows.append(
                {
                    "horizon": horizon,
                    "fit_rows": int(len(fit_positions)),
                    "evaluation_rows": int(len(score_positions)),
                    "relative_mse_utility": None,
                    "status": "insufficient_rows",
                }
            )
            continue
        score = _fit_and_score(
            frame,
            future_target,
            fit_positions,
            score_positions,
            [feature],
            config.ridge_alpha,
        )
        train_median = float(
            pd.to_numeric(frame[feature].iloc[fit_positions], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .median()
        )
        if not math.isfinite(train_median):
            train_median = 0.0
        persistence_prediction = (
            pd.to_numeric(frame[feature].iloc[score_positions], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(train_median)
            .to_numpy(dtype=float)
        )
        future_values = future_target.iloc[score_positions].to_numpy(dtype=float)
        persistence_metrics = _metrics(future_values, persistence_prediction)
        persistence_mse = float(persistence_metrics["mse"])
        predictability_utility = (
            float(1.0 - float(score["model"]["mse"]) / persistence_mse)
            if persistence_mse > 1e-15
            else 0.0
        )
        horizon_rows.append(
            {
                "horizon": horizon,
                "fit_rows": int(len(fit_positions)),
                "evaluation_rows": int(len(score_positions)),
                "relative_mse_utility": predictability_utility,
                "correlation": float(score["model"]["correlation"]),
                "model_mse": float(score["model"]["mse"]),
                "persistence_baseline_mse": persistence_mse,
                "status": "ok",
            }
        )
    utilities = [
        float(row["relative_mse_utility"])
        for row in horizon_rows
        if row.get("relative_mse_utility") is not None
    ]
    return {
        "mean_relative_mse_utility": float(np.mean(utilities)) if utilities else 0.0,
        "horizons": horizon_rows,
        "target_kind": "future_same_world_feature",
        "baseline_kind": "last_observed_feature_value_persistence",
        "fit_and_labels_restricted_to_each_segment": True,
    }


def _temporal_feature_scores(
    frame: pd.DataFrame,
    target: pd.Series,
    folds: Sequence[_PurgedFold],
    config: WorldFeatureAuditConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in frame.columns:
        fold_metrics: list[dict[str, Any]] = []
        coefficients: list[float] = []
        for fold in folds:
            downstream_score = _fit_and_score(
                frame,
                target,
                fold.train_positions,
                fold.validation_positions,
                [feature],
                config.ridge_alpha,
            )
            future_world_score = _future_world_feature_score(
                frame,
                feature,
                fold.train_positions,
                fold.validation_positions,
                config,
            )
            downstream_weight, future_world_weight = config.normalized_utility_weights
            combined_utility = (
                downstream_weight * float(downstream_score["relative_mse_utility"])
                + future_world_weight * float(future_world_score["mean_relative_mse_utility"])
            )
            coefficient = float(downstream_score["coefficients"].get(feature, 0.0))
            coefficients.append(coefficient)
            fold_metrics.append(
                {
                    "fold": int(fold.fold),
                    "train_rows": int(len(fold.train_positions)),
                    "validation_rows": int(len(fold.validation_positions)),
                    "purge_rows": int(len(fold.purge_positions)),
                    "utility": combined_utility,
                    "downstream_return_utility": float(
                        downstream_score["relative_mse_utility"]
                    ),
                    "future_world_predictability": float(
                        future_world_score["mean_relative_mse_utility"]
                    ),
                    "future_world_horizons": future_world_score["horizons"],
                    "mse": float(downstream_score["model"]["mse"]),
                    "baseline_mse": float(downstream_score["baseline"]["mse"]),
                    "correlation": float(downstream_score["model"]["correlation"]),
                    "coefficient": coefficient,
                }
            )
        utilities = np.asarray([item["utility"] for item in fold_metrics], dtype=float)
        downstream_utilities = np.asarray(
            [item["downstream_return_utility"] for item in fold_metrics], dtype=float
        )
        future_world_utilities = np.asarray(
            [item["future_world_predictability"] for item in fold_metrics], dtype=float
        )
        mean_utility = float(np.mean(utilities)) if len(utilities) else float("-inf")
        median_utility = float(np.median(utilities)) if len(utilities) else float("-inf")
        utility_std = float(np.std(utilities)) if len(utilities) else float("inf")
        signs = np.sign(np.asarray(coefficients, dtype=float))
        nonzero_signs = signs[signs != 0.0]
        if len(nonzero_signs):
            positive = float(np.mean(nonzero_signs > 0.0))
            sign_consistency = max(positive, 1.0 - positive)
        else:
            sign_consistency = 0.0
        stable_utility = mean_utility - float(config.stability_penalty) * utility_std
        utility_score = stable_utility - float(config.sign_instability_penalty) * (1.0 - sign_consistency)
        rows.append(
            {
                "feature": feature,
                "fold_count": int(len(fold_metrics)),
                "mean_utility": mean_utility,
                "mean_downstream_return_utility": float(np.mean(downstream_utilities)),
                "mean_future_world_predictability": float(np.mean(future_world_utilities)),
                "future_world_target_horizons": list(config.world_target_horizons),
                "median_utility": median_utility,
                "utility_std": utility_std,
                "positive_utility_fold_share": float(np.mean(utilities > 0.0)) if len(utilities) else 0.0,
                "coefficient_sign_consistency": float(sign_consistency),
                "stable_utility": stable_utility,
                "utility_score": utility_score,
                "fold_metrics": fold_metrics,
                "fit_segment": "purged_chronological_folds_within_train",
            }
        )
    return sorted(rows, key=lambda item: (-float(item["utility_score"]), item["feature"]))


class _UnionFind:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _redundancy_clusters(
    frame: pd.DataFrame,
    train_positions: np.ndarray,
    feature_scores: Sequence[Mapping[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    features = list(frame.columns)
    if not features:
        return []
    train = frame.iloc[train_positions]
    medians = train.median(axis=0, skipna=True).fillna(0.0)
    correlation = train.fillna(medians).corr().abs().fillna(0.0)
    groups = _UnionFind(features)
    for left_position, left in enumerate(features):
        for right in features[left_position + 1 :]:
            if float(correlation.loc[left, right]) >= float(threshold):
                groups.union(left, right)
    components: dict[str, list[str]] = {}
    for feature in features:
        components.setdefault(groups.find(feature), []).append(feature)
    score_lookup = {str(row["feature"]): float(row["utility_score"]) for row in feature_scores}
    clusters: list[dict[str, Any]] = []
    ordered_components = sorted(components.values(), key=lambda members: min(members))
    for number, members in enumerate(ordered_components):
        ranked = sorted(members, key=lambda name: (-score_lookup.get(name, float("-inf")), name))
        representative = ranked[0]
        pairwise = []
        for member_position, left in enumerate(sorted(members)):
            for right in sorted(members)[member_position + 1 :]:
                pairwise.append(float(correlation.loc[left, right]))
        clusters.append(
            {
                "cluster_id": f"cluster_{number:04d}",
                "members": sorted(members),
                "size": int(len(members)),
                "representative": representative,
                "rejected_members": [member for member in sorted(members) if member != representative],
                "max_absolute_correlation": max(pairwise) if pairwise else 0.0,
                "correlation_fit_segment": "train_only",
            }
        )
    return clusters


def _selection_stage(
    world_name: str,
    frame: pd.DataFrame,
    target: pd.Series,
    train_positions: np.ndarray,
    selection_positions: np.ndarray,
    feature_scores: Sequence[Mapping[str, Any]],
    clusters: Sequence[Mapping[str, Any]],
    config: WorldFeatureAuditConfig,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    score_lookup = {str(row["feature"]): row for row in feature_scores}
    candidates: list[str] = []
    rejected: dict[str, list[str]] = {}
    for cluster in clusters:
        representative = str(cluster["representative"])
        row = score_lookup[representative]
        if int(row["fold_count"]) < int(config.min_utility_folds):
            rejected.setdefault(representative, []).append("insufficient_temporal_utility_folds")
        elif float(row["stable_utility"]) < float(config.min_train_utility):
            rejected.setdefault(representative, []).append("below_train_utility_threshold")
        else:
            candidates.append(representative)

    budget = config.budget_for(world_name)
    pool_size = max(budget, budget * int(config.candidate_pool_multiplier))
    ranked_train_candidates = sorted(
        candidates,
        key=lambda feature: (-float(score_lookup[feature]["utility_score"]), feature),
    )
    candidates = ranked_train_candidates[:pool_size]
    for feature in ranked_train_candidates[pool_size:]:
        rejected.setdefault(feature, []).append("outside_preselection_candidate_pool")
    selection_scores: list[dict[str, Any]] = []
    selection_future_lookup: dict[str, float] = {}
    eligible: list[tuple[str, float]] = []
    for feature in candidates:
        downstream_score = _fit_and_score(
            frame,
            target,
            train_positions,
            selection_positions,
            [feature],
            config.ridge_alpha,
        )
        future_world_score = _future_world_feature_score(
            frame,
            feature,
            train_positions,
            selection_positions,
            config,
        )
        downstream_weight, future_world_weight = config.normalized_utility_weights
        selection_downstream_utility = float(downstream_score["relative_mse_utility"])
        selection_future_world_predictability = float(
            future_world_score["mean_relative_mse_utility"]
        )
        selection_utility = (
            downstream_weight * selection_downstream_utility
            + future_world_weight * selection_future_world_predictability
        )
        selection_future_lookup[feature] = selection_future_world_predictability
        train_utility = float(score_lookup[feature]["utility_score"])
        combined = (
            (1.0 - float(config.selection_weight)) * train_utility
            + float(config.selection_weight) * selection_utility
        )
        is_eligible = selection_utility >= float(config.min_selection_utility)
        if is_eligible:
            eligible.append((feature, combined))
        else:
            rejected.setdefault(feature, []).append("below_selection_utility_threshold")
        selection_scores.append(
            {
                "feature": feature,
                "selection_rows": int(len(selection_positions)),
                "train_utility_score": train_utility,
                "selection_relative_mse_utility": selection_utility,
                "selection_downstream_return_utility": selection_downstream_utility,
                "selection_future_world_predictability": selection_future_world_predictability,
                "selection_future_world_horizons": future_world_score["horizons"],
                "selection_mse": float(downstream_score["model"]["mse"]),
                "selection_baseline_mse": float(downstream_score["baseline"]["mse"]),
                "selection_correlation": float(downstream_score["model"]["correlation"]),
                "combined_selection_score": combined,
                "eligible": bool(is_eligible),
                "fit_segment": "train",
                "score_segment": "selection_only",
            }
        )
    ranked = [feature for feature, _ in sorted(eligible, key=lambda item: (-item[1], item[0]))]

    prefix_scores: list[dict[str, Any]] = []
    start = 0 if config.allow_empty_selection else min(1, len(ranked))
    for count in range(start, min(budget, len(ranked)) + 1):
        features = ranked[:count]
        downstream_score = _fit_and_score(
            frame,
            target,
            train_positions,
            selection_positions,
            features,
            config.ridge_alpha,
        )
        downstream_utility = float(downstream_score["relative_mse_utility"])
        future_world_predictability = (
            float(np.mean([selection_future_lookup[feature] for feature in features]))
            if features
            else 0.0
        )
        downstream_weight, future_world_weight = config.normalized_utility_weights
        combined_utility = (
            downstream_weight * downstream_utility
            + future_world_weight * future_world_predictability
        )
        objective = combined_utility - float(config.selection_complexity_penalty) * count
        prefix_scores.append(
            {
                "feature_count": int(count),
                "features": list(features),
                "selection_relative_mse_utility": combined_utility,
                "selection_downstream_return_utility": downstream_utility,
                "selection_future_world_predictability": future_world_predictability,
                "selection_mse": float(downstream_score["model"]["mse"]),
                "selection_baseline_mse": float(downstream_score["baseline"]["mse"]),
                "selection_correlation": float(downstream_score["model"]["correlation"]),
                "complexity_penalized_objective": objective,
                "score_segment": "selection_only",
            }
        )
    if not prefix_scores:
        selected: list[str] = []
    else:
        winner = max(
            prefix_scores,
            key=lambda row: (
                float(row["complexity_penalized_objective"]),
                -int(row["feature_count"]),
            ),
        )
        selected = list(winner["features"])
    for feature in ranked:
        if feature not in selected:
            reason = "feature_budget_exceeded" if ranked.index(feature) >= budget else "not_in_best_selection_prefix"
            rejected.setdefault(feature, []).append(reason)
    return selected, selection_scores, prefix_scores, rejected


def _representation_coverage_scores(
    frame: pd.DataFrame,
    train_positions: np.ndarray,
    clusters: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rank nonredundant state inputs without looking at targets or later splits."""

    train = frame.iloc[train_positions]
    rows: list[dict[str, Any]] = []
    for feature in frame.columns:
        values = pd.to_numeric(train[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = values.dropna().astype(float)
        observation_coverage = float(len(finite) / max(len(values), 1))
        if len(finite) < 2:
            unique_ratio = 0.0
            change_activity = 0.0
            temporal_stability = 0.0
        else:
            unique_ratio = float(min(1.0, finite.nunique() / min(len(finite), 50)))
            diffs = finite.diff().dropna().abs()
            scale = float(finite.quantile(0.75) - finite.quantile(0.25))
            tolerance = max(abs(scale) * 1e-6, 1e-12)
            change_activity = float(np.mean(diffs.to_numpy(dtype=float) > tolerance)) if len(diffs) else 0.0
            midpoint = max(1, len(finite) // 2)
            early = finite.iloc[:midpoint]
            late = finite.iloc[midpoint:]
            if late.empty:
                temporal_stability = 1.0
            else:
                overall_iqr = max(abs(scale), 1e-12)
                median_shift = abs(float(late.median()) - float(early.median())) / overall_iqr
                early_iqr = max(abs(float(early.quantile(0.75) - early.quantile(0.25))), 1e-12)
                late_iqr = max(abs(float(late.quantile(0.75) - late.quantile(0.25))), 1e-12)
                scale_shift = abs(math.log(late_iqr / early_iqr))
                temporal_stability = float(0.5 * math.exp(-median_shift) + 0.5 * math.exp(-scale_shift))
        state_variation = float(0.5 * unique_ratio + 0.5 * change_activity)
        coverage_score = float(
            0.45 * observation_coverage
            + 0.30 * state_variation
            + 0.25 * temporal_stability
        )
        rows.append(
            {
                "feature": feature,
                "observation_coverage": observation_coverage,
                "unique_ratio": unique_ratio,
                "change_activity": change_activity,
                "state_variation": state_variation,
                "temporal_stability": temporal_stability,
                "representation_coverage_score": coverage_score,
                "fit_segment": "train_only",
                "target_used": False,
                "selection_used": False,
                "test_used": False,
            }
        )
    score_lookup = {str(row["feature"]): float(row["representation_coverage_score"]) for row in rows}
    cluster_lookup: dict[str, tuple[str, str]] = {}
    for cluster in clusters:
        members = [str(member) for member in cluster["members"]]
        coverage_representative = max(
            members,
            key=lambda feature: (score_lookup.get(feature, float("-inf")), feature),
        )
        if isinstance(cluster, dict):
            cluster["coverage_representative"] = coverage_representative
        for member in members:
            cluster_lookup[member] = (str(cluster["cluster_id"]), coverage_representative)
    for row in rows:
        cluster_id, coverage_representative = cluster_lookup[str(row["feature"])]
        row["cluster_id"] = cluster_id
        row["coverage_cluster_representative"] = coverage_representative
        row["eligible_as_nonredundant_coverage_representative"] = bool(
            str(row["feature"]) == coverage_representative
        )
    return sorted(
        rows,
        key=lambda row: (-float(row["representation_coverage_score"]), str(row["feature"])),
    )


def _apply_representation_coverage(
    world_name: str,
    frame: pd.DataFrame,
    train_positions: np.ndarray,
    supervised_selected: Sequence[str],
    clusters: Sequence[Mapping[str, Any]],
    config: WorldFeatureAuditConfig,
) -> tuple[list[str], list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    coverage_scores = _representation_coverage_scores(frame, train_positions, clusters)
    selected = list(supervised_selected)
    minimum = config.minimum_for(world_name)
    budget = config.budget_for(world_name)
    coverage_selected: list[str] = []
    if len(selected) < minimum:
        for row in coverage_scores:
            feature = str(row["feature"])
            if not row["eligible_as_nonredundant_coverage_representative"]:
                continue
            cluster = next(
                cluster
                for cluster in clusters
                if str(cluster["cluster_id"]) == str(row["cluster_id"])
            )
            if any(str(member) in selected for member in cluster["members"]):
                continue
            selected.append(feature)
            coverage_selected.append(feature)
            if len(selected) >= minimum or len(selected) >= budget:
                break
    score_lookup = {str(row["feature"]): row for row in coverage_scores}
    metadata: list[dict[str, Any]] = []
    for rank, feature in enumerate(selected, start=1):
        basis = "supervised_downstream_utility" if feature in supervised_selected else "representation_coverage"
        metadata.append(
            {
                "feature": feature,
                "selection_basis": basis,
                "selected_rank": int(rank),
                "coverage_backfill_reason": (
                    None
                    if basis == "supervised_downstream_utility"
                    else f"minimum_world_features={minimum}"
                ),
                "representation_coverage_score": score_lookup.get(feature, {}).get(
                    "representation_coverage_score"
                ),
                "selection_fit_segments": (
                    ["train", "selection"]
                    if basis == "supervised_downstream_utility"
                    else ["train_only"]
                ),
                "test_used_for_selection": False,
            }
        )
    return selected, coverage_selected, coverage_scores, metadata


def _test_ablation(
    frame: pd.DataFrame,
    target: pd.Series,
    train_positions: np.ndarray,
    selection_positions: np.ndarray,
    test_positions: np.ndarray,
    selected: Sequence[str],
    config: WorldFeatureAuditConfig,
) -> dict[str, Any]:
    fit_positions = np.sort(np.concatenate([train_positions, selection_positions])).astype(int)
    baseline = _fit_and_score(
        frame,
        target,
        fit_positions,
        test_positions,
        [],
        config.ridge_alpha,
    )
    full = _fit_and_score(
        frame,
        target,
        fit_positions,
        test_positions,
        list(selected),
        config.ridge_alpha,
    )
    all_sanitized_raw = _fit_and_score(
        frame,
        target,
        fit_positions,
        test_positions,
        list(frame.columns),
        config.ridge_alpha,
    )
    rows: list[dict[str, Any]] = []
    full_mse = float(full["model"]["mse"])
    full_utility = float(full["relative_mse_utility"])
    for feature in selected:
        remaining = [candidate for candidate in selected if candidate != feature]
        score = _fit_and_score(
            frame,
            target,
            fit_positions,
            test_positions,
            remaining,
            config.ridge_alpha,
        )
        rows.append(
            {
                "ablated_feature": feature,
                "remaining_features": remaining,
                "test_rows": int(len(test_positions)),
                "full_model_mse": full_mse,
                "ablated_model_mse": float(score["model"]["mse"]),
                "mse_degradation_when_removed": float(score["model"]["mse"]) - full_mse,
                "full_model_utility": full_utility,
                "ablated_model_utility": float(score["relative_mse_utility"]),
                "utility_loss_when_removed": full_utility - float(score["relative_mse_utility"]),
                "ablated_model_correlation": float(score["model"]["correlation"]),
                "score_segment": "untouched_test_post_selection",
                "used_for_selection": False,
            }
        )
    return {
        "fit_rows": int(len(fit_positions)),
        "test_rows": int(len(test_positions)),
        "baseline": baseline,
        "full_model": full,
        "all_sanitized_raw_model": all_sanitized_raw,
        "selected_vs_all_sanitized_raw": {
            "selected_mse": float(full["model"]["mse"]),
            "all_sanitized_raw_mse": float(all_sanitized_raw["model"]["mse"]),
            "selected_minus_all_raw_mse": float(full["model"]["mse"])
            - float(all_sanitized_raw["model"]["mse"]),
            "selected_utility": float(full["relative_mse_utility"]),
            "all_sanitized_raw_utility": float(
                all_sanitized_raw["relative_mse_utility"]
            ),
            "selected_minus_all_raw_utility": float(full["relative_mse_utility"])
            - float(all_sanitized_raw["relative_mse_utility"]),
            "all_raw_scope": (
                "All features that passed train-only sanitation; invalid/non-numeric raw "
                "columns remain excluded for model safety."
            ),
            "used_for_selection": False,
        },
        "leave_one_feature_out": rows,
        "test_opened_after_selected_manifest_frozen": True,
        "used_for_selection": False,
    }


def _leakage_audit(
    split: Mapping[str, Any],
    folds: Sequence[_PurgedFold],
    rejected: Mapping[str, Sequence[str]],
    config: WorldFeatureAuditConfig,
    coverage_selected: Sequence[str],
    target_horizon_validation: Mapping[str, Any],
) -> dict[str, Any]:
    positions = split["_positions"]
    train = positions["train"]
    selection = positions["selection"]
    test = positions["test"]
    purge = config.effective_purge_gap
    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        observed_gap = int(fold.validation_positions[0] - fold.train_positions[-1] - 1)
        fold_rows.append(
            {
                "fold": int(fold.fold),
                "train_end_position": int(fold.train_positions[-1]),
                "validation_start_position": int(fold.validation_positions[0]),
                "observed_purge_rows": observed_gap,
                "required_purge_rows": int(purge),
                "purge_satisfied": bool(observed_gap >= purge),
            }
        )
    train_selection_gap = int(selection[0] - train[-1] - 1)
    selection_test_gap = int(test[0] - selection[-1] - 1)
    target_like_rejections = sorted(
        feature
        for feature, reasons in rejected.items()
        if "forbidden_target_like_name" in reasons
    )
    target_content_rejections = sorted(
        feature
        for feature, reasons in rejected.items()
        if any(
            reason == "near_exact_target_content_on_training_segment"
            or reason.startswith("suspicious_future_target_lead:")
            for reason in reasons
        )
    )
    return {
        "chronological_index_required": True,
        "segments_disjoint": bool(
            set(train).isdisjoint(selection)
            and set(train).isdisjoint(test)
            and set(selection).isdisjoint(test)
        ),
        "target_horizon_rows": int(config.target_horizon),
        "target_horizon_validation": dict(target_horizon_validation),
        "configured_purge_gap_rows": int(config.purge_gap_rows),
        "effective_purge_gap_rows": int(purge),
        "purge_is_at_least_target_horizon": bool(purge >= int(config.target_horizon)),
        "train_selection_observed_gap_rows": train_selection_gap,
        "selection_test_observed_gap_rows": selection_test_gap,
        "outer_purges_satisfied": bool(
            train_selection_gap >= purge and selection_test_gap >= purge
        ),
        "temporal_folds": fold_rows,
        "all_temporal_fold_purges_satisfied": bool(
            fold_rows and all(row["purge_satisfied"] for row in fold_rows)
        ),
        "sanitation_fit_segment": "train_only",
        "redundancy_fit_segment": "train_only",
        "utility_fit_segment": "purged_folds_within_train_only",
        "utility_feature_eligibility_prefilter": (
            "Sanitation and redundancy are fit once on the complete outer-train segment."
        ),
        "utility_folds_use_fold_local_prefilter": False,
        "utility_fold_interpretation": (
            "Temporal stability diagnostics conditional on the outer-train prefilter; "
            "not a fully nested feature-selection estimate."
        ),
        "selection_score_segment": "selection_only",
        "representation_coverage_fit_segment": "train_only",
        "representation_coverage_target_used": False,
        "representation_coverage_selection_segment_used": False,
        "representation_coverage_test_used": False,
        "representation_coverage_selected_columns": list(coverage_selected),
        "test_access_stage": "post_selection_ablation_only",
        "test_used_for_selection": False,
        "full_input_hash_is_provenance_only": True,
        "full_input_hash_used_for_selection": False,
        "target_like_columns_rejected": target_like_rejections,
        "target_content_columns_rejected": target_content_rejections,
        "target_content_leak_checks_fit_segment": "train_only",
    }


def audit_world_frame(
    world_name: str,
    frame: pd.DataFrame,
    target: pd.Series,
    config: WorldFeatureAuditConfig | None = None,
) -> dict[str, Any]:
    """Audit and select one world's features without using test data to select.

    The returned mapping is deliberately JSON-friendly and contains all
    evidence needed to reproduce the decision: sanitation, temporal fold
    scores, redundancy clusters, selection-only scores, the frozen selected
    manifest, and post-selection test ablations.
    """

    cfg = config or WorldFeatureAuditConfig()
    target_horizon_validation = _validate_target_horizon(target, cfg)
    aligned_frame, aligned_target = _align_frame_target(frame, target)
    if len(aligned_frame) < int(cfg.min_aligned_rows):
        raise ValueError(
            f"{world_name}: {len(aligned_frame)} aligned rows is below "
            f"min_aligned_rows={cfg.min_aligned_rows}."
        )
    split = _outer_split(aligned_frame.index, cfg)
    positions = split["_positions"]
    numeric_frame, sanitation, rejected = _sanitation_audit(
        aligned_frame,
        aligned_target,
        positions["train"],
        cfg,
    )
    folds = _purged_folds(len(positions["train"]), cfg)
    if len(folds) < int(cfg.min_utility_folds):
        raise ValueError(
            f"{world_name}: only {len(folds)} purged utility folds can be built; "
            f"min_utility_folds={cfg.min_utility_folds}, train_rows={len(positions['train'])}, "
            f"min_fold_train_rows={cfg.min_fold_train_rows}, purge={cfg.effective_purge_gap}."
        )
    feature_scores = _temporal_feature_scores(numeric_frame, aligned_target, folds, cfg)
    clusters = _redundancy_clusters(
        numeric_frame,
        positions["train"],
        feature_scores,
        cfg.redundancy_threshold,
    )
    for cluster in clusters:
        representative = str(cluster["representative"])
        for member in cluster["rejected_members"]:
            rejected.setdefault(str(member), []).append(f"redundant_with:{representative}")
    supervised_selected, selection_scores, prefix_scores, selection_rejected = _selection_stage(
        world_name,
        numeric_frame,
        aligned_target,
        positions["train"],
        positions["selection"],
        feature_scores,
        clusters,
        cfg,
    )
    for feature, reasons in selection_rejected.items():
        rejected.setdefault(feature, []).extend(reasons)
    selected, coverage_selected, coverage_scores, selected_metadata = _apply_representation_coverage(
        world_name,
        numeric_frame,
        positions["train"],
        supervised_selected,
        clusters,
        cfg,
    )
    rejected = {
        feature: list(dict.fromkeys(reasons))
        for feature, reasons in sorted(rejected.items())
        if feature not in selected
    }
    for feature in aligned_frame.columns:
        if feature not in selected and feature not in rejected:
            rejected[feature] = ["not_selected_after_train_selection_audit"]
    rejected = dict(sorted(rejected.items()))

    # The selected list is frozen before this function is called.  Nothing in
    # the returned test block is read by the selection stage above.
    test_ablation = _test_ablation(
        numeric_frame,
        aligned_target,
        positions["train"],
        positions["selection"],
        positions["test"],
        selected,
        cfg,
    )
    leakage = _leakage_audit(
        split,
        folds,
        rejected,
        cfg,
        coverage_selected,
        target_horizon_validation,
    )
    public_split = {key: value for key, value in split.items() if key != "_positions"}
    return _json_safe(
        {
            "world": str(world_name),
            "target_name": str(aligned_target.name),
            "target_horizon_validation": target_horizon_validation,
            "input_content_hash_sha256": _pandas_content_hash(aligned_frame),
            "target_content_hash_sha256": _pandas_content_hash(aligned_target),
            "training_input_content_hash_sha256": _pandas_content_hash(
                aligned_frame.iloc[positions["train"]]
            ),
            "training_target_content_hash_sha256": _pandas_content_hash(
                aligned_target.iloc[positions["train"]]
            ),
            "input_rows": int(len(frame)),
            "aligned_rows": int(len(aligned_frame)),
            "input_feature_count": int(frame.shape[1]),
            "sanitized_feature_count": int(numeric_frame.shape[1]),
            "feature_budget": int(cfg.budget_for(world_name)),
            "minimum_world_features": int(cfg.minimum_for(world_name)),
            "coverage_minimum_met": bool(len(selected) >= cfg.minimum_for(world_name)),
            "coverage_minimum_shortfall": int(
                max(0, cfg.minimum_for(world_name) - len(selected))
            ),
            "selected_columns": list(selected),
            "supervised_selected_columns": list(supervised_selected),
            "coverage_selected_columns": list(coverage_selected),
            "selected_feature_metadata": selected_metadata,
            "selected_feature_count": int(len(selected)),
            "rejected_features": rejected,
            "split_timestamps": public_split,
            "sanitation": sanitation,
            "feature_stability_utility": feature_scores,
            "redundancy_clusters": clusters,
            "selection_only_scores": selection_scores,
            "selection_prefix_scores": prefix_scores,
            "representation_coverage_scores": coverage_scores,
            "untouched_test_ablation": test_ablation,
            "leakage_purge_audit": leakage,
        }
    )


def select_world_features(
    world_name: str,
    frame: pd.DataFrame,
    target: pd.Series,
    config: WorldFeatureAuditConfig | None = None,
) -> list[str]:
    """Return the causally selected column list for one world."""

    return list(audit_world_frame(world_name, frame, target, config)["selected_columns"])


def audit_all_worlds(
    world_frames: Mapping[str, pd.DataFrame],
    target: pd.Series,
    config: WorldFeatureAuditConfig | None = None,
) -> dict[str, Any]:
    """Run the same causal audit contract over all five production worlds."""

    cfg = config or WorldFeatureAuditConfig()
    missing = [world for world in WORLD_NAMES if world not in world_frames]
    if missing and cfg.require_all_worlds:
        raise ValueError(f"Missing required world frames: {missing}")
    ordered_worlds = [world for world in WORLD_NAMES if world in world_frames]
    ordered_worlds.extend(sorted(set(world_frames) - set(ordered_worlds)))
    ordered_target = _as_ordered_target(target)
    ordered_frames = {world: _as_ordered_frame(world_frames[world]) for world in ordered_worlds}
    common_index = ordered_target.dropna().index
    for world in ordered_worlds:
        common_index = common_index.intersection(ordered_frames[world].index)
    common_index = common_index.sort_values()
    if len(common_index) < int(cfg.min_aligned_rows):
        raise ValueError(
            f"Only {len(common_index)} rows remain after intersecting every world and target; "
            f"min_aligned_rows={cfg.min_aligned_rows}."
        )
    aligned_target = ordered_target.reindex(common_index)
    aligned_frames = {
        world: ordered_frames[world].reindex(common_index)
        for world in ordered_worlds
    }
    worlds = {
        world: audit_world_frame(world, aligned_frames[world], aligned_target, cfg)
        for world in ordered_worlds
    }
    all_world_minimums_met = all(
        bool(result["coverage_minimum_met"])
        for result in worlds.values()
    )
    config_payload = _json_safe(asdict(cfg))
    audit_contract_hash = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    implementation_path = Path(__file__).resolve()
    return _json_safe(
        {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "target_name": str(target.name or "downstream_target"),
            "config": config_payload,
            "audit_contract_hash_sha256": audit_contract_hash,
            "implementation_content_hashes_sha256": {
                str(implementation_path): _file_content_hash(implementation_path),
            },
            "source_revision": None,
            "source_revision_status": (
                "No revision supplied by the dataframe API; implementation content hash recorded."
            ),
            "world_input_content_hashes_sha256": {
                world: result["input_content_hash_sha256"]
                for world, result in worlds.items()
            },
            "target_content_hash_sha256": _pandas_content_hash(aligned_target),
            "common_aligned_rows": int(len(common_index)),
            "common_index_start": _index_value(common_index[0]),
            "common_index_end": _index_value(common_index[-1]),
            "all_worlds_share_identical_split_timestamps": bool(
                len(
                    {
                        json.dumps(result["split_timestamps"], sort_keys=True)
                        for result in worlds.values()
                    }
                )
                == 1
            ),
            "world_order": ordered_worlds,
            "selected_columns_by_world": {
                world: list(result["selected_columns"])
                for world, result in worlds.items()
            },
            "all_world_minimums_met": all_world_minimums_met,
            "worlds": worlds,
            "global_leakage_statement": (
                "All sanitation, redundancy, and temporal utility decisions use training rows only; "
                "selection rows choose the manifest; test rows are opened only for post-selection ablation."
            ),
        }
    )


def _flatten_report_rows(audit: Mapping[str, Any], field_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for world, result in audit.get("worlds", {}).items():
        for item in result.get(field_name, []):
            row = {"world": world, **dict(item)}
            for key, value in list(row.items()):
                if isinstance(value, (list, dict, tuple)):
                    row[key] = json.dumps(_json_safe(value), sort_keys=True)
            rows.append(row)
    return rows


def save_feature_audit_report(
    audit: Mapping[str, Any],
    output_dir: str | Path = "reports/world_jepa/feature_audit",
    manifest_path: str | Path | None = None,
) -> dict[str, str]:
    """Persist the selected manifest and auditable CSV/Markdown evidence."""

    if "worlds" not in audit:
        world = str(audit.get("world", "world"))
        audit = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "target_name": audit.get("target_name", "downstream_target"),
            "config": {},
            "world_order": [world],
            "selected_columns_by_world": {world: audit.get("selected_columns", [])},
            "worlds": {world: dict(audit)},
        }
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    resolved_manifest_path = (
        Path(manifest_path) if manifest_path is not None else root / "feature_manifest.json"
    )
    resolved_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    full_audit_manifest_path = root / "selected_feature_manifest.json"
    sanitation_path = root / "sanitation_report.csv"
    feature_score_path = root / "feature_stability_utility.csv"
    redundancy_path = root / "redundancy_clusters.csv"
    selection_path = root / "selection_only_scores.csv"
    prefix_path = root / "selection_prefix_scores.csv"
    coverage_path = root / "representation_coverage_scores.csv"
    ablation_path = root / "ablation_report.csv"
    leakage_path = root / "leakage_purge_audit.json"
    summary_path = root / "summary.md"

    safe_audit = _json_safe(audit)
    resolved_manifest_path.write_text(json.dumps(safe_audit, indent=2, sort_keys=True) + "\n")
    full_audit_manifest_path.write_text(json.dumps(safe_audit, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(_flatten_report_rows(safe_audit, "sanitation")).to_csv(sanitation_path, index=False)
    pd.DataFrame(_flatten_report_rows(safe_audit, "feature_stability_utility")).to_csv(
        feature_score_path, index=False
    )
    pd.DataFrame(_flatten_report_rows(safe_audit, "selection_only_scores")).to_csv(
        selection_path, index=False
    )
    pd.DataFrame(_flatten_report_rows(safe_audit, "selection_prefix_scores")).to_csv(
        prefix_path, index=False
    )
    pd.DataFrame(_flatten_report_rows(safe_audit, "representation_coverage_scores")).to_csv(
        coverage_path, index=False
    )

    redundancy_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    leakage_payload: dict[str, Any] = {}
    for world, result in safe_audit.get("worlds", {}).items():
        for cluster in result.get("redundancy_clusters", []):
            redundancy_rows.append(
                {
                    "world": world,
                    "cluster_id": cluster["cluster_id"],
                    "representative": cluster["representative"],
                    "coverage_representative": cluster.get(
                        "coverage_representative", cluster["representative"]
                    ),
                    "size": cluster["size"],
                    "members": json.dumps(cluster["members"]),
                    "rejected_members": json.dumps(cluster["rejected_members"]),
                    "max_absolute_correlation": cluster["max_absolute_correlation"],
                }
            )
        test = result.get("untouched_test_ablation", {})
        full_model = test.get("full_model", {})
        ablation_rows.append(
            {
                "world": world,
                "record_type": "full_model",
                "ablated_feature": None,
                "test_rows": test.get("test_rows"),
                "test_mse": full_model.get("model", {}).get("mse"),
                "test_relative_mse_utility": full_model.get("relative_mse_utility"),
                "test_correlation": full_model.get("model", {}).get("correlation"),
                "used_for_selection": False,
            }
        )
        all_raw_model = test.get("all_sanitized_raw_model", {})
        selected_vs_raw = test.get("selected_vs_all_sanitized_raw", {})
        ablation_rows.append(
            {
                "world": world,
                "record_type": "all_sanitized_raw_model",
                "ablated_feature": None,
                "test_rows": test.get("test_rows"),
                "test_mse": all_raw_model.get("model", {}).get("mse"),
                "test_relative_mse_utility": all_raw_model.get(
                    "relative_mse_utility"
                ),
                "test_correlation": all_raw_model.get("model", {}).get(
                    "correlation"
                ),
                "selected_minus_all_raw_mse": selected_vs_raw.get(
                    "selected_minus_all_raw_mse"
                ),
                "selected_minus_all_raw_utility": selected_vs_raw.get(
                    "selected_minus_all_raw_utility"
                ),
                "used_for_selection": False,
            }
        )
        for row in test.get("leave_one_feature_out", []):
            ablation_rows.append({"world": world, "record_type": "leave_one_out", **dict(row)})
        leakage_payload[world] = result.get("leakage_purge_audit", {})
    pd.DataFrame(redundancy_rows).to_csv(redundancy_path, index=False)
    pd.DataFrame(ablation_rows).to_csv(ablation_path, index=False)
    leakage_path.write_text(json.dumps(_json_safe(leakage_payload), indent=2, sort_keys=True) + "\n")

    summary_lines = [
        "# World-JEPA Feature Audit",
        "",
        f"Generated: `{safe_audit.get('generated_at_utc')}`",
        "",
        "Feature selection is train/selection only. Test results below are post-selection diagnostics and never feed back into the manifest.",
        "",
        "| World | Input | Sanitized | Supervised | Coverage | Selected | Min | Min met | Budget | Selected test utility | All-raw test utility |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|",
    ]
    for world in safe_audit.get("world_order", []):
        result = safe_audit["worlds"][world]
        test_utility = (
            result.get("untouched_test_ablation", {})
            .get("full_model", {})
            .get("relative_mse_utility")
        )
        utility_text = "n/a" if test_utility is None else f"{float(test_utility):.4f}"
        all_raw_utility = (
            result.get("untouched_test_ablation", {})
            .get("all_sanitized_raw_model", {})
            .get("relative_mse_utility")
        )
        all_raw_utility_text = (
            "n/a" if all_raw_utility is None else f"{float(all_raw_utility):.4f}"
        )
        summary_lines.append(
            f"| {world} | {result['input_feature_count']} | {result['sanitized_feature_count']} | "
            f"{len(result.get('supervised_selected_columns', []))} | "
            f"{len(result.get('coverage_selected_columns', []))} | {result['selected_feature_count']} | "
            f"{result.get('minimum_world_features', 0)} | "
            f"{'yes' if result.get('coverage_minimum_met') else 'NO'} | "
            f"{result['feature_budget']} | {utility_text} | {all_raw_utility_text} |"
        )
    summary_lines.extend(["", "## Selected columns", ""])
    for world in safe_audit.get("world_order", []):
        selected = safe_audit["worlds"][world].get("selected_columns", [])
        coverage = set(safe_audit["worlds"][world].get("coverage_selected_columns", []))
        rendered = [f"`{name}`{' _(coverage)_' if name in coverage else ''}" for name in selected]
        summary_lines.append(f"- **{world}:** {', '.join(rendered) if rendered else '_baseline only_'}")
    summary_lines.extend(
        [
            "",
            "## Leakage and purge contract",
            "",
            "- Sanitation and redundancy are fit on training rows only.",
            "- Feature utility uses expanding chronological folds with a purge at least as large as the target horizon.",
            "- Supervised scores combine downstream return utility with future same-world predictability at the configured world horizons; future-world utility is measured against a last-value persistence baseline.",
            "- Selection-only scores and feature-count choice use the selection segment.",
            "- If supervised selection is below the configured world minimum, nonredundant features are backfilled by train-only observation coverage, temporal stability, and state variation.",
            "- The test segment is untouched until the selected column list is frozen, then used only for ablation reporting.",
            "- Test ablations include selected features versus all train-sanitized raw features.",
            "",
        ]
    )
    summary_path.write_text("\n".join(summary_lines))
    return {
        "manifest": str(resolved_manifest_path),
        "selected_feature_manifest": str(full_audit_manifest_path),
        "sanitation": str(sanitation_path),
        "feature_stability_utility": str(feature_score_path),
        "redundancy_clusters": str(redundancy_path),
        "selection_only_scores": str(selection_path),
        "selection_prefix_scores": str(prefix_path),
        "representation_coverage_scores": str(coverage_path),
        "ablation_report": str(ablation_path),
        "leakage_purge_audit": str(leakage_path),
        "summary": str(summary_path),
    }


__all__ = [
    "WORLD_NAMES",
    "WorldFeatureAuditConfig",
    "audit_world_frame",
    "select_world_features",
    "audit_all_worlds",
    "save_feature_audit_report",
]
