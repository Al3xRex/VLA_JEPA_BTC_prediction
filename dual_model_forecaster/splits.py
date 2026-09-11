from __future__ import annotations

from typing import Any

import pandas as pd

from dual_model_forecaster.utils import format_date


def build_outer_folds(index: pd.Index, config: dict[str, Any]) -> list[dict[str, Any]]:
    split_cfg = config["splits"]
    min_train_days = int(split_cfg["min_train_days"])
    test_days = int(split_cfg["outer_test_days"])
    step_days = int(split_cfg["outer_step_days"])
    max_folds = int(split_cfg["max_outer_folds"])

    folds: list[dict[str, Any]] = []
    total = len(index)
    fold_id = 0
    test_end = total
    while test_end - test_days >= min_train_days:
        test_start = test_end - test_days
        train_end = test_start
        train_start = 0
        folds.append(
            {
                "fold_id": f"fold_{fold_id:02d}",
                "train_start_pos": train_start,
                "train_end_pos": train_end,
                "test_start_pos": test_start,
                "test_end_pos": test_end,
                "train_start": format_date(index[train_start]),
                "train_end": format_date(index[train_end - 1]),
                "test_start": format_date(index[test_start]),
                "test_end": format_date(index[test_end - 1]),
            }
        )
        fold_id += 1
        if fold_id >= max_folds:
            break
        test_end -= step_days

    folds.reverse()
    for idx, fold in enumerate(folds):
        fold["fold_id"] = f"fold_{idx:02d}"
    return folds


def select_outer_fold(folds: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    fold_index = int(config["splits"]["outer_fold_index"])
    return folds[fold_index]


def build_inner_segments(
    index: pd.Index,
    train_end_pos: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    split_cfg = config["splits"]
    min_train_days = int(split_cfg["inner_min_train_days"])
    segment_days = int(split_cfg["inner_segment_days"])

    segments: list[dict[str, Any]] = []
    predict_start = min_train_days
    segment_id = 0
    while predict_start < train_end_pos:
        predict_end = min(predict_start + segment_days, train_end_pos)
        fit_end = predict_start
        if predict_end <= fit_end:
            break
        segments.append(
            {
                "segment_id": f"inner_{segment_id:02d}",
                "fit_end_pos": fit_end,
                "predict_start_pos": predict_start,
                "predict_end_pos": predict_end,
                "fit_end": format_date(index[fit_end - 1]),
                "predict_start": format_date(index[predict_start]),
                "predict_end": format_date(index[predict_end - 1]),
            }
        )
        predict_start = predict_end
        segment_id += 1
    return segments


def build_split_manifest(index: pd.Index, config: dict[str, Any]) -> dict[str, Any]:
    outer_folds = build_outer_folds(index=index, config=config)
    manifest = []
    for fold in outer_folds:
        manifest.append(
            {
                "outer_fold": fold,
                "inner_segments": build_inner_segments(
                    index=index,
                    train_end_pos=int(fold["train_end_pos"]),
                    config=config,
                ),
            }
        )
    return {
        "row_count": int(len(index)),
        "first_timestamp": format_date(index.min()),
        "last_timestamp": format_date(index.max()),
        "folds": manifest,
    }
