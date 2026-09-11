from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from dual_model_forecaster.utils import ensure_matplotlib_cache


SPECIALIST_COLORS = {
    "structure": "#345995",
    "environment": "#2a9d8f",
    "edges": "#d1495b",
    "movement": "#edae49",
}

HORIZON_COLORS = {
    1: "#345995",
    3: "#2a9d8f",
    7: "#d1495b",
    15: "#6d597a",
}

PRESSURE_COLORS = {
    "reversion_pressure": "#345995",
    "vol_pressure": "#d1495b",
    "tail_pressure": "#edae49",
    "uncertainty_proxy": "#6d597a",
}


def _prepare_timeseries(timeseries: pd.DataFrame) -> pd.DataFrame:
    if timeseries.empty or "timestamp" not in timeseries.columns:
        return pd.DataFrame()
    frame = timeseries.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    for column in [
        "horizon",
        "jepa_norm",
        "jepa_delta_norm",
        "kalman_alignment",
        "reversion_pressure",
        "vol_pressure",
        "tail_pressure",
        "uncertainty_proxy",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan)


def _recent_close(close: pd.Series, start: pd.Timestamp | None) -> pd.Series:
    if close.empty:
        return close
    history = pd.to_numeric(close, errors="coerce").dropna()
    if start is not None:
        history = history.loc[pd.DatetimeIndex(history.index) >= pd.Timestamp(start)]
    return history


def _first_kalman_panel(panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    for panel in panels.values():
        if not panel.empty:
            return panel
    return pd.DataFrame()


def _format_time_axis(ax: object, mdates: object) -> None:
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _save_alignment_overlay(
    *,
    frame: pd.DataFrame,
    panels: Mapping[str, pd.DataFrame],
    close: pd.Series,
    path: Path,
) -> str:
    ensure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    start = pd.Timestamp(frame["timestamp"].min()) if not frame.empty else None
    close_history = _recent_close(close, start)
    kalman_panel = _first_kalman_panel(panels)
    if start is not None and not kalman_panel.empty:
        kalman_panel = kalman_panel.loc[pd.DatetimeIndex(kalman_panel.index) >= start]

    fig, (ax_price, ax_align) = plt.subplots(
        2,
        1,
        figsize=(13, 7.4),
        gridspec_kw={"height_ratios": [2.7, 1.4]},
        constrained_layout=True,
        sharex=False,
    )

    if not close_history.empty:
        ax_price.plot(
            close_history.index,
            close_history.to_numpy(dtype=float),
            color="#1f2937",
            linewidth=1.9,
            label="BTC close",
        )
    if not kalman_panel.empty and "kalman__kalman_fair_value_price" in kalman_panel.columns:
        fair_value = pd.to_numeric(kalman_panel["kalman__kalman_fair_value_price"], errors="coerce").dropna()
        if not fair_value.empty:
            ax_price.plot(
                fair_value.index,
                fair_value.to_numpy(dtype=float),
                color="#0f766e",
                linewidth=1.45,
                alpha=0.90,
                label="Kalman fair value",
            )
    ax_price.set_title("JEPA/Kalman Alignment Overlay")
    ax_price.set_ylabel("BTC price")
    ax_price.grid(alpha=0.22)
    ax_price.legend(loc="upper left", ncol=2, frameon=False)
    _format_time_axis(ax_price, mdates)

    for specialist, subset in frame.groupby("specialist"):
        series = subset.groupby("timestamp")["kalman_alignment"].mean().dropna()
        if series.empty:
            continue
        ax_align.plot(
            series.index,
            series.to_numpy(dtype=float),
            color=SPECIALIST_COLORS.get(str(specialist), "#4b5563"),
            linewidth=1.35,
            alpha=0.82,
            label=str(specialist),
        )
    aggregate = frame.groupby("timestamp")["kalman_alignment"].mean().dropna()
    if not aggregate.empty:
        ax_align.plot(
            aggregate.index,
            aggregate.to_numpy(dtype=float),
            color="#111827",
            linewidth=2.4,
            label="ensemble mean",
        )
    ax_align.axhline(0.0, color="#6b7280", linestyle="--", linewidth=0.9)
    ax_align.set_ylabel("Alignment")
    ax_align.set_ylim(-1.05, 1.05)
    ax_align.grid(alpha=0.22)
    ax_align.legend(loc="upper left", ncol=5, frameon=False)
    _format_time_axis(ax_align, mdates)

    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path.resolve())


def _save_pressure_overlay(frame: pd.DataFrame, close: pd.Series, path: Path) -> str:
    ensure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    aggregate = frame.groupby("timestamp")[list(PRESSURE_COLORS)].mean().dropna(how="all")
    start = pd.Timestamp(aggregate.index.min()) if not aggregate.empty else None
    close_history = _recent_close(close, start)

    fig, ax = plt.subplots(figsize=(13, 4.9), constrained_layout=True)
    for metric, color in PRESSURE_COLORS.items():
        series = aggregate.get(metric, pd.Series(dtype=float)).dropna()
        if series.empty:
            continue
        ax.plot(
            series.index,
            series.to_numpy(dtype=float),
            color=color,
            linewidth=1.8,
            label=metric.replace("_", " "),
        )
    ax.axhline(0.0, color="#6b7280", linestyle="--", linewidth=0.9)
    ax.set_title("JEPA Pressure Overlay")
    ax.set_ylabel("Semantic pressure")
    ax.grid(alpha=0.22)

    handles, labels = ax.get_legend_handles_labels()
    if not close_history.empty:
        ax_price = ax.twinx()
        normalized_close = close_history / float(close_history.iloc[0]) - 1.0
        ax_price.plot(
            normalized_close.index,
            normalized_close.to_numpy(dtype=float),
            color="#111827",
            linewidth=1.1,
            alpha=0.22,
            label="BTC close change",
        )
        ax_price.set_ylabel("BTC close change")
        price_handles, price_labels = ax_price.get_legend_handles_labels()
        handles += price_handles
        labels += price_labels

    if handles:
        ax.legend(handles, labels, loc="upper left", ncol=3, frameon=False)
    _format_time_axis(ax, mdates)

    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path.resolve())


def _save_norm_overlay(frame: pd.DataFrame, path: Path) -> str:
    ensure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    specialists = [str(value) for value in frame["specialist"].dropna().unique()]
    if not specialists:
        specialists = ["JEPA"]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 7.6),
        constrained_layout=True,
        sharex=True,
    )
    flat_axes = list(axes.ravel())
    for ax, specialist in zip(flat_axes, specialists):
        subset = frame.loc[frame["specialist"].astype(str) == specialist]
        for horizon, horizon_subset in subset.groupby("horizon"):
            series = horizon_subset.groupby("timestamp")["jepa_norm"].mean().dropna()
            if series.empty:
                continue
            horizon_int = int(horizon)
            ax.plot(
                series.index,
                series.to_numpy(dtype=float),
                color=HORIZON_COLORS.get(horizon_int, "#4b5563"),
                linewidth=1.45,
                alpha=0.88,
                label=f"{horizon_int}d",
            )
        ax.set_title(str(specialist))
        ax.set_ylabel("Latent norm")
        ax.grid(alpha=0.22)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", ncol=4, frameon=False)
        _format_time_axis(ax, mdates)

    for ax in flat_axes[len(specialists) :]:
        ax.axis("off")

    fig.suptitle("JEPA Latent Norm Overlay", fontsize=14)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path.resolve())


def write_jepa_visuals(
    *,
    artifact_dir: str | Path,
    timeseries: pd.DataFrame,
    panels: Mapping[str, pd.DataFrame],
    close: pd.Series,
) -> dict[str, str]:
    frame = _prepare_timeseries(timeseries)
    if frame.empty:
        return {}

    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "alignment_overlay": _save_alignment_overlay(
            frame=frame,
            panels=panels,
            close=close,
            path=output_dir / "jepa_alignment_overlay.png",
        ),
        "pressure_overlay": _save_pressure_overlay(
            frame=frame,
            close=close,
            path=output_dir / "jepa_pressure_overlay.png",
        ),
        "latent_norm_overlay": _save_norm_overlay(
            frame=frame,
            path=output_dir / "jepa_latent_norm_overlay.png",
        ),
    }
    return paths
