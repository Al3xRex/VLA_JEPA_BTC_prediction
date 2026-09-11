from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dual_model_forecaster.world_jepa.world_encoder import DEFAULT_WORLD_NAMES


_METADATA_ONLY_TARGET_FAMILIES = frozenset(
    {
        "observation_age",
        "change_age",
        "validity",
        "missingness",
        "staleness",
    }
)
_TARGET_WINDOW_MODES = frozenset({"forward", "disjoint_horizon_bins"})


@dataclass(frozen=True)
class WorldJEPASampleIndex:
    world_name: str
    world_id: int
    as_of_position: int
    as_of_timestamp: Any


def _as_frame(panel: pd.DataFrame | np.ndarray | torch.Tensor) -> pd.DataFrame:
    if isinstance(panel, pd.DataFrame):
        frame = panel.copy()
    elif isinstance(panel, torch.Tensor):
        values = panel.detach().cpu().numpy()
        if values.ndim != 2:
            raise ValueError("A tensor panel must have shape [T, F].")
        frame = pd.DataFrame(values)
    else:
        values = np.asarray(panel)
        if values.ndim != 2:
            raise ValueError("An array panel must have shape [T, F].")
        frame = pd.DataFrame(values)
    if frame.index.has_duplicates:
        raise ValueError("World JEPA panels require a unique chronological index.")
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index()
    if frame.shape[1] == 0:
        raise ValueError("World JEPA panels require at least one feature column.")
    return frame.apply(pd.to_numeric, errors="coerce")


def _encode_labels(
    labels: pd.Series | Sequence[Any] | np.ndarray | torch.Tensor | None,
    *,
    index: pd.Index,
) -> tuple[np.ndarray, dict[Any, int]]:
    if labels is None:
        return np.zeros(len(index), dtype=np.int64), {}
    if isinstance(labels, pd.Series):
        aligned = labels.reindex(index)
    elif isinstance(labels, torch.Tensor):
        values = labels.detach().cpu().numpy()
        if len(values) != len(index):
            raise ValueError("Label length must match panel length.")
        aligned = pd.Series(values, index=index)
    else:
        values = np.asarray(labels)
        if len(values) != len(index):
            raise ValueError("Label length must match panel length.")
        aligned = pd.Series(values, index=index)

    numeric = pd.to_numeric(aligned, errors="coerce")
    if numeric.notna().all() and (numeric >= 0).all() and np.allclose(numeric, np.rint(numeric)):
        return numeric.astype(np.int64).to_numpy(), {}
    observed = sorted({value for value in aligned.dropna().tolist()}, key=lambda value: str(value))
    mapping = {value: idx + 1 for idx, value in enumerate(observed)}
    encoded = aligned.map(mapping).fillna(0).astype(np.int64).to_numpy()
    return encoded, mapping


def _is_metadata_only_target_column(column: Any) -> bool:
    family = str(column).strip().lower().split("__", 1)[0]
    return family in _METADATA_ONLY_TARGET_FAMILIES


def _semantic_target_indices(
    columns: pd.Index,
    requested: Sequence[str] | None,
) -> np.ndarray:
    string_columns = [str(column) for column in columns]
    if requested is None:
        indices = [
            index
            for index, column in enumerate(columns)
            if not _is_metadata_only_target_column(column)
        ]
    else:
        requested_names = [str(column) for column in requested]
        if not requested_names:
            raise ValueError("semantic_target_columns cannot be empty.")
        missing = sorted(set(requested_names).difference(string_columns))
        if missing:
            raise ValueError(f"Unknown semantic_target_columns: {missing}.")
        requested_set = set(requested_names)
        indices = [index for index, column in enumerate(string_columns) if column in requested_set]
    if not indices:
        raise ValueError(
            "No semantic target features remain after excluding metadata-only channels; "
            "provide semantic_target_columns explicitly if this schema is intentional."
        )
    return np.asarray(indices, dtype=np.int64)


def _target_offset_blocks(
    horizons: np.ndarray,
    *,
    target_block_length: int,
    mode: str,
) -> np.ndarray:
    """Build fixed-width future offset blocks; ``-1`` denotes structural padding."""

    mode = str(mode).lower()
    if mode not in _TARGET_WINDOW_MODES:
        raise ValueError(f"target_window_mode must be one of {sorted(_TARGET_WINDOW_MODES)}.")
    offsets = np.rint(horizons).astype(np.int64)
    if mode == "forward":
        return np.stack(
            [
                np.arange(offset, offset + int(target_block_length), dtype=np.int64)
                for offset in offsets
            ]
        )

    if len(np.unique(offsets)) != len(offsets):
        raise ValueError("disjoint_horizon_bins requires unique horizons.")
    by_horizon: dict[int, np.ndarray] = {}
    previous = 0
    for horizon in sorted(int(value) for value in offsets):
        # Each horizon owns the causal interval since the preceding horizon.
        # Long bins retain their most horizon-proximal rows; short bins are
        # left-padded so target blocks keep a stable tensor shape.
        real_offsets = np.arange(previous + 1, horizon + 1, dtype=np.int64)
        real_offsets = real_offsets[-int(target_block_length) :]
        padded = np.full(int(target_block_length), -1, dtype=np.int64)
        padded[-len(real_offsets) :] = real_offsets
        by_horizon[horizon] = padded
        previous = horizon
    return np.stack([by_horizon[int(offset)] for offset in offsets])


class WorldJEPADataset(Dataset):
    """Strictly causal samples for a single semantic world.

    ``target_window_mode='forward'`` preserves the original fixed-length block
    behavior. ``'disjoint_horizon_bins'`` treats each horizon as an endpoint
    and constructs non-overlapping intervals between consecutive horizons,
    with structural left-padding to retain a fixed tensor shape. Target rows
    are valid only when at least one semantic (non-metadata) feature exists.
    """

    def __init__(
        self,
        panel: pd.DataFrame | np.ndarray | torch.Tensor,
        *,
        world_name: str,
        horizons: Sequence[float] = (1.0, 3.0, 7.0, 15.0),
        context_length: int = 128,
        target_block_length: int = 1,
        allow_partial_context: bool = False,
        min_context: int = 1,
        require_all_targets: bool = True,
        stride: int = 1,
        target_window_mode: str = "forward",
        semantic_target_columns: Sequence[str] | None = None,
        state_labels: pd.Series | Sequence[Any] | np.ndarray | torch.Tensor | None = None,
        regime_labels: pd.Series | Sequence[Any] | np.ndarray | torch.Tensor | None = None,
        world_names: Sequence[str] = DEFAULT_WORLD_NAMES,
    ) -> None:
        super().__init__()
        self.panel = _as_frame(panel)
        self.values = self.panel.to_numpy(dtype=np.float32)
        self.index = self.panel.index
        self.feature_columns = [str(column) for column in self.panel.columns]
        self.world_name = str(world_name)
        names = tuple(str(name) for name in world_names)
        if self.world_name not in names:
            names = (*names, self.world_name)
        self.world_names = names
        self.world_to_id = {name: idx for idx, name in enumerate(names)}
        self.world_id = int(self.world_to_id[self.world_name])
        self.context_length = int(context_length)
        self.target_block_length = int(target_block_length)
        self.allow_partial_context = bool(allow_partial_context)
        self.min_context = int(min_context)
        self.require_all_targets = bool(require_all_targets)
        self.stride = int(stride)
        self.target_window_mode = str(target_window_mode).lower()
        if self.context_length <= 0:
            raise ValueError("context_length must be positive.")
        if self.target_block_length <= 0:
            raise ValueError("target_block_length must be positive.")
        if self.min_context <= 0 or self.min_context > self.context_length:
            raise ValueError("min_context must be in [1, context_length].")
        if self.stride <= 0:
            raise ValueError("stride must be positive.")

        self.horizons = np.asarray([float(value) for value in horizons], dtype=np.float32)
        if self.horizons.ndim != 1 or len(self.horizons) == 0:
            raise ValueError("horizons must be a non-empty one-dimensional sequence.")
        if not np.isfinite(self.horizons).all() or (self.horizons <= 0.0).any():
            raise ValueError("Dataset horizons must be finite and strictly positive.")
        rounded = np.rint(self.horizons).astype(np.int64)
        if not np.allclose(self.horizons, rounded):
            raise ValueError("Dataset horizons must map to integer row offsets.")
        self.horizon_offsets = rounded
        self.target_offset_blocks = _target_offset_blocks(
            self.horizons,
            target_block_length=self.target_block_length,
            mode=self.target_window_mode,
        )
        self.target_required_mask = self.target_offset_blocks >= 0
        self.semantic_target_indices = _semantic_target_indices(
            self.panel.columns,
            semantic_target_columns,
        )
        self.semantic_target_feature_mask = np.zeros(self.values.shape[1], dtype=bool)
        self.semantic_target_feature_mask[self.semantic_target_indices] = True

        self.state_ids, self.state_mapping = _encode_labels(state_labels, index=self.index)
        self.regime_ids, self.regime_mapping = _encode_labels(regime_labels, index=self.index)
        first_position = self.min_context - 1 if self.allow_partial_context else self.context_length - 1
        last_position = len(self.panel) - 1
        if self.require_all_targets:
            last_position -= int(self.target_offset_blocks[self.target_required_mask].max())
        self.sample_positions = list(range(first_position, max(last_position + 1, first_position), self.stride))
        if not self.require_all_targets:
            self.sample_positions = list(range(first_position, len(self.panel), self.stride))
        elif self.sample_positions:
            # A target row containing no observed features is padding, not a
            # valid self-supervised target. Filter it at construction time so
            # ``__getitem__`` never fails nondeterministically during loading.
            filtered_positions: list[int] = []
            for position in self.sample_positions:
                complete = True
                for offset_block in self.target_offset_blocks:
                    required_offsets = offset_block[offset_block >= 0]
                    target_positions = position + required_offsets
                    if (target_positions >= len(self.values)).any():
                        complete = False
                        break
                    block = self.values[target_positions]
                    semantic_finite = np.isfinite(block[:, self.semantic_target_indices])
                    if not semantic_finite.any(axis=1).all():
                        complete = False
                        break
                if complete:
                    filtered_positions.append(position)
            self.sample_positions = filtered_positions

    def __len__(self) -> int:
        return len(self.sample_positions)

    def sample_index(self, item: int) -> WorldJEPASampleIndex:
        position = int(self.sample_positions[item])
        return WorldJEPASampleIndex(
            world_name=self.world_name,
            world_id=self.world_id,
            as_of_position=position,
            as_of_timestamp=self.index[position],
        )

    def __getitem__(self, item: int) -> dict[str, Any]:
        position = int(self.sample_positions[item])
        context_start = max(0, position - self.context_length + 1)
        context_values = self.values[context_start : position + 1]
        valid_context_rows = int(len(context_values))
        left_padding = self.context_length - valid_context_rows

        x = np.zeros((self.context_length, self.values.shape[1]), dtype=np.float32)
        feature_mask = np.zeros_like(x, dtype=bool)
        finite_context = np.isfinite(context_values)
        x[left_padding:] = np.where(finite_context, context_values, 0.0)
        feature_mask[left_padding:] = finite_context
        padding_mask = np.ones(self.context_length, dtype=bool)
        padding_mask[left_padding:] = False

        horizon_count = len(self.horizon_offsets)
        target_shape = (horizon_count, self.target_block_length, self.values.shape[1])
        target_x_blocks = np.zeros(target_shape, dtype=np.float32)
        target_feature_blocks = np.zeros(target_shape, dtype=bool)
        target_semantic_feature_blocks = np.zeros(target_shape, dtype=bool)
        target_padding_blocks = np.ones((horizon_count, self.target_block_length), dtype=bool)
        target_semantic_blocks = np.zeros((horizon_count, self.target_block_length), dtype=bool)
        target_timestamp_blocks: list[list[Any | None]] = []
        for horizon_index, offset_block in enumerate(self.target_offset_blocks):
            timestamps: list[Any | None] = []
            for block_index, offset in enumerate(offset_block):
                if int(offset) < 0:
                    timestamps.append(None)
                    continue
                target_position = position + int(offset)
                if target_position >= len(self.values):
                    timestamps.append(None)
                    continue
                target_values = self.values[target_position]
                finite_target = np.isfinite(target_values)
                semantic_finite_target = finite_target & self.semantic_target_feature_mask
                has_semantic_target = bool(semantic_finite_target.any())
                target_x_blocks[horizon_index, block_index] = np.where(finite_target, target_values, 0.0)
                # The context encoder still receives metadata channels, but the
                # teacher is explicitly masked to semantic value/velocity
                # channels. Otherwise age/validity counters can become an easy
                # target that dominates the representation objective.
                target_feature_blocks[horizon_index, block_index] = semantic_finite_target
                target_semantic_feature_blocks[horizon_index, block_index] = (
                    semantic_finite_target
                )
                target_padding_blocks[horizon_index, block_index] = not has_semantic_target
                target_semantic_blocks[horizon_index, block_index] = has_semantic_target
                timestamps.append(self.index[target_position])
            target_timestamp_blocks.append(timestamps)

        missing_required_targets = target_padding_blocks & self.target_required_mask
        if self.require_all_targets and missing_required_targets.any():
            raise RuntimeError("Internal error: a required future target is unavailable.")
        if self.target_block_length == 1:
            target_x: np.ndarray = target_x_blocks[:, 0, :]
            target_feature_mask: np.ndarray = target_feature_blocks[:, 0, :]
            target_semantic_feature_mask: np.ndarray = target_semantic_feature_blocks[:, 0, :]
            target_padding_mask: np.ndarray = target_padding_blocks[:, 0]
            target_semantic_mask: np.ndarray = target_semantic_blocks[:, 0]
            target_required_mask: np.ndarray = self.target_required_mask[:, 0]
            target_timestamps: list[Any | None] | list[list[Any | None]] = [
                timestamps[0] for timestamps in target_timestamp_blocks
            ]
        else:
            target_x = target_x_blocks
            target_feature_mask = target_feature_blocks
            target_semantic_feature_mask = target_semantic_feature_blocks
            target_padding_mask = target_padding_blocks
            target_semantic_mask = target_semantic_blocks
            target_required_mask = self.target_required_mask.copy()
            target_timestamps = target_timestamp_blocks
        return {
            "x": torch.from_numpy(x),
            "horizons": torch.from_numpy(self.horizons.copy()),
            "padding_mask": torch.from_numpy(padding_mask),
            "time_mask": torch.from_numpy(~padding_mask),
            "feature_mask": torch.from_numpy(feature_mask),
            "target_x": torch.from_numpy(target_x),
            "target_padding_mask": torch.from_numpy(target_padding_mask),
            "target_time_mask": torch.from_numpy(~target_padding_mask),
            "target_feature_mask": torch.from_numpy(target_feature_mask),
            "target_semantic_feature_mask": torch.from_numpy(target_semantic_feature_mask),
            "target_semantic_mask": torch.from_numpy(target_semantic_mask),
            "target_required_mask": torch.from_numpy(target_required_mask),
            "world_id": torch.tensor(self.world_id, dtype=torch.long),
            "world_name": self.world_name,
            "state_id": torch.tensor(int(self.state_ids[position]), dtype=torch.long),
            "regime_id": torch.tensor(int(self.regime_ids[position]), dtype=torch.long),
            "as_of_position": torch.tensor(position, dtype=torch.long),
            "as_of_timestamp": self.index[position],
            "target_timestamps": target_timestamps,
        }


def temporal_adjacency_pairs(
    as_of_positions: torch.Tensor,
    world_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Find true adjacent samples regardless of shuffled batch order.

    Returned pairs are directed from the earlier sample to the next row. No
    assumption is made that neighboring batch entries are neighboring times.
    """

    positions = torch.as_tensor(as_of_positions, dtype=torch.long)
    if positions.ndim != 1:
        raise ValueError("as_of_positions must have shape [B].")
    if world_ids is None:
        worlds = torch.zeros_like(positions)
    else:
        worlds = torch.as_tensor(world_ids, dtype=torch.long, device=positions.device)
        if worlds.shape != positions.shape:
            raise ValueError("world_ids must match as_of_positions.")
    pairs: list[tuple[int, int]] = []
    for world in torch.unique(worlds).tolist():
        batch_indices = torch.nonzero(worlds == int(world), as_tuple=False).flatten().tolist()
        position_to_indices: dict[int, list[int]] = {}
        for batch_index in batch_indices:
            position_to_indices.setdefault(int(positions[batch_index]), []).append(int(batch_index))
        for position, left_indices in position_to_indices.items():
            right_indices = position_to_indices.get(position + 1, [])
            for left in left_indices:
                for right in right_indices:
                    pairs.append((left, right))
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long, device=positions.device)
    return torch.tensor(pairs, dtype=torch.long, device=positions.device)


def world_jepa_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty world JEPA batch.")
    feature_counts = {int(row["x"].shape[-1]) for row in batch}
    if len(feature_counts) != 1:
        raise ValueError("A world JEPA batch must contain a single feature schema; use one loader per world.")
    horizons = torch.stack([row["horizons"] for row in batch])
    if not torch.allclose(horizons, horizons[0].unsqueeze(0).expand_as(horizons)):
        raise ValueError("All samples in a batch must share the same horizon grid.")
    collated: dict[str, Any] = {
        "x": torch.stack([row["x"] for row in batch]),
        "horizons": horizons,
        "padding_mask": torch.stack([row["padding_mask"] for row in batch]),
        "time_mask": torch.stack([row["time_mask"] for row in batch]),
        "feature_mask": torch.stack([row["feature_mask"] for row in batch]),
        "target_x": torch.stack([row["target_x"] for row in batch]),
        "target_padding_mask": torch.stack([row["target_padding_mask"] for row in batch]),
        "target_time_mask": torch.stack([row["target_time_mask"] for row in batch]),
        "target_feature_mask": torch.stack([row["target_feature_mask"] for row in batch]),
        "target_semantic_feature_mask": torch.stack(
            [row["target_semantic_feature_mask"] for row in batch]
        ),
        "target_semantic_mask": torch.stack([row["target_semantic_mask"] for row in batch]),
        "target_required_mask": torch.stack([row["target_required_mask"] for row in batch]),
        "world_ids": torch.stack([row["world_id"] for row in batch]),
        "state_ids": torch.stack([row["state_id"] for row in batch]),
        "regime_ids": torch.stack([row["regime_id"] for row in batch]),
        "as_of_positions": torch.stack([row["as_of_position"] for row in batch]),
        "world_names": [row["world_name"] for row in batch],
        "as_of_timestamps": [row["as_of_timestamp"] for row in batch],
        "target_timestamps": [row["target_timestamps"] for row in batch],
    }
    collated["adjacency_pairs"] = temporal_adjacency_pairs(
        collated["as_of_positions"],
        collated["world_ids"],
    )
    return collated


def build_world_datasets(
    panels: Mapping[str, pd.DataFrame | np.ndarray | torch.Tensor],
    **kwargs: Any,
) -> dict[str, WorldJEPADataset]:
    """Build isolated per-world datasets for any subset of the five worlds."""

    options = dict(kwargs)
    world_names = tuple(str(name) for name in options.pop("world_names", tuple(panels)))
    return {
        str(name): WorldJEPADataset(
            panel,
            world_name=str(name),
            world_names=world_names,
            **options,
        )
        for name, panel in panels.items()
    }


world_collate = world_jepa_collate
