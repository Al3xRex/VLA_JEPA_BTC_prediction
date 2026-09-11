from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from database_interraction import (
    duckdb_connection,
    ensure_macro_series_table,
    load_table_as_dataframe,
    upsert_dataframe,
)
from get_data.get_fred import DB_PATH, TABLE_NAME

MANUAL_DIR = Path(__file__).resolve().parent.parent / "manual"
SOURCE_NAME = "manual"
KEY_COLUMNS = ["date", "source", "series_id"]


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return normalized.strip("_") or "series"


def _normalize_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame(columns=["date", "source", "series_id", "series_name", "value"])

    normalized = dataframe.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], utc=True, errors="coerce").dt.date
    normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")
    normalized["source"] = SOURCE_NAME
    normalized = normalized[["date", "source", "series_id", "series_name", "value"]]
    normalized = normalized.dropna(subset=["date", "value"])
    normalized = normalized.drop_duplicates(
        subset=["date", "source", "series_id"],
        keep="last",
    )
    return normalized


def _parse_highcharts_json(file_path: Path) -> pd.DataFrame:
    payload = json.loads(file_path.read_text())
    if isinstance(payload, dict):
        payload = [payload]

    rows: list[dict[str, Any]] = []
    file_prefix = _slugify(file_path.stem)
    for series in payload:
        series_name = str(series.get("name") or file_path.stem).strip()
        series_id = f"{file_prefix}__{_slugify(series_name)}"
        for point in series.get("data", []):
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            rows.append(
                {
                    "date": pd.to_datetime(point[0], unit="ms", utc=True),
                    "series_id": series_id,
                    "series_name": series_name,
                    "value": point[1],
                }
            )

    return _normalize_dataframe(pd.DataFrame(rows))


def _parse_manual_csv(file_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(file_path)
    if dataframe.empty or len(dataframe.columns) < 2:
        return pd.DataFrame(columns=["date", "source", "series_id", "series_name", "value"])

    date_column = dataframe.columns[0]
    value_columns = [column for column in dataframe.columns if column != date_column]
    melted = dataframe.melt(
        id_vars=[date_column],
        value_vars=value_columns,
        var_name="series_name",
        value_name="value",
    ).rename(columns={date_column: "date"})

    file_prefix = _slugify(file_path.stem)
    melted["series_id"] = melted["series_name"].map(
        lambda series_name: f"{file_prefix}__{_slugify(str(series_name))}"
    )
    return _normalize_dataframe(melted)


def load_manual_file(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    if path.suffix.lower() == ".csv":
        return _parse_manual_csv(path)
    return _parse_highcharts_json(path)


def _load_existing_manual_rows(connection, table_name: str) -> pd.DataFrame:
    dataframe = load_table_as_dataframe(
        connection,
        table_name,
        columns=["date", "source", "series_id", "series_name", "value"],
        filters={"source": SOURCE_NAME},
        order_by=["date", "source", "series_id"],
    )
    if dataframe.empty:
        return pd.DataFrame(columns=["date", "source", "series_id", "series_name", "value"])

    dataframe = dataframe.copy()
    dataframe["date"] = pd.to_datetime(dataframe["date"], utc=True, errors="coerce").dt.date
    dataframe["value"] = pd.to_numeric(dataframe["value"], errors="coerce")
    dataframe["source"] = SOURCE_NAME
    dataframe = dataframe.dropna(subset=["date", "value"])
    return dataframe[["date", "source", "series_id", "series_name", "value"]]


def _manual_rows_to_upsert(
    parsed: pd.DataFrame,
    existing: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if parsed.empty:
        return parsed, {"new_rows": 0, "changed_rows": 0, "unchanged_rows": 0}
    if existing.empty:
        return parsed, {
            "new_rows": int(len(parsed)),
            "changed_rows": 0,
            "unchanged_rows": 0,
        }

    comparison = parsed.merge(
        existing,
        on=KEY_COLUMNS,
        how="left",
        suffixes=("", "_existing"),
        indicator=True,
    )
    is_new = comparison["_merge"].eq("left_only")
    is_existing = comparison["_merge"].eq("both")
    value_changed = is_existing & ~comparison["value"].eq(
        comparison["value_existing"]
    )
    series_name_changed = is_existing & ~comparison[
        "series_name"
    ].fillna("").eq(comparison["series_name_existing"].fillna(""))
    should_upsert = is_new | value_changed | series_name_changed

    changed_rows = comparison.loc[should_upsert, parsed.columns].copy()
    counts = {
        "new_rows": int(is_new.sum()),
        "changed_rows": int((should_upsert & ~is_new).sum()),
        "unchanged_rows": int((~should_upsert).sum()),
    }
    return changed_rows, counts


def update_manual_macro_data(
    manual_dir: str | Path = MANUAL_DIR,
    db_path: str = DB_PATH,
    table_name: str = TABLE_NAME,
) -> dict[str, Any]:
    manual_path = Path(manual_dir)
    if not manual_path.exists() or not manual_path.is_dir():
        print(f"[WARN] Manual directory not found: {manual_path}")
        return {
            "source": SOURCE_NAME,
            "db_path": db_path,
            "table": table_name,
            "file_count": 0,
            "rows_upserted": 0,
            "skipped": True,
        }

    frames: list[pd.DataFrame] = []
    processed_files: list[str] = []

    for file_path in sorted(manual_path.iterdir()):
        if not file_path.is_file() or file_path.name.lower() == "getdata.md":
            continue

        try:
            dataframe = load_manual_file(file_path)
        except Exception as exc:
            print(f"[ERR] Failed to parse {file_path.name}: {exc}")
            continue
        if dataframe.empty:
            print(f"[WARN] No usable manual data found in {file_path.name}.")
            continue

        processed_files.append(file_path.name)
        frames.append(dataframe)
        print(f"[OK] Loaded manual file {file_path.name} ({len(dataframe)} rows)")

    if not frames:
        print("[WARN] No manual macro data was loaded.")
        return {
            "source": SOURCE_NAME,
            "db_path": db_path,
            "table": table_name,
            "file_count": 0,
            "rows_upserted": 0,
        }

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=KEY_COLUMNS,
        keep="last",
    )

    with duckdb_connection(db_path) as connection:
        ensure_macro_series_table(connection, table_name)
        existing = _load_existing_manual_rows(connection, table_name)
        rows_to_upsert, delta_counts = _manual_rows_to_upsert(combined, existing)
        if rows_to_upsert.empty:
            rows_upserted = 0
        else:
            rows_upserted = upsert_dataframe(
                connection,
                table_name,
                rows_to_upsert,
                key_columns=KEY_COLUMNS,
            )

    if rows_upserted:
        print(f"[INFO] Manual macro upsert complete. Rows written: {rows_upserted}")
    else:
        print("[INFO] Manual macro local files already imported. No rows written.")
    return {
        "source": SOURCE_NAME,
        "db_path": db_path,
        "table": table_name,
        "file_count": len(processed_files),
        "files": processed_files,
        "rows_parsed": int(len(combined)),
        "existing_rows": int(len(existing)),
        **delta_counts,
        "rows_upserted": int(rows_upserted),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load manual macro files into the shared macro DuckDB table.",
    )
    parser.add_argument(
        "--manual-dir",
        default=str(MANUAL_DIR),
        help="Directory containing manual macro files.",
    )
    parser.add_argument(
        "--db-path",
        default=DB_PATH,
        help="DuckDB file that stores macro series.",
    )
    parser.add_argument(
        "--table",
        default=TABLE_NAME,
        help="DuckDB table used for macro series.",
    )
    args = parser.parse_args()

    summary = update_manual_macro_data(
        manual_dir=args.manual_dir,
        db_path=args.db_path,
        table_name=args.table,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
