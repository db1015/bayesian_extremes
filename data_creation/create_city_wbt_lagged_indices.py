#!/usr/bin/env python3
"""
Create city-level wet-bulb temperature tables merged with lagged RONI/DMI indices.

This script extracts daily peak wet-bulb temperature from the repository's
``DailyPeakState`` NetCDF files at selected Arabian Peninsula cities, aggregates
the daily values to monthly statistics, constructs lagged RONI and DMI predictors,
and saves merged tables used for exploratory lag analysis and downstream model
preparation.

Inputs
------
- data/DailyPeakState/DailyPeakState-*.nc
- data/sst/roni_dmi_monthly_1950_2025.csv

Outputs
-------
- city_daily_wbt.csv
- city_monthly_wbt.csv
- city_monthly_wbt_JJAS.csv
- roni_dmi_monthly.csv
- roni_dmi_monthly_lagged_0_6.csv
- city_wbt_roni_dmi_lagged_allmonths.csv
- city_wbt_roni_dmi_lagged_JJAS.csv

By default, outputs are written under:
    data/wbt_sst_city_runs/

The script uses repository-relative paths and can be run from any directory.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# REPOSITORY PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

DEFAULT_WBT_GLOB = str(
    DATA_DIR
    / "DailyPeakState"
    / "DailyPeakState-*.nc"
)

DEFAULT_INDEX_CSV = (
    DATA_DIR
    / "sst"
    / "roni_dmi_monthly_1950_2025.csv"
)

DEFAULT_OUTPUT_DIR = (
    DATA_DIR
    / "wbt_sst_city_runs"
)


# ============================================================
# ANALYSIS SETTINGS
# ============================================================

CITIES = {
    "muscat": {"lat": 23.5880, "lon": 58.3829},
    "doha": {"lat": 25.2854, "lon": 51.5310},
    "dubai": {"lat": 25.2048, "lon": 55.2708},
    "jeddah": {"lat": 21.4858, "lon": 39.1925},
    "aden": {"lat": 12.7855, "lon": 45.0187},
}

WBT_VAR = "wbt_daily_peak"
MAX_LAG = 6
JJAS_MONTHS = [6, 7, 8, 9]


# ============================================================
# HELPERS
# ============================================================

def to_0360(lon: float) -> float:
    """Convert longitude to a 0--360 convention."""
    return lon % 360


def infer_coord_name(
    ds: xr.Dataset,
    candidates: list[str],
) -> str | None:
    """Return the first matching coordinate/dimension name."""
    for candidate in candidates:
        if candidate in ds.coords or candidate in ds.dims:
            return candidate

    return None


def standardize_time_coord(
    ds: xr.Dataset,
) -> tuple[xr.Dataset, str | None]:
    """Standardize ``day`` to ``time`` when necessary."""
    if "time" in ds.coords or "time" in ds.dims:
        return ds, "time"

    if "day" in ds.coords or "day" in ds.dims:
        ds = ds.rename({"day": "time"})
        return ds, "time"

    return ds, None


def open_one_file(path: str | Path) -> xr.Dataset:
    """
    Open one DailyPeakState file.

    ``h5netcdf`` is preferred for consistency with the analysis workflow,
    with a fallback to xarray's default backend.
    """
    try:
        return xr.open_dataset(
            path,
            engine="h5netcdf",
        )
    except Exception:
        return xr.open_dataset(path)


def detect_file_structure(
    sample_path: str | Path,
) -> tuple[str, str, bool]:
    """
    Detect latitude/longitude names and longitude convention from a sample file.
    """
    with open_one_file(sample_path) as sample:
        sample, time_name = standardize_time_coord(sample)

        if WBT_VAR not in sample.data_vars:
            raise ValueError(
                f"{WBT_VAR!r} not found in sample file "
                f"{Path(sample_path).name}."
            )

        lat_name = infer_coord_name(
            sample,
            ["lat", "latitude", "y"],
        )

        lon_name = infer_coord_name(
            sample,
            ["lon", "longitude", "x"],
        )

        if time_name is None:
            raise ValueError(
                "Could not identify a time coordinate in the "
                "DailyPeakState files."
            )

        if lat_name is None or lon_name is None:
            raise ValueError(
                "Could not infer latitude/longitude coordinate names."
            )

        use_0360 = (
            float(sample[lon_name].max()) > 180
        )

    return lat_name, lon_name, use_0360


# ============================================================
# DAILY CITY EXTRACTION
# ============================================================

def extract_city_daily_wbt(
    files: list[str],
) -> pd.DataFrame:
    """Extract daily peak WBT at the nearest grid cell for each city."""
    lat_name, lon_name, use_0360 = detect_file_structure(
        files[0]
    )

    print("Detected input structure:")
    print(f"  latitude coordinate : {lat_name}")
    print(f"  longitude coordinate: {lon_name}")
    print(f"  longitude 0--360    : {use_0360}")

    rows = []

    for index, source_file in enumerate(
        files,
        start=1,
    ):
        source_path = Path(source_file)

        if (
            index % 25 == 0
            or index == 1
            or index == len(files)
        ):
            print(
                f"Processing file {index}/{len(files)}: "
                f"{source_path.name}"
            )

        try:
            with open_one_file(source_path) as ds:
                ds, time_name = standardize_time_coord(ds)

                if time_name is None:
                    print(
                        f"  skipping {source_path.name}: "
                        "no time coordinate"
                    )
                    continue

                if WBT_VAR not in ds.data_vars:
                    print(
                        f"  skipping {source_path.name}: "
                        f"{WBT_VAR} missing"
                    )
                    continue

                # Re-detect coordinates in case individual files vary.
                this_lat_name = infer_coord_name(
                    ds,
                    ["lat", "latitude", "y"],
                )

                this_lon_name = infer_coord_name(
                    ds,
                    ["lon", "longitude", "x"],
                )

                if (
                    this_lat_name is None
                    or this_lon_name is None
                ):
                    print(
                        f"  skipping {source_path.name}: "
                        "could not identify lat/lon coordinates"
                    )
                    continue

                da = ds[WBT_VAR]

                this_use_0360 = (
                    float(ds[this_lon_name].max()) > 180
                )

                for city, metadata in CITIES.items():
                    lon = (
                        to_0360(metadata["lon"])
                        if this_use_0360
                        else metadata["lon"]
                    )

                    point = da.sel(
                        {
                            this_lat_name: metadata["lat"],
                            this_lon_name: lon,
                        },
                        method="nearest",
                    )

                    series = point.to_series()
                    frame = series.reset_index()

                    value_col = frame.columns[-1]
                    frame = frame.rename(
                        columns={
                            value_col: WBT_VAR
                        }
                    )

                    # Locate the time column after reset_index().
                    actual_time_col = next(
                        (
                            column
                            for column in ["time", "date", "day"]
                            if column in frame.columns
                        ),
                        None,
                    )

                    if actual_time_col is None:
                        actual_time_col = next(
                            (
                                column
                                for column in frame.columns
                                if np.issubdtype(
                                    frame[column].dtype,
                                    np.datetime64,
                                )
                            ),
                            None,
                        )

                    if actual_time_col is None:
                        print(
                            f"  skipping {city} in {source_path.name}: "
                            "no datetime column after extraction"
                        )
                        continue

                    keep_cols = [
                        actual_time_col,
                        WBT_VAR,
                    ]

                    if this_lat_name in frame.columns:
                        keep_cols.append(
                            this_lat_name
                        )

                    if this_lon_name in frame.columns:
                        keep_cols.append(
                            this_lon_name
                        )

                    frame = frame[
                        keep_cols
                    ].copy()

                    frame = frame.rename(
                        columns={
                            actual_time_col: "time",
                            this_lat_name: "latitude",
                            this_lon_name: "longitude",
                        }
                    )

                    frame["city"] = city

                    rows.append(frame)

        except Exception as error:
            print(
                f"  failed on {source_path.name}: {error}"
            )

    if not rows:
        raise RuntimeError(
            "No city WBT rows were extracted from the input files."
        )

    city_daily_wbt = pd.concat(
        rows,
        ignore_index=True,
    )

    city_daily_wbt["time"] = pd.to_datetime(
        city_daily_wbt["time"]
    )

    city_daily_wbt = (
        city_daily_wbt
        .sort_values(
            ["city", "time"]
        )
        .reset_index(drop=True)
    )

    duplicates = city_daily_wbt.duplicated(
        subset=["city", "time"]
    )

    if duplicates.any():
        raise RuntimeError(
            "Duplicate city/date rows were found after combining "
            "DailyPeakState files."
        )

    return city_daily_wbt


# ============================================================
# MONTHLY WBT METRICS
# ============================================================

def create_monthly_wbt_tables(
    city_daily_wbt: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate daily city WBT to monthly mean, p95, and p99."""
    data = city_daily_wbt.copy()

    data["ym"] = (
        data["time"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    city_monthly_wbt = (
        data
        .groupby(
            ["city", "ym"],
            as_index=False,
        )
        .agg(
            wbt_mean=(
                WBT_VAR,
                "mean",
            ),
            wbt_p95=(
                WBT_VAR,
                lambda values: np.nanpercentile(
                    values,
                    95,
                ),
            ),
            wbt_p99=(
                WBT_VAR,
                lambda values: np.nanpercentile(
                    values,
                    99,
                ),
            ),
            n_days=(
                WBT_VAR,
                "count",
            ),
        )
    )

    city_monthly_wbt["year"] = (
        city_monthly_wbt["ym"].dt.year
    )

    city_monthly_wbt["month"] = (
        city_monthly_wbt["ym"].dt.month
    )

    city_monthly_wbt_jjas = city_monthly_wbt[
        city_monthly_wbt["month"].isin(
            JJAS_MONTHS
        )
    ].copy()

    return (
        city_monthly_wbt,
        city_monthly_wbt_jjas,
    )


# ============================================================
# RONI / DMI LAGS
# ============================================================

def load_index_table(
    index_csv: Path,
) -> pd.DataFrame:
    """Load monthly RONI/DMI indices and standardize the month coordinate."""
    if not index_csv.exists():
        raise FileNotFoundError(
            f"RONI/DMI CSV not found: {index_csv}"
        )

    idx = pd.read_csv(index_csv)

    year_col = next(
        (
            column
            for column in idx.columns
            if column.lower() == "year"
        ),
        None,
    )

    month_col = next(
        (
            column
            for column in idx.columns
            if column.lower() == "month"
        ),
        None,
    )

    roni_col = next(
        (
            column
            for column in idx.columns
            if "roni" in column.lower()
        ),
        None,
    )

    dmi_col = next(
        (
            column
            for column in idx.columns
            if "dmi" in column.lower()
        ),
        None,
    )

    if roni_col is None or dmi_col is None:
        raise ValueError(
            "Could not identify RONI/DMI columns in the index CSV."
        )

    if (
        year_col is not None
        and month_col is not None
    ):
        idx["ym"] = pd.to_datetime(
            {
                "year": idx[year_col].astype(int),
                "month": idx[month_col].astype(int),
                "day": 1,
            }
        )

    else:
        time_col = next(
            (
                column
                for column in idx.columns
                if column.lower()
                in {"time", "date", "datetime"}
            ),
            None,
        )

        if time_col is None:
            raise ValueError(
                "Could not infer a date column in the RONI/DMI CSV."
            )

        idx["ym"] = (
            pd.to_datetime(
                idx[time_col]
            )
            .dt.to_period("M")
            .dt.to_timestamp()
        )

    idx_monthly = (
        idx[
            [
                "ym",
                roni_col,
                dmi_col,
            ]
        ]
        .rename(
            columns={
                roni_col: "RONI",
                dmi_col: "DMI",
            }
        )
        .sort_values("ym")
        .drop_duplicates("ym")
        .reset_index(drop=True)
    )

    return idx_monthly


def add_index_lags(
    idx_monthly: pd.DataFrame,
) -> pd.DataFrame:
    """Create monthly RONI/DMI lags from 0 through MAX_LAG."""
    idx_lagged = idx_monthly.copy()

    for variable in [
        "RONI",
        "DMI",
    ]:
        for lag in range(
            MAX_LAG + 1
        ):
            idx_lagged[
                f"{variable}_lag{lag}"
            ] = idx_lagged[
                variable
            ].shift(lag)

    return idx_lagged


# ============================================================
# COMMAND LINE
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create city WBT tables merged with lagged "
            "RONI and DMI indices."
        )
    )

    parser.add_argument(
        "--wbt-glob",
        default=DEFAULT_WBT_GLOB,
        help=(
            "Glob for DailyPeakState NetCDF files. "
            "Default: <repository>/data/DailyPeakState/DailyPeakState-*.nc"
        ),
    )

    parser.add_argument(
        "--index-csv",
        type=Path,
        default=DEFAULT_INDEX_CSV,
        help=(
            "Monthly RONI/DMI CSV."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Output directory for generated city/index tables."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    files = sorted(
        glob.glob(
            args.wbt_glob
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No DailyPeakState files matched: {args.wbt_glob}"
        )

    index_csv = (
        args.index_csv
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("CITY WBT + LAGGED RONI/DMI TABLE CREATION")
    print("=" * 72)
    print(
        f"DailyPeakState files: {len(files)}"
    )
    print(
        f"Input glob          : {args.wbt_glob}"
    )
    print(
        f"RONI/DMI CSV        : {index_csv}"
    )
    print(
        f"Maximum lag         : {MAX_LAG} months"
    )
    print(
        f"Output directory    : {output_dir}"
    )
    print("=" * 72)

    city_daily_wbt = extract_city_daily_wbt(
        files
    )

    (
        city_monthly_wbt,
        city_monthly_wbt_jjas,
    ) = create_monthly_wbt_tables(
        city_daily_wbt
    )

    idx_monthly = load_index_table(
        index_csv
    )

    idx_lagged = add_index_lags(
        idx_monthly
    )

    merged_allmonths = city_monthly_wbt.merge(
        idx_lagged,
        on="ym",
        how="inner",
    )

    merged_jjas = city_monthly_wbt_jjas.merge(
        idx_lagged,
        on="ym",
        how="inner",
    )

    outputs = {
        "city_daily_wbt.csv":
            city_daily_wbt,

        "city_monthly_wbt.csv":
            city_monthly_wbt,

        "city_monthly_wbt_JJAS.csv":
            city_monthly_wbt_jjas,

        "roni_dmi_monthly.csv":
            idx_monthly,

        "roni_dmi_monthly_lagged_0_6.csv":
            idx_lagged,

        "city_wbt_roni_dmi_lagged_allmonths.csv":
            merged_allmonths,

        "city_wbt_roni_dmi_lagged_JJAS.csv":
            merged_jjas,
    }

    print("\nSaving:")

    for filename, table in outputs.items():
        path = (
            output_dir
            / filename
        )

        table.to_csv(
            path,
            index=False,
        )

        print(
            f"  {path.name}"
        )

    print(
        "\nCompleted city WBT / teleconnection table creation."
    )


if __name__ == "__main__":
    main()
