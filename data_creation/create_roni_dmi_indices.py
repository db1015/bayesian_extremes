#!/usr/bin/env python3
"""
Create monthly RONI and Indian Ocean Dipole (DMI) indices from ERA5 SST.

This script computes monthly SST anomalies for the canonical Niño 3.4,
tropical-mean, western tropical Indian Ocean (WTIO), and southeastern
tropical Indian Ocean (SETIO) regions, then derives:

    RONI = Niño 3.4 anomaly - tropical 20S--20N SST anomaly
    DMI  = WTIO anomaly - SETIO anomaly

Monthly anomalies are calculated relative to a 1991--2020 calendar-month
climatology. SST anomalies are equivalent in Kelvin and degrees Celsius.

The raw ERA5 SST files are external inputs and are not distributed with
this repository.

Example
-------
From the repository root:

    python data_creation/create_roni_dmi_indices.py \
        --sst-glob "/path/to/era5_sst/era5_sst_*.nc"

By default, the output CSV is written to:

    data/sst/roni_dmi_monthly_1950_2025.csv
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# REPOSITORY PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data"
    / "sst"
    / "roni_dmi_monthly_1950_2025.csv"
)


# ============================================================
# ANALYSIS SETTINGS
# ============================================================

START = "1950-01-01"
END = "2025-12-31"

CLIM_START = "1991-01-01"
CLIM_END = "2020-12-31"

# Canonical index regions.
# Longitudes are expressed on a -180 to 180 grid.
NINO34 = {
    "lat_min": -5.0,
    "lat_max": 5.0,
    "lon_min": -170.0,
    "lon_max": -120.0,
}

TROP20 = {
    "lat_min": -20.0,
    "lat_max": 20.0,
    "lon_min": -180.0,
    "lon_max": 180.0,
}

WTIO = {
    "lat_min": -10.0,
    "lat_max": 10.0,
    "lon_min": 50.0,
    "lon_max": 70.0,
}

SETIO = {
    "lat_min": -10.0,
    "lat_max": 0.0,
    "lon_min": 90.0,
    "lon_max": 110.0,
}


# ============================================================
# HELPERS
# ============================================================

def pick_sst_var(ds: xr.Dataset) -> xr.DataArray:
    """Identify the SST variable in an ERA5 dataset."""
    for variable in ds.data_vars:
        name = variable.lower()

        if (
            name in {"sst", "sea_surface_temperature"}
            or "sst" in name
            or "sea_surface_temperature" in name
        ):
            return ds[variable]

    # Fallback for simple single-variable files.
    return ds[list(ds.data_vars)[0]]


def fix_time_and_expver(ds: xr.Dataset) -> xr.Dataset:
    """Standardize ERA5 time coordinate and remove ``expver``."""
    if "valid_time" in ds.coords or "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    if "expver" in ds.dims:
        if (
            "expver" in ds.coords
            and np.any(ds["expver"].values == 1)
        ):
            ds = ds.sel(expver=1)
        else:
            ds = ds.isel(expver=0)

        ds = ds.drop_vars(
            "expver",
            errors="ignore",
        )

    if (
        "expver" in ds.coords
        and "expver" not in ds.dims
    ):
        ds = ds.drop_vars(
            "expver",
            errors="ignore",
        )

    return ds


def standardize_longitude(
    da: xr.DataArray,
) -> xr.DataArray:
    """Convert 0--360 longitude coordinates to -180--180 if necessary."""
    lon_name = (
        "longitude"
        if "longitude" in da.coords
        else "lon"
    )

    lon = da[lon_name]

    if float(lon.max()) > 180:
        lon_new = ((lon + 180) % 360) - 180

        da = (
            da.assign_coords(
                {lon_name: lon_new}
            )
            .sortby(lon_name)
        )

    return da


def subset_box(
    da: xr.DataArray,
    box: dict[str, float],
) -> xr.DataArray:
    """Subset a DataArray to one rectangular index region."""
    lat_name = (
        "latitude"
        if "latitude" in da.coords
        else "lat"
    )

    lon_name = (
        "longitude"
        if "longitude" in da.coords
        else "lon"
    )

    lat = da[lat_name]

    if lat[0] > lat[-1]:
        da = da.sel(
            {
                lat_name: slice(
                    box["lat_max"],
                    box["lat_min"],
                )
            }
        )
    else:
        da = da.sel(
            {
                lat_name: slice(
                    box["lat_min"],
                    box["lat_max"],
                )
            }
        )

    return da.sel(
        {
            lon_name: slice(
                box["lon_min"],
                box["lon_max"],
            )
        }
    )


def area_weighted_mean(
    da: xr.DataArray,
) -> xr.DataArray:
    """Calculate a cosine-latitude weighted spatial mean."""
    lat_name = (
        "latitude"
        if "latitude" in da.coords
        else "lat"
    )

    lon_name = (
        "longitude"
        if "longitude" in da.coords
        else "lon"
    )

    weights = np.cos(
        np.deg2rad(
            da[lat_name]
        )
    )

    return da.weighted(
        weights
    ).mean(
        dim=(
            lat_name,
            lon_name,
        )
    )


def monthly_region_series(
    sst: xr.DataArray,
    box: dict[str, float],
) -> pd.Series:
    """Return the area-weighted monthly SST series for one region."""
    region = subset_box(
        sst,
        box,
    )

    if (
        region.size == 0
        or region.sizes.get("time", 0) == 0
    ):
        raise ValueError(
            f"Region subset is empty for box: {box}"
        )

    return area_weighted_mean(
        region
    ).to_series()


# ============================================================
# INDEX CREATION
# ============================================================

def build_raw_region_means(
    input_files: list[str],
) -> pd.DataFrame:
    """
    Build monthly area-weighted SST means for all four index regions.
    """
    records = []

    for source_file in input_files:
        source_path = Path(source_file)

        print(
            f"Opening: {source_path.name}"
        )

        with xr.open_dataset(source_path) as ds:
            ds = fix_time_and_expver(ds)

            sst = pick_sst_var(ds)
            sst = standardize_longitude(sst)
            sst = sst.sel(
                time=slice(
                    START,
                    END,
                )
            )

            if sst.sizes.get("time", 0) == 0:
                print(
                    f"  skipped {source_path.name} "
                    f"(no overlap with {START}--{END})"
                )
                continue

            nino = monthly_region_series(
                sst,
                NINO34,
            )

            trop = monthly_region_series(
                sst,
                TROP20,
            )

            wtio = monthly_region_series(
                sst,
                WTIO,
            )

            setio = monthly_region_series(
                sst,
                SETIO,
            )

            frame = pd.concat(
                [
                    nino,
                    trop,
                    wtio,
                    setio,
                ],
                axis=1,
            )

            frame.columns = [
                "nino34",
                "trop20",
                "wtio",
                "setio",
            ]

            records.append(frame)

    if not records:
        raise RuntimeError(
            "No valid SST records were created from the input files."
        )

    raw = (
        pd.concat(records)
        .sort_index()
    )

    raw.index = (
        pd.to_datetime(raw.index)
        .to_period("M")
        .to_timestamp()
    )

    raw = raw.loc[
        "1950-01-01":"2025-12-01"
    ].copy()

    # Duplicate months should not occur in the source workflow.
    if raw.index.duplicated().any():
        duplicates = raw.index[
            raw.index.duplicated()
        ]

        raise RuntimeError(
            "Duplicate monthly timestamps were found in the ERA5 SST "
            f"inputs. Example duplicates: {duplicates[:5].tolist()}"
        )

    return raw


def calculate_monthly_anomalies(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate calendar-month SST anomalies relative to 1991--2020.
    """
    baseline = raw.loc[
        CLIM_START:CLIM_END
    ].copy()

    if baseline.empty:
        raise RuntimeError(
            "The requested 1991--2020 climatology period is absent "
            "from the input SST record."
        )

    climatology = (
        baseline
        .groupby(
            baseline.index.month
        )
        .mean()
    )

    anomalies = raw.copy()

    for month in range(1, 13):
        mask = (
            anomalies.index.month
            == month
        )

        anomalies.loc[mask, :] = (
            raw.loc[mask, :]
            - climatology.loc[
                month,
                :
            ].values
        )

    return anomalies


def create_indices(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Derive RONI and DMI from monthly regional SST anomalies.
    """
    anomalies = calculate_monthly_anomalies(
        raw
    )

    roni = (
        anomalies["nino34"]
        - anomalies["trop20"]
    )

    dmi = (
        anomalies["wtio"]
        - anomalies["setio"]
    )

    output = pd.DataFrame(
        {
            "time": roni.index,
            "RONI": roni.values,
            "DMI": dmi.values,
        }
    )

    return output


# ============================================================
# COMMAND LINE
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create monthly RONI and DMI indices from ERA5 SST."
        )
    )

    parser.add_argument(
        "--sst-glob",
        default=os.environ.get(
            "ERA5_SST_GLOB"
        ),
        help=(
            "Glob for ERA5 SST NetCDF files, e.g. "
            "'/path/to/era5_sst/era5_sst_*.nc'. "
            "May also be supplied through ERA5_SST_GLOB."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output CSV path. Default: "
            "<repository>/data/sst/roni_dmi_monthly_1950_2025.csv"
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    if not args.sst_glob:
        raise SystemExit(
            "No ERA5 SST input was specified.\n"
            "Use --sst-glob '/path/to/era5_sst_*.nc' "
            "or set ERA5_SST_GLOB."
        )

    input_files = sorted(
        glob.glob(args.sst_glob)
    )

    if not input_files:
        raise FileNotFoundError(
            f"No SST files matched: {args.sst_glob}"
        )

    output_path = (
        args.output
        .expanduser()
        .resolve()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("RONI / DMI INDEX CREATION")
    print("=" * 72)
    print(
        f"Input files       : {len(input_files)}"
    )
    print(
        f"Input glob        : {args.sst_glob}"
    )
    print(
        f"Analysis period   : {START} to {END}"
    )
    print(
        f"Climatology period: {CLIM_START} to {CLIM_END}"
    )
    print(
        f"Output            : {output_path}"
    )
    print("=" * 72)

    raw = build_raw_region_means(
        input_files
    )

    output = create_indices(
        raw
    )

    output.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nWrote: {output_path}"
    )

    print("\nFirst rows:")
    print(
        output.head().to_string(
            index=False
        )
    )

    print("\nLast rows:")
    print(
        output.tail().to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
