#!/usr/bin/env python3
"""
Download ERA5 pressure-level data from the Google ARCO raw bucket,
subset to the Arabian Peninsula, compute UTC daily means, and save
monthly NetCDFs.

Examples
--------
Test one variable/month:
    python data_scrape_era5_raw_gcs.py test --year 2000 --month 1 --var geopotential

Run full period for one variable:
    python data_scrape_era5_raw_gcs.py full --var geopotential

Run custom range for one variable:
    python data_scrape_era5_raw_gcs.py full --var specific_humidity --start-year 1980 --end-year 2024
"""

from __future__ import annotations

import argparse
import calendar
from io import BytesIO
from pathlib import Path

import gcsfs
import xarray as xr

# =========================
# User settings
# =========================
DEFAULT_START_YEAR = 1980
DEFAULT_END_YEAR = 2024

LAT_MAX = 39
LAT_MIN = 5
LON_MIN = 29
LON_MAX = 65

OUT_ROOT = Path("../data/era5")
SKIP_EXISTING = True

RAW_ROOT = "gcp-public-data-arco-era5/raw/date-variable-pressure_level"

# Requested output name -> search aliases + level
SPECS = {
    "geopotential": {
        "file_aliases": ["geopotential", "z", "129"],
        "data_aliases": ["z", "geopotential"],
        "level": 500,
    },
    "vertical_velocity": {
        "file_aliases": ["vertical_velocity", "w", "135"],
        "data_aliases": ["w", "vertical_velocity"],
        "level": 500,
    },
    "specific_humidity": {
        "file_aliases": ["specific_humidity", "q", "133"],
        "data_aliases": ["q", "specific_humidity"],
        "level": 925,
    },
    "u_component_of_wind": {
        "file_aliases": ["u_component_of_wind", "u", "131"],
        "data_aliases": ["u", "u_component_of_wind"],
        "level": 925,
    },
    "v_component_of_wind": {
        "file_aliases": ["v_component_of_wind", "v", "132"],
        "data_aliases": ["v", "v_component_of_wind"],
        "level": 925,
    },
}

FS = gcsfs.GCSFileSystem(token="anon")


# =========================
# Helpers
# =========================
def path_matches_var(path_str, spec):
    p = Path(path_str)
    parent = p.parent.name.lower()
    filename = p.name.lower()

    return (
        parent in [a.lower() for a in spec["file_aliases"]]
        and filename == f"{spec['level']}.nc"
    )
    
def output_path(var_name: str, level: int, year: int, month: int) -> Path:
    out_dir = OUT_ROOT / var_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{var_name}_{level}_{year}_{month:02d}.nc"


def raw_day_prefix(year: int, month: int, day: int) -> str:
    return f"{RAW_ROOT}/{year:04d}/{month:02d}/{day:02d}"


def detect_data_var(ds: xr.Dataset, aliases: list[str]) -> str:
    for name in aliases:
        if name in ds.data_vars:
            return name
    raise ValueError(
        f"Could not find any of {aliases} in dataset vars: {list(ds.data_vars)}"
    )


def detect_coord(ds: xr.Dataset, candidates: list[str], label: str) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError(f"Could not find {label} among {candidates}")


def month_time_bounds(year: int, month: int) -> tuple[str, str]:
    ndays = calendar.monthrange(year, month)[1]
    start = f"{year}-{month:02d}-01T00:00:00"
    end = f"{year}-{month:02d}-{ndays:02d}T23:59:59"
    return start, end



def find_raw_file(year: int, month: int, day: int, var_name: str) -> str:
    """
    Find the raw ERA5 pressure-level file for a given day/variable.

    Returns a gs:// URL.
    """
    prefix = raw_day_prefix(year, month, day)
    spec = SPECS[var_name]
    aliases = set(a.lower() for a in spec["file_aliases"])
    level_file = f"{spec['level']}.nc"

    entries = FS.ls(prefix, detail=False)

    for entry in entries:
        var_dir = Path(entry).name.lower()
        if var_dir not in aliases:
            continue

        try:
            subfiles = FS.ls(entry, detail=False)
        except Exception:
            continue

        for sf in subfiles:
            if Path(sf).name.lower() == level_file:
                return "gs://" + sf

    raise FileNotFoundError(
        f"No raw file found for {var_name} on {year}-{month:02d}-{day:02d} "
        f"under gs://{prefix}"
    )


def open_remote_file(url: str) -> xr.Dataset:
    """
    Open a remote GCS raw file from bytes in memory.

    ARCO raw pressure-level files here are NetCDF (.nc), and the most reliable
    way in this workflow is to read bytes with gcsfs and open with h5netcdf
    or scipy.
    """
    path = url.replace("gs://", "")

    with FS.open(path, "rb") as f:
        data = f.read()

    bio = BytesIO(data)

    # Try h5netcdf first
    try:
        return xr.open_dataset(bio, engine="h5netcdf")
    except Exception:
        bio.seek(0)

    # Then scipy
    try:
        return xr.open_dataset(bio, engine="scipy")
    except Exception:
        bio.seek(0)

    raise ValueError(f"Could not open remote file as NetCDF: {url}")


def subset_one_day(year: int, month: int, day: int, var_name: str) -> xr.DataArray:
    spec = SPECS[var_name]
    url = find_raw_file(year, month, day, var_name)

    print(f"Opening {url}")
    ds = open_remote_file(url)

    data_var = detect_data_var(ds, spec["data_aliases"])
    lat_coord = detect_coord(ds, ["latitude", "lat"], "latitude")
    lon_coord = detect_coord(ds, ["longitude", "lon"], "longitude")

    da = ds[data_var]

    # Select level if there is a pressure coordinate
    possible_level_coords = ["isobaricInhPa", "pressure_level", "level"]
    level_coord = None
    for c in possible_level_coords:
        if c in da.coords or c in da.dims:
            level_coord = c
            break

    if level_coord is not None:
        da = da.sel({level_coord: spec["level"]})

    # Handle descending latitude
    lat_vals = da[lat_coord].values
    if lat_vals[0] > lat_vals[-1]:
        lat_slice = slice(LAT_MAX, LAT_MIN)
    else:
        lat_slice = slice(LAT_MIN, LAT_MAX)

    da = da.sel(
        {
            lat_coord: lat_slice,
            lon_coord: slice(LON_MIN, LON_MAX),
        }
    )

    return da

def process_month(var_name: str, year: int, month: int) -> None:
    spec = SPECS[var_name]
    out_file = output_path(var_name, spec["level"], year, month)

    if SKIP_EXISTING and out_file.exists():
        print(f"✓ exists, skipping: {out_file}")
        return

    print(f"\n=== Processing {var_name} {spec['level']} hPa for {year}-{month:02d} ===")

    ndays = calendar.monthrange(year, month)[1]
    daily_arrays = []

    for day in range(1, ndays + 1):
        try:
            da = subset_one_day(year, month, day, var_name)
            daily_arrays.append(da)
        except Exception as e:
            print(f"✗ failed for {year}-{month:02d}-{day:02d}: {e}")

    if not daily_arrays:
        raise RuntimeError(f"No daily arrays were collected for {var_name} {year}-{month:02d}")

    ds_month = xr.concat(daily_arrays, dim="time").sortby("time")

    # If source is hourly/subdaily, compute UTC daily means
    if ds_month.sizes.get("time", 0) > ndays:
        da_out = ds_month.resample(time="1D").mean()
    else:
        da_out = ds_month

    da_out = da_out.rename(var_name)

    # Drop scalar level coords if present
    for c in ["isobaricInhPa", "pressure_level", "level"]:
        if c in da_out.coords:
            try:
                da_out = da_out.drop_vars(c, errors="ignore")
            except Exception:
                pass

    encoding = {var_name: {"zlib": True, "complevel": 4}}
    da_out.to_netcdf(out_file, encoding=encoding)
    print(f"✓ saved: {out_file}")


def run_test(year: int, month: int, var_name: str) -> None:
    process_month(var_name, year, month)
    print("\nTest run complete.")


def run_full(start_year: int, end_year: int, var_name: str) -> None:
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            process_month(var_name, year, month)
    print("\nFull run complete.")


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create monthly ERA5 pressure-level files from Google ARCO raw bucket."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_test = subparsers.add_parser("test", help="Run one test month for one variable")
    p_test.add_argument("--year", type=int, required=True)
    p_test.add_argument("--month", type=int, required=True)
    p_test.add_argument("--var", choices=sorted(SPECS.keys()), required=True)

    p_full = subparsers.add_parser("full", help="Run full period for one variable")
    p_full.add_argument("--var", choices=sorted(SPECS.keys()), required=True)
    p_full.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    p_full.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "test":
        if not (1 <= args.month <= 12):
            raise ValueError("Month must be between 1 and 12.")
        run_test(args.year, args.month, args.var)

    elif args.mode == "full":
        if args.start_year > args.end_year:
            raise ValueError("start-year must be <= end-year.")
        run_full(args.start_year, args.end_year, args.var)


if __name__ == "__main__":
    main()


