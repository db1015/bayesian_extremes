#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
Download daily-mean ERA5 pressure-level fields for the Arabian Peninsula region.

Usage:
    python download_era5_daily_means.py geopotential
    python download_era5_daily_means.py vertical_velocity
    python download_era5_daily_means.py specific_humidity
    python download_era5_daily_means.py u_component_of_wind
    python download_era5_daily_means.py v_component_of_wind
"""

import os
import sys
import zipfile
import calendar
from pathlib import Path

import cdsapi

# =========================
# User settings
# =========================
START_YEAR = 1980
END_YEAR = 2024

# AP bounding box padded by 5 degrees
LAT_MAX = 39
LAT_MIN = 5
LON_MIN = 29
LON_MAX = 65

# CDS area order: [north, west, south, east]
AREA = [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX]

OUT_ROOT = Path("../data/era5")
DATASET = "derived-era5-pressure-levels-daily-statistics"

VARIABLES = {
    "geopotential": "500",
    "vertical_velocity": "500",
    "specific_humidity": "925",
    "u_component_of_wind": "925",
    "v_component_of_wind": "925",
}

DAILY_STATISTIC = "daily_mean"
SUBDAILY_SAMPLING = "1_hourly"
TIME_ZONE = "utc+00:00"
SKIP_EXISTING = True


def month_days(year: int, month: int) -> list[str]:
    n_days = calendar.monthrange(year, month)[1]
    return [f"{day:02d}" for day in range(1, n_days + 1)]


def expected_nc_path(var: str, level: str, year: int, month: int) -> Path:
    out_dir = OUT_ROOT / var
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{var}_{level}_{year}_{month:02d}.nc"


def zip_target_path(var: str, level: str, year: int, month: int) -> Path:
    out_dir = OUT_ROOT / var
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{var}_{level}_{year}_{month:02d}.zip"


def extract_single_nc(zip_path: Path, final_nc_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        nc_members = [m for m in members if m.endswith(".nc")]

        if len(nc_members) == 0:
            raise RuntimeError(f"No .nc file found in {zip_path}")
        if len(nc_members) > 1:
            raise RuntimeError(f"Expected one .nc in {zip_path}, found {len(nc_members)}")

        tmp_extract_dir = zip_path.parent / f"tmp_extract_{zip_path.stem}"
        tmp_extract_dir.mkdir(exist_ok=True)

        zf.extract(nc_members[0], path=tmp_extract_dir)
        extracted = tmp_extract_dir / nc_members[0]

        os.replace(extracted, final_nc_path)

        try:
            parent = extracted.parent
            while parent != tmp_extract_dir and parent.exists():
                if not any(parent.iterdir()):
                    parent.rmdir()
                parent = parent.parent
            if tmp_extract_dir.exists() and not any(tmp_extract_dir.iterdir()):
                tmp_extract_dir.rmdir()
        except Exception:
            pass


def build_request(var: str, level: str, year: int, month: int) -> dict:
    return {
        "product_type": "reanalysis",
        "variable": [var],
        "pressure_level": [level],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "daily_statistic": DAILY_STATISTIC,
        "frequency": SUBDAILY_SAMPLING,
        "time_zone": TIME_ZONE,
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "zip",
    }


def main():
    if len(sys.argv) != 2:
        valid = ", ".join(VARIABLES.keys())
        raise SystemExit(f"Usage: python {sys.argv[0]} <variable>\nValid variables: {valid}")

    var = sys.argv[1]
    if var not in VARIABLES:
        valid = ", ".join(VARIABLES.keys())
        raise SystemExit(f"Unknown variable: {var}\nValid variables: {valid}")

    level = VARIABLES[var]
    client = cdsapi.Client()

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            final_nc = expected_nc_path(var, level, year, month)
            zip_path = zip_target_path(var, level, year, month)

            if SKIP_EXISTING and final_nc.exists():
                print(f"✓ exists, skipping: {final_nc}")
                continue

            request = build_request(var, level, year, month)

            print(f"Downloading {var} @ {level} hPa for {year}-{month:02d}")

            try:
                client.retrieve(DATASET, request, str(zip_path))
                extract_single_nc(zip_path, final_nc)
                zip_path.unlink(missing_ok=True)
                print(f"✓ saved: {final_nc}")
            except Exception as e:
                print(f"✗ failed for {var} @ {level} hPa {year}-{month:02d}: {e}")
                continue


if __name__ == "__main__":
    main()

