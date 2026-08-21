#!/usr/bin/env python3

from pathlib import Path
import re
import numpy as np
import xarray as xr

ERA5_ROOT = Path("../data/era5")
DERIVED_ROOT = Path("../data/era5_derived")
ANOM_ROOT = Path("../data/era5_anomalies")
CLIM_ROOT = Path("../data/era5_climatology")

ANOM_ROOT.mkdir(parents=True, exist_ok=True)
CLIM_ROOT.mkdir(parents=True, exist_ok=True)

VARIABLE_DIRS = {
    "geopotential": ERA5_ROOT / "geopotential",
    "vertical_velocity": ERA5_ROOT / "vertical_velocity",
    "specific_humidity": ERA5_ROOT / "specific_humidity",
    "u_component_of_wind": ERA5_ROOT / "u_component_of_wind",
    "v_component_of_wind": ERA5_ROOT / "v_component_of_wind",
    "wind_speed_925": DERIVED_ROOT / "wind_speed_925",
    "moisture_flux_u_925": DERIVED_ROOT / "moisture_flux_u_925",
    "moisture_flux_v_925": DERIVED_ROOT / "moisture_flux_v_925",
    "moisture_flux_mag_925": DERIVED_ROOT / "moisture_flux_mag_925",
}

FILE_RE = re.compile(r".*_(\d{4})_(\d{2})\.nc$")


def get_var_name(ds: xr.Dataset) -> str:
    if len(ds.data_vars) != 1:
        raise ValueError(f"Expected 1 variable, got {list(ds.data_vars)}")
    return list(ds.data_vars)[0]


def parse_year_month(path: Path):
    m = FILE_RE.match(path.name)
    if not m:
        raise ValueError(f"Could not parse year/month from {path.name}")
    return int(m.group(1)), int(m.group(2))


def open_da(path: Path) -> xr.DataArray:
    ds = xr.open_dataset(path)
    var_name = get_var_name(ds)
    da = ds[var_name].sortby("time")
    return da


for var_label, var_dir in VARIABLE_DIRS.items():
    print(f"\n===== Working on {var_label} =====")

    files = sorted(var_dir.glob("*.nc"))
    if not files:
        print(f"No files found in {var_dir}, skipping.")
        continue

    out_clim_dir = CLIM_ROOT / var_label
    out_anom_dir = ANOM_ROOT / var_label
    out_clim_dir.mkdir(parents=True, exist_ok=True)
    out_anom_dir.mkdir(parents=True, exist_ok=True)

    clim_file = out_clim_dir / f"{var_label}_monthly_climatology_1980_2024.nc"

    # -----------------------------
    # Pass 1: build monthly climatology
    # -----------------------------
    monthly_sum = {}
    monthly_count = {}

    for month in range(1, 13):
        monthly_sum[month] = None
        monthly_count[month] = 0

    for fp in files:
        year, month = parse_year_month(fp)
        print(f"Reading for climatology: {fp.name}")

        da = open_da(fp)
        this_mean = da.mean("time", skipna=True)

        if monthly_sum[month] is None:
            monthly_sum[month] = this_mean
        else:
            monthly_sum[month] = monthly_sum[month] + this_mean

        monthly_count[month] += 1

    clim_list = []
    for month in range(1, 13):
        if monthly_count[month] == 0:
            raise RuntimeError(f"No files found for month {month} in {var_label}")
        clim_month = (monthly_sum[month] / monthly_count[month]).expand_dims(month=[month])
        clim_list.append(clim_month)

    clim = xr.concat(clim_list, dim="month").rename(var_label)

    if not clim_file.exists():
        clim.to_netcdf(
            clim_file,
            encoding={clim.name: {"zlib": True, "complevel": 4}}
        )
        print(f"Saved climatology: {clim_file}")
    else:
        print(f"Climatology already exists: {clim_file}")

    # -----------------------------
    # Pass 2: save monthly anomalies
    # -----------------------------
    for fp in files:
        year, month = parse_year_month(fp)
        out_file = out_anom_dir / f"{var_label}_anom_{year}_{month:02d}.nc"

        if out_file.exists():
            print(f"Skipping existing anomaly file: {out_file.name}")
            continue

        print(f"Writing anomaly: {fp.name}")

        da = open_da(fp)
        clim_month = clim.sel(month=month).drop_vars("month", errors="ignore")
        anom = (da - clim_month).rename(f"{var_label}_anom")

        anom.to_netcdf(
            out_file,
            encoding={anom.name: {"zlib": True, "complevel": 4}}
        )

    print(f"Done with {var_label}")




