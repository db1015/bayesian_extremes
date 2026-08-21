#!/usr/bin/env python3
"""
Create basin-scale ERA5 sea-surface-temperature (SST) anomalies.

This script masks ERA5 SST fields to the marine basins used in the
Arabian Peninsula humid-heat analysis and computes monthly SST anomalies
relative to a 1991--2020 climatology.

The raw ERA5 SST files are treated as an external input and are not
distributed with this repository. Basin polygons are expected under the
repository ``data/shapefiles`` directory.

Workflow
--------
1. Open each ERA5 SST NetCDF file.
2. Standardize the time, longitude, and ERA5 ``expver`` dimensions.
3. Spatially subset and mask SST to each basin polygon.
4. Write temporary masked NetCDF pieces.
5. Concatenate the pieces across time.
6. Compute monthly SST anomalies relative to 1991--2020.
7. Save one anomaly NetCDF per basin.

Example
-------
From the repository root:

    python data_creation/create_basin_sst_anomalies.py \
        --sst-glob "/path/to/era5_sst/era5_sst_*.nc"

Optional arguments allow custom shapefile, output, and worker locations.

Notes
-----
The script preserves the basin-layer configuration used in the original
analysis. If a shapefile contains multiple features, all features are
unioned into one basin geometry before masking.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
from pathlib import Path

import dask
import geopandas as gpd
import numpy as np
import regionmask
import xarray as xr
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster


# ============================================================
# REPOSITORY PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SHAPE_DIR = (
    REPO_ROOT
    / "data"
    / "shapefiles"
)

DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "data"
    / "sst"
    / "basin_anoms"
)


# ============================================================
# ANALYSIS SETTINGS
# ============================================================

START = "1950-01-01"
END = "2025-12-31"

CLIM_START = "1991-01-01"
CLIM_END = "2020-12-31"

PAD = 0.5

# These are the basin polygon files used in the original analysis.
# Paths are interpreted relative to --shape-dir.
BASIN_LAYER_RELATIVE_PATHS = {
    "arabian_gulf": Path("ecoregions") / "ecoregions.shp",
    "gulf_oman": Path("iho") / "iho.shp",
    "red_and_aden": Path("provinces") / "provinces.shp",
}


# ============================================================
# HELPERS
# ============================================================

def fix_time_and_expver(ds: xr.Dataset) -> xr.Dataset:
    """Standardize ERA5 time coordinate and remove the ``expver`` dimension."""
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


def standardize_longitude(
    da: xr.DataArray,
) -> xr.DataArray:
    """Convert 0--360 longitudes to -180--180 when necessary."""
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


def subset_bbox(
    da: xr.DataArray,
    bounds,
) -> xr.DataArray:
    """Subset a DataArray to a geographic bounding box."""
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

    minx, miny, maxx, maxy = bounds
    lat = da[lat_name]

    if lat[0] > lat[-1]:
        return da.sel(
            {
                lat_name: slice(maxy, miny),
                lon_name: slice(minx, maxx),
            }
        )

    return da.sel(
        {
            lat_name: slice(miny, maxy),
            lon_name: slice(minx, maxx),
        }
    )


def make_mask(
    da_subset_2d: xr.DataArray,
    geom,
    basin_name: str,
) -> xr.DataArray:
    """Create a regionmask mask for one basin polygon."""
    lat_name = (
        "latitude"
        if "latitude" in da_subset_2d.coords
        else "lat"
    )

    lon_name = (
        "longitude"
        if "longitude" in da_subset_2d.coords
        else "lon"
    )

    regions = regionmask.Regions(
        [geom],
        names=[basin_name],
        abbrevs=[basin_name],
    )

    return regions.mask(
        da_subset_2d[lon_name],
        da_subset_2d[lat_name],
    )


def load_basin_geometry(
    shapefile_path: Path,
):
    """Read one basin shapefile and return a WGS84 union geometry."""
    if not shapefile_path.exists():
        raise FileNotFoundError(
            f"Basin shapefile not found: {shapefile_path}"
        )

    gdf = gpd.read_file(shapefile_path)

    if len(gdf) == 0:
        raise ValueError(
            f"{shapefile_path}: GeoDataFrame contains zero features."
        )

    if gdf.crs is None:
        raise ValueError(
            f"{shapefile_path}: shapefile CRS is missing."
        )

    gdf = gdf.to_crs("EPSG:4326")

    if len(gdf) == 1:
        return gdf.geometry.iloc[0]

    # GeoPandas/Shapely compatibility across versions.
    try:
        return gdf.geometry.union_all()
    except AttributeError:
        return gdf.geometry.unary_union


# ============================================================
# BASIN PROCESSING
# ============================================================

def build_basin_anomalies(
    basin_name: str,
    shapefile_path: Path,
    input_files: list[str],
    output_dir: Path,
    temp_root: Path,
) -> Path:
    """Build one basin's monthly SST-anomaly NetCDF file."""
    print(f"\n{'=' * 72}")
    print(f"BASIN: {basin_name}")
    print("=" * 72)

    geom = load_basin_geometry(
        shapefile_path
    )

    minx, miny, maxx, maxy = geom.bounds

    bounds = (
        minx - PAD,
        miny - PAD,
        maxx + PAD,
        maxy + PAD,
    )

    basin_tmp_dir = (
        temp_root
        / f"{basin_name}_pieces"
    )

    basin_tmp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove old temporary pieces from previous runs.
    for old_file in basin_tmp_dir.glob("*.nc"):
        old_file.unlink()

    tmp_paths = []

    # --------------------------------------------------------
    # 1. Create one masked temporary piece per source file.
    # --------------------------------------------------------

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
                time=slice(START, END)
            )

            if sst.sizes.get("time", 0) == 0:
                print(
                    f"  skipped {source_path.name} "
                    f"(no overlap with {START}--{END})"
                )
                continue

            sub = subset_bbox(
                sst,
                bounds,
            )

            lat_name = (
                "latitude"
                if "latitude" in sub.coords
                else "lat"
            )

            lon_name = (
                "longitude"
                if "longitude" in sub.coords
                else "lon"
            )

            if (
                sub.sizes.get(lat_name, 0) == 0
                or sub.sizes.get(lon_name, 0) == 0
            ):
                raise ValueError(
                    f"{basin_name}: bbox subset returned an empty grid "
                    f"for {source_path.name}.\n"
                    f"Bounds: {bounds}\n"
                    f"Longitude range: "
                    f"{float(sst[lon_name].min())} to "
                    f"{float(sst[lon_name].max())}\n"
                    f"Latitude range: "
                    f"{float(sst[lat_name].min())} to "
                    f"{float(sst[lat_name].max())}"
                )

            # Region mask only needs one spatial slice.
            mask = make_mask(
                sub.isel(
                    time=0,
                    drop=True,
                ),
                geom,
                basin_name,
            )

            sub_masked = sub.where(
                mask == 0
            )

            piece = xr.Dataset(
                {
                    "sst": sub_masked
                }
            )

            output_piece = (
                basin_tmp_dir
                / (
                    f"{basin_name}__"
                    f"{source_path.stem}.nc"
                )
            )

            piece.to_netcdf(
                output_piece
            )

            tmp_paths.append(
                str(output_piece)
            )

            print(
                f"  wrote piece: {output_piece.name}"
            )

    if not tmp_paths:
        raise RuntimeError(
            f"{basin_name}: no valid SST pieces were created."
        )

    # --------------------------------------------------------
    # 2. Concatenate pieces and compute anomalies.
    # --------------------------------------------------------

    ds_basin = xr.open_mfdataset(
        tmp_paths,
        combine="by_coords",
        parallel=True,
        chunks="auto",
    )

    try:
        sst_basin = ds_basin["sst"]

        climatology = (
            sst_basin
            .sel(
                time=slice(
                    CLIM_START,
                    CLIM_END,
                )
            )
            .groupby("time.month")
            .mean("time")
        )

        sst_anomaly = (
            sst_basin.groupby("time.month")
            - climatology
        ).rename("sst_anom")

        print(
            "Computing SST anomalies..."
        )

        with ProgressBar():
            sst_anomaly_loaded = (
                sst_anomaly
                .astype("float32")
                .load()
            )

    finally:
        ds_basin.close()

    output = xr.Dataset(
        {
            "sst_anom": sst_anomaly_loaded
        }
    )

    output.attrs.update(
        {
            "description": (
                "ERA5 monthly sea-surface-temperature anomalies "
                f"for the {basin_name} basin."
            ),
            "analysis_period": f"{START} to {END}",
            "climatology_period": (
                f"{CLIM_START} to {CLIM_END}"
            ),
            "anomaly_definition": (
                "Monthly SST minus the corresponding "
                "1991--2020 calendar-month climatology."
            ),
            "created_by": (
                "create_basin_sst_anomalies.py"
            ),
        }
    )

    output["sst_anom"].attrs.update(
        {
            "long_name": (
                "Sea-surface-temperature anomaly"
            ),
            "units": "K",
        }
    )

    output_file = (
        output_dir
        / (
            f"era5_sst_anom_{basin_name}_"
            "1950_2025.nc"
        )
    )

    encoding = {
        "sst_anom": {
            "dtype": "float32",
        }
    }

    print(
        f"Writing: {output_file}"
    )

    with ProgressBar():
        output.to_netcdf(
            output_file,
            encoding=encoding,
        )

    print(
        f"Finished: {output_file.name}"
    )

    return output_file


# ============================================================
# COMMAND LINE
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create basin-scale ERA5 SST anomalies "
            "for the Arabian Peninsula analysis."
        )
    )

    parser.add_argument(
        "--sst-glob",
        default=os.environ.get(
            "ERA5_SST_GLOB"
        ),
        help=(
            "Glob for input ERA5 SST NetCDF files, e.g. "
            "'/path/to/era5_sst/era5_sst_*.nc'. "
            "May also be supplied through ERA5_SST_GLOB."
        ),
    )

    parser.add_argument(
        "--shape-dir",
        type=Path,
        default=DEFAULT_SHAPE_DIR,
        help=(
            "Directory containing basin shapefiles. "
            "Default: <repository>/data/shapefiles"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=(
            "Output directory for basin anomaly NetCDF files."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=int(
            os.environ.get(
                "SLURM_CPUS_PER_TASK",
                min(os.cpu_count() or 1, 8),
            )
        ),
        help=(
            "Number of local Dask workers. "
            "Defaults to SLURM_CPUS_PER_TASK when available, "
            "otherwise up to 8 local CPU cores."
        ),
    )

    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help=(
            "Retain temporary masked NetCDF pieces after completion."
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

    shape_dir = (
        args.shape_dir
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

    temp_root = (
        output_dir
        / "_tmp_basin_build"
    )

    temp_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    basin_layers = {
        basin_name: (
            shape_dir
            / relative_path
        )
        for basin_name, relative_path
        in BASIN_LAYER_RELATIVE_PATHS.items()
    }

    print("=" * 72)
    print("ERA5 BASIN SST ANOMALY CREATION")
    print("=" * 72)
    print(
        f"Input files : {len(input_files)}"
    )
    print(
        f"Input glob  : {args.sst_glob}"
    )
    print(
        f"Shape dir   : {shape_dir}"
    )
    print(
        f"Output dir  : {output_dir}"
    )
    print(
        f"Dask workers: {args.workers}"
    )
    print("=" * 72)

    # Avoid machine-specific dashboard/proxy configuration.
    dask.config.set(
        {
            "distributed.dashboard.link": (
                "http://localhost:{port}/status"
            )
        }
    )

    cluster = LocalCluster(
        n_workers=args.workers,
        threads_per_worker=1,
        processes=True,
    )

    client = Client(cluster)

    try:
        print(
            f"Dask dashboard: {client.dashboard_link}"
        )

        for (
            basin_name,
            shapefile_path,
        ) in basin_layers.items():

            build_basin_anomalies(
                basin_name=basin_name,
                shapefile_path=shapefile_path,
                input_files=input_files,
                output_dir=output_dir,
                temp_root=temp_root,
            )

    finally:
        client.close()
        cluster.close()

        if (
            not args.keep_temp
            and temp_root.exists()
        ):
            shutil.rmtree(
                temp_root
            )

    print(
        "\nAll basin SST anomaly files created."
    )


if __name__ == "__main__":
    main()
