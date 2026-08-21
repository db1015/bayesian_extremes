#!/usr/bin/env python3
"""
Atmospheric mechanism composites with month-block bootstrap significance.

Columns:
    El Niño - ENSO neutral
    La Niña - ENSO neutral
    pIOD - IOD neutral
    nIOD - IOD neutral

Rows:
    500 hPa geopotential height anomaly
    500 hPa vertical velocity anomaly
    925 hPa specific humidity anomaly + wind vectors
    925 hPa moisture-flux magnitude anomaly + flux vectors

Significance:
    95% month-block bootstrap CI. Scalar fields are stippled where the CI
    excludes zero. Vector fields are shown as composite phase-minus-neutral
    anomalies but are not separately significance-tested.

Terrain:
    Rows 3-4 are masked/gray where GMTED2010 elevation exceeds the approximate
    standard-atmosphere height of the 925-hPa surface (~760 m).

Expensive bootstrap products are saved to NetCDF BEFORE plotting so that figure
styling can later be changed without rerunning the bootstrap.
"""

# Limit nested numerical-library threading. Bootstrap parallelism is controlled
# explicitly with ThreadPoolExecutor below.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import glob
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe

import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# PATHS
# ============================================================
ANOM_ROOT = Path("../data/era5_anomalies")
FIG_DIR = Path("../figures/mechanism_composites")
FIG_DIR.mkdir(parents=True, exist_ok=True)

EVENT_CSV = Path("../data/wbt_sst_city_runs/city_daily_wbt_JJAS_with_lagged_phases.csv")
ELEVATION_FILE = Path("../data/elevation/GMTED2010_15n060_0250deg.nc")

PRODUCTS_PATH = FIG_DIR / "mechanism_phase_vs_neutral_bootstrap_products.nc"
PNG_PATH = FIG_DIR / "mechanism_phase_vs_neutral_bootstrap_16panel.png"
PDF_PATH = FIG_DIR / "mechanism_phase_vs_neutral_bootstrap_16panel_manuscript.pdf"

# ============================================================
# USER SETTINGS
# ============================================================
CITY = None
# CITY = "dubai"
# CITY = ["dubai", "doha"]
# CITY = None  -> all cities pooled

MONTHS = [6, 7, 8, 9]
DAY_MODE = "p95"  # "p95" or "all"

WBT_COL = "wbt_daily_peak"
DATE_COL = "time"
CITY_COL = "city"

LON_MIN, LON_MAX = 29, 65
LAT_MIN, LAT_MAX = 5, 39

# Existing classification convention in these mechanism scripts.
ENSO_POS_THRESH = 1.0
ENSO_NEG_THRESH = -1.0
IOD_POS_THRESH = 1.0
IOD_NEG_THRESH = -1.0
ENSO_LAG = 2
IOD_LAG = 1

# Bootstrap
N_BOOT = 10000
CI_LOW = 2.5
CI_HIGH = 97.5
BOOT_SEED = 58
N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", "32"))

# Plotting
QUIVER_STRIDE = 6
ROBUST_PCT = 98
STIPPLE_STRIDE = 1
STIPPLE_SIZE = 1.6
STIPPLE_ALPHA = 0.62

# 925-hPa terrain approximation
TERRAIN_925_M = 760.0

# Unit conversions
G = 9.81
KGKG_TO_GKG = 1000.0

OPEN_MFDATASET_KW = dict(
    combine="by_coords",
    parallel=False,
    coords="minimal",
    compat="override",
    engine="h5netcdf",
)

ERL_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "path.simplify": False,
    "savefig.transparent": False,
}

PANEL_LABELS = [
    "(a)", "(b)", "(c)", "(d)",
    "(e)", "(f)", "(g)", "(h)",
    "(i)", "(j)", "(k)", "(l)",
    "(m)", "(n)", "(o)", "(p)",
]

COLUMN_KEYS = ["el_nino", "la_nina", "piod", "niod"]
COLUMN_TITLES = ["El Niño", "La Niña", "pIOD", "nIOD"]

# ============================================================
# BASIC HELPERS
# ============================================================
def get_lat_lon_names(obj):
    lat_name = "latitude" if "latitude" in obj.coords else "lat"
    lon_name = "longitude" if "longitude" in obj.coords else "lon"
    return lat_name, lon_name


def find_time_coord(obj):
    for cand in ["time", "date", "valid_time"]:
        if cand in obj.coords or cand in obj.dims:
            return cand
    raise KeyError(
        f"Could not find time coordinate. coords={list(obj.coords)}, dims={list(obj.dims)}"
    )


def classify_enso(val):
    if pd.isna(val):
        return np.nan
    if val >= ENSO_POS_THRESH:
        return "El Niño"
    if val <= ENSO_NEG_THRESH:
        return "La Niña"
    return "Neutral"


def classify_iod(val):
    if pd.isna(val):
        return np.nan
    if val >= IOD_POS_THRESH:
        return "pIOD"
    if val <= IOD_NEG_THRESH:
        return "nIOD"
    return "Neutral"


def centered_levels_from_arrays(arrays, nlev=21, pct=ROBUST_PCT):
    vals = []
    for a in arrays:
        x = np.asarray(a).ravel()
        x = x[np.isfinite(x)]
        if x.size:
            vals.append(x)

    if not vals:
        return np.linspace(-1, 1, nlev)

    vals = np.concatenate(vals)
    vmax = np.nanpercentile(np.abs(vals), pct)

    if not np.isfinite(vmax) or vmax == 0:
        vmax = np.nanmax(np.abs(vals))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    return np.linspace(-vmax, vmax, nlev)


# ============================================================
# EVENT / PHASE TABLE
# ============================================================
def load_event_phase_dates(day_mode=DAY_MODE):
    """
    Build event-date collections for all six phase categories:
      El Niño, La Niña, ENSO Neutral, pIOD, nIOD, IOD Neutral.

    Important:
      Monthly phase classifications are built from the unfiltered event CSV
      before the p95-day filter is applied. This preserves precursor months.
    """
    raw = pd.read_csv(EVENT_CSV)
    raw[DATE_COL] = pd.to_datetime(raw[DATE_COL])

    if CITY is not None:
        cities = [CITY] if isinstance(CITY, str) else list(CITY)
        raw = raw[raw[CITY_COL].isin(cities)].copy()

    raw["month_start"] = raw[DATE_COL].dt.to_period("M").dt.to_timestamp()

    # Build monthly teleconnection table FIRST.
    monthly = raw[raw["lag"] == 0].copy()
    if monthly.empty:
        raise ValueError("No lag==0 rows found in event CSV")

    monthly = (
        monthly[["month_start", "RONI_lag", "DMI_lag"]]
        .drop_duplicates(subset=["month_start"])
        .sort_values("month_start")
        .reset_index(drop=True)
    )

    monthly["enso_phase_now"] = monthly["RONI_lag"].apply(classify_enso)
    monthly["iod_phase_now"] = monthly["DMI_lag"].apply(classify_iod)

    # Shift source phase forward to its response month.
    enso_shifted = monthly[["month_start", "enso_phase_now"]].copy()
    enso_shifted["month_start"] += pd.DateOffset(months=ENSO_LAG)
    enso_shifted = enso_shifted.rename(
        columns={"enso_phase_now": "enso_phase_shifted"}
    )

    iod_shifted = monthly[["month_start", "iod_phase_now"]].copy()
    iod_shifted["month_start"] += pd.DateOffset(months=IOD_LAG)
    iod_shifted = iod_shifted.rename(
        columns={"iod_phase_now": "iod_phase_shifted"}
    )

    df = raw.merge(
        monthly[["month_start", "enso_phase_now", "iod_phase_now"]],
        on="month_start",
        how="left",
    )
    df = df.merge(enso_shifted, on="month_start", how="left")
    df = df.merge(iod_shifted, on="month_start", how="left")

    df["month_num"] = df[DATE_COL].dt.month

    # Preserve the fallback behavior from the existing mechanism script.
    df["enso_phase_final"] = df["enso_phase_shifted"]
    enso_fallback = (
        df["enso_phase_final"].isna()
        & df["month_num"].isin([6, 7])
    )
    df.loc[enso_fallback, "enso_phase_final"] = df.loc[
        enso_fallback, "enso_phase_now"
    ]

    df["iod_phase_final"] = df["iod_phase_shifted"]
    iod_fallback = (
        df["iod_phase_final"].isna()
        & df["month_num"].isin([6])
    )
    df.loc[iod_fallback, "iod_phase_final"] = df.loc[
        iod_fallback, "iod_phase_now"
    ]

    # Restrict to JJAS.
    df = df[df["month_num"].isin(MONTHS)].copy()

    # Now apply HHE-day selection.
    if day_mode.lower() == "p95":
        df["thresh"] = df.groupby(CITY_COL)[WBT_COL].transform(
            lambda x: x.quantile(0.95)
        )
        df = df[df[WBT_COL] >= df["thresh"]].copy()
        mode_label = "p95"
    elif day_mode.lower() == "all":
        mode_label = "all"
    else:
        raise ValueError("DAY_MODE must be 'p95' or 'all'")

    phase_dates = {}

    phase_specs = [
        ("el_nino", "enso_phase_final", "El Niño"),
        ("la_nina", "enso_phase_final", "La Niña"),
        ("enso_neutral", "enso_phase_final", "Neutral"),
        ("piod", "iod_phase_final", "pIOD"),
        ("niod", "iod_phase_final", "nIOD"),
        ("iod_neutral", "iod_phase_final", "Neutral"),
    ]

    for key, col, phase in specs:
        sub = df[df[col] == phase]
    
        out[key] = pd.DatetimeIndex(
            sorted(
                pd.to_datetime(sub[DATE_COL])
                .normalize()
                .unique()
            )
        )

    print(f"\nEvent counts after filtering ({mode_label} mode):")
    for key in [
        "el_nino", "la_nina", "enso_neutral",
        "piod", "niod", "iod_neutral"
    ]:
        print(f"  {key:>13s}: {len(phase_dates[key])} unique dates")

    print("\nFallback counts before event filtering:")
    print(f"  ENSO fallback rows: {int(enso_fallback.sum())}")
    print(f"  IOD  fallback rows: {int(iod_fallback.sum())}")

    return phase_dates, mode_label


def dates_to_month_blocks(dates, common_dates):
    """
    Convert requested event dates into integer-index blocks grouped by month,
    using the aligned atmospheric common_dates axis.
    """
    common_dates = pd.DatetimeIndex(pd.to_datetime(common_dates).normalize())
    lookup = pd.Series(
        np.arange(len(common_dates), dtype=np.int32),
        index=common_dates,
    )

    dates = pd.DatetimeIndex(pd.to_datetime(dates).normalize())
    dates = dates.intersection(common_dates)

    if len(dates) == 0:
        raise ValueError("No requested event dates overlap atmospheric data")

    periods = dates.to_period("M")
    blocks = []

    for ym in periods.unique().sort_values():
        dmonth = dates[periods == ym]
        idx = lookup.loc[dmonth].to_numpy(dtype=np.int32)
        blocks.append(idx)

    return blocks


# ============================================================
# ATMOSPHERIC DATA
# ============================================================
def open_all_anoms(var_name):
    files = sorted((ANOM_ROOT / var_name).glob("*.nc"))

    if not files:
        raise FileNotFoundError(
            f"No anomaly files found for {var_name} in "
            f"{ANOM_ROOT / var_name}"
        )

    ds = xr.open_mfdataset(
        files,
        **OPEN_MFDATASET_KW
    )

    if len(ds.data_vars) != 1:
        raise ValueError(
            f"Expected one variable in {var_name}; "
            f"found {list(ds.data_vars)}"
        )

    da = ds[list(ds.data_vars)[0]]

    time_name = find_time_coord(da)

    if time_name != "time":
        da = da.rename({time_name: "time"})

    da = da.sortby("time")

    lat_name, lon_name = get_lat_lon_names(da)

    # --------------------------------------------------------
    # Robust spatial subset regardless of coordinate direction
    # --------------------------------------------------------
    lat_vals = da[lat_name].values
    lon_vals = da[lon_name].values

    if lat_vals[0] < lat_vals[-1]:
        lat_slice = slice(LAT_MIN, LAT_MAX)
    else:
        lat_slice = slice(LAT_MAX, LAT_MIN)

    if lon_vals[0] < lon_vals[-1]:
        lon_slice = slice(LON_MIN, LON_MAX)
    else:
        lon_slice = slice(LON_MAX, LON_MIN)

    da = da.sel({
        lat_name: lat_slice,
        lon_name: lon_slice,
    })

    if da.sizes.get(lat_name, 0) == 0:
        raise ValueError(
            f"{var_name}: latitude subset is empty. "
            f"Original latitude range was "
            f"{lat_vals.min()} to {lat_vals.max()}."
        )

    if da.sizes.get(lon_name, 0) == 0:
        raise ValueError(
            f"{var_name}: longitude subset is empty. "
            f"Original longitude range was "
            f"{lon_vals.min()} to {lon_vals.max()}."
        )

    return da


def load_aligned_fields():
    """
    Open all eight fields, align to one shared daily time/grid, convert units,
    and materialize them into float32 NumPy arrays exactly once.
    """
    print("\nOpening atmospheric anomaly fields...")

    fields = {
        "z": open_all_anoms("geopotential") / G,
        "omega": open_all_anoms("vertical_velocity"),
        "q": open_all_anoms("specific_humidity") * KGKG_TO_GKG,
        "u": open_all_anoms("u_component_of_wind"),
        "v": open_all_anoms("v_component_of_wind"),
        "mfmag": open_all_anoms("moisture_flux_mag_925") * KGKG_TO_GKG,
        "mfu": open_all_anoms("moisture_flux_u_925") * KGKG_TO_GKG,
        "mfv": open_all_anoms("moisture_flux_v_925") * KGKG_TO_GKG,
    }

    print("Aligning all fields to common dates/grid...")
    names = list(fields)
    aligned = xr.align(
        *[fields[n] for n in names],
        join="inner",
        copy=False,
    )
    fields = dict(zip(names, aligned))

    ref = fields["z"]
    lat_name, lon_name = get_lat_lon_names(ref)

    # Ensure common dimension order.
    for name in names:
        fields[name] = fields[name].transpose("time", lat_name, lon_name)

    print("Loading atmospheric fields into RAM once...")
    arrays = {}
    for name in names:
        print(f"  loading {name}...")
        arrays[name] = np.asarray(
            fields[name].values,
            dtype=np.float32,
        )

    times = pd.to_datetime(ref["time"].values).normalize()
    lats = np.asarray(ref[lat_name].values)
    lons = np.asarray(ref[lon_name].values)

    print(
        f"Common atmospheric grid: time={len(times)}, "
        f"lat={len(lats)}, lon={len(lons)}"
    )

    return arrays, times, lats, lons


# ============================================================
# TERRAIN MASK
# ============================================================
def load_terrain_mask(target_lats, target_lons):
    """
    Autodetect the elevation variable and interpolate the GMTED2010 field
    to the atmospheric grid if coordinates are not already identical.
    """
    if not ELEVATION_FILE.exists():
        raise FileNotFoundError(f"Elevation file not found: {ELEVATION_FILE}")

    ds = xr.open_dataset(ELEVATION_FILE)

    if not ds.data_vars:
        raise ValueError(f"No data variables found in {ELEVATION_FILE}")

    # Prefer obvious elevation names; otherwise use the first numeric variable.
    preferred = [
        "elevation", "elev", "z", "Band1", "gmted2010", "topography", "orog"
    ]
    var_name = None

    for cand in preferred:
        if cand in ds.data_vars:
            var_name = cand
            break

    if var_name is None:
        numeric = [
            name for name, da in ds.data_vars.items()
            if np.issubdtype(da.dtype, np.number)
        ]
        if not numeric:
            raise ValueError(
                f"Could not identify numeric elevation variable. "
                f"Variables={list(ds.data_vars)}"
            )
        var_name = numeric[0]

    elev = ds[var_name].squeeze(drop=True)
    lat_name, lon_name = get_lat_lon_names(elev)

    elev = elev.sortby(lat_name).sortby(lon_name)
    elev = elev.sel(
        {
            lat_name: slice(LAT_MIN, LAT_MAX),
            lon_name: slice(LON_MIN, LON_MAX),
        }
    )

    # Rename for straightforward interpolation.
    if lat_name != "latitude" or lon_name != "longitude":
        elev = elev.rename(
            {
                lat_name: "latitude",
                lon_name: "longitude",
            }
        )

    target_lat_da = xr.DataArray(target_lats, dims="latitude")
    target_lon_da = xr.DataArray(target_lons, dims="longitude")

    same_grid = (
        elev.sizes.get("latitude") == len(target_lats)
        and elev.sizes.get("longitude") == len(target_lons)
        and np.allclose(elev["latitude"].values, target_lats)
        and np.allclose(elev["longitude"].values, target_lons)
    )

    if not same_grid:
        print(
            "Elevation grid does not exactly match atmospheric grid; "
            "interpolating elevation once with nearest-neighbor."
        )
        elev = elev.interp(
            latitude=target_lat_da,
            longitude=target_lon_da,
            method="nearest",
        )

    elev_np = np.asarray(elev.values, dtype=np.float32)

    # If units suggest km, convert to m.
    units = str(elev.attrs.get("units", "")).lower()
    if units in {"km", "kilometer", "kilometers"}:
        elev_np *= 1000.0

    terrain_mask = np.isfinite(elev_np) & (elev_np >= TERRAIN_925_M)

    print(
        f"Terrain mask: {terrain_mask.sum()} cells >= {TERRAIN_925_M:.0f} m "
        f"({100.0 * terrain_mask.mean():.1f}% of grid)"
    )

    return terrain_mask, elev_np, var_name


# ============================================================
# COMPOSITES / BOOTSTRAP
# ============================================================
SCALAR_NAMES = ["z", "omega", "q", "mfmag"]
VECTOR_NAMES = ["u", "v", "mfu", "mfv"]


def mean_fields(arrays, indices, names):
    out = {}
    for name in names:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out[name] = np.nanmean(
                arrays[name][indices, :, :],
                axis=0,
            ).astype(np.float32, copy=False)
    return out


def observed_comparison(arrays, phase_blocks, neutral_blocks):
    pidx = np.concatenate(phase_blocks)
    nidx = np.concatenate(neutral_blocks)

    pmean = mean_fields(arrays, pidx, SCALAR_NAMES + VECTOR_NAMES)
    nmean = mean_fields(arrays, nidx, SCALAR_NAMES + VECTOR_NAMES)

    return {
        name: (pmean[name] - nmean[name]).astype(np.float32, copy=False)
        for name in SCALAR_NAMES + VECTOR_NAMES
    }


def bootstrap_one(seed, scalar_arrays, phase_blocks, neutral_blocks):
    rng = np.random.default_rng(seed)

    pdraw = rng.integers(
        0, len(phase_blocks), size=len(phase_blocks)
    )
    ndraw = rng.integers(
        0, len(neutral_blocks), size=len(neutral_blocks)
    )

    pidx = np.concatenate([phase_blocks[i] for i in pdraw])
    nidx = np.concatenate([neutral_blocks[i] for i in ndraw])

    out = np.empty(
        (
            len(SCALAR_NAMES),
            scalar_arrays[SCALAR_NAMES[0]].shape[1],
            scalar_arrays[SCALAR_NAMES[0]].shape[2],
        ),
        dtype=np.float32,
    )

    for j, name in enumerate(SCALAR_NAMES):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pmean = np.nanmean(scalar_arrays[name][pidx], axis=0)
            nmean = np.nanmean(scalar_arrays[name][nidx], axis=0)
        out[j] = pmean - nmean

    return out


def bootstrap_comparison(
    label,
    arrays,
    phase_blocks,
    neutral_blocks,
    seed,
):
    print(
        f"\n{label}: {len(phase_blocks)} phase months vs "
        f"{len(neutral_blocks)} neutral months"
    )

    observed = observed_comparison(
        arrays,
        phase_blocks,
        neutral_blocks,
    )

    nlat, nlon = observed["z"].shape
    boot = np.empty(
        (N_BOOT, len(SCALAR_NAMES), nlat, nlon),
        dtype=np.float32,
    )

    scalar_arrays = {name: arrays[name] for name in SCALAR_NAMES}

    seed_seq = np.random.SeedSequence(seed)
    child = seed_seq.spawn(N_BOOT)
    seeds = [
        int(s.generate_state(1, dtype=np.uint32)[0])
        for s in child
    ]

    print(
        f"{label}: running {N_BOOT} month-block replicates "
        f"with {N_WORKERS} workers..."
    )

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(
                bootstrap_one,
                seeds[b],
                scalar_arrays,
                phase_blocks,
                neutral_blocks,
            ): b
            for b in range(N_BOOT)
        }

        done = 0
        for future in as_completed(futures):
            b = futures[future]
            boot[b] = future.result()
            done += 1

            if done % 100 == 0 or done == N_BOOT:
                print(f"{label}: {done}/{N_BOOT}")

    print(f"{label}: calculating 95% bootstrap CI...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ci_low = np.nanpercentile(
            boot, CI_LOW, axis=0
        ).astype(np.float32)
        ci_high = np.nanpercentile(
            boot, CI_HIGH, axis=0
        ).astype(np.float32)

    sig = (
        ((ci_low > 0.0) | (ci_high < 0.0))
        & np.isfinite(ci_low)
        & np.isfinite(ci_high)
    )

    ci_low_dict = {
        name: ci_low[j] for j, name in enumerate(SCALAR_NAMES)
    }
    ci_high_dict = {
        name: ci_high[j] for j, name in enumerate(SCALAR_NAMES)
    }
    sig_dict = {
        name: sig[j] for j, name in enumerate(SCALAR_NAMES)
    }

    del boot

    return observed, ci_low_dict, ci_high_dict, sig_dict


# ============================================================
# SAVE PLOTTING PRODUCTS
# ============================================================
def save_products(
    observed,
    ci_low,
    ci_high,
    sig,
    terrain_mask,
    elevation,
    lats,
    lons,
    mode_label,
    elevation_var,
):
    data_vars = {}

    for phase_key in COLUMN_KEYS:
        for field in SCALAR_NAMES + VECTOR_NAMES:
            data_vars[f"{phase_key}_{field}"] = (
                ("latitude", "longitude"),
                observed[phase_key][field],
            )

        for field in SCALAR_NAMES:
            data_vars[f"{phase_key}_{field}_ci_low"] = (
                ("latitude", "longitude"),
                ci_low[phase_key][field],
            )
            data_vars[f"{phase_key}_{field}_ci_high"] = (
                ("latitude", "longitude"),
                ci_high[phase_key][field],
            )
            data_vars[f"{phase_key}_{field}_sig"] = (
                ("latitude", "longitude"),
                sig[phase_key][field].astype(np.int8),
            )

    data_vars["terrain_925_mask"] = (
        ("latitude", "longitude"),
        terrain_mask.astype(np.int8),
    )
    data_vars["elevation_m"] = (
        ("latitude", "longitude"),
        elevation.astype(np.float32),
    )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "latitude": lats,
            "longitude": lons,
        },
        attrs={
            "description": (
                "Atmospheric phase-minus-neutral composites with 95% "
                "month-block bootstrap confidence intervals and significance."
            ),
            "day_mode": mode_label,
            "bootstrap_replicates": N_BOOT,
            "bootstrap_ci_low": CI_LOW,
            "bootstrap_ci_high": CI_HIGH,
            "bootstrap_seed": BOOT_SEED,
            "enso_lag_months": ENSO_LAG,
            "iod_lag_months": IOD_LAG,
            "enso_threshold": ENSO_POS_THRESH,
            "iod_threshold": IOD_POS_THRESH,
            "terrain_925_threshold_m": TERRAIN_925_M,
            "elevation_source": str(ELEVATION_FILE),
            "elevation_variable": elevation_var,
        },
    )

    ds.to_netcdf(PRODUCTS_PATH)
    print(f"\nSaved plotting products to: {PRODUCTS_PATH}")


# ============================================================
# PLOTTING
# ============================================================
def style_map(ax, show_grid_labels=False):
    ax.set_extent(
        [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
        crs=ccrs.PlateCarree(),
    )

    ax.coastlines(resolution="50m", linewidth=0.6, color="0.35")
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.35,
        edgecolor="0.45",
    )

    gl = ax.gridlines(
        draw_labels=show_grid_labels,
        linewidth=0.3,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )

    if show_grid_labels:
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(30, 66, 5))
        gl.ylocator = mticker.FixedLocator(np.arange(5, 40, 5))
        gl.xlabel_style = {"size": 7}
        gl.ylabel_style = {"size": 7}


def add_panel_label(ax, label):
    ax.text(
        0.02, 0.98, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        fontweight="bold",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.8,
            pad=1.2,
        ),
        zorder=30,
    )


def add_stippling(ax, sig_mask, lons, lats, terrain_mask=None):
    mask = np.asarray(sig_mask, dtype=bool).copy()

    if terrain_mask is not None:
        mask &= ~terrain_mask

    if STIPPLE_STRIDE > 1:
        mask = mask[::STIPPLE_STRIDE, ::STIPPLE_STRIDE]
        lon_use = lons[::STIPPLE_STRIDE]
        lat_use = lats[::STIPPLE_STRIDE]
    else:
        lon_use = lons
        lat_use = lats

    yy, xx = np.where(mask)
    if yy.size == 0:
        return

    ax.scatter(
        lon_use[xx],
        lat_use[yy],
        s=STIPPLE_SIZE,
        c="black",
        marker=".",
        linewidths=0,
        alpha=STIPPLE_ALPHA,
        transform=ccrs.PlateCarree(),
        zorder=12,
    )


def add_terrain_gray(ax, terrain_mask, lons, lats):
    ax.contourf(
        lons,
        lats,
        terrain_mask.astype(float),
        levels=[0.5, 1.5],
        colors=["0.72"],
        transform=ccrs.PlateCarree(),
        zorder=10,
    )


def add_quiver_np(
    ax,
    u,
    v,
    lons,
    lats,
    terrain_mask=None,
    stride=QUIVER_STRIDE,
):
    uplot = np.asarray(u, dtype=float).copy()
    vplot = np.asarray(v, dtype=float).copy()

    if terrain_mask is not None:
        uplot[terrain_mask] = np.nan
        vplot[terrain_mask] = np.nan

    u2 = uplot[::stride, ::stride]
    v2 = vplot[::stride, ::stride]
    lon2 = lons[::stride]
    lat2 = lats[::stride]
    lon2d, lat2d = np.meshgrid(lon2, lat2)

    return ax.quiver(
        lon2d,
        lat2d,
        u2,
        v2,
        transform=ccrs.PlateCarree(),
        scale=None,
        width=0.0022,
        headwidth=3.7,
        headlength=4.2,
        minlength=0.05,
        pivot="middle",
        color="black",
        path_effects=[
            pe.Stroke(linewidth=1.05, foreground="white"),
            pe.Normal(),
        ],
        zorder=15,
    )


def plot_16panel(observed, sig, terrain_mask, lats, lons):
    row_specs = [
        ("z", "PuOr", "500 hPa $Z_g$", "m", None),
        ("omega", "PuOr", "500 hPa $\\Omega$", "Pa s$^{-1}$", None),
        ("q", "BrBG", "925 hPa $q$", "g kg$^{-1}$", ("u", "v")),
        (
            "mfmag",
            "BrBG",
            r"925 hPa $q\mathbf{v}$",
            "g kg$^{-1}$ m s$^{-1}$",
            ("mfu", "mfv"),
        ),
    ]

    levels = {}
    for field, _, _, _, _ in row_specs:
        levels[field] = centered_levels_from_arrays(
            [observed[k][field] for k in COLUMN_KEYS],
            nlev=21,
        )

    proj = ccrs.PlateCarree()

    with mpl.rc_context(ERL_RC):
        fig, axes = plt.subplots(
            4,
            4,
            figsize=(10.5, 10.8),
            subplot_kw={"projection": proj},
            constrained_layout=False,
        )

        fig.subplots_adjust(
            left=0.09,
            right=0.985,
            top=0.94,
            bottom=0.055,
            wspace=0.055,
            hspace=0.22,
        )

        # Column headers
        for c, title in enumerate(COLUMN_TITLES):
            pos = axes[0, c].get_position()
            fig.text(
                (pos.x0 + pos.x1) / 2,
                0.975,
                title,
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
            )

        mappables = {}
        panel_idx = 0

        for r, (field, cmap, row_label, cbar_label, vectors) in enumerate(row_specs):
            is_925 = r >= 2

            for c, key in enumerate(COLUMN_KEYS):
                ax = axes[r, c]
                show_labels = (c == 0)
                style_map(ax, show_grid_labels=show_labels)

                plot_field = observed[key][field].copy()
                if is_925:
                    plot_field = np.where(terrain_mask, np.nan, plot_field)

                cf = ax.contourf(
                    lons,
                    lats,
                    plot_field,
                    levels=levels[field],
                    cmap=cmap,
                    extend="both",
                    transform=proj,
                    zorder=1,
                )

                if c == 0:
                    mappables[field] = cf

                add_stippling(
                    ax,
                    sig[key][field],
                    lons,
                    lats,
                    terrain_mask=terrain_mask if is_925 else None,
                )

                if vectors is not None:
                    u_name, v_name = vectors
                    qv = add_quiver_np(
                        ax,
                        observed[key][u_name],
                        observed[key][v_name],
                        lons,
                        lats,
                        terrain_mask=terrain_mask,
                    )

                    # One key per row, placed in the last column only.
                    if c == 3:
                        if field == "q":
                            ax.quiverkey(
                                qv,
                                0.48,
                                -0.055,
                                5,
                                "5 m s$^{-1}$",
                                labelpos="E",
                                coordinates="axes",
                                fontproperties={"size": 7},
                            )
                        else:
                            ax.quiverkey(
                                qv,
                                0.42,
                                -0.055,
                                15,
                                "15 g kg$^{-1}$ m s$^{-1}$",
                                labelpos="E",
                                coordinates="axes",
                                fontproperties={"size": 7},
                            )

                if is_925:
                    add_terrain_gray(ax, terrain_mask, lons, lats)

                add_panel_label(ax, PANEL_LABELS[panel_idx])
                panel_idx += 1

            # Left-side row label
            pos = axes[r, 0].get_position()
            fig.text(
                0.025,
                (pos.y0 + pos.y1) / 2,
                row_label,
                ha="center",
                va="center",
                rotation=90,
                fontsize=8.5,
                fontweight="bold",
            )

            cbar = fig.colorbar(
                mappables[field],
                ax=axes[r, :],
                orientation="horizontal",
                pad=0.07,
                shrink=0.89,
            )
            cbar.set_label(cbar_label)
            cbar.ax.tick_params(labelsize=7)

        fig.savefig(
            PNG_PATH,
            dpi=300,
            bbox_inches="tight",
        )

        fig.savefig(
            PDF_PATH,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )

        plt.close(fig)

    print(f"Saved PNG to: {PNG_PATH}")
    print(f"Saved PDF to: {PDF_PATH}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading HHE event dates and lagged phase classifications...")
    phase_dates, mode_label = load_event_phase_dates(DAY_MODE)

    arrays, common_dates, lats, lons = load_aligned_fields()

    print("\nBuilding event-month blocks on atmospheric time axis...")

    blocks = {
        key: dates_to_month_blocks(dates, common_dates)
        for key, dates in phase_dates.items()
    }

    for key, val in blocks.items():
        print(f"  {key:>13s}: {len(val)} event months")

    comparisons = {
        "el_nino": (
            "El Niño vs ENSO neutral",
            blocks["el_nino"],
            blocks["enso_neutral"],
            BOOT_SEED + 1,
        ),
        "la_nina": (
            "La Niña vs ENSO neutral",
            blocks["la_nina"],
            blocks["enso_neutral"],
            BOOT_SEED + 2,
        ),
        "piod": (
            "pIOD vs IOD neutral",
            blocks["piod"],
            blocks["iod_neutral"],
            BOOT_SEED + 3,
        ),
        "niod": (
            "nIOD vs IOD neutral",
            blocks["niod"],
            blocks["iod_neutral"],
            BOOT_SEED + 4,
        ),
    }

    observed = {}
    ci_low = {}
    ci_high = {}
    sig = {}

    for key in COLUMN_KEYS:
        label, pblocks, nblocks, seed = comparisons[key]
        (
            observed[key],
            ci_low[key],
            ci_high[key],
            sig[key],
        ) = bootstrap_comparison(
            label,
            arrays,
            pblocks,
            nblocks,
            seed,
        )

    print("\nLoading elevation and constructing 925-hPa terrain mask...")
    terrain_mask, elevation, elevation_var = load_terrain_mask(
        lats,
        lons,
    )

    # SAVE BEFORE PLOTTING.
    print("\nSaving plotting-ready bootstrap products before rendering...")
    save_products(
        observed,
        ci_low,
        ci_high,
        sig,
        terrain_mask,
        elevation,
        lats,
        lons,
        mode_label,
        elevation_var,
    )

    print("\nRendering 4x4 atmospheric composite figure...")
    plot_16panel(
        observed,
        sig,
        terrain_mask,
        lats,
        lons,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
