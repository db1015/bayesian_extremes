#!/usr/bin/env python3
"""
1980-2024 JJAS moisture composites conditioned on GRID-CELL-LOCAL p95 WBT days.

Columns: El Nino-neutral, La Nina-neutral, pIOD-neutral, nIOD-neutral
Rows: 925-hPa q+winds, 925-hPa moisture flux+vectors, swvl1, swvl2, SST.
ENSO/IOD thresholds = +/-0.5; lags = 2/1 months. No bootstrap significance.
All fields are placed on the DailyPeakState grid before applying the local HHE mask.
Plotting-ready products are saved before rendering.
"""

import os
import glob
import warnings
from pathlib import Path

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
# PATHS / SETTINGS
# ============================================================
DAILY_PEAK_GLOB = "../data/DailyPeakState/DailyPeakState-*.nc"
ANOM_ROOT = Path("../data/era5_anomalies")
SOIL_GLOB = "../data/land/soil_moisture/era5_land_soil_moisture_*.nc"
SST_DIR = Path("/home/k16v981/my_work/data/era5/era5_sst")
PHASE_CSV = Path("../data/sst/roni_dmi_monthly_1950_2025.csv")
ELEVATION_FILE = Path("../data/elevation/GMTED2010_15n060_0250deg.nc")

FIG_DIR = Path("../figures/moisture_workup")
FIG_DIR.mkdir(parents=True, exist_ok=True)
PRODUCTS_PATH = FIG_DIR / "moisture_workup_local_hhe_phase_vs_neutral_1980_2024_products.nc"
PNG_PATH = FIG_DIR / "moisture_workup_local_hhe_phase_vs_neutral_1980_2024.png"
PDF_PATH = FIG_DIR / "moisture_workup_local_hhe_phase_vs_neutral_1980_2024_manuscript.pdf"

START_YEAR, END_YEAR = 1980, 2024
MONTHS = [6, 7, 8, 9]
WBT_VAR = "wbt_daily_peak"
PCTL = 0.95
ENSO_LAG, IOD_LAG = 2, 1
ENSO_POS_THRESH, ENSO_NEG_THRESH = 0.5, -0.5
IOD_POS_THRESH, IOD_NEG_THRESH = 0.5, -0.5
LON_MIN, LON_MAX = 29, 65
LAT_MIN, LAT_MAX = 5, 39
KGKG_TO_GKG = 1000.0
TERRAIN_925_M = 760.0
QUIVER_STRIDE = 6
ROBUST_PCT = 98
MIN_EVENT_COUNT = 3

COLUMN_KEYS = ["el_nino", "la_nina", "piod", "niod"]
COLUMN_TITLES = ["El Niño", "La Niña", "pIOD", "nIOD"]
PANEL_LABELS = [f"({chr(97+i)})" for i in range(20)]
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
    "savefig.transparent": False,
}

# ============================================================
# GENERIC HELPERS
# ============================================================
def get_lat_lon_names(obj):
    lat = next((n for n in ["latitude", "lat", "y"] if n in obj.coords or n in obj.dims), None)
    lon = next((n for n in ["longitude", "lon", "x"] if n in obj.coords or n in obj.dims), None)
    if lat is None or lon is None:
        raise ValueError(f"Could not detect lat/lon: coords={list(obj.coords)}, dims={list(obj.dims)}")
    return lat, lon


def find_time_name(obj):
    for n in ["time", "valid_time", "date"]:
        if n in obj.coords or n in obj.dims:
            return n
    raise ValueError(f"Could not detect time: coords={list(obj.coords)}, dims={list(obj.dims)}")

def standardize_names(da, require_time=True):
    rename = {}

    # latitude
    if "latitude" not in da.coords and "latitude" not in da.dims:
        for cand in ["lat", "y"]:
            if cand in da.coords or cand in da.dims:
                rename[cand] = "latitude"
                break

    # longitude
    if "longitude" not in da.coords and "longitude" not in da.dims:
        for cand in ["lon", "x"]:
            if cand in da.coords or cand in da.dims:
                rename[cand] = "longitude"
                break

    # time only when required
    if require_time:
        if "time" not in da.coords and "time" not in da.dims:
            for cand in ["valid_time", "date"]:
                if cand in da.coords or cand in da.dims:
                    rename[cand] = "time"
                    break
            else:
                raise ValueError(
                    f"Could not detect time: "
                    f"coords={list(da.coords)}, dims={list(da.dims)}"
                )

    if rename:
        da = da.rename(rename)

    return da

def standardize_time(da):
    if "time" in da.coords or "time" in da.dims:
        return da

    for cand in ["valid_time", "date", "day"]:
        if cand in da.coords or cand in da.dims:
            return da.rename({cand: "time"})

    raise ValueError(
        f"Could not detect time coordinate: "
        f"coords={list(da.coords)}, dims={list(da.dims)}"
    )


def subset_ap(da):
    da = standardize_names(
        da,
        require_time=False
    )

    da = (
        da
        .sortby("latitude")
        .sortby("longitude")
    )

    da = da.sel(
        latitude=slice(LAT_MIN, LAT_MAX),
        longitude=slice(LON_MIN, LON_MAX),
    )

    if da.sizes.get("latitude", 0) == 0:
        raise RuntimeError("AP latitude subset is empty")

    if da.sizes.get("longitude", 0) == 0:
        raise RuntimeError("AP longitude subset is empty")

    return da

def restrict_period_jjas(da):
    da = standardize_names(da)
    da = da.sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))
    return da.where(da.time.dt.month.isin(MONTHS), drop=True)


def regrid_to_target(da, target_lat, target_lon, method="linear"):
    da = subset_ap(da)
    same = (
        da.sizes.get("latitude") == len(target_lat)
        and da.sizes.get("longitude") == len(target_lon)
        and np.allclose(da.latitude.values, target_lat)
        and np.allclose(da.longitude.values, target_lon)
    )
    if same:
        return da
    return da.interp(
        latitude=xr.DataArray(target_lat, dims="latitude"),
        longitude=xr.DataArray(target_lon, dims="longitude"),
        method=method,
    )


def centered_levels(arrays, nlev=21, pct=ROBUST_PCT):
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
# PHASE TABLE: ONE ROW PER MONTH, +/-0.5, LAGGED TO RESPONSE MONTH
# ============================================================
def classify_enso(x):
    if pd.isna(x): return np.nan
    if x >= ENSO_POS_THRESH: return "El Niño"
    if x <= ENSO_NEG_THRESH: return "La Niña"
    return "Neutral"


def classify_iod(x):
    if pd.isna(x): return np.nan
    if x >= IOD_POS_THRESH: return "pIOD"
    if x <= IOD_NEG_THRESH: return "nIOD"
    return "Neutral"


def load_phase_table():
    df = pd.read_csv(PHASE_CSV)
    req = {"time", "RONI", "DMI"}
    if req.difference(df.columns):
        raise ValueError(f"Phase CSV missing {sorted(req.difference(df.columns))}")
    df["time"] = pd.to_datetime(df["time"])
    df["ym"] = df.time.dt.to_period("M")
    df = (
        df.groupby("ym", as_index=False)
          .agg(RONI=("RONI", "mean"), DMI=("DMI", "mean"))
          .sort_values("ym")
          .reset_index(drop=True)
    )
    df["time"] = df.ym.dt.to_timestamp()
    df["RONI_lagged"] = df.RONI.shift(ENSO_LAG)
    df["DMI_lagged"] = df.DMI.shift(IOD_LAG)
    df["enso_phase"] = df.RONI_lagged.apply(classify_enso)
    df["iod_phase"] = df.DMI_lagged.apply(classify_iod)
    df = df[
        (df.time.dt.year >= START_YEAR)
        & (df.time.dt.year <= END_YEAR)
        & df.time.dt.month.isin(MONTHS)
    ].copy()
    print("\nJJAS response-month phase counts, 1980-2024:")
    print("ENSO:\n", df.enso_phase.value_counts(dropna=False))
    print("IOD:\n", df.iod_phase.value_counts(dropna=False))
    return df


def daily_phase_labels(times, phase_df):
    ym = pd.DatetimeIndex(pd.to_datetime(times)).to_period("M")
    lookup = phase_df.set_index("ym")
    enso = pd.Series(ym).map(lookup["enso_phase"]).to_numpy(dtype=object)
    iod = pd.Series(ym).map(lookup["iod_phase"]).to_numpy(dtype=object)
    return enso, iod

# ============================================================
# DAILY PEAK WBT AND GRID-CELL-LOCAL HHE MASK
# ============================================================
def preprocess_peak(ds):
    if "day" in ds.dims:
        ds = ds.rename({"day": "time"})
    elif "day" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"day": "time"})
    if "latitude" in ds.coords:
        ds = ds.sortby("latitude")
    if "longitude" in ds.coords:
        ds = ds.sortby("longitude")
    return ds


def open_daily_peak_wbt():
    files = sorted(glob.glob(DAILY_PEAK_GLOB))
    if not files:
        raise FileNotFoundError(f"No DailyPeakState files matched {DAILY_PEAK_GLOB}")
    print(f"\nOpening {len(files)} DailyPeakState files...")
    ds = xr.open_mfdataset(
        files, combine="by_coords", preprocess=preprocess_peak,
        parallel=False, engine="h5netcdf", coords="minimal", compat="override"
    )
    if WBT_VAR not in ds:
        raise KeyError(f"{WBT_VAR} absent; vars={list(ds.data_vars)}")
    wbt = restrict_period_jjas(subset_ap(ds[WBT_VAR]))
    print("Loading 1980-2024 JJAS daily-peak WBT...")
    wbt = wbt.load()
    # Normalize daily timestamp convention once.
    wbt = wbt.assign_coords(time=pd.DatetimeIndex(pd.to_datetime(wbt.time.values)).normalize())
    print(f"WBT grid: time={wbt.sizes['time']}, lat={wbt.sizes['latitude']}, lon={wbt.sizes['longitude']}")
    return wbt


def build_local_hhe_mask(wbt):
    print("Computing each pixel's 1980-2024 JJAS p95 WBT threshold...")
    p95 = wbt.quantile(PCTL, dim="time", skipna=True)
    if "quantile" in p95.dims:
        p95 = p95.squeeze("quantile", drop=True)
    hhe = np.isfinite(wbt) & np.isfinite(p95) & (wbt >= p95)
    total = hhe.sum("time").astype(np.int16)
    positive = total.where(total > 0)
    print(
        "Local HHE count per valid pixel: "
        f"median={float(positive.median()):.0f}, "
        f"min={int(positive.min())}, max={int(total.max())}"
    )
    return hhe, p95.astype(np.float32), total

# ============================================================
# ENVIRONMENTAL DATA
# ============================================================
def open_anom_field(folder):
    files = sorted((ANOM_ROOT / folder).glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No files in {ANOM_ROOT / folder}")
    ds = xr.open_mfdataset(
        files, combine="by_coords", parallel=False, engine="h5netcdf",
        coords="minimal", compat="override"
    )
    if len(ds.data_vars) != 1:
        raise ValueError(f"Expected one var in {folder}; got {list(ds.data_vars)}")
    return restrict_period_jjas(subset_ap(ds[list(ds.data_vars)[0]]))


def load_atmos_fields():
    print("\nOpening atmospheric moisture fields...")
    return {
        "q": open_anom_field("specific_humidity") * KGKG_TO_GKG,
        "u": open_anom_field("u_component_of_wind"),
        "v": open_anom_field("v_component_of_wind"),
        "mfmag": open_anom_field("moisture_flux_mag_925") * KGKG_TO_GKG,
        "mfu": open_anom_field("moisture_flux_u_925") * KGKG_TO_GKG,
        "mfv": open_anom_field("moisture_flux_v_925") * KGKG_TO_GKG,
    }


def load_soil_fields():
    files = sorted(glob.glob(SOIL_GLOB))
    if not files:
        raise FileNotFoundError(f"No soil files matched {SOIL_GLOB}")
    print(f"\nOpening {len(files)} ERA5-Land soil-moisture files...")
    ds = xr.open_mfdataset(
        files, combine="by_coords", parallel=False, engine="h5netcdf",
        coords="minimal", compat="override"
    )
    t = find_time_name(ds)
    if t != "time":
        ds = ds.rename({t: "time"})
    for var in ["swvl1", "swvl2"]:
        if var not in ds:
            raise KeyError(f"{var} absent; vars={list(ds.data_vars)}")
    return {
        "swvl1": restrict_period_jjas(subset_ap(ds.swvl1)),
        "swvl2": restrict_period_jjas(subset_ap(ds.swvl2)),
    }


def preprocess_sst(ds):
    # ERA5 expver can occur in only some annual files.
    if "expver" in ds.dims:
        ds = ds.max("expver", skipna=True)
    if "expver" in ds.coords and "expver" not in ds.dims:
        ds = ds.drop_vars("expver")
    return ds


def load_daily_sst():
    files = [str(SST_DIR / f"era5_sst_{y}.nc") for y in range(START_YEAR, END_YEAR + 1)]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} SST files; first={missing[0]}")
    print(f"\nOpening {len(files)} yearly SST files...")
    ds = xr.open_mfdataset(
        files, combine="by_coords", preprocess=preprocess_sst,
        parallel=False, engine="h5netcdf", coords="minimal", compat="override"
    )
    name = next((n for n in ["sst", "sea_surface_temperature"] if n in ds.data_vars), None)
    if name is None:
        if len(ds.data_vars) != 1:
            raise ValueError(f"Cannot identify SST variable; vars={list(ds.data_vars)}")
        name = list(ds.data_vars)[0]
    sst = subset_ap(ds[name])
    sst = standardize_time(sst)
    sst = sst.sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))
    print("Computing daily-mean SST from 6-hourly ERA5...")
    sst = sst.resample(time="1D").mean(skipna=True)
    sst = restrict_period_jjas(sst)
    units = str(ds[name].attrs.get("units", "")).lower()
    if units in {"k", "kelvin", "degrees_k", "degree_k"}:
        sst = sst - 273.15
        sst.attrs["units"] = "degC"
    return sst

# ============================================================
# ALIGN EVERYTHING TO DAILY-PEAK GRID/TIME
# ============================================================
def align_field(da, target_time, target_lat, target_lon, name):
    print(f"  aligning {name}...")
    da = regrid_to_target(standardize_names(da), target_lat, target_lon, method="linear")
    norm_time = pd.DatetimeIndex(pd.to_datetime(da.time.values)).normalize()
    da = da.assign_coords(time=norm_time)
    if pd.DatetimeIndex(da.time.values).has_duplicates:
        da = da.groupby("time").mean(skipna=True)
    da = da.reindex(time=target_time)
    return da.transpose("time", "latitude", "longitude")

# ============================================================
# LOCAL EVENT-CONDITIONED PHASE MINUS NEUTRAL COMPOSITES
# ============================================================
def phase_mask(labels, label, time):
    return xr.DataArray(labels == label, dims="time", coords={"time": time})


def local_composite(da, hhe, pmask):
    select = hhe & pmask & np.isfinite(da)
    count = select.sum("time").astype(np.int16)
    comp = da.where(select).mean("time", skipna=True)
    return comp.where(count >= MIN_EVENT_COUNT).astype(np.float32), count


def build_composites(fields, hhe, enso_labels, iod_labels):
    time = hhe.time
    masks = {
        "el_nino": phase_mask(enso_labels, "El Niño", time),
        "la_nina": phase_mask(enso_labels, "La Niña", time),
        "enso_neutral": phase_mask(enso_labels, "Neutral", time),
        "piod": phase_mask(iod_labels, "pIOD", time),
        "niod": phase_mask(iod_labels, "nIOD", time),
        "iod_neutral": phase_mask(iod_labels, "Neutral", time),
    }
    baseline = {"el_nino": "enso_neutral", "la_nina": "enso_neutral", "piod": "iod_neutral", "niod": "iod_neutral"}
    comps = {k: {} for k in COLUMN_KEYS}
    phase_counts = {k: {} for k in COLUMN_KEYS}
    neutral_counts = {k: {} for k in COLUMN_KEYS}

    for fname, da in fields.items():
        print(f"\nLocal-HHE composites: {fname}")
        cache = {}
        for state in ["el_nino", "la_nina", "enso_neutral", "piod", "niod", "iod_neutral"]:
            cache[state] = local_composite(da, hhe, masks[state])
        for key in COLUMN_KEYS:
            base = baseline[key]
            pcomp, pc = cache[key]
            ncomp, nc = cache[base]
            valid = (pc >= MIN_EVENT_COUNT) & (nc >= MIN_EVENT_COUNT)
            comps[key][fname] = (pcomp - ncomp).where(valid).load()
            phase_counts[key][fname] = pc.load()
            neutral_counts[key][fname] = nc.load()
    return comps, phase_counts, neutral_counts

# ============================================================
# TERRAIN
# ============================================================
def load_terrain(target_lat, target_lon):
    ds = xr.open_dataset(ELEVATION_FILE)
    preferred = ["elevation", "elev", "z", "Band1", "gmted2010", "topography", "orog"]
    var = next((v for v in preferred if v in ds.data_vars), None)
    if var is None:
        numeric = [v for v, da in ds.data_vars.items() if np.issubdtype(da.dtype, np.number)]
        if not numeric:
            raise ValueError("No numeric elevation variable found")
        var = numeric[0]
    elev = subset_ap(ds[var].squeeze(drop=True))
    elev = elev.interp(
        latitude=xr.DataArray(target_lat, dims="latitude"),
        longitude=xr.DataArray(target_lon, dims="longitude"),
        method="nearest",
    )
    arr = np.asarray(elev.values, dtype=np.float32)
    if str(elev.attrs.get("units", "")).lower() in {"km", "kilometer", "kilometers"}:
        arr *= 1000.0
    return np.isfinite(arr) & (arr >= TERRAIN_925_M), arr

# ============================================================
# SAVE PRODUCTS BEFORE PLOTTING
# ============================================================
def save_products(comps, phase_counts, neutral_counts, p95, total, terrain, elevation):
    vars_out = {
        "wbt_p95_threshold": (("latitude", "longitude"), p95.values.astype(np.float32)),
        "total_local_hhe_count": (("latitude", "longitude"), total.values.astype(np.int16)),
        "terrain_925_mask": (("latitude", "longitude"), terrain.astype(np.int8)),
        "elevation_m": (("latitude", "longitude"), elevation.astype(np.float32)),
    }
    for phase in COLUMN_KEYS:
        for field in comps[phase]:
            vars_out[f"{phase}_{field}"] = (("latitude", "longitude"), comps[phase][field].values.astype(np.float32))
            vars_out[f"{phase}_{field}_phase_event_count"] = (("latitude", "longitude"), phase_counts[phase][field].values.astype(np.int16))
            vars_out[f"{phase}_{field}_neutral_event_count"] = (("latitude", "longitude"), neutral_counts[phase][field].values.astype(np.int16))

    ds = xr.Dataset(
        vars_out,
        coords={"latitude": p95.latitude.values, "longitude": p95.longitude.values},
        attrs={
            "description": "1980-2024 moisture phase-minus-neutral composites conditioned on grid-cell-local p95 daily peak WBT days.",
            "local_hhe_definition": "wbt_daily_peak >= each grid cell's 1980-2024 JJAS p95",
            "wbt_percentile": PCTL,
            "enso_threshold": ENSO_POS_THRESH,
            "iod_threshold": IOD_POS_THRESH,
            "enso_lag_months": ENSO_LAG,
            "iod_lag_months": IOD_LAG,
            "start_year": START_YEAR,
            "end_year": END_YEAR,
            "months": "JJAS",
            "minimum_event_count": MIN_EVENT_COUNT,
            "terrain_925_threshold_m": TERRAIN_925_M,
        },
    )
    ds.to_netcdf(PRODUCTS_PATH)
    print(f"\nSaved plotting-ready products: {PRODUCTS_PATH}")

# ============================================================
# PLOT
# ============================================================
def style_map(ax, labels=False):
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.coastlines(resolution="50m", linewidth=0.75, color="0.15", zorder=20)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.50, edgecolor="0.20", zorder=20)
    gl = ax.gridlines(draw_labels=labels, linewidth=0.25, color="0.45", alpha=0.45, linestyle="--")
    if labels:
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(30, 66, 5))
        gl.ylocator = mticker.FixedLocator(np.arange(5, 40, 5))
        gl.xlabel_style = {"size": 7}
        gl.ylabel_style = {"size": 7}


def panel_label(ax, label):
    ax.text(0.02, 0.98, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=9, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=1.2), zorder=30)


def add_terrain(ax, terrain, lons, lats):
    ax.contourf(lons, lats, terrain.astype(float), levels=[0.5, 1.5], colors=["0.88"],
                transform=ccrs.PlateCarree(), zorder=10)


def add_quiver(ax, u, v, lons, lats, terrain):
    uu = np.asarray(u, float).copy(); vv = np.asarray(v, float).copy()
    uu[terrain] = np.nan; vv[terrain] = np.nan
    uu = uu[::QUIVER_STRIDE, ::QUIVER_STRIDE]
    vv = vv[::QUIVER_STRIDE, ::QUIVER_STRIDE]
    xx, yy = np.meshgrid(lons[::QUIVER_STRIDE], lats[::QUIVER_STRIDE])
    return ax.quiver(
        xx, yy, uu, vv, transform=ccrs.PlateCarree(), scale=None,
        width=0.0020, headwidth=3.7, headlength=4.2, minlength=0.05, pivot="middle",
        color="black", path_effects=[pe.Stroke(linewidth=1.0, foreground="white"), pe.Normal()], zorder=15
    )


def plot_workup(comps, terrain):
    rows = [
        ("q", "BrBG", r"925 hPa $q$", "g kg$^{-1}$", ("u", "v"), True),
        ("mfmag", "BrBG", r"925 hPa $q\mathbf{v}$", "g kg$^{-1}$ m s$^{-1}$", ("mfu", "mfv"), True),
        ("swvl1", "BrBG", "Soil moisture layer 1", r"m$^3$ m$^{-3}$", None, False),
        ("swvl2", "BrBG", "Soil moisture layer 2", r"m$^3$ m$^{-3}$", None, False),
        ("sst", "coolwarm", "SST", r"$^\circ$C", None, False),
    ]
    levels = {field: centered_levels([comps[k][field].values for k in COLUMN_KEYS]) for field, *_ in rows}
    ref = comps["el_nino"]["q"]
    lats, lons = ref.latitude.values, ref.longitude.values
    proj = ccrs.PlateCarree()

    with mpl.rc_context(ERL_RC):
        fig, axes = plt.subplots(5, 4, figsize=(11.5, 12.8), subplot_kw={"projection": proj}, constrained_layout=False)
        fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.04, wspace=0.045, hspace=0.18)
        for c, title in enumerate(COLUMN_TITLES):
            axes[0, c].set_title(title, fontsize=10, fontweight="bold")

        pi = 0
        maps = {}
        for r, (field, cmap, row_label, cbar_label, vectors, is925) in enumerate(rows):
            for c, key in enumerate(COLUMN_KEYS):
                ax = axes[r, c]
                style_map(ax, labels=(c == 0))
                vals = comps[key][field].values.astype(float).copy()
                if is925:
                    vals[terrain] = np.nan
                cf = ax.contourf(lons, lats, vals, levels=levels[field], cmap=cmap, extend="both", transform=proj, zorder=1)
                if c == 0:
                    maps[field] = cf
                if vectors is not None:
                    un, vn = vectors
                    qv = add_quiver(ax, comps[key][un].values, comps[key][vn].values, lons, lats, terrain)
                    if c == 3 and field == "q":
                        ax.quiverkey(qv, 0.49, -0.055, 5, "5 m s$^{-1}$", labelpos="E", coordinates="axes", fontproperties={"size": 7})
                    elif c == 3 and field == "mfmag":
                        ax.quiverkey(qv, 0.43, -0.055, 15, "15 g kg$^{-1}$ m s$^{-1}$", labelpos="E", coordinates="axes", fontproperties={"size": 7})
                if is925:
                    add_terrain(ax, terrain, lons, lats)
                panel_label(ax, PANEL_LABELS[pi]); pi += 1

            axes[r, 0].text(-0.25, 0.50, row_label, transform=axes[r, 0].transAxes,
                            rotation=90, ha="center", va="center", fontsize=8.5, fontweight="bold")
            cb = fig.colorbar(maps[field], ax=axes[r, :], orientation="horizontal", pad=0.045, shrink=0.84, fraction=0.025, aspect=45)
            cb.set_label(cbar_label)
            cb.ax.tick_params(labelsize=7)

        fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
        fig.savefig(PDF_PATH, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    print(f"Saved PNG: {PNG_PATH}")
    print(f"Saved PDF: {PDF_PATH}")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 64)
    print("LOCAL-HHE MOISTURE WORKUP")
    print(f"Period: {START_YEAR}-{END_YEAR} JJAS")
    print(f"Phase thresholds: ENSO/IOD +/-{ENSO_POS_THRESH}")
    print(f"Local HHE threshold: p{int(PCTL*100)} WBT")
    print("=" * 64)

    phase_df = load_phase_table()
    wbt = open_daily_peak_wbt()
    hhe, p95, total = build_local_hhe_mask(wbt)
    enso_labels, iod_labels = daily_phase_labels(wbt.time.values, phase_df)

    print("\nDaily phase-day counts on WBT axis:")
    for lab in ["El Niño", "La Niña", "Neutral"]:
        print(f"  ENSO {lab:>8s}: {np.sum(enso_labels == lab)}")
    for lab in ["pIOD", "nIOD", "Neutral"]:
        print(f"  IOD  {lab:>8s}: {np.sum(iod_labels == lab)}")

    target_lat = wbt.latitude.values
    target_lon = wbt.longitude.values
    target_time = pd.DatetimeIndex(pd.to_datetime(wbt.time.values)).normalize()
    hhe = hhe.assign_coords(time=target_time)

    raw = {**load_atmos_fields(), **load_soil_fields(), "sst": load_daily_sst()}
    print("\nAligning all environmental fields to DailyPeakState grid/time...")
    fields = {name: align_field(da, target_time, target_lat, target_lon, name) for name, da in raw.items()}

    print("\nMaterializing aligned environmental fields once...")

    for name in fields:
        print(f"  loading {name}...")
        fields[name] = fields[name].load()

    print("\nComputing local-HHE phase-minus-neutral composites...")
    comps, phase_counts, neutral_counts = build_composites(fields, hhe, enso_labels, iod_labels)
    terrain, elevation = load_terrain(target_lat, target_lon)

    # Save before plotting.
    save_products(comps, phase_counts, neutral_counts, p95, total, terrain, elevation)
    print("\nPlotting...")
    plot_workup(comps, terrain)
    print("\nDone.")


if __name__ == "__main__":
    main()
