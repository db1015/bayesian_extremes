#!/usr/bin/env python3
"""
Generate ENSO- and IOD-phase composites of extreme humid heat
across the Arabian Peninsula.

Daily p95 wet-bulb temperature (Tw), concurrent air temperature (Ta),
and specific humidity (q) are compared between lagged ENSO/IOD phases
and their respective neutral conditions during JJAS.

Statistical significance is assessed using a month-block bootstrap that
preserves all daily observations within each sampled month.

Final analysis settings:
    ENSO lag: 2 months
    IOD lag: 1 month
    phase thresholds: +/- 0.5
    bootstrap replicates: 1000
    bootstrap seed: 58

The script saves both manuscript figures and a NetCDF containing the
phase-minus-neutral anomaly fields and significance masks.
"""

# Prevent nested BLAS/OpenMP threading. We parallelize bootstrap replicates ourselves.
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
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# PATHS
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
FIGURE_DIR = REPO_ROOT / "figures"

DAILY_PEAK_GLOB = str(
    DATA_DIR / "DailyPeakState" / "DailyPeakState-*.nc"
)

PHASE_CSV = (
    DATA_DIR / "sst" / "roni_dmi_monthly_1950_2025.csv"
)

OUT_DIR = FIGURE_DIR / "phase_maps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEMP_VAR = "t2m_at_wbt_daily_peak"
Q_VAR = "q_at_wbt_daily_peak"
WBT_VAR = "wbt_daily_peak"

LAT_MIN, LAT_MAX = 10, 34
LON_MIN, LON_MAX = 34, 60

PCTL = 0.95
MIN_COUNT = 10
USE_MONTHS = [6, 7, 8, 9]

# ENSO settings
ENSO_LAG = 2
ENSO_POS_THRESH = 0.5
ENSO_NEG_THRESH = -0.5

# IOD settings
IOD_LAG = 1
IOD_POS_THRESH = 0.5
IOD_NEG_THRESH = -0.5

# Bootstrap settings
N_BOOT = 1000
BOOT_CI_LOW = 2.5
BOOT_CI_HIGH = 97.5
BOOT_SEED = 58
N_WORKERS = int(
    os.environ.get(
        "SLURM_CPUS_PER_TASK",
        os.cpu_count() or 1,
    )
)

# Stippling
STIPPLE_SIZE = 2.0
STIPPLE_ALPHA = 0.65
STIPPLE_STRIDE = 1

# Output
PNG_PATH = OUT_DIR / "enso_iod_phase_p95_anoms_relative_neutral_12panel_bootstrap.png"
PDF_PATH = OUT_DIR / "enso_iod_phase_p95_anoms_relative_neutral_12panel_bootstrap_manuscript.pdf"

# ============================================================
# PHASE CLASSIFICATION
# ============================================================
def classify_enso_from_roni(val):
    if pd.isna(val):
        return np.nan
    if val >= ENSO_POS_THRESH:
        return "El Nino"
    if val <= ENSO_NEG_THRESH:
        return "La Nina"
    return "Neutral"


def classify_iod_from_dmi(val):
    if pd.isna(val):
        return np.nan
    if val >= IOD_POS_THRESH:
        return "pIOD"
    if val <= IOD_NEG_THRESH:
        return "nIOD"
    return "Neutral"


# ============================================================
# DATA LOADING
# ============================================================
def standardize_time_dim(ds: xr.Dataset) -> xr.Dataset:
    if "day" in ds.dims:
        ds = ds.rename({"day": "time"})
    if "day" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"day": "time"})

    ds = ds.sortby("latitude")
    ds = ds.sortby("longitude")

    ds = ds.sel(
        latitude=slice(LAT_MIN, LAT_MAX),
        longitude=slice(LON_MIN, LON_MAX),
    )
    return ds


def open_daily_peak_dataset() -> xr.Dataset:
    files = sorted(glob.glob(DAILY_PEAK_GLOB))
    if not files:
        raise FileNotFoundError(f"No files matched:\n{DAILY_PEAK_GLOB}")

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        preprocess=standardize_time_dim,
        engine="h5netcdf",
    )

    needed = [WBT_VAR, TEMP_VAR, Q_VAR]
    missing = [v for v in needed if v not in ds.data_vars]
    if missing:
        raise ValueError(
            f"Missing required variables: {missing}\n"
            f"Available variables include: {list(ds.data_vars)}"
        )

    return ds


def load_phase_table() -> pd.DataFrame:
    df = pd.read_csv(PHASE_CSV)

    required = ["time", "RONI", "DMI"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in PHASE_CSV: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["time"] = pd.to_datetime(df["time"])
    df["ym"] = df["time"].dt.to_period("M")

    # One value per calendar month.
    df = (
        df.sort_values("time")
        .groupby("ym", as_index=False)
        .first()
        .copy()
    )
    df = df.sort_values("ym").reset_index(drop=True)

    # Apply lags BEFORE restricting to JJAS.
    df["RONI_lagged"] = df["RONI"].shift(ENSO_LAG)
    df["DMI_lagged"] = df["DMI"].shift(IOD_LAG)

    df["enso_phase_lagged"] = df["RONI_lagged"].map(classify_enso_from_roni)
    df["iod_phase_lagged"] = df["DMI_lagged"].map(classify_iod_from_dmi)

    df["month"] = df["ym"].dt.month.astype(int)

    if USE_MONTHS is not None:
        df = df[df["month"].isin(USE_MONTHS)].copy()

    return df


def attach_monthly_phases(ds: xr.Dataset, phase_df: pd.DataFrame) -> xr.Dataset:
    time_index = pd.to_datetime(ds["time"].values)

    if USE_MONTHS is not None:
        keep = pd.Series(time_index).dt.month.isin(USE_MONTHS).to_numpy()
        ds = ds.isel(time=np.where(keep)[0])
        time_index = pd.to_datetime(ds["time"].values)

    ym = pd.Series(time_index).dt.to_period("M")
    phase_lookup = phase_df.set_index("ym")

    enso_vals = ym.map(phase_lookup["enso_phase_lagged"]).to_numpy()
    iod_vals = ym.map(phase_lookup["iod_phase_lagged"]).to_numpy()

    ds = ds.assign_coords(
        enso_phase_lagged=("time", enso_vals),
        iod_phase_lagged=("time", iod_vals),
    )

    return ds


# ============================================================
# MONTH-BLOCK SETUP
# ============================================================
def build_month_groups(time_values):
    """
    Return:
      month_periods: one Period[M] per unique month, in chronological order
      month_indices: list of integer daily-index arrays, one array per month
    """
    periods = pd.PeriodIndex(pd.to_datetime(time_values), freq="M")
    unique_months = periods.unique().sort_values()

    month_indices = []
    for ym in unique_months:
        month_indices.append(np.where(periods == ym)[0].astype(np.int32))

    return unique_months, month_indices


def month_phase_labels(ds, month_periods, phase_coord):
    """
    Pull one phase label per unique month. All days in a month should carry
    the same monthly teleconnection classification.
    """
    periods_daily = pd.PeriodIndex(pd.to_datetime(ds["time"].values), freq="M")
    phase_daily = np.asarray(ds[phase_coord].values, dtype=object)

    labels = []
    for ym in month_periods:
        idx = np.where(periods_daily == ym)[0]
        vals = pd.Series(phase_daily[idx]).dropna().unique()

        if len(vals) == 0:
            labels.append(np.nan)
        elif len(vals) == 1:
            labels.append(vals[0])
        else:
            raise ValueError(
                f"Month {ym} has multiple {phase_coord} labels: {vals}"
            )

    return np.asarray(labels, dtype=object)


def select_month_blocks(month_indices, month_labels, wanted_label):
    selected = [
        month_indices[i]
        for i, label in enumerate(month_labels)
        if label == wanted_label
    ]

    if not selected:
        raise ValueError(f"No months found for phase '{wanted_label}'")

    return selected


# ============================================================
# P95 / BOOTSTRAP HELPERS
# ============================================================
def spatial_p95(data_3var, daily_idx):
    """
    data_3var shape: (3, time, lat, lon)
    daily_idx: 1-D integer indices into time

    Returns shape: (3, lat, lon), float32
    """
    sample = data_3var[:, daily_idx, :, :]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = np.nanpercentile(
            sample,
            PCTL * 100.0,
            axis=1,
        )

    valid_count = np.sum(np.isfinite(sample), axis=1)
    out = np.where(valid_count >= MIN_COUNT, out, np.nan)

    return out.astype(np.float32, copy=False)


def observed_phase_minus_neutral(data_3var, phase_months, neutral_months):
    phase_idx = np.concatenate(phase_months)
    neutral_idx = np.concatenate(neutral_months)

    phase_p95 = spatial_p95(data_3var, phase_idx)
    neutral_p95 = spatial_p95(data_3var, neutral_idx)

    return (phase_p95 - neutral_p95).astype(np.float32, copy=False)


def _bootstrap_one(
    seed,
    data_3var,
    phase_months,
    neutral_months,
):
    """
    One month-block bootstrap replicate.

    Resamples phase months with replacement and neutral months with replacement,
    preserving all daily observations within each sampled month.
    """
    rng = np.random.default_rng(seed)

    phase_draw = rng.integers(
        0,
        len(phase_months),
        size=len(phase_months),
    )
    neutral_draw = rng.integers(
        0,
        len(neutral_months),
        size=len(neutral_months),
    )

    phase_idx = np.concatenate([phase_months[i] for i in phase_draw])
    neutral_idx = np.concatenate([neutral_months[i] for i in neutral_draw])

    phase_p95 = spatial_p95(data_3var, phase_idx)
    neutral_p95 = spatial_p95(data_3var, neutral_idx)

    return (phase_p95 - neutral_p95).astype(np.float32, copy=False)


def bootstrap_ci_for_comparison(
    name,
    data_3var,
    phase_months,
    neutral_months,
    n_boot=N_BOOT,
    n_workers=N_WORKERS,
    seed=BOOT_SEED,
):
    """
    Bootstrap one phase-vs-neutral comparison.

    Returns:
      observed: (3, lat, lon)
      ci_low:   (3, lat, lon)
      ci_high:  (3, lat, lon)
      sig:      (3, lat, lon), boolean
    """
    print(
        f"\n{name}: "
        f"{len(phase_months)} phase months vs "
        f"{len(neutral_months)} neutral months"
    )

    observed = observed_phase_minus_neutral(
        data_3var,
        phase_months,
        neutral_months,
    )

    nlat = data_3var.shape[2]
    nlon = data_3var.shape[3]

    # ~120 MB for 1000 reps on a 100x100 grid with 3 variables.
    boot = np.empty(
        (n_boot, 3, nlat, nlon),
        dtype=np.float32,
    )

    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(n_boot)
    int_seeds = [
        int(s.generate_state(1, dtype=np.uint32)[0])
        for s in child_seeds
    ]

    print(
        f"{name}: running {n_boot} month-block bootstrap replicates "
        f"with {n_workers} threads..."
    )

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _bootstrap_one,
                int_seeds[b],
                data_3var,
                phase_months,
                neutral_months,
            ): b
            for b in range(n_boot)
        }

        completed = 0

        for future in as_completed(futures):
            b = futures[future]
            boot[b] = future.result()

            completed += 1
            if completed % 100 == 0 or completed == n_boot:
                print(f"{name}: {completed}/{n_boot} replicates complete")

    print(f"{name}: calculating {BOOT_CI_LOW:.1f}-{BOOT_CI_HIGH:.1f}% CI...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ci_low = np.nanpercentile(
            boot,
            BOOT_CI_LOW,
            axis=0,
        ).astype(np.float32)
        ci_high = np.nanpercentile(
            boot,
            BOOT_CI_HIGH,
            axis=0,
        ).astype(np.float32)

    sig = (
        ((ci_low > 0.0) | (ci_high < 0.0))
        & np.isfinite(observed)
        & np.isfinite(ci_low)
        & np.isfinite(ci_high)
    )

    # Release the large bootstrap stack before moving to the next comparison.
    del boot

    return observed, ci_low, ci_high, sig


# ============================================================
# PLOTTING HELPERS
# ============================================================
def nice_cbar_limit(*arrays, percentile=98):
    vals = []

    for arr in arrays:
        x = np.asarray(arr).ravel()
        x = x[np.isfinite(x)]
        if x.size:
            vals.append(x)

    if not vals:
        return 1.0

    vals = np.concatenate(vals)
    vmax = np.nanpercentile(np.abs(vals), percentile)
    return float(max(vmax, 0.25))


def add_map_features(ax):
    ax.set_extent(
        [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
        crs=ccrs.PlateCarree(),
    )

    ax.add_feature(cfeature.LAND, facecolor="0.92", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        linestyle="--",
        alpha=0.5,
    )

    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()


def add_panel_label(ax, label):
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.8,
            pad=1.5,
        ),
        zorder=10,
    )


def add_significance_stippling(ax, sig_mask, lons, lats):
    """
    Add black stippling where the 95% bootstrap CI excludes zero.
    """
    if STIPPLE_STRIDE > 1:
        sig_use = sig_mask[::STIPPLE_STRIDE, ::STIPPLE_STRIDE]
        lon_use = lons[::STIPPLE_STRIDE]
        lat_use = lats[::STIPPLE_STRIDE]
    else:
        sig_use = sig_mask
        lon_use = lons
        lat_use = lats

    yy, xx = np.where(sig_use)

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
        zorder=8,
    )


# ============================================================
# PLOTTING
# ============================================================
def plot_main_12panel(
    observed_maps,
    significance_masks,
    lats,
    lons,
    png_path,
    pdf_path,
):
    """
    observed_maps and significance_masks are dictionaries with keys:
      el_nino, la_nina, piod, niod

    Each value has shape (3, lat, lon), where variable order is:
      0 = Tw
      1 = Ta at max Tw
      2 = q at max Tw
    """
    proj = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(16, 10),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )

    # Make plotting copies so q can be expressed in g/kg without changing source.
    plot_maps = {
        key: value.copy()
        for key, value in observed_maps.items()
    }

    for key in plot_maps:
        plot_maps[key][2] *= 1000.0

    # Preserve original shared temperature scale across Tw + Ta.
    vmax_temp = nice_cbar_limit(
        plot_maps["el_nino"][0],
        plot_maps["la_nina"][0],
        plot_maps["piod"][0],
        plot_maps["niod"][0],
        plot_maps["el_nino"][1],
        plot_maps["la_nina"][1],
        plot_maps["piod"][1],
        plot_maps["niod"][1],
    )

    norm_temp = TwoSlopeNorm(
        vmin=-vmax_temp,
        vcenter=0.0,
        vmax=vmax_temp,
    )

    vmax_q = nice_cbar_limit(
        plot_maps["el_nino"][2],
        plot_maps["la_nina"][2],
        plot_maps["piod"][2],
        plot_maps["niod"][2],
    )

    norm_q = TwoSlopeNorm(
        vmin=-vmax_q,
        vcenter=0.0,
        vmax=vmax_q,
    )

    column_keys = ["el_nino", "la_nina", "piod", "niod"]
    column_titles = ["El Niño", "La Niña", "pIOD", "nIOD"]

    panel_labels = [
        "(a)", "(b)", "(c)", "(d)",
        "(e)", "(f)", "(g)", "(h)",
        "(i)", "(j)", "(k)", "(l)",
    ]

    mappable_temp = None
    mappable_q = None
    label_i = 0

    # Row 0: Tw
    for j, key in enumerate(column_keys):
        ax = axes[0, j]
        add_map_features(ax)

        field = plot_maps[key][0]

        mappable_temp = ax.pcolormesh(
            lons,
            lats,
            field,
            transform=proj,
            cmap="coolwarm",
            norm=norm_temp,
            shading="auto",
        )

        add_significance_stippling(
            ax,
            significance_masks[key][0],
            lons,
            lats,
        )

        add_panel_label(ax, panel_labels[label_i])
        label_i += 1

    # Row 1: Ta
    for j, key in enumerate(column_keys):
        ax = axes[1, j]
        add_map_features(ax)

        field = plot_maps[key][1]

        mappable_temp = ax.pcolormesh(
            lons,
            lats,
            field,
            transform=proj,
            cmap="coolwarm",
            norm=norm_temp,
            shading="auto",
        )

        add_significance_stippling(
            ax,
            significance_masks[key][1],
            lons,
            lats,
        )

        add_panel_label(ax, panel_labels[label_i])
        label_i += 1

    # Row 2: q
    for j, key in enumerate(column_keys):
        ax = axes[2, j]
        add_map_features(ax)

        field = plot_maps[key][2]

        mappable_q = ax.pcolormesh(
            lons,
            lats,
            field,
            transform=proj,
            cmap="BrBG",
            norm=norm_q,
            shading="auto",
        )

        add_significance_stippling(
            ax,
            significance_masks[key][2],
            lons,
            lats,
        )

        add_panel_label(ax, panel_labels[label_i])
        label_i += 1

    # Column headers
    for j, title in enumerate(column_titles):
        axes[0, j].set_title(
            title,
            fontsize=12,
            fontweight="bold",
        )

    # Row labels
    row_labels = [
        r"$T_w$",
        r"$T_a$ at max $T_w$",
        r"$q$ at max $T_w$",
    ]

    for i, label in enumerate(row_labels):
        axes[i, 0].text(
            -0.19,
            0.5,
            label,
            transform=axes[i, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    # Colorbars
    cbar_temp = fig.colorbar(
        mappable_temp,
        ax=list(axes[0, :]) + list(axes[1, :]),
        shrink=0.92,
        pad=0.03,
    )
    cbar_temp.set_label("Difference from neutral-phase p95 (°C)")

    cbar_q = fig.colorbar(
        mappable_q,
        ax=list(axes[2, :]),
        shrink=0.92,
        pad=0.03,
    )
    cbar_q.set_label("Difference from neutral-phase p95 (g/kg)")

    # Save both outputs from the same already-computed figure.
    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main():
    print("Opening DailyPeakState files...")
    ds = open_daily_peak_dataset()

    print("Loading and lagging RONI/DMI table...")
    phase_df = load_phase_table()

    print("Attaching lagged ENSO and IOD phases...")
    ds = attach_monthly_phases(ds, phase_df)

    # --------------------------------------------------------
    # Load ONLY the three required JJAS fields into RAM ONCE.
    # This is the only expensive source-data materialization.
    # --------------------------------------------------------
    print("Loading JJAS AP Tw/Ta/q fields into RAM once...")

    work = ds[[WBT_VAR, TEMP_VAR, Q_VAR]].load()

    lats = np.asarray(work["latitude"].values)
    lons = np.asarray(work["longitude"].values)

    # Use float32 to reduce memory bandwidth and bootstrap temporary memory.
    data_3var = np.stack(
        [
            np.asarray(work[WBT_VAR].values, dtype=np.float32),
            np.asarray(work[TEMP_VAR].values, dtype=np.float32),
            np.asarray(work[Q_VAR].values, dtype=np.float32),
        ],
        axis=0,
    )

    print(
        "Loaded array shape "
        f"(variable, time, lat, lon) = {data_3var.shape}"
    )

    # --------------------------------------------------------
    # Build month blocks and one phase label per month.
    # --------------------------------------------------------
    month_periods, month_indices = build_month_groups(work["time"].values)

    enso_month_labels = month_phase_labels(
        ds,
        month_periods,
        "enso_phase_lagged",
    )
    iod_month_labels = month_phase_labels(
        ds,
        month_periods,
        "iod_phase_lagged",
    )

    # ENSO month blocks
    enso_el_months = select_month_blocks(
        month_indices,
        enso_month_labels,
        "El Nino",
    )
    enso_la_months = select_month_blocks(
        month_indices,
        enso_month_labels,
        "La Nina",
    )
    enso_neutral_months = select_month_blocks(
        month_indices,
        enso_month_labels,
        "Neutral",
    )

    # IOD month blocks
    iod_pos_months = select_month_blocks(
        month_indices,
        iod_month_labels,
        "pIOD",
    )
    iod_neg_months = select_month_blocks(
        month_indices,
        iod_month_labels,
        "nIOD",
    )
    iod_neutral_months = select_month_blocks(
        month_indices,
        iod_month_labels,
        "Neutral",
    )

    print("\nMonth counts:")
    print(f"  El Niño: {len(enso_el_months)}")
    print(f"  La Niña: {len(enso_la_months)}")
    print(f"  ENSO neutral: {len(enso_neutral_months)}")
    print(f"  pIOD: {len(iod_pos_months)}")
    print(f"  nIOD: {len(iod_neg_months)}")
    print(f"  IOD neutral: {len(iod_neutral_months)}")

    # --------------------------------------------------------
    # Bootstrap four scientific comparisons.
    # Each replicate computes Tw, Ta, and q together.
    # --------------------------------------------------------
    observed_maps = {}
    significance_masks = {}

    observed_maps["el_nino"], _, _, significance_masks["el_nino"] = (
        bootstrap_ci_for_comparison(
            "El Niño vs ENSO neutral",
            data_3var,
            enso_el_months,
            enso_neutral_months,
            seed=BOOT_SEED + 1,
        )
    )

    observed_maps["la_nina"], _, _, significance_masks["la_nina"] = (
        bootstrap_ci_for_comparison(
            "La Niña vs ENSO neutral",
            data_3var,
            enso_la_months,
            enso_neutral_months,
            seed=BOOT_SEED + 2,
        )
    )

    observed_maps["piod"], _, _, significance_masks["piod"] = (
        bootstrap_ci_for_comparison(
            "pIOD vs IOD neutral",
            data_3var,
            iod_pos_months,
            iod_neutral_months,
            seed=BOOT_SEED + 3,
        )
    )

    observed_maps["niod"], _, _, significance_masks["niod"] = (
        bootstrap_ci_for_comparison(
            "nIOD vs IOD neutral",
            data_3var,
            iod_neg_months,
            iod_neutral_months,
            seed=BOOT_SEED + 4,
        )
    )


    # ========================================================
    # SAVE FINAL PLOTTING PRODUCTS
    # ========================================================
    # Everything expensive is finished at this point.
    # This file can be used later to remake the figure without
    # rerunning any bootstrap calculations.
    # ========================================================

    print("\nSaving plotting products...")

    products = xr.Dataset(
        {
            # El Niño
            "el_nino_wbt_anom": (
                ("latitude", "longitude"),
                observed_maps["el_nino"][0],
            ),
            "el_nino_t_anom": (
                ("latitude", "longitude"),
                observed_maps["el_nino"][1],
            ),
            "el_nino_q_anom": (
                ("latitude", "longitude"),
                observed_maps["el_nino"][2],
            ),
            "el_nino_wbt_sig": (
                ("latitude", "longitude"),
                significance_masks["el_nino"][0].astype(np.int8),
            ),
            "el_nino_t_sig": (
                ("latitude", "longitude"),
                significance_masks["el_nino"][1].astype(np.int8),
            ),
            "el_nino_q_sig": (
                ("latitude", "longitude"),
                significance_masks["el_nino"][2].astype(np.int8),
            ),

            # La Niña
            "la_nina_wbt_anom": (
                ("latitude", "longitude"),
                observed_maps["la_nina"][0],
            ),
            "la_nina_t_anom": (
                ("latitude", "longitude"),
                observed_maps["la_nina"][1],
            ),
            "la_nina_q_anom": (
                ("latitude", "longitude"),
                observed_maps["la_nina"][2],
            ),
            "la_nina_wbt_sig": (
                ("latitude", "longitude"),
                significance_masks["la_nina"][0].astype(np.int8),
            ),
            "la_nina_t_sig": (
                ("latitude", "longitude"),
                significance_masks["la_nina"][1].astype(np.int8),
            ),
            "la_nina_q_sig": (
                ("latitude", "longitude"),
                significance_masks["la_nina"][2].astype(np.int8),
            ),

            # pIOD
            "piod_wbt_anom": (
                ("latitude", "longitude"),
                observed_maps["piod"][0],
            ),
            "piod_t_anom": (
                ("latitude", "longitude"),
                observed_maps["piod"][1],
            ),
            "piod_q_anom": (
                ("latitude", "longitude"),
                observed_maps["piod"][2],
            ),
            "piod_wbt_sig": (
                ("latitude", "longitude"),
                significance_masks["piod"][0].astype(np.int8),
            ),
            "piod_t_sig": (
                ("latitude", "longitude"),
                significance_masks["piod"][1].astype(np.int8),
            ),
            "piod_q_sig": (
                ("latitude", "longitude"),
                significance_masks["piod"][2].astype(np.int8),
            ),

            # nIOD
            "niod_wbt_anom": (
                ("latitude", "longitude"),
                observed_maps["niod"][0],
            ),
            "niod_t_anom": (
                ("latitude", "longitude"),
                observed_maps["niod"][1],
            ),
            "niod_q_anom": (
                ("latitude", "longitude"),
                observed_maps["niod"][2],
            ),
            "niod_wbt_sig": (
                ("latitude", "longitude"),
                significance_masks["niod"][0].astype(np.int8),
            ),
            "niod_t_sig": (
                ("latitude", "longitude"),
                significance_masks["niod"][1].astype(np.int8),
            ),
            "niod_q_sig": (
                ("latitude", "longitude"),
                significance_masks["niod"][2].astype(np.int8),
            ),
        },
        coords={
            "latitude": lats,
            "longitude": lons,
        },
        attrs={
            "bootstrap_replicates": N_BOOT,
            "bootstrap_ci_low": BOOT_CI_LOW,
            "bootstrap_ci_high": BOOT_CI_HIGH,
            "bootstrap_seed": BOOT_SEED,
            "enso_lag_months": ENSO_LAG,
            "iod_lag_months": IOD_LAG,
            "pctl": PCTL,
        },
    )

    products_path = (
        OUT_DIR /
        "enso_iod_relative_composites_bootstrap_products.nc"
    )

    products.to_netcdf(products_path)

    print(f"Saved plotting products to: {products_path}")

    # ========================================================
    # PLOT
    # ========================================================

    print("\nPlotting PNG + manuscript PDF...")

    plot_main_12panel(
        observed_maps,
        significance_masks,
        lats,
        lons,
        PNG_PATH,
        PDF_PATH,
    )

    print("\nDone.")
    print(f"Saved plotting data to: {products_path}")
    print(f"Saved PNG to: {PNG_PATH}")
    print(f"Saved PDF to: {PDF_PATH}")




if __name__ == "__main__":
    main()


# In[ ]: