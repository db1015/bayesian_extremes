#!/usr/bin/env python3
# coding: utf-8

"""
MAKE FOUR MAP-BASED CITY-INSET FIGURES

This is a plotting-only script. It does not refit any Bayesian model and does
not rerun posterior aggregation. It reads the CSV outputs already produced by
Models 1, 2, 5, and 6 and creates four independent map figures:

1. Bernoulli ENSO/IOD occurrence effects
2. GPD ENSO/IOD magnitude effects (p97.5 + p99 together)
3. Adjacent-basin warming effects (p97.5 + p99 together)
4. Joint ENSO/IOD + adjacent-basin warming effects (p97.5 + p99 together)

Figures 1-2 use all ten cities.
Figures 3-4 use the eight cities in the SST models (Riyadh and Medina omitted).

All four figures share the same AP map, inset positions, connector anchors,
terrain, coastline/border styling, fonts, and panel geometry.
"""

from pathlib import Path
import glob

import cmocean
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D


# ============================================================
# PATHS
# ============================================================

# Repository root:
# bayesian_extremes/
#   data/
#   figures/
#   map_fig_generation/
#
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
FIGURE_DIR = REPO_ROOT / "figures"

VAR = "wbt_daily_peak"

ETOPO_PATH = (
    DATA_DIR
    / "elevation"
    / "ETOPO_2022_v1_60s_N90W180_bed.nc"
)

DAILY_STATE_GLOB = str(
    DATA_DIR
    / "DailyPeakState"
    / "DailyPeakState-*.nc"
)


# ---- Model 1: Bernoulli ENSO/IOD
BERNOULLI_CSV = (
    DATA_DIR
    / f"{VAR}_daily_city_runs_bernoulli"
    / f"{VAR}_bernoulli_scenario_impacts.csv"
)

BERNOULLI_FIG_DIR = FIGURE_DIR / "city_roni_dmi_bernoulli"

BERNOULLI_OUTPUT = (
    BERNOULLI_FIG_DIR
    / f"{VAR}_bernoulli_probability_change_map_insets.pdf"
)

RONI_DMI_CSV = (
    DATA_DIR
    / "sst"
    / "roni_dmi_monthly_1950_2025.csv"
)


# ---- Model 2: GPD ENSO/IOD
GPD_CSV = (
    DATA_DIR
    / f"{VAR}_daily_city_runs"
    / f"{VAR}_pointwise_extreme_changes.csv"
)

GPD_FIG_DIR = FIGURE_DIR / "city_roni_dmi"

GPD_OUTPUT = (
    GPD_FIG_DIR
    / f"{VAR}_gpd_quantile_change_map_insets.pdf"
)


# ---- Model 5: adjacent-basin warming
BASIN_WARMING_DIR = (
    DATA_DIR
    / "wbt_sst_city_runs"
)

BASIN_WARMING_CSV = (
    BASIN_WARMING_DIR
    / f"{VAR}_city_response_to_adjacent_basin_warming_ALLBASINS_JJAS.csv"
)

BASIN_BASELINE_CSV = (
    BASIN_WARMING_DIR
    / f"{VAR}_baseline_extremes_city_JJAS.csv"
)

BASIN_WARMING_FIG_DIR = (
    FIGURE_DIR
    / "wbt_city_basin_warming_byvar"
)

BASIN_WARMING_OUTPUT = (
    BASIN_WARMING_FIG_DIR
    / f"{VAR}_city_basin_warming_map_insets.pdf"
)


# ---- Model 6: joint ENSO/IOD + local-basin warming
JOINT_CSV = (
    DATA_DIR
    / "joint_enso_local_sst_city_runs"
    / "joint_wbt_daily_peak_enso_iod_local_sst_experiments.csv"
)

JOINT_FIG_DIR = (
    FIGURE_DIR
    / "joint_enso_local_sst"
)

JOINT_OUTPUT = (
    JOINT_FIG_DIR
    / "joint_enso_local_sst_compound_map_insets.pdf"
)


# ============================================================
# CITY LOCATIONS / DISPLAY
# ============================================================

CITY_COORDS = {
    "riyadh":      (46.6753, 24.7136),
    "jeddah":      (39.1925, 21.4858),
    "aden":        (45.0187, 12.8855),
    "muscat":      (58.3829, 23.5880),
    "doha":        (51.5200, 25.2760),
    "dubai":       (55.2962, 25.2770),
    "dammam":      (50.0888, 26.4207),
    "medina":      (39.5692, 24.5247),
    "kuwait_city": (47.9774, 29.3759),
    "basra":       (47.7835, 30.5085),
}

CITY_LABELS = {
    "riyadh": "Riyadh",
    "jeddah": "Jeddah",
    "aden": "Aden",
    "muscat": "Muscat",
    "doha": "Doha",
    "dubai": "Dubai",
    "dammam": "Dammam",
    "medina": "Medina",
    "kuwait_city": "Kuwait City",
    "basra": "Basra",
}

# The exact finalized inset positions from the first two figures.
INSET_POSITIONS = {
    "kuwait_city": [0.295, 0.590, 0.180, 0.175],
    "jeddah":      [0.175, 0.255, 0.180, 0.175],
    "aden":        [0.195, 0.040, 0.180, 0.175],
    "basra":       [0.125, 0.785, 0.180, 0.175],
    "medina":      [0.095, 0.480, 0.180, 0.175],
    "riyadh":      [0.440, 0.075, 0.180, 0.175],
    "dammam":      [0.655, 0.785, 0.180, 0.175],
    "doha":        [0.680, 0.560, 0.180, 0.175],
    "dubai":       [0.490, 0.325, 0.180, 0.175],
    "muscat":      [0.635, 0.100, 0.180, 0.175],
}

# Connector attachment points within each inset:
# (0,0)=bottom-left, (1,1)=top-right.
CONNECTOR_ANCHORS = {
    "basra":       (1.00, 0.50),
    "dammam":      (0.00, 0.00),
    "kuwait_city": (1.00, 0.50),
    "medina":      (1.00, 0.40),
    "jeddah":      (0.60, 1.00),
    "aden":        (1.00, 0.50),
    "riyadh":      (0.30, 1.00),
    "doha":        (0.00, 0.50),
    "dubai":       (0.60, 1.00),
    "muscat":      (0.60, 1.00),
}

ALL_CITIES = list(INSET_POSITIONS.keys())

# Models 5 and 6 do not contain Riyadh or Medina.
COASTAL_CITIES = [
    "kuwait_city",
    "jeddah",
    "aden",
    "basra",
    "dammam",
    "doha",
    "dubai",
    "muscat",
]


# ============================================================
# MODEL-SPECIFIC CONFIGURATION
# ============================================================

# Models 1-2: ENSO/IOD scenarios.
SCENARIOS = [
    "El Niño (+1,0)",
    "La Niña (-1,0)",
    "pIOD (0,+1)",
    "nIOD (0,-1)",
    "La Niña + pIOD (-1,+1)",
    "La Niña + nIOD (-1,-1)",
    "Strong El Niño (+2,0)",
    "Strong La Niña (-2,0)",
    "Super La Niña (-2.5,0)",
]

scenario_cmap = plt.get_cmap("tab10")
SCENARIO_STYLE = {
    scenario: {
        "color": scenario_cmap(i % 10),
        "marker": "o",
    }
    for i, scenario in enumerate(SCENARIOS)
}

# Shared p97.5 / p99 encoding.
# The +/-0.16 gives the two estimates a small but clear vertical separation.
QUANTILE_STYLES = {
    0.975: {
        "marker": "o",
        "offset": -0.16,
        "label": "Overall p97.5",
    },
    0.99: {
        "marker": "s",
        "offset": +0.16,
        "label": "Overall p99",
    },
}

# Model 5: adjacent-basin warming experiments.
WARM_ORDER = ["+0.5C", "+1C", "+1.5C", "+2C"]

WARM_LABELS = {
    "+0.5C": "+0.5°C basin warming",
    "+1C": "+1.0°C basin warming",
    "+1.5C": "+1.5°C basin warming",
    "+2C": "+2.0°C basin warming",
}

# Preserve the colors already used by the Model 5 aggregation/plotting script.
WARM_COLORS = {
    "+0.5C": "#4575b4",
    "+1C": "#74add1",
    "+1.5C": "#fdae61",
    "+2C": "#d73027",
}

CITY_TO_BASIN = {
    "muscat": "gulf_oman",
    "doha": "arabian_gulf",
    "dubai": "arabian_gulf",
    "dammam": "arabian_gulf",
    "kuwait_city": "arabian_gulf",
    "basra": "arabian_gulf",
    "jeddah": "red_sea",
    "aden": "gulf_aden",
}

# Model 6: same ten compound experiments as the original Figure 7 script,
# but with shorter legend labels so the map legend stays manageable.
# tuple = (display label, N_sd, D_sd, basin_warming_C)
JOINT_SCENARIOS = [
    ("La Niña +1°C",              -1.0,  0.0, 1.0),
    ("La Niña+pIOD +1°C",         -1.0, +1.0, 1.0),
    ("La Niña+nIOD +1°C",         -1.0, -1.0, 1.0),
    ("Strong La Niña +1°C",       -2.0,  0.0, 1.0),
    ("Super La Niña +1°C",        -2.5,  0.0, 1.0),
    ("La Niña +2°C",              -1.0,  0.0, 2.0),
    ("La Niña+pIOD +2°C",         -1.0, +1.0, 2.0),
    ("La Niña+nIOD +2°C",         -1.0, -1.0, 2.0),
    ("Strong La Niña +2°C",       -2.0,  0.0, 2.0),
    ("Super La Niña +2°C",        -2.5,  0.0, 2.0),
]

joint_cmap = plt.get_cmap("tab10")
JOINT_COLORS = {
    label: joint_cmap(i % 10)
    for i, (label, _, _, _) in enumerate(JOINT_SCENARIOS)
}


# ============================================================
# STYLE / MAP
# ============================================================

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
})

geo = ccrs.PlateCarree()

borders = cfeature.NaturalEarthFeature(
    "cultural",
    "admin_0_boundary_lines_land",
    "10m",
    facecolor="none",
    edgecolor="0.55",
)

coastline = cfeature.NaturalEarthFeature(
    "physical",
    "coastline",
    "10m",
    facecolor="none",
    edgecolor="0.40",
)


# ============================================================
# ETOPO
# ============================================================

if not ETOPO_PATH.exists():
    raise FileNotFoundError(f"ETOPO file not found: {ETOPO_PATH}")

_etopo_ds = xr.open_dataset(ETOPO_PATH, engine="h5netcdf")
_zname = "z" if "z" in _etopo_ds.data_vars else list(_etopo_ds.data_vars)[0]

_etopo_ds = _etopo_ds.assign_coords(
    lon=((_etopo_ds.lon + 360) % 360)
).sortby("lon")

if _etopo_ds.lat[0] > _etopo_ds.lat[-1]:
    _etopo_ds = _etopo_ds.sortby("lat")

ETOPO = _etopo_ds[_zname].sel(
    lon=slice(35, 62),
    lat=slice(10, 35),
)


def add_etopo(ax):
    lon = ETOPO.lon.values
    lat = ETOPO.lat.values
    elev = ETOPO.values

    step = max(1, int(max(elev.shape) / 500))

    lon = lon[::step]
    lat = lat[::step]
    elev = elev[::step, ::step]

    norm = TwoSlopeNorm(
        vmin=-5000,
        vcenter=0,
        vmax=3000,
    )

    ax.pcolormesh(
        lon,
        lat,
        elev,
        cmap=cmocean.cm.topo,
        norm=norm,
        shading="auto",
        alpha=0.25,
        transform=geo,
        rasterized=True,
        zorder=0,
    )


def make_map_base(cities_to_plot):
    """Create the common AP map and draw only the requested city dots."""
    fig = plt.figure(figsize=(12.5, 8.0))

    ax_map = fig.add_axes(
        [0.19, 0.07, 0.62, 0.86],
        projection=geo,
    )

    ax_map.set_extent(
        [35, 62, 10, 35],
        crs=geo,
    )

    add_etopo(ax_map)

    ax_map.add_feature(
        borders,
        linewidth=0.28,
        alpha=0.65,
        zorder=4,
    )

    ax_map.add_feature(
        coastline,
        linewidth=0.42,
        alpha=0.75,
        zorder=5,
    )

    for city in cities_to_plot:
        lon, lat = CITY_COORDS[city]

        ax_map.scatter(
            lon,
            lat,
            s=18,
            facecolor="black",
            edgecolor="white",
            linewidth=0.5,
            transform=geo,
            zorder=20,
        )

    return fig, ax_map

def get_bernoulli_scenario_month_counts():
    """
    Count JJAS months in the historical RONI/DMI record that achieve each
    Bernoulli scenario.

    RONI and DMI are standardized, then lagged to match the Bernoulli model:
      ENSO: 2 months
      IOD : 1 month

    For single-mode scenarios, the other index must remain neutral (|z| < 1).
    Strong/super events are counted as reaching that threshold or stronger.
    """
    if not os.path.exists(RONI_DMI_CSV):
        raise FileNotFoundError(
            f"Missing RONI/DMI CSV: {RONI_DMI_CSV}"
        )

    x = pd.read_csv(RONI_DMI_CSV)

    required = {"time", "RONI", "DMI"}
    missing = required.difference(x.columns)

    if missing:
        raise KeyError(
            "RONI/DMI CSV missing required columns: "
            + ", ".join(sorted(missing))
        )

    x["time"] = pd.to_datetime(x["time"])
    x = x.sort_values("time").reset_index(drop=True)

    x["time"] = pd.to_datetime(x["time"])
    x["month"] = x["time"].dt.to_period("M")
    
    x = (
        x.groupby("month", as_index=False)
         .agg({
             "RONI": "mean",
             "DMI": "mean",
         })
    )
    
    x["time"] = x["month"].dt.to_timestamp()
    x = x.sort_values("time").reset_index(drop=True)
    
    # Standardize the two indices.
    x["N"] = (
        (x["RONI"] - x["RONI"].mean())
        / x["RONI"].std(ddof=0)
    )

    x["D"] = (
        (x["DMI"] - x["DMI"].mean())
        / x["DMI"].std(ddof=0)
    )

    # Predictor values associated with each response month.
    x["N_lag"] = x["N"].shift(2)
    x["D_lag"] = x["D"].shift(1)

    # Bernoulli analysis is JJAS.
    x = x[x["time"].dt.month.isin([6, 7, 8, 9])].copy()

    N = x["N_lag"]
    D = x["D_lag"]

    neutral_N = N.abs() < 1.0
    neutral_D = D.abs() < 1.0

    counts = {
        "El Niño (+1,0)":
            int(((N >= 1.0) & neutral_D).sum()),

        "La Niña (-1,0)":
            int(((N <= -1.0) & neutral_D).sum()),

        "pIOD (0,+1)":
            int((neutral_N & (D >= 1.0)).sum()),

        "nIOD (0,-1)":
            int((neutral_N & (D <= -1.0)).sum()),

        "La Niña + pIOD (-1,+1)":
            int(((N <= -1.0) & (D >= 1.0)).sum()),

        "La Niña + nIOD (-1,-1)":
            int(((N <= -1.0) & (D <= -1.0)).sum()),

        "Strong El Niño (+2,0)":
            int(((N >= 2.0) & neutral_D).sum()),

        "Strong La Niña (-2,0)":
            int(((N <= -2.0) & neutral_D).sum()),

        "Super La Niña (-2.5,0)":
            int(((N <= -2.5) & neutral_D).sum()),
    }

    print("\nHistorical JJAS scenario-month counts:")
    for scenario in SCENARIOS:
        print(f"{scenario:32s} {counts[scenario]:3d}")

    return counts


def add_connectors(fig, ax_map, inset_axes):
    """Connect map city dots to the manually chosen inset anchor positions."""
    fig.canvas.draw()

    def map_point_to_figure(lon, lat):
        display_xy = ax_map.transData.transform((lon, lat))
        return fig.transFigure.inverted().transform(display_xy)

    for city, ax_inset in inset_axes.items():
        lon, lat = CITY_COORDS[city]
        x_city, y_city = map_point_to_figure(lon, lat)

        box = ax_inset.get_position()

        fx, fy = CONNECTOR_ANCHORS[city]

        x_panel = box.x0 + fx * box.width
        y_panel = box.y0 + fy * box.height

        connector = Line2D(
            [x_city, x_panel],
            [y_city, y_panel],
            transform=fig.transFigure,
            linewidth=0.65,
            color="0.35",
            zorder=2,
        )

        fig.add_artist(connector)


def style_inset(ax, title, xlim, nrows):
    """Apply the shared inset style."""
    ax.axvline(
        0,
        color="0.4",
        linestyle="--",
        linewidth=0.6,
        zorder=1,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(nrows - 0.5, -0.5)
    ax.set_yticks([])

    ax.tick_params(
        axis="x",
        labelsize=6,
        length=2,
        pad=1,
    )

    ax.set_title(
        title,
        fontsize=7.5,
        pad=2,
        fontweight="bold",
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("0.45")


def padded_xlim(low_values, high_values, include_zero=False):
    """Global x limits with the same scale across every city inset."""
    xmin = float(np.nanmin(np.asarray(low_values, dtype=float)))
    xmax = float(np.nanmax(np.asarray(high_values, dtype=float)))

    if include_zero:
        xmin = min(0.0, xmin)
        xmax = max(0.0, xmax)

    span = xmax - xmin
    pad = 0.08 * span if span > 0 else 0.2

    return xmin - pad, xmax + pad


def quantile_legend_handles():
    return [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            color="0.25",
            markerfacecolor="0.25",
            markersize=4,
            label="Overall p97.5",
        ),
        Line2D(
            [0], [0],
            marker="s",
            linestyle="none",
            color="0.25",
            markerfacecolor="0.25",
            markersize=4,
            label="Overall p99",
        ),
    ]



# ============================================================
# BASELINE QUANTILE HELPERS
# ============================================================

def detect_lat_lon_names(ds):
    lat_name = next(
        (name for name in ["lat", "latitude", "y"]
         if name in ds.coords or name in ds.dims),
        None,
    )
    lon_name = next(
        (name for name in ["lon", "longitude", "x"]
         if name in ds.coords or name in ds.dims),
        None,
    )

    if lat_name is None or lon_name is None:
        raise ValueError("Could not detect latitude/longitude coordinates.")

    return lat_name, lon_name


def compute_empirical_city_quantiles():
    """
    Reproduce the empirical p97.5 / p99 city values used by the original
    ENSO/IOD GPD post-processing script.

    These are calculated from all finite DailyPeakState values at the nearest
    city grid cell. They are contextual empirical values, not a new model fit.

    NetCDF files are opened explicitly with the h5netcdf engine.
    """
    files = sorted(glob.glob(DAILY_STATE_GLOB))

    if not files:
        raise FileNotFoundError(
            f"No DailyPeakState files matched: {DAILY_STATE_GLOB}"
        )

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=False,
        engine="h5netcdf",
    )

    try:
        if VAR not in ds:
            raise KeyError(f"{VAR} not found in DailyPeakState dataset.")

        da = ds[VAR]
        lat_name, lon_name = detect_lat_lon_names(ds)

        lon_is_360 = np.nanmax(ds[lon_name].values) > 180

        out = {}

        for city in ALL_CITIES:
            lon, lat = CITY_COORDS[city]
            lon_use = lon % 360 if lon_is_360 else lon

            point = da.sel(
                {
                    lat_name: lat,
                    lon_name: lon_use,
                },
                method="nearest",
            )

            values = np.asarray(point.values).ravel()
            values = values[np.isfinite(values)]

            out[city] = {
                q: (
                    float(np.nanquantile(values, q))
                    if values.size
                    else np.nan
                )
                for q in QUANTILE_STYLES
            }

        return out

    finally:
        ds.close()


def load_basin_baselines():
    """
    Load the Model 5 reconstructed neutral baseline overall p97.5 / p99
    values already calculated by the original adjacent-basin warming script.
    """
    if not os.path.exists(BASIN_BASELINE_CSV):
        raise FileNotFoundError(
            f"Missing Model 5 baseline CSV: {BASIN_BASELINE_CSV}"
        )

    baseline_df = pd.read_csv(BASIN_BASELINE_CSV)
    baseline_df["city"] = baseline_df["city"].astype(str)
    baseline_df["quantile"] = baseline_df["quantile"].astype(float)

    out = {}

    for city in COASTAL_CITIES:
        city_df = baseline_df[baseline_df["city"] == city]

        out[city] = {}

        for q in QUANTILE_STYLES:
            row = city_df[np.isclose(city_df["quantile"], q)]

            out[city][q] = (
                float(row.iloc[0]["baseline_mean"])
                if not row.empty
                else np.nan
            )

    return out


def joint_baseline_quantiles(df):
    """
    Extract the Model 6 neutral ENSO / neutral IOD / zero-warming p97.5 and
    p99 levels from the experiment CSV.
    """
    out = {}

    for city in COASTAL_CITIES:
        city_df = df[df["city"] == city]
        out[city] = {}

        for q in QUANTILE_STYLES:
            row = city_df[
                np.isclose(city_df["N_sd"].astype(float), 0.0)
                & np.isclose(city_df["D_sd"].astype(float), 0.0)
                & np.isclose(
                    city_df["basin_warming_C"].astype(float),
                    0.0,
                )
                & np.isclose(
                    city_df["overall_quantile"].astype(float),
                    q,
                )
            ]

            if len(row) != 1:
                raise RuntimeError(
                    "Expected exactly one neutral/no-warming baseline row for "
                    f"{city}, q={q}; found {len(row)}."
                )

            out[city][q] = float(row.iloc[0]["tw_level_mean_C"])

    return out


def add_quantile_baseline_text(ax, values):
    """
    Add compact p97.5 / p99 baseline context to the upper-right of an inset.
    """
    p975 = values.get(0.975, np.nan)
    p99 = values.get(0.99, np.nan)

    lines = []

    if np.isfinite(p975):
        lines.append(f"p97.5={p975:.2f}°C")

    if np.isfinite(p99):
        lines.append(f"p99={p99:.2f}°C")

    if not lines:
        return

    ax.text(
        0.98,
        0.94,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.6,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.68,
            pad=1.0,
        ),
        zorder=10,
    )


# ============================================================
# FIGURE 1 — BERNOULLI ENSO/IOD OCCURRENCE
# ============================================================

def plot_bernoulli_map():
    if not os.path.exists(BERNOULLI_CSV):
        raise FileNotFoundError(f"Missing Bernoulli CSV: {BERNOULLI_CSV}")

    df = pd.read_csv(BERNOULLI_CSV)
    df = df[df["scenario"].isin(SCENARIOS)].copy()
    df["city"] = df["city"].astype(str)

    scenario_month_counts = get_bernoulli_scenario_month_counts()

    xlim = padded_xlim(
        100.0 * df["delta_p_hdi_low"],
        100.0 * df["delta_p_hdi_high"],
        include_zero=False,
    )

    fig, ax_map = make_map_base(ALL_CITIES)
    inset_axes = {}

    for city in ALL_CITIES:
        ax = fig.add_axes(INSET_POSITIONS[city])
        inset_axes[city] = ax

        city_df = (
            df[df["city"] == city]
            .set_index("scenario")
            .reindex(SCENARIOS)
        )

        for k, scenario in enumerate(SCENARIOS):
            if scenario not in city_df.index:
                continue

            row = city_df.loc[scenario]
            if pd.isna(row["delta_p_mean"]):
                continue

            mean = 100.0 * float(row["delta_p_mean"])
            low = 100.0 * float(row["delta_p_hdi_low"])
            high = 100.0 * float(row["delta_p_hdi_high"])

            color = SCENARIO_STYLE[scenario]["color"]

            ax.errorbar(
                mean,
                k,
                xerr=[[mean - low], [high - mean]],
                fmt="o",
                color=color,
                ecolor=color,
                markersize=4.5,
                linewidth=1.25,
                capsize=1.5,
                zorder=3,
            )

        style_inset(
            ax,
            CITY_LABELS[city],
            xlim,
            len(SCENARIOS),
        )

        baseline = 100.0 * float(np.nanmean(city_df["p_base_mean"].to_numpy()))
        ax.text(
            0.98,
            0.94,
            f"$p_0$={baseline:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.8,
        )

    add_connectors(fig, ax_map, inset_axes)

    fig.text(
        0.50,
        0.025,
        "Change in exceedance probability (%)",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=SCENARIO_STYLE[scenario]["color"],
            markeredgecolor=SCENARIO_STYLE[scenario]["color"],
            markersize=4.5,
            label=f"{scenario}  {scenario_month_counts[scenario]}",
        )
        for scenario in SCENARIOS
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.475, 0.995),
        ncol=3,
        fontsize=6.5,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.35,
    )

    BERNOULLI_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        BERNOULLI_OUTPUT,
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved Bernoulli map: {BERNOULLI_OUTPUT}")
    plt.close(fig)


# ============================================================
# FIGURE 2 — GPD ENSO/IOD MAGNITUDE
# ============================================================

def plot_gpd_map():
    if not os.path.exists(GPD_CSV):
        raise FileNotFoundError(f"Missing GPD CSV: {GPD_CSV}")

    # Reproduce the empirical p97.5 / p99 context from the original
    # post-processing script using h5netcdf.
    empirical_baselines = compute_empirical_city_quantiles()

    df = pd.read_csv(GPD_CSV)
    df = df[df["scenario"].isin(SCENARIOS)].copy()
    df["city"] = df["city"].astype(str)

    xlim = padded_xlim(
        df["delta_hdi_low"],
        df["delta_hdi_high"],
        include_zero=False,
    )

    fig, ax_map = make_map_base(ALL_CITIES)
    inset_axes = {}

    for city in ALL_CITIES:
        ax = fig.add_axes(INSET_POSITIONS[city])
        inset_axes[city] = ax

        city_df = df[df["city"] == city]

        for k, scenario in enumerate(SCENARIOS):
            scenario_df = city_df[city_df["scenario"] == scenario]
            color = SCENARIO_STYLE[scenario]["color"]

            for q, qstyle in QUANTILE_STYLES.items():
                row = scenario_df[
                    np.isclose(
                        scenario_df["overall_quantile"].astype(float),
                        q,
                    )
                ]

                if row.empty:
                    continue

                row = row.iloc[0]

                mean = float(row["delta_mean"])
                low = float(row["delta_hdi_low"])
                high = float(row["delta_hdi_high"])
                y = k + qstyle["offset"]

                ax.errorbar(
                    mean,
                    y,
                    xerr=[[mean - low], [high - mean]],
                    fmt=qstyle["marker"],
                    color=color,
                    ecolor=color,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markersize=3.0,
                    linewidth=0.75,
                    capsize=1.4,
                    zorder=3,
                )

        style_inset(
            ax,
            CITY_LABELS[city],
            xlim,
            len(SCENARIOS),
        )

        add_quantile_baseline_text(
            ax,
            empirical_baselines[city],
        )

    add_connectors(fig, ax_map, inset_axes)

    fig.text(
        0.50,
        0.025,
        "",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    scenario_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=SCENARIO_STYLE[scenario]["color"],
            markeredgecolor=SCENARIO_STYLE[scenario]["color"],
            markersize=4,
            label=scenario,
        )
        for scenario in SCENARIOS
    ]

    legend1 = fig.legend(
        handles=scenario_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        fontsize=6.5,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.35,
    )

    fig.legend(
        handles=quantile_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=2,
        fontsize=6.3,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.4,
    )
    fig.add_artist(legend1)

    os.makedirs(GPD_FIG_DIR, exist_ok=True)
    fig.savefig(
        GPD_OUTPUT,
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved GPD map: {GPD_OUTPUT}")
    plt.close(fig)


# ============================================================
# FIGURE 3 — ADJACENT-BASIN WARMING
# ============================================================

def plot_basin_warming_map():
    if not os.path.exists(BASIN_WARMING_CSV):
        raise FileNotFoundError(
            f"Missing adjacent-basin warming CSV: {BASIN_WARMING_CSV}"
        )

    # These neutral reconstructed p97.5 / p99 values were already
    # calculated by the original Model 5 aggregation script.
    basin_baselines = load_basin_baselines()

    df = pd.read_csv(BASIN_WARMING_CSV)
    df["city"] = df["city"].astype(str)

    # Keep WBT only if the CSV carries a target-variable column.
    if "target_var" in df.columns:
        df = df[df["target_var"] == VAR].copy()

    # Retain only each city's assigned adjacent basin.
    if "basin_warmed" in df.columns:
        df = df[
            df.apply(
                lambda row: (
                    row["city"] in CITY_TO_BASIN
                    and CITY_TO_BASIN[row["city"]] == row["basin_warmed"]
                ),
                axis=1,
            )
        ].copy()

    df = df[
        df["city"].isin(COASTAL_CITIES)
        & df["warming"].isin(WARM_ORDER)
    ].copy()

    quantile_col = (
        "overall_quantile"
        if "overall_quantile" in df.columns
        else "quantile"
    )
    df[quantile_col] = df[quantile_col].astype(float)

    required = {
        "city",
        "warming",
        quantile_col,
        "delta_mean",
        "delta_hdi_low",
        "delta_hdi_high",
    }
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(
            "Adjacent-basin warming CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    xlim = padded_xlim(
        df["delta_hdi_low"],
        df["delta_hdi_high"],
        include_zero=True,
    )

    fig, ax_map = make_map_base(COASTAL_CITIES)
    inset_axes = {}

    for city in COASTAL_CITIES:
        ax = fig.add_axes(INSET_POSITIONS[city])
        inset_axes[city] = ax

        city_df = df[df["city"] == city]

        for k, warming in enumerate(WARM_ORDER):
            warming_df = city_df[city_df["warming"] == warming]
            color = WARM_COLORS[warming]

            for q, qstyle in QUANTILE_STYLES.items():
                row = warming_df[
                    np.isclose(
                        warming_df[quantile_col],
                        q,
                    )
                ]

                if row.empty:
                    continue

                row = row.iloc[0]

                mean = float(row["delta_mean"])
                low = float(row["delta_hdi_low"])
                high = float(row["delta_hdi_high"])
                y = k + qstyle["offset"]

                ax.errorbar(
                    mean,
                    y,
                    xerr=[[mean - low], [high - mean]],
                    fmt=qstyle["marker"],
                    color=color,
                    ecolor=color,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markersize=3.0,
                    linewidth=0.75,
                    capsize=1.4,
                    zorder=3,
                )

        style_inset(
            ax,
            CITY_LABELS[city],
            xlim,
            len(WARM_ORDER),
        )

        add_quantile_baseline_text(
            ax,
            basin_baselines[city],
        )

    add_connectors(fig, ax_map, inset_axes)

    fig.text(
        0.50,
        0.025,
        "",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    warming_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=WARM_COLORS[warming],
            markeredgecolor=WARM_COLORS[warming],
            markersize=4,
            label=WARM_LABELS[warming],
        )
        for warming in WARM_ORDER
    ]

    legend1 = fig.legend(
        handles=warming_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=4,
        fontsize=6.5,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.35,
    )

    fig.legend(
        handles=quantile_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=2,
        fontsize=6.3,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.4,
    )
    fig.add_artist(legend1)

    os.makedirs(BASIN_WARMING_FIG_DIR, exist_ok=True)
    fig.savefig(
        BASIN_WARMING_OUTPUT,
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved adjacent-basin warming map: {BASIN_WARMING_OUTPUT}")
    plt.close(fig)


# ============================================================
# FIGURE 4 — JOINT ENSO/IOD + BASIN WARMING
# ============================================================

def select_joint_row(city_df, N, D, warming, quantile):
    match = city_df[
        np.isclose(city_df["N_sd"].astype(float), N)
        & np.isclose(city_df["D_sd"].astype(float), D)
        & np.isclose(city_df["basin_warming_C"].astype(float), warming)
        & np.isclose(city_df["overall_quantile"].astype(float), quantile)
    ]

    if len(match) != 1:
        raise RuntimeError(
            "Expected exactly one joint-model row for "
            f"N={N}, D={D}, warming={warming}, q={quantile}; "
            f"found {len(match)}."
        )

    return match.iloc[0]


def plot_joint_compound_map():
    if not os.path.exists(JOINT_CSV):
        raise FileNotFoundError(f"Missing joint-model CSV: {JOINT_CSV}")

    df = pd.read_csv(JOINT_CSV)
    df["city"] = df["city"].astype(str)
    df = df[df["city"].isin(COASTAL_CITIES)].copy()

    # Neutral ENSO / neutral IOD / zero-warming baseline levels are
    # already present in the Model 6 experiment table.
    joint_baselines = joint_baseline_quantiles(df)

    required = {
        "city",
        "N_sd",
        "D_sd",
        "basin_warming_C",
        "overall_quantile",
        "delta_total_mean_C",
        "delta_total_hdi_low_C",
        "delta_total_hdi_high_C",
    }
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(
            "Joint-model CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    selected_lows = []
    selected_highs = []

    for city in COASTAL_CITIES:
        city_df = df[df["city"] == city]

        for _, N, D, warming in JOINT_SCENARIOS:
            for q in QUANTILE_STYLES:
                row = select_joint_row(
                    city_df,
                    N=N,
                    D=D,
                    warming=warming,
                    quantile=q,
                )
                selected_lows.append(float(row["delta_total_hdi_low_C"]))
                selected_highs.append(float(row["delta_total_hdi_high_C"]))

    xlim = padded_xlim(
        selected_lows,
        selected_highs,
        include_zero=True,
    )

    fig, ax_map = make_map_base(COASTAL_CITIES)
    inset_axes = {}

    for city in COASTAL_CITIES:
        ax = fig.add_axes(INSET_POSITIONS[city])
        inset_axes[city] = ax

        city_df = df[df["city"] == city]

        for k, (label, N, D, warming) in enumerate(JOINT_SCENARIOS):
            color = JOINT_COLORS[label]

            for q, qstyle in QUANTILE_STYLES.items():
                row = select_joint_row(
                    city_df,
                    N=N,
                    D=D,
                    warming=warming,
                    quantile=q,
                )

                mean = float(row["delta_total_mean_C"])
                low = float(row["delta_total_hdi_low_C"])
                high = float(row["delta_total_hdi_high_C"])
                y = k + qstyle["offset"]

                ax.errorbar(
                    mean,
                    y,
                    xerr=[[mean - low], [high - mean]],
                    fmt=qstyle["marker"],
                    color=color,
                    ecolor=color,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markersize=3.0,
                    linewidth=0.75,
                    capsize=1.4,
                    zorder=3,
                )

        style_inset(
            ax,
            CITY_LABELS[city],
            xlim,
            len(JOINT_SCENARIOS),
        )

        add_quantile_baseline_text(
            ax,
            joint_baselines[city],
        )

    add_connectors(fig, ax_map, inset_axes)

    fig.text(
        0.50,
        0.025,
        "",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    joint_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=JOINT_COLORS[label],
            markeredgecolor=JOINT_COLORS[label],
            markersize=4,
            label=label,
        )
        for label, _, _, _ in JOINT_SCENARIOS
    ]

    # Five columns -> two rows: +1 C block then +2 C block.
    legend1 = fig.legend(
        handles=joint_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.998),
        ncol=5,
        fontsize=5.7,
        frameon=False,
        columnspacing=0.85,
        handletextpad=0.25,
    )

    fig.legend(
        handles=quantile_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.936),
        ncol=2,
        fontsize=6.3,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.4,
    )
    fig.add_artist(legend1)

    os.makedirs(JOINT_FIG_DIR, exist_ok=True)
    fig.savefig(
        JOINT_OUTPUT,
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved joint compound map: {JOINT_OUTPUT}")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n=== 1 / 4: Bernoulli ENSO/IOD occurrence map ===")
    plot_bernoulli_map()

    print("\n=== 2 / 4: GPD ENSO/IOD magnitude map ===")
    plot_gpd_map()

    print("\n=== 3 / 4: Adjacent-basin warming map ===")
    plot_basin_warming_map()

    print("\n=== 4 / 4: Joint ENSO/IOD + basin warming map ===")
    plot_joint_compound_map()

    print("\nDone. Four map figures created.")


if __name__ == "__main__":
    main()
