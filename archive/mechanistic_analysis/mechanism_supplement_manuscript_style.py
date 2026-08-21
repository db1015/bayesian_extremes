#!/usr/bin/env python3
"""
supplemental figures of atmospheric composites on all JJAS and p95 days
"""

from pathlib import Path
import os
import contextlib

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ============================================================
# Paths
# ============================================================
ANOM_ROOT = Path("../data/era5_anomalies")
FIG_DIR = Path("../figures/mechanism_supplement")
FIG_DIR.mkdir(parents=True, exist_ok=True)

EVENT_CSV = Path("../data/wbt_sst_city_runs/city_daily_wbt_JJAS_with_lagged_phases.csv")

# ============================================================
# User settings
# ============================================================
CITY = None
# Examples:
# CITY = "dubai"
# CITY = ["dubai", "doha"]
# CITY = None   -> all cities pooled

MONTHS = [6, 7, 8, 9]   # JJAS
WBT_COL = "wbt_daily_peak"
DATE_COL = "time"
CITY_COL = "city"

# ------------------------------------------------------------
# plotting mode:
# "p95" = only p95 WBT days
# "all" = all JJAS days in each phase
# ------------------------------------------------------------
DAY_MODE = "p95"   # "p95" or "all"

# ------------------------------------------------------------
# climate mode switch:
# "enso" -> La Niña vs El Niño
# "iod"  -> nIOD vs pIOD
# ------------------------------------------------------------
CLIMATE_MODE = "iod"   # "enso" or "iod"

if CLIMATE_MODE.lower() == "enso":
    POS_PHASE = "La Niña"
    NEG_PHASE = "El Niño"
    PAIR_TAG = "lanina_vs_elnino"
elif CLIMATE_MODE.lower() == "iod":
    POS_PHASE = "nIOD"
    NEG_PHASE = "pIOD"
    PAIR_TAG = "niod_vs_piod"
else:
    raise ValueError("CLIMATE_MODE must be 'enso' or 'iod'")

# map extent
LON_MIN, LON_MAX = 29, 65
LAT_MIN, LAT_MAX = 5, 39

# vector thinning
QUIVER_STRIDE = 6

# percentile used for symmetric robust color scaling
ROBUST_PCT = 98

# geopotential -> geopotential height conversion
G = 9.81

# unit conversions
KGKG_TO_GKG = 1000.0

# xarray open settings
OPEN_MFDATASET_KW = dict(
    combine="by_coords",
    parallel=False,
    coords="minimal",
    compat="override",
    engine="h5netcdf",
)

# small threshold to avoid divide-by-zero in vector diagnostics
EPS = 1e-10


# ============================================================
# Figure style — matched to manuscript composites
# ============================================================
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

PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

# ============================================================
# Helpers
# ============================================================
@contextlib.contextmanager
def suppress_stderr():
    old_stderr_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(devnull)
        os.close(old_stderr_fd)


def get_lat_lon_names(da: xr.DataArray):
    lat_name = "latitude" if "latitude" in da.coords else "lat"
    lon_name = "longitude" if "longitude" in da.coords else "lon"
    return lat_name, lon_name


def find_time_coord(obj):
    for cand in ["time", "date", "valid_time"]:
        if cand in obj.coords:
            return cand
        if cand in obj.dims:
            return cand
    raise KeyError(
        f"Could not find a time coordinate/dimension. "
        f"Available coords={list(obj.coords)}, dims={list(obj.dims)}"
    )


def centered_levels_from_arrays(arrays, nlev=21, pct=ROBUST_PCT):
    vals = []
    for a in arrays:
        x = np.asarray(a).ravel()
        x = x[np.isfinite(x)]
        if x.size > 0:
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


def positive_levels_from_arrays(arrays, nlev=21, pct=ROBUST_PCT):
    vals = []
    for a in arrays:
        x = np.asarray(a).ravel()
        x = x[np.isfinite(x)]
        x = np.abs(x)
        if x.size > 0:
            vals.append(x)

    if not vals:
        return np.linspace(0, 1, nlev)

    vals = np.concatenate(vals)
    vmax = np.nanpercentile(vals, pct)

    if not np.isfinite(vmax) or vmax == 0:
        vmax = np.nanmax(vals)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    return np.linspace(0, vmax, nlev)


def angle_diff_deg(u1, v1, u2, v2):
    """
    Signed minimal angular difference in degrees between two vector fields:
    angle1 - angle2, wrapped to [-180, 180].
    """
    a1 = np.degrees(np.arctan2(v1, u1))
    a2 = np.degrees(np.arctan2(v2, u2))
    d = a1 - a2
    d = ((d + 180.0) % 360.0) - 180.0
    return d


def style_map(ax, show_grid_labels=True):
    ax.set_extent(
        [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
        crs=ccrs.PlateCarree(),
    )

    ax.coastlines(
        resolution="50m",
        linewidth=0.6,
        color="0.35",
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.35,
        edgecolor="0.45",
    )
    ax.add_feature(
        cfeature.LAKES.with_scale("50m"),
        linewidth=0.25,
        edgecolor="0.55",
        facecolor="none",
    )
    ax.add_feature(
        cfeature.RIVERS.with_scale("50m"),
        linewidth=0.25,
        edgecolor="0.60",
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
        zorder=20,
    )


def add_quiver(ax, u, v, stride=QUIVER_STRIDE, scale=None):
    lat_name, lon_name = get_lat_lon_names(u)

    u2 = u.isel({
        lat_name: slice(None, None, stride),
        lon_name: slice(None, None, stride),
    })
    v2 = v.isel({
        lat_name: slice(None, None, stride),
        lon_name: slice(None, None, stride),
    })

    lon2d, lat2d = np.meshgrid(
        u2[lon_name].values,
        u2[lat_name].values,
    )

    qv = ax.quiver(
        lon2d,
        lat2d,
        u2.values,
        v2.values,
        transform=ccrs.PlateCarree(),
        scale=scale,
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
    )

    return qv


def open_all_monthly_anoms(var_name: str) -> xr.DataArray:
    files = sorted((ANOM_ROOT / var_name).glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No anomaly files found for {var_name} in {ANOM_ROOT / var_name}")

    with suppress_stderr():
        ds = xr.open_mfdataset(files, **OPEN_MFDATASET_KW)

    if len(ds.data_vars) != 1:
        raise ValueError(f"Expected 1 variable in {var_name}, got {list(ds.data_vars)}")

    da_name = list(ds.data_vars)[0]
    da = ds[da_name]

    time_name = find_time_coord(da)
    if time_name != "time":
        da = da.rename({time_name: "time"})

    da = da.sortby("time")
    return da


def classify_enso(val):
    if pd.isna(val):
        return np.nan
    if val >= 1:
        return "El Niño"
    elif val <= -1:
        return "La Niña"
    else:
        return "Neutral"


def classify_iod(val):
    if pd.isna(val):
        return np.nan
    if val >= 1:
        return "pIOD"
    elif val <= -1:
        return "nIOD"
    else:
        return "Neutral"


def load_event_dates(day_mode=DAY_MODE):
    df = pd.read_csv(EVENT_CSV)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    if CITY is not None:
        if isinstance(CITY, str):
            cities = [CITY]
        else:
            cities = list(CITY)
        df = df[df[CITY_COL].isin(cities)].copy()

    # keep full year first so precursor months are available
    df["month_start"] = df[DATE_COL].dt.to_period("M").dt.to_timestamp()

    # choose day mode
    if day_mode.lower() == "p95":
        df["thresh"] = df.groupby(CITY_COL)[WBT_COL].transform(lambda x: x.quantile(0.95))
        df = df[df[WBT_COL] >= df["thresh"]].copy()
        mode_label = "p95"
    elif day_mode.lower() == "all":
        mode_label = "all"
    else:
        raise ValueError("DAY_MODE must be 'p95' or 'all'")

    # build monthly climate-mode table from lag==0
    monthly = df[df["lag"] == 0].copy()
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

    # shifted influence windows
    enso_shifted = monthly[["month_start", "enso_phase_now"]].copy()
    enso_shifted["month_start"] = enso_shifted["month_start"] + pd.DateOffset(months=2)
    enso_shifted = enso_shifted.rename(columns={"enso_phase_now": "enso_phase_shifted"})

    iod_shifted = monthly[["month_start", "iod_phase_now"]].copy()
    iod_shifted["month_start"] = iod_shifted["month_start"] + pd.DateOffset(months=1)
    iod_shifted = iod_shifted.rename(columns={"iod_phase_now": "iod_phase_shifted"})

    # attach current and shifted monthly labels
    df = df.merge(
        monthly[["month_start", "enso_phase_now", "iod_phase_now"]],
        on="month_start",
        how="left",
    )
    df = df.merge(enso_shifted, on="month_start", how="left")
    df = df.merge(iod_shifted, on="month_start", how="left")

    # fallback for early JJAS truncation
    df["month_num"] = df[DATE_COL].dt.month

    df["enso_phase_final"] = df["enso_phase_shifted"]
    enso_fallback = df["enso_phase_final"].isna() & df["month_num"].isin([6, 7])
    df.loc[enso_fallback, "enso_phase_final"] = df.loc[enso_fallback, "enso_phase_now"]

    df["iod_phase_final"] = df["iod_phase_shifted"]
    iod_fallback = df["iod_phase_final"].isna() & df["month_num"].isin([6])
    df.loc[iod_fallback, "iod_phase_final"] = df.loc[iod_fallback, "iod_phase_now"]

    # now restrict to JJAS
    df = df[df["month_num"].isin(MONTHS)].copy()

    # collect dates
    phase_dates = {}
    for phase in ["La Niña", "El Niño"]:
        sub = df[df["enso_phase_final"] == phase].copy()
        phase_dates[phase] = pd.DatetimeIndex(sorted(sub[DATE_COL].dt.normalize().unique()))

    for phase in ["nIOD", "pIOD"]:
        sub = df[df["iod_phase_final"] == phase].copy()
        phase_dates[phase] = pd.DatetimeIndex(sorted(sub[DATE_COL].dt.normalize().unique()))

    print(f"\nEvent counts after filtering ({mode_label} mode):")
    for k in ["La Niña", "El Niño", "nIOD", "pIOD"]:
        print(f"{k:>12s}: {len(phase_dates[k])} unique dates")

    print("\nFallback counts:")
    print(f"ENSO fallback used: {enso_fallback.sum()} rows")
    print(f"IOD  fallback used: {iod_fallback.sum()} rows")

    return phase_dates, mode_label


def composite_on_dates(da: xr.DataArray, dates: pd.DatetimeIndex) -> xr.DataArray:
    if len(dates) == 0:
        raise ValueError("No dates provided for composite")

    da = da.assign_coords(date=("time", pd.to_datetime(da["time"].values).normalize()))
    target_dates = pd.to_datetime(dates).normalize()

    mask = np.isin(da["date"].values, target_dates.values)
    da_sel = da.isel(time=mask)

    if da_sel.sizes.get("time", 0) == 0:
        raise ValueError("No matching anomaly times found for requested dates")

    return da_sel.mean("time", skipna=True)


def load_core_fields():
    z = open_all_monthly_anoms("geopotential") / G
    z = z.rename("geopotential_height_anom")

    omega = open_all_monthly_anoms("vertical_velocity")
    omega = omega.rename("vertical_velocity_anom")

    q = open_all_monthly_anoms("specific_humidity") * KGKG_TO_GKG
    q = q.rename("specific_humidity_anom_gkg")

    u = open_all_monthly_anoms("u_component_of_wind")
    u = u.rename("u_wind_anom")

    v = open_all_monthly_anoms("v_component_of_wind")
    v = v.rename("v_wind_anom")

    mfmag = open_all_monthly_anoms("moisture_flux_mag_925") * KGKG_TO_GKG
    mfmag = mfmag.rename("moisture_flux_mag_925_anom_gkg_ms")

    mfu = open_all_monthly_anoms("moisture_flux_u_925") * KGKG_TO_GKG
    mfu = mfu.rename("moisture_flux_u_925_anom_gkg_ms")

    mfv = open_all_monthly_anoms("moisture_flux_v_925") * KGKG_TO_GKG
    mfv = mfv.rename("moisture_flux_v_925_anom_gkg_ms")

    return {
        "z": z,
        "omega": omega,
        "q": q,
        "u": u,
        "v": v,
        "mfmag": mfmag,
        "mfu": mfu,
        "mfv": mfv,
    }


def build_phase_composites(fields, phase_dates, pos_phase=POS_PHASE, neg_phase=NEG_PHASE):
    if pos_phase not in phase_dates or neg_phase not in phase_dates:
        raise ValueError(f"Missing requested phases: {pos_phase}, {neg_phase}")

    out = {
        pos_phase: {},
        neg_phase: {},
        "n_pos": len(phase_dates[pos_phase]),
        "n_neg": len(phase_dates[neg_phase]),
    }

    for name, da in fields.items():
        out[pos_phase][name] = composite_on_dates(da, phase_dates[pos_phase])
        out[neg_phase][name] = composite_on_dates(da, phase_dates[neg_phase])

    return out


# ============================================================
# Supplement 1: raw composites, no subtraction
# ============================================================
def plot_raw_composites(comps, mode_label, outpath):
    """
    Supplemental raw phase composites styled to match the manuscript figure.

    Columns = the two climate phases.
    Rows    = 500 hPa Zg, 925 hPa q + winds, 925 hPa moisture flux.
    Vertical velocity is intentionally omitted.
    """
    proj = ccrs.PlateCarree()

    pos = comps[POS_PHASE]
    neg = comps[NEG_PHASE]

    z_levels = centered_levels_from_arrays(
        [pos["z"].values, neg["z"].values],
        nlev=21,
    )
    q_levels = centered_levels_from_arrays(
        [pos["q"].values, neg["q"].values],
        nlev=21,
    )
    mf_levels = centered_levels_from_arrays(
        [pos["mfmag"].values, neg["mfmag"].values],
        nlev=21,
    )

    fields = [
        ("z", "PuOr", z_levels, None, None, "500 hPa $Z_g$", "m"),
        ("q", "BrBG", q_levels, "u", "v", "925 hPa $q$", "g kg$^{-1}$"),
        (
            "mfmag",
            "BrBG",
            mf_levels,
            "mfu",
            "mfv",
            r"925 hPa $q\mathbf{v}$",
            "g kg$^{-1}$ m s$^{-1}$",
        ),
    ]

    col_comps = [pos, neg]

    with mpl.rc_context(ERL_RC):
        fig, axes = plt.subplots(
            3, 2,
            figsize=(7.1, 8.2),
            subplot_kw={"projection": proj},
            constrained_layout=False,
        )

        fig.subplots_adjust(
            left=0.10,
            right=0.98,
            top=0.91,
            bottom=0.06,
            wspace=0.06,
            hspace=0.20,
        )

        day_text = (
            "p95 WBT JJAS days"
            if mode_label == "p95"
            else "all JJAS days"
        )

        mode_name = "ENSO" if CLIMATE_MODE.lower() == "enso" else "IOD"

        fig.text(
            0.50, 0.985,
            f"{mode_name} composites on {day_text}",
            ha="center", va="top",
            fontsize=10, fontweight="bold",
        )

        fig.text(
            0.29, 0.945,
            f"{POS_PHASE}\n(n={comps['n_pos']})",
            ha="center", va="top",
            fontsize=9.5, fontweight="bold",
        )
        fig.text(
            0.74, 0.945,
            f"{NEG_PHASE}\n(n={comps['n_neg']})",
            ha="center", va="top",
            fontsize=9.5, fontweight="bold",
        )

        panel_idx = 0
        mappables = {}

        for r, (
            field_name,
            cmap,
            levels,
            u_name,
            v_name,
            row_label,
            cbar_label,
        ) in enumerate(fields):

            for c, comp in enumerate(col_comps):
                ax = axes[r, c]

                show_grid_labels = (r == 0 and c == 0)
                style_map(
                    ax,
                    show_grid_labels=show_grid_labels,
                )

                field = comp[field_name]
                lat_name, lon_name = get_lat_lon_names(field)

                cf = ax.contourf(
                    field[lon_name],
                    field[lat_name],
                    field,
                    levels=levels,
                    cmap=cmap,
                    extend="both",
                    transform=proj,
                )

                if c == 0:
                    mappables[field_name] = cf

                if u_name is not None and v_name is not None:
                    qv = add_quiver(
                        ax,
                        comp[u_name],
                        comp[v_name],
                        stride=QUIVER_STRIDE,
                        scale=None,
                    )

                    if field_name == "q":
                        ax.quiverkey(
                            qv,
                            0.50, -0.045,
                            5,
                            "5 m s$^{-1}$",
                            labelpos="E",
                            coordinates="axes",
                            fontproperties={"size": 7},
                        )

                    elif field_name == "mfmag":
                        ax.quiverkey(
                            qv,
                            0.50, -0.045,
                            15,
                            "15 g kg$^{-1}$ m s$^{-1}$",
                            labelpos="E",
                            coordinates="axes",
                            fontproperties={"size": 7},
                        )

                add_panel_label(
                    ax,
                    PANEL_LABELS[panel_idx],
                )
                panel_idx += 1

            pos_left = axes[r, 0].get_position()
            pos_right = axes[r, 1].get_position()

            xmid = (pos_left.x1 + pos_right.x0) / 2
            ymid = (pos_left.y0 + pos_left.y1) / 2

            fig.text(
                xmid,
                ymid,
                row_label,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

        cbar1 = fig.colorbar(
            mappables["z"],
            ax=axes[0, :],
            orientation="horizontal",
            pad=0.08,
            shrink=0.90,
        )
        cbar1.set_label("m")

        cbar2 = fig.colorbar(
            mappables["q"],
            ax=axes[1, :],
            orientation="horizontal",
            pad=0.08,
            shrink=0.90,
        )
        cbar2.set_label("g kg$^{-1}$")

        cbar3 = fig.colorbar(
            mappables["mfmag"],
            ax=axes[2, :],
            orientation="horizontal",
            pad=0.08,
            shrink=0.90,
        )
        cbar3.set_label("g kg$^{-1}$ m s$^{-1}$")

        for cbar in [cbar1, cbar2, cbar3]:
            cbar.ax.tick_params(labelsize=7)

        fig.savefig(
            outpath,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )

        plt.close(fig)






# ============================================================
# Main
# ============================================================
def main():
    print(f"Selected climate mode: {CLIMATE_MODE} ({POS_PHASE} vs {NEG_PHASE})")
    
    print("Loading phase dates...")
    phase_dates, mode_label = load_event_dates(day_mode=DAY_MODE)

    print("\nLoading fields...")
    fields = load_core_fields()

    print(f"\nBuilding composites for {POS_PHASE} and {NEG_PHASE}...")
    comps = build_phase_composites(fields, phase_dates, pos_phase=POS_PHASE, neg_phase=NEG_PHASE)

    out1 = FIG_DIR / f"{PAIR_TAG}_{mode_label}_raw_composites_manuscript_style.pdf"

    print("\nPlotting raw composites...")
    plot_raw_composites(comps, mode_label, out1)
    print(f"Saved {out1}")

    print("\nDone.")


if __name__ == "__main__":
    main()