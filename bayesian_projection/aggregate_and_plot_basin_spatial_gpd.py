#!/usr/bin/env python3
"""
AGGREGATE AND PLOT — SPATIAL BASIN GPD ENSO/IOD EXPERIMENTS
===========================================================

Purpose
-------
SI: Figures in support of 2.4 in manuscript
This script combines the former posterior-experiment and mapping notebooks for
Model 4 of 6. For one basin, it:

  1. loads the existing GPD posterior;
  2. reconstructs the retained POT grid cells using the exact fitting rules;
  3. evaluates each ENSO/IOD experiment;
  4. writes cell-level and basin-summary CSV files;
  5. produces the existing 3x3 posterior-mean delta-sigma map.

Experiment quantity
-------------------
For each retained grid cell and posterior draw:

    sigma_0 = exp(a_s)

    delta_log_sigma =
        bN_s * N + bD_s * D + bND_s * N * D

    delta_sigma =
        sigma_0 * (exp(delta_log_sigma) - 1).

Maps show the posterior mean delta_sigma in kelvin. Text annotations summarize
the posterior distribution of the spatially averaged delta_sigma.

Consistency decisions
---------------------
* Geometry recovery repeats monthly resampling, JJAS filtering, wet-cell
  selection, cell-specific p95 thresholds, and MIN_EVENTS filtering.
* Recovered retained-cell count must equal the posterior ``space`` dimension.
* Internal data and directory names retain ``arabian_gulf``.
* Manuscript-facing labels use ``Persian Gulf``.
* Existing figure directory and PNG filename are retained.
"""

import argparse
import os

import arviz as az
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate and map spatial basin GPD experiments."
    )
    parser.add_argument(
        "--basin",
        default="gulf_aden",
        choices=["arabian_gulf", "red_sea", "gulf_oman", "gulf_aden"],
    )
    parser.add_argument("--q", type=float, default=0.95)
    parser.add_argument("--min-events", type=int, default=5)
    parser.add_argument("--hdi-prob", type=float, default=0.94)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


ARGS = parse_args()
BASIN = ARGS.basin
Q = ARGS.q
MIN_EVENTS = ARGS.min_events
HDI_PROB = ARGS.hdi_prob

IDATA_PATH = f"../data/sst/gpd_{BASIN}_roni_dmi_idata.nc"
BASIN_NC = (
    f"../data/sst/basin_anoms/"
    f"era5_sst_anom_{BASIN}_1950_2025.nc"
)

OUT_DIR = f"../figures/roni_iod_{BASIN}_extremes"
os.makedirs(OUT_DIR, exist_ok=True)

TABLE_DIR = "../data/sst/gpd_spatial_experiment_tables"
os.makedirs(TABLE_DIR, exist_ok=True)

PNG_OUT = os.path.join(
    OUT_DIR,
    f"roni_iod_{BASIN}_extremes_dsigmaK_3x3.png",
)
PDF_OUT = os.path.join(
    OUT_DIR,
    f"roni_iod_{BASIN}_extremes_dsigmaK_3x3.pdf",
)
CELL_CSV = os.path.join(
    TABLE_DIR,
    f"gpd_{BASIN}_roni_dmi_experiment_cells.csv",
)
SUMMARY_CSV = os.path.join(
    TABLE_DIR,
    f"gpd_{BASIN}_roni_dmi_experiment_summary.csv",
)

BASIN_LABELS = {
    "arabian_gulf": "Persian Gulf",
    "red_sea": "Red Sea",
    "gulf_oman": "Gulf of Oman",
    "gulf_aden": "Gulf of Aden",
}

BASIN_EXTENTS = {
    "arabian_gulf": [47.0, 57.0, 22.0, 31.0],
    "red_sea": [32.0, 44.5, 12.0, 30.5],
    "gulf_oman": [53.0, 61.5, 21.0, 27.5],
    "gulf_aden": [41.0, 56.5, 10.0, 17.5],
}
EXTENT = BASIN_EXTENTS[BASIN]

EXPERIMENTS = [
    {"name": "El Niño (+1,0)", "N": +1.0, "D": 0.0},
    {"name": "La Niña (-1,0)", "N": -1.0, "D": 0.0},
    {"name": "pIOD (0,+1)", "N": 0.0, "D": +1.0},
    {"name": "nIOD (0,-1)", "N": 0.0, "D": -1.0},
    {"name": "Joint + (+1,+1)", "N": +1.0, "D": +1.0},
    {"name": "Opposing El Niño (+1,-1)", "N": +1.0, "D": -1.0},
    {"name": "Joint - (-1,-1)", "N": -1.0, "D": -1.0},
    {"name": "Opposing La Niña (-1,+1)", "N": -1.0, "D": +1.0},
    {
        "name": "Strong Opposing - (-1.5,+1.5)",
        "N": -1.5,
        "D": +1.5,
    },
]

ERL_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "savefig.transparent": False,
    "path.simplify": False,
}


def recover_space_geometry(path, q, min_events, expected_space):
    dataset = xr.open_dataset(path)
    sst = dataset["sst_anom"]

    lat_name = "latitude" if "latitude" in sst.coords else "lat"
    lon_name = "longitude" if "longitude" in sst.coords else "lon"

    if float(sst[lon_name].max()) > 180:
        longitude = sst[lon_name]
        shifted = ((longitude + 180) % 360) - 180
        sst = sst.assign_coords({lon_name: shifted}).sortby(lon_name)

    initial_time = pd.DatetimeIndex(
        pd.to_datetime(sst["time"].values)
    )
    monthly = (
        (initial_time.day == 1).all()
        and (initial_time.hour == 0).all()
        and (initial_time.minute == 0).all()
    )
    if not monthly:
        sst = sst.resample(time="MS").mean(skipna=True)

    jjas = pd.DatetimeIndex(
        pd.to_datetime(sst["time"].values)
    ).month.isin([6, 7, 8, 9])
    sst = sst.isel(time=jjas)

    stacked = sst.stack(space=(lat_name, lon_name))
    valid_space = np.isfinite(stacked.isel(time=0).values)
    stacked = stacked.isel(space=valid_space)

    values = stacked.values.astype("float32")
    space_index = stacked["space"].to_index()
    lats = np.array([item[0] for item in space_index])
    lons = np.array([item[1] for item in space_index])

    threshold = np.nanquantile(values, q, axis=0)
    exceedance = values > threshold[None, :]
    _, space_idx = np.where(exceedance)
    counts = np.bincount(space_idx, minlength=values.shape[1])
    keep_space = counts >= min_events

    kept_lats = lats[keep_space]
    kept_lons = lons[keep_space]

    lat_values = sst[lat_name].values.astype(float)
    lon_values = sst[lon_name].values.astype(float)
    lon2d, lat2d = np.meshgrid(lon_values, lat_values)

    lat_to_i = {value: index for index, value in enumerate(lat_values)}
    lon_to_j = {value: index for index, value in enumerate(lon_values)}
    space_to_ij = np.column_stack([
        np.array([lat_to_i[value] for value in kept_lats], dtype=int),
        np.array([lon_to_j[value] for value in kept_lons], dtype=int),
    ])

    dataset.close()

    recovered = int(keep_space.sum())
    if recovered != expected_space:
        raise ValueError(
            f"Recovered {recovered} cells, but posterior has "
            f"{expected_space}. Confirm Q={q}, MIN_EVENTS={min_events}, "
            "JJAS filtering, and the source basin file."
        )

    return kept_lons, kept_lats, lon2d, lat2d, space_to_ij


def stack_draws(da):
    return da.stack(sample=("chain", "draw")).transpose("sample", "space")


def posterior_coefficients(idata):
    posterior = idata.posterior
    return (
        stack_draws(posterior["a_s"]),
        stack_draws(posterior["bN_s"]),
        stack_draws(posterior["bD_s"]),
        stack_draws(posterior["bND_s"]),
    )


def delta_sigma_draws(idata, n_value, d_value):
    a_s, b_n, b_d, b_nd = posterior_coefficients(idata)
    sigma_zero = np.exp(a_s)
    delta_log = (
        b_n * n_value
        + b_d * d_value
        + b_nd * (n_value * d_value)
    )
    return sigma_zero * (np.exp(delta_log) - 1.0)


def summarize_experiments(idata, longitudes, latitudes):
    cell_rows = []
    summary_rows = []

    for experiment in EXPERIMENTS:
        draws = delta_sigma_draws(
            idata,
            experiment["N"],
            experiment["D"],
        )
        posterior_mean = draws.mean(dim="sample").values
        low = az.hdi(
            draws.transpose("space", "sample").values,
            hdi_prob=HDI_PROB,
        )[:, 0]
        high = az.hdi(
            draws.transpose("space", "sample").values,
            hdi_prob=HDI_PROB,
        )[:, 1]

        for space in range(len(longitudes)):
            cell_rows.append({
                "basin": BASIN,
                "basin_label": BASIN_LABELS[BASIN],
                "experiment": experiment["name"],
                "N": experiment["N"],
                "D": experiment["D"],
                "space": space,
                "longitude": float(longitudes[space]),
                "latitude": float(latitudes[space]),
                "delta_sigma_mean_K": float(posterior_mean[space]),
                "delta_sigma_hdi_low_K": float(low[space]),
                "delta_sigma_hdi_high_K": float(high[space]),
            })

        basin_draws = draws.mean(dim="space").values
        basin_low, basin_high = az.hdi(
            basin_draws,
            hdi_prob=HDI_PROB,
        )
        summary_rows.append({
            "basin": BASIN,
            "basin_label": BASIN_LABELS[BASIN],
            "experiment": experiment["name"],
            "N": experiment["N"],
            "D": experiment["D"],
            "basin_mean_delta_sigma_K": float(
                basin_draws.mean()
            ),
            "basin_mean_hdi_low_K": float(basin_low),
            "basin_mean_hdi_high_K": float(basin_high),
        })

    cell_df = pd.DataFrame(cell_rows)
    summary_df = pd.DataFrame(summary_rows)
    cell_df.to_csv(CELL_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print(f"✅ saved cell table: {CELL_CSV}")
    print(f"✅ saved basin summary: {SUMMARY_CSV}")
    return cell_df, summary_df


def field_from_space(values, lat2d, space_to_ij):
    field = np.full(lat2d.shape, np.nan)
    for space, (row, column) in enumerate(space_to_ij):
        field[row, column] = values[space]
    return field


def add_map_features(axis, show_labels=False):
    axis.coastlines(linewidth=0.8)
    axis.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.5,
    )
    axis.add_feature(cfeature.LAND, alpha=0.15)
    axis.add_feature(cfeature.OCEAN, alpha=0.05)

    gridlines = axis.gridlines(
        draw_labels=show_labels,
        linewidth=0.2,
        alpha=0.4,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    if not show_labels:
        gridlines.left_labels = False
        gridlines.bottom_labels = False


def make_maps(
    idata,
    summary_df,
    lon2d,
    lat2d,
    space_to_ij,
):
    means = []
    for experiment in EXPERIMENTS:
        draws = delta_sigma_draws(
            idata,
            experiment["N"],
            experiment["D"],
        )
        means.append(draws.mean(dim="sample").values)

    all_values = np.concatenate(means)
    finite = all_values[np.isfinite(all_values)]
    limit = np.nanpercentile(np.abs(finite), 99)
    if not np.isfinite(limit) or limit == 0:
        limit = np.nanmax(np.abs(finite))
    if not np.isfinite(limit) or limit == 0:
        limit = 1.0

    norm = mcolors.TwoSlopeNorm(
        vmin=-limit,
        vcenter=0.0,
        vmax=limit,
    )

    with mpl.rc_context(ERL_RC):
        fig, axes = plt.subplots(
            nrows=3,
            ncols=3,
            figsize=(13, 11),
            subplot_kw={"projection": ccrs.PlateCarree()},
            constrained_layout=True,
        )
        axes = axes.ravel()
        mappable = None

        for index, experiment in enumerate(EXPERIMENTS):
            axis = axes[index]
            axis.set_extent(EXTENT, crs=ccrs.PlateCarree())

            field = field_from_space(
                means[index],
                lat2d,
                space_to_ij,
            )
            mappable = axis.pcolormesh(
                lon2d,
                lat2d,
                field,
                transform=ccrs.PlateCarree(),
                shading="auto",
                cmap="RdBu_r",
                norm=norm,
            )
            add_map_features(axis, show_labels=index == 0)
            axis.set_title(experiment["name"], fontsize=11)

            row = summary_df[
                summary_df["experiment"] == experiment["name"]
            ].iloc[0]
            text = (
                f"{BASIN_LABELS[BASIN]} mean Δσ: "
                f"{row['basin_mean_delta_sigma_K']:+.3f} K\n"
                f"{int(HDI_PROB * 100)}% HDI: "
                f"[{row['basin_mean_hdi_low_K']:+.3f}, "
                f"{row['basin_mean_hdi_high_K']:+.3f}] K"
            )
            axis.text(
                0.02,
                0.02,
                text,
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=9,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.75,
                    "edgecolor": "none",
                    "pad": 3,
                },
            )

        colorbar = fig.colorbar(
            mappable,
            ax=axes,
            orientation="vertical",
            shrink=0.9,
            pad=0.02,
        )
        colorbar.set_label("Δσ (K)")

        fig.savefig(PNG_OUT, dpi=250, bbox_inches="tight")
        fig.savefig(PDF_OUT, dpi=300, bbox_inches="tight")

        if ARGS.show:
            plt.show()
        plt.close(fig)

    print(f"✅ saved PNG: {PNG_OUT}")
    print(f"✅ saved PDF: {PDF_OUT}")


def main():
    if not os.path.exists(IDATA_PATH):
        raise FileNotFoundError(f"Missing posterior: {IDATA_PATH}")

    idata = az.from_netcdf(IDATA_PATH)
    expected_space = idata.posterior.sizes["space"]

    longitudes, latitudes, lon2d, lat2d, mapping = (
        recover_space_geometry(
            BASIN_NC,
            q=Q,
            min_events=MIN_EVENTS,
            expected_space=expected_space,
        )
    )

    _, summary_df = summarize_experiments(
        idata,
        longitudes,
        latitudes,
    )
    make_maps(
        idata,
        summary_df,
        lon2d,
        lat2d,
        mapping,
    )


if __name__ == "__main__":
    main()
