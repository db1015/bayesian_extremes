#!/usr/bin/env python3
# coding: utf-8
'''
============================================================
MODEL 2 OF 6 — SUPPLEMENTAL POST-PROCESSING FOR SECTION 2.2
Aggregate overall-distribution GPD scenario effects and reproduce the supplemental figure
============================================================

PURPOSE
-------
This is the single post-processing entry point for
city_remote_hier_gpd_pipeline.py. It does not refit the model. It:
  1. reads the existing GPD summary CSV and saved InferenceData files;
  2. computes changes in the reconstructed overall daily p97.5 and p99
     under fixed, standardized RONI/DMI scenarios relative to N=0, D=0;
  3. writes the existing pointwise-extreme-change CSV; and
  4. reproduces the existing multi-city supplemental PNG and PDF.

POSTERIOR CALCULATION CHOICES
-----------------------------
1. For each posterior draw:
      sigma_0 = exp(a)
      sigma_1 = sigma_0 * exp(bN*N + bD*D + bND*N*D)
   Shape xi and the fitted threshold u are held fixed within that draw.
2. The fitted threshold is the empirical overall p95. Requested overall
   daily quantiles are converted to conditional GPD probabilities using

       p_tail = (p_overall - 0.95) / 0.05.

   Therefore, overall p97.5 uses conditional GPD p50 and overall p99 uses
   conditional GPD p80. Reported effects are reconstructed overall-quantile
   changes, while the threshold-exceedance probability remains fixed at 5%.
3. The xi -> 0 exponential limit is used for numerical stability.
4. Uncertainty is summarized with 94% highest-density intervals.
5. The pooled parameter order must exactly match the fit:
      Doha, Dubai, Dammam.
6. The small p97.5/p99 annotations reproduce the prior notebook behavior:
   they are empirical overall quantiles from all finite DailyPeakState values
   at the nearest city cell, without a JJAS restriction. They provide raw-value
   context and are separate from the JJAS POT fit.

EXISTING OUTPUT LOCATIONS RETAINED
----------------------------------
CSV:
  ../data/<var>_daily_city_runs/<var>_pointwise_extreme_changes.csv
Figures:
  ../figures/city_roni_dmi/
============================================================
'''

import argparse
import glob
import os

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


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
mpl.rcParams.update(ERL_RC)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate and plot city GPD ENSO/IOD scenario effects."
    )
    parser.add_argument("--netid", default="k16v981")
    parser.add_argument("--base-data-dir", default="../data")
    parser.add_argument("--fig-dir", default="../figures/city_roni_dmi")
    parser.add_argument("--data-dir", default=None,
                        help="Optional DailyPeakState directory override")
    parser.add_argument("--glob", default=None,
                        help="Optional DailyPeakState glob override")
    parser.add_argument("--vars", default="wbt_daily_peak",
                        help="Comma-separated variables")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


SCENARIOS = [
    {"name": "El Niño (+1,0)",                 "N": +1.0, "D":  0.0},
    {"name": "La Niña (-1,0)",                 "N": -1.0, "D":  0.0},
    {"name": "pIOD (0,+1)",                    "N":  0.0, "D": +1.0},
    {"name": "nIOD (0,-1)",                    "N":  0.0, "D": -1.0},
    {"name": "La Niña + pIOD (-1,+1)",         "N": -1.0, "D": +1.0},
    {"name": "La Niña + nIOD (-1,-1)",         "N": -1.0, "D": -1.0},
    {"name": "Strong La Niña (-2,0)",          "N": -2.0, "D":  0.0},
    {"name": "Super La Niña (-2.5,0)",         "N": -2.5, "D":  0.0},
    {"name": "Strong La Niña + nIOD (-2,-1)",  "N": -2.0, "D": -1.0},
    {"name": "Strong El Niño (+2,0)",           "N": +2.0, "D":  0.0},
    {"name": "El Niño + pIOD (+1,+1)",          "N": +1.0, "D": +1.0},
    {"name": "El Niño + nIOD (+1,-1)",          "N": +1.0, "D": -1.0},
]

SCENARIOS_TO_PLOT = [
    "El Niño (+1,0)",
    "La Niña (-1,0)",
    "La Niña + nIOD (-1,-1)",
    "La Niña + pIOD (-1,+1)",
    "Strong El Niño (+2,0)",
    "Strong La Niña (-2,0)",
    "Super La Niña (-2.5,0)",
]

THRESHOLD_PROBABILITY = 0.95
OVERALL_Q_LEVELS = [0.975, 0.99]

POOLED_SPACE_ORDER = {
    "gulf_coastal_pooled": ["doha", "dubai", "dammam"],
}

CITY_ORDER = [
    "muscat", "doha", "dubai", "dammam", "kuwait_city",
    "basra", "jeddah", "aden", "medina", "riyadh",
]

CITY_LABELS = {
    "muscat": "Muscat",
    "doha": "Doha",
    "dubai": "Dubai",
    "dammam": "Dammam",
    "kuwait_city": "Kuwait City",
    "basra": "Basra",
    "jeddah": "Jeddah",
    "aden": "Aden",
    "medina": "Medina",
    "riyadh": "Riyadh",
}

CITY_COORDS = {
    "muscat": (23.5880, 58.3829),
    "doha": (25.2854, 51.5310),
    "dubai": (25.2048, 55.2708),
    "dammam": (26.4207, 50.0888),
    "kuwait_city": (29.3759, 47.9774),
    "basra": (30.5085, 47.7835),
    "jeddah": (21.4858, 39.1925),
    "aden": (12.7855, 45.0187),
    "medina": (24.5247, 39.5692),
    "riyadh": (24.7136, 46.6753),
}

VAR_LABELS = {
    "wbt_daily_peak": "Wet-Bulb Temperature",
}



def overall_to_conditional_probability(
    overall_probability,
    threshold_probability=THRESHOLD_PROBABILITY,
):
    """Map an overall-distribution quantile to the conditional GPD quantile."""
    if not threshold_probability <= overall_probability < 1.0:
        raise ValueError(
            "Overall probability must be at least the POT threshold "
            "probability and less than 1."
        )
    return (
        (overall_probability - threshold_probability)
        / (1.0 - threshold_probability)
    )


def quantile_label(probability):
    percentile = 100.0 * probability
    if float(percentile).is_integer():
        return f"p{int(percentile)}"
    return f"p{percentile:g}"


def gpd_quantile(u, sigma, xi, q, xi_tol=1e-6):
    sigma, xi = np.broadcast_arrays(
        np.asarray(sigma, dtype=float),
        np.asarray(xi, dtype=float),
    )
    result = np.empty_like(sigma)
    near_zero = np.abs(xi) < xi_tol
    result[near_zero] = (
        u + sigma[near_zero] * np.log(1.0 / (1.0 - q))
    )
    result[~near_zero] = (
        u
        + sigma[~near_zero] / xi[~near_zero]
        * ((1.0 - q) ** (-xi[~near_zero]) - 1.0)
    )
    return result


def stack_samples(da):
    return da.stack(sample=("chain", "draw")).values


def pretty_city(city):
    return CITY_LABELS.get(city, city.replace("_", " ").title())


def get_var_paths(base_data_dir, var):
    run_dir = os.path.join(base_data_dir, f"{var}_daily_city_runs")
    summary_csv = os.path.join(
        run_dir, f"{var}_daily_gpd_city_roni_dmi_summary.csv"
    )
    return run_dir, summary_csv


def load_idata(run_dir, run_id):
    path = os.path.join(run_dir, f"idata_{run_id}.nc")
    if not os.path.exists(path):
        return None, path
    return az.from_netcdf(path), path


def pooled_lookup_key(pooled_group, base_run):
    if pooled_group in POOLED_SPACE_ORDER:
        return pooled_group
    if base_run in POOLED_SPACE_ORDER:
        return base_run
    for key in POOLED_SPACE_ORDER:
        if key in str(pooled_group) or key in str(base_run):
            return key
    return None


def compute_impacts_for_var(base_data_dir, var):
    run_dir, summary_csv = get_var_paths(base_data_dir, var)
    if not os.path.exists(summary_csv):
        print(f"WARNING: missing summary CSV for {var}: {summary_csv}")
        return None

    summary_df = pd.read_csv(summary_csv)
    rows = []

    for _, record in summary_df.iterrows():
        city = record["city"]
        run_id = str(record["run_id"])
        if city == "ALL":
            continue

        if ":" in run_id:
            base_run, city_name = run_id.split(":", 1)
            idata, expected = load_idata(run_dir, base_run)
            if idata is None:
                print(f"WARNING: missing pooled idata: {expected}")
                continue

            key = pooled_lookup_key(record.get("pooled_group", ""), base_run)
            if key is None:
                raise ValueError(
                    f"Unknown pooled order for pooled_group="
                    f"'{record.get('pooled_group', '')}', base_run='{base_run}'."
                )
            order = POOLED_SPACE_ORDER[key]
            if city_name not in order:
                raise ValueError(f"City '{city_name}' not found in {order}")
            s = order.index(city_name)

            post = idata.posterior
            xi = stack_samples(post["xi"])
            a = stack_samples(post["a_s"].isel(space=s))
            b_n = stack_samples(post["bN_s"].isel(space=s))
            b_d = stack_samples(post["bD_s"].isel(space=s))
            b_nd = stack_samples(post["bND_s"].isel(space=s))
        else:
            idata, expected = load_idata(run_dir, run_id)
            if idata is None:
                print(f"WARNING: missing single-city idata: {expected}")
                continue

            post = idata.posterior
            xi = stack_samples(post["xi"])
            a = stack_samples(post["a"])
            b_n = stack_samples(post["bN"])
            b_d = stack_samples(post["bD"])
            b_nd = stack_samples(post["bND"])

        sigma_0 = np.exp(a)
        u = float(record["u"])

        for scenario in SCENARIOS:
            n_value = scenario["N"]
            d_value = scenario["D"]
            delta_log_sigma = (
                b_n * n_value
                + b_d * d_value
                + b_nd * n_value * d_value
            )
            sigma_1 = sigma_0 * np.exp(delta_log_sigma)

            for overall_q in OVERALL_Q_LEVELS:
                conditional_q = overall_to_conditional_probability(overall_q)
                x_0 = gpd_quantile(u, sigma_0, xi, conditional_q)
                x_1 = gpd_quantile(u, sigma_1, xi, conditional_q)
                delta = x_1 - x_0
                low, high = az.hdi(delta, hdi_prob=0.94)

                rows.append({
                    "var": var,
                    "city": city,
                    "run_id": run_id,
                    "scenario": scenario["name"],
                    "overall_quantile": overall_q,
                    "conditional_gpd_quantile": conditional_q,
                    "delta_mean": float(delta.mean()),
                    "delta_hdi_low": float(low),
                    "delta_hdi_high": float(high),
                    "N": n_value,
                    "D": d_value,
                })

    if not rows:
        return None

    result = pd.DataFrame(rows)
    result["city"] = pd.Categorical(
        result["city"], categories=CITY_ORDER, ordered=True
    )
    result = result.sort_values(
        ["city", "overall_quantile", "scenario"]
    ).reset_index(drop=True)

    out_csv = os.path.join(run_dir, f"{var}_pointwise_extreme_changes.csv")
    result.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)
    return result


def detect_lat_lon_names(ds):
    lat_name = next(
        (n for n in ["lat", "latitude", "y"] if n in ds.coords or n in ds.dims),
        None,
    )
    lon_name = next(
        (n for n in ["lon", "longitude", "x"] if n in ds.coords or n in ds.dims),
        None,
    )
    if lat_name is None or lon_name is None:
        raise ValueError("Could not detect latitude/longitude coordinates.")
    return lat_name, lon_name


def compute_actual_city_quantiles(data_glob, city_coords, var_name):
    files = sorted(glob.glob(data_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {data_glob}")

    ds = xr.open_mfdataset(
        files, combine="by_coords", parallel=False, engine="h5netcdf"
    )
    try:
        if var_name not in ds:
            raise KeyError(f"{var_name} not found in DailyPeakState dataset.")

        da = ds[var_name]
        lat_name, lon_name = detect_lat_lon_names(ds)
        lon_is_360 = np.nanmax(ds[lon_name].values) > 180

        city_quantiles = {}
        for city, (lat, lon) in city_coords.items():
            lon_use = lon % 360 if lon_is_360 else lon
            point = da.sel(
                {lat_name: lat, lon_name: lon_use},
                method="nearest",
            )
            values = np.asarray(point.values).ravel()
            values = values[np.isfinite(values)]
            city_quantiles[city] = {
                q: (
                    float(np.nanquantile(values, q))
                    if values.size else np.nan
                )
                for q in OVERALL_Q_LEVELS
            }
        return city_quantiles
    finally:
        ds.close()


def compute_global_xlim(df):
    sub = df[df["scenario"].isin(SCENARIOS_TO_PLOT)]
    if sub.empty:
        return -1.0, 1.0
    x_min = np.nanmin(sub["delta_hdi_low"].values)
    x_max = np.nanmax(sub["delta_hdi_high"].values)
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        return -1.0, 1.0
    span = x_max - x_min
    pad = 0.10 * span if span > 0 else 0.2
    return x_min - pad, x_max + pad


def plot_var(impact_df, actual_quantiles, var, fig_dir, show=False):
    impact_df = impact_df[
        impact_df["scenario"].isin(SCENARIOS_TO_PLOT)
    ].copy()
    if impact_df.empty:
        print(f"No matching scenarios for {var}")
        return

    present = impact_df["city"].dropna().astype(str).unique().tolist()
    cities = [city for city in CITY_ORDER if city in present]
    if not cities:
        print(f"No matching cities for {var}")
        return

    xlim = compute_global_xlim(impact_df)
    nrows = len(cities)
    ncols = len(OVERALL_Q_LEVELS)
    fig_height = max(1.75 * nrows + 0.8, 10.5)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7.2, fig_height),
        sharex=True,
        sharey=False,
    )
    axes = np.asarray(axes)
    if nrows == 1:
        axes = axes[np.newaxis, :]
    if ncols == 1:
        axes = axes[:, np.newaxis]

    cmap = plt.get_cmap("tab10")
    city_colors = {
        city: cmap(i % 10) for i, city in enumerate(cities)
    }

    y_spacing = 4.0
    y = np.arange(len(SCENARIOS_TO_PLOT)) * y_spacing

    for i, city in enumerate(cities):
        city_df = impact_df[impact_df["city"].astype(str) == city]
        color = city_colors[city]

        for j, q in enumerate(OVERALL_Q_LEVELS):
            ax = axes[i, j]
            sub = (
                city_df[city_df["overall_quantile"] == q]
                .set_index("scenario")
                .reindex(SCENARIOS_TO_PLOT)
            )

            mean = sub["delta_mean"].to_numpy()
            low = sub["delta_hdi_low"].to_numpy()
            high = sub["delta_hdi_high"].to_numpy()

            ax.errorbar(
                mean,
                y,
                xerr=[mean - low, high - mean],
                fmt="o",
                capsize=2.5,
                linewidth=1.1,
                color=color,
                ecolor=color,
                markersize=4.0,
            )
            ax.axvline(0, linestyle="--", linewidth=0.8, color="0.45")

            dy = 0.3 * y_spacing
            for k, value in enumerate(mean):
                if np.isfinite(value):
                    ax.text(
                        value,
                        y[k] + dy,
                        f"{value:.2f}",
                        ha="center",
                        va="top",
                        fontsize=7.5,
                        color=color,
                    )

            q_value = actual_quantiles.get(city, {}).get(q, np.nan)
            ax.text(
                0.98,
                0.95,
                f"{quantile_label(q)}: {q_value:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
                color=color,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "0.8",
                    "alpha": 0.95,
                    "pad": 2.0,
                },
            )

            ax.set_xlim(xlim)
            ax.set_yticks(y)
            if j == 0:
                ax.set_yticklabels(SCENARIOS_TO_PLOT, fontsize=7.5)
                ax.set_ylabel(pretty_city(city), fontsize=8.5, color=color)
            else:
                ax.set_yticklabels([])

            if i == 0:
                ax.set_title(f"Overall daily {quantile_label(q)}", fontsize=9)
            if i == nrows - 1:
                ax.set_xlabel("Change in overall daily WBT quantile", fontsize=8)

            ax.set_ylim(
                y[-1] + 1.1 * y_spacing,
                -1.1 * y_spacing,
            )

    os.makedirs(fig_dir, exist_ok=True)
    title = fig.suptitle(
        f"{VAR_LABELS.get(var, var)} response to ENSO / IOD scenarios",
        fontsize=10,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    png_out = os.path.join(
        fig_dir, f"{var}_all_cities_roni_dmi_bigpanel.png"
    )
    fig.savefig(png_out, dpi=300, bbox_inches="tight")
    print("Saved PNG:", png_out)

    title.remove()
    fig.subplots_adjust(top=0.985)
    pdf_out = os.path.join(
        fig_dir, f"{var}_all_cities_roni_dmi_bigpanel_manuscript.pdf"
    )
    fig.savefig(pdf_out)
    print("Saved PDF:", pdf_out)

    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    variables = [v.strip() for v in args.vars.split(",") if v.strip()]

    base_dir = (
        f"/home/{args.netid}/my_work/code/arabian_peninsula/"
        "bayesian_extremes/data"
    )
    daily_state_dir = args.data_dir or os.path.join(base_dir, "DailyPeakState")
    data_glob = args.glob or os.path.join(
        daily_state_dir, "DailyPeakState-*.nc"
    )

    for var in variables:
        print(f"\n=== Processing {var} ===")
        impact_df = compute_impacts_for_var(args.base_data_dir, var)
        if impact_df is None:
            continue

        actual_quantiles = compute_actual_city_quantiles(
            data_glob,
            CITY_COORDS,
            var,
        )
        plot_var(
            impact_df,
            actual_quantiles,
            var,
            args.fig_dir,
            show=args.show,
        )


if __name__ == "__main__":
    main()