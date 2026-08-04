#!/usr/bin/env python3
# coding: utf-8
"""
Aggregate baseline POT/GPD extremes and plot adjacent-basin SST warming impacts.

This script:

1. Loads each basin-specific fitted posterior.
2. Computes baseline unconditional p95 and p99 values for each city.
3. Loads the existing adjacent-basin warming impact CSV.
4. Produces one two-column figure per target variable.

Modeling is unchanged. Internal file and basin keys retain ``arabian_gulf``.
All plot-facing labels use ``Persian Gulf``.
"""

import os
import pickle
from collections import defaultdict

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------
# ERL manuscript style
# --------------------------------------------------
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


# --------------------------------------------------
# Paths
# --------------------------------------------------
NETID = "k16v981"

OUT_DIR = (
    f"/home/{NETID}/my_work/code/arabian_peninsula/"
    "bayesian_extremes/data/wbt_sst_city_runs"
)

FIG_DIR = "../figures/wbt_city_basin_warming_byvar"
os.makedirs(FIG_DIR, exist_ok=True)


# --------------------------------------------------
# Scientific configuration
# --------------------------------------------------
VALID_BASINS = [
    "gulf_oman",
    "arabian_gulf",
    "red_sea",
    "gulf_aden",
]

TARGET_VARS = [
    "wbt_daily_peak",
    "tau_at_wbt_daily_peak",
    "t2m_at_wbt_daily_peak",
    "q_at_wbt_daily_peak",
]

CITY_ORDER = [
    "dammam",
    "doha",
    "dubai",
    "jeddah",
    "kuwait_city",
    "basra",
    "aden",
    "muscat",
]

CITY_TO_BASIN = {
    "muscat":      "gulf_oman",
    "doha":        "arabian_gulf",
    "dubai":       "arabian_gulf",
    "dammam":      "arabian_gulf",
    "kuwait_city": "arabian_gulf",
    "basra":       "arabian_gulf",
    "jeddah":      "red_sea",
    "aden":        "gulf_aden",
}

CITY_LABELS = {
    "muscat": "Muscat",
    "doha": "Doha",
    "dubai": "Dubai",
    "dammam": "Dammam",
    "kuwait_city": "Kuwait City",
    "basra": "Basra",
    "jeddah": "Jeddah",
    "aden": "Aden",
}

SEA_LABELS = {
    "arabian_gulf": "Persian Gulf",
    "red_sea": "Red Sea",
    "gulf_aden": "Gulf of Aden",
    "gulf_oman": "Gulf of Oman",
}

VAR_LABELS = {
    "wbt_daily_peak": "Daily Maximum Wet-Bulb Temperature",
    "tau_at_wbt_daily_peak": r"$\tau_{\mathrm{MSE}}$ at Maximum $T_w$",
    "t2m_at_wbt_daily_peak": r"$T_a$ at Maximum $T_w$",
    "q_at_wbt_daily_peak": r"Specific Humidity at Maximum $T_w$",
}

VAR_UNITS = {
    "wbt_daily_peak": r"$^\circ$C",
    "tau_at_wbt_daily_peak": r"kJ kg$^{-1}$",
    "t2m_at_wbt_daily_peak": r"$^\circ$C",
    "q_at_wbt_daily_peak": r"g kg$^{-1}$",
}

Q_LEVELS = [0.95, 0.99]

# These match WARMING_EXPTS in the fitting pipeline.
WARM_ORDER = ["+0.5C", "+1C", "+1.5C", "+2C"]

WARM_COLORS = {
    "+0.5C": "#4575b4",
    "+1C": "#74add1",
    "+1.5C": "#fdae61",
    "+2C": "#d73027",
}

WARMING_EXPTS = {
    "+0.5C": 0.5,
    "+1C": 1.0,
    "+1.5C": 1.5,
    "+2C": 2.0,
}


# --------------------------------------------------
# Paths and loading helpers
# --------------------------------------------------
def idata_path(target_var, basin):
    return os.path.join(
        OUT_DIR,
        f"idata_city_hier_{target_var}_vs_sst_{basin}_JJAS.nc",
    )


def meta_path(target_var, basin):
    return os.path.join(
        OUT_DIR,
        f"meta_city_hier_{target_var}_vs_sst_{basin}_JJAS.pkl",
    )


def impact_csv_path(target_var):
    return os.path.join(
        OUT_DIR,
        (
            f"{target_var}_city_response_to_adjacent_basin_"
            "warming_ALLBASINS_JJAS.csv"
        ),
    )


def baseline_csv_path(target_var):
    return os.path.join(
        OUT_DIR,
        f"{target_var}_baseline_extremes_city_JJAS.csv",
    )


def stack_samples(data_array):
    return data_array.stack(
        sample=("chain", "draw")
    ).values


def load_idata_any(target_var, basin):
    nc_path = idata_path(target_var, basin)
    pkl_path = nc_path.replace(".nc", ".pkl")

    if os.path.exists(nc_path):
        try:
            return az.from_netcdf(
                nc_path,
                engine="netcdf4",
            ), nc_path
        except Exception as exc:
            print(
                f"⚠️ failed to open NetCDF for basin='{basin}', "
                f"var='{target_var}': {exc}"
            )

    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as handle:
                idata = pickle.load(handle)
            return idata, pkl_path
        except Exception as exc:
            print(
                f"⚠️ failed to open pickle for basin='{basin}', "
                f"var='{target_var}': {exc}"
            )

    raise FileNotFoundError(
        f"No readable idata found for target_var='{target_var}', "
        f"basin='{basin}'."
    )


# --------------------------------------------------
# POT/GPD baseline helper
# --------------------------------------------------
def gpd_unconditional_quantile(
    u,
    sigma,
    xi,
    q,
    zeta_u,
    xi_tol=1e-6,
):
    """
    Unconditional quantile for a POT/GPD model.

    The fitted threshold exceedance probability is represented empirically
    by zeta_u = n_exceedances / n_days.
    """
    u = np.asarray(u)
    sigma = np.asarray(sigma)
    xi = np.asarray(xi)

    out_shape = np.broadcast(xi, sigma).shape
    threshold_quantile = 1.0 - zeta_u

    if q <= threshold_quantile + 1e-12:
        return np.broadcast_to(
            u,
            out_shape,
        ).astype(float)

    conditional_survival = (1.0 - q) / zeta_u

    out = np.empty(
        out_shape,
        dtype=float,
    )
    near_zero = np.abs(xi) < xi_tol

    out[~near_zero] = (
        u
        + sigma[~near_zero]
        / xi[~near_zero]
        * (
            conditional_survival ** (-xi[~near_zero])
            - 1.0
        )
    )

    out[near_zero] = (
        u
        + sigma[near_zero]
        * np.log(1.0 / conditional_survival)
    )

    return out


# --------------------------------------------------
# Baseline aggregation
# --------------------------------------------------
def aggregate_baseline_one_var(target_var):
    print("=" * 80)
    print(f"BASELINE AGGREGATION: {target_var}")
    print("=" * 80)

    basin_to_cities = defaultdict(list)

    for city, basin in CITY_TO_BASIN.items():
        basin_to_cities[basin].append(city)

    all_rows = []

    for basin, target_cities in basin_to_cities.items():
        mpath = meta_path(
            target_var,
            basin,
        )

        if not os.path.exists(mpath):
            print(
                f"⚠️ missing metadata for basin='{basin}', "
                f"var='{target_var}'"
            )
            continue

        try:
            idata, used_path = load_idata_any(
                target_var,
                basin,
            )
            print(f"✅ loaded {basin}: {used_path}")
        except Exception as exc:
            print(
                f"⚠️ could not load idata for basin='{basin}', "
                f"var='{target_var}': {exc}"
            )
            continue

        meta = pd.read_pickle(mpath)

        cities = list(meta["cities"])
        u_by_city = meta["u_by_city"]
        n_days_by_city = meta["n_days_by_city"]
        n_exc_by_city = meta["n_exc_by_city"]

        post = idata.posterior
        xi = stack_samples(post["xi"])
        a_city = post["a_city"].stack(
            sample=("chain", "draw")
        )

        city_to_i = {
            city: i
            for i, city in enumerate(cities)
        }

        cities_keep = [
            city
            for city in cities
            if city in target_cities
        ]

        for city in cities_keep:
            i = city_to_i[city]
            u = float(u_by_city[city])

            n_days = int(n_days_by_city[city])
            n_exc = int(n_exc_by_city[city])
            zeta_u = n_exc / n_days

            a = a_city.isel(city=i).values
            sigma0 = np.exp(a)

            for q in Q_LEVELS:
                x0 = gpd_unconditional_quantile(
                    u,
                    sigma0,
                    xi,
                    q,
                    zeta_u,
                )

                mean = float(np.mean(x0))
                low, high = az.hdi(
                    x0,
                    hdi_prob=0.94,
                )

                all_rows.append(
                    {
                        "target_var": target_var,
                        "city": city,
                        "basin": basin,
                        "quantile": q,
                        "baseline_mean": mean,
                        "baseline_hdi_low": float(low),
                        "baseline_hdi_high": float(high),
                        "n_days": n_days,
                        "n_exc": n_exc,
                        "zeta_u": zeta_u,
                        "u": u,
                        "months": "".join(
                            str(month)
                            for month in meta["months"]
                        ),
                    }
                )

    baseline_df = pd.DataFrame(all_rows)

    if baseline_df.empty:
        raise RuntimeError(
            f"No baseline rows were produced for {target_var}."
        )

    baseline_df["city"] = pd.Categorical(
        baseline_df["city"],
        categories=CITY_ORDER,
        ordered=True,
    )

    baseline_df = baseline_df.sort_values(
        ["city", "quantile"]
    ).reset_index(drop=True)

    out_csv = baseline_csv_path(target_var)
    baseline_df.to_csv(
        out_csv,
        index=False,
    )

    print(f"✅ wrote baseline CSV: {out_csv}")
    print(
        baseline_df[
            [
                "city",
                "basin",
                "quantile",
                "baseline_mean",
                "baseline_hdi_low",
                "baseline_hdi_high",
            ]
        ].to_string(index=False)
    )

    return baseline_df


# --------------------------------------------------
# Data preparation for plotting
# --------------------------------------------------
def load_plot_data(target_var):
    impact_path = impact_csv_path(target_var)
    baseline_path = baseline_csv_path(target_var)

    if not os.path.exists(impact_path):
        raise FileNotFoundError(
            f"Missing warming-impact CSV: {impact_path}"
        )

    if not os.path.exists(baseline_path):
        raise FileNotFoundError(
            f"Missing baseline CSV: {baseline_path}"
        )

    impact_df = pd.read_csv(impact_path)
    baseline_df = pd.read_csv(baseline_path)

    # Retain only the city and its assigned adjacent basin.
    impact_df = impact_df[
        impact_df.apply(
            lambda row: (
                CITY_TO_BASIN.get(row["city"])
                == row["basin_warmed"]
            ),
            axis=1,
        )
    ].copy()

    impact_df = impact_df[
        impact_df["city"].isin(CITY_ORDER)
    ].copy()

    baseline_df = baseline_df[
        baseline_df["city"].isin(CITY_ORDER)
    ].copy()

    impact_df["quantile"] = (
        impact_df["quantile"].astype(float)
    )
    baseline_df["quantile"] = (
        baseline_df["quantile"].astype(float)
    )

    # Convert specific humidity from kg/kg to g/kg.
    if target_var == "q_at_wbt_daily_peak":
        for column in [
            "delta_mean",
            "delta_hdi_low",
            "delta_hdi_high",
        ]:
            impact_df[column] = (
                1000.0 * impact_df[column]
            )

        for column in [
            "baseline_mean",
            "baseline_hdi_low",
            "baseline_hdi_high",
            "u",
        ]:
            if column in baseline_df.columns:
                baseline_df[column] = (
                    1000.0 * baseline_df[column]
                )

    return impact_df, baseline_df


# --------------------------------------------------
# Plotting
# --------------------------------------------------
def plot_one_var(target_var, impact_df, baseline_df):
    cities_present = set(
        impact_df["city"].dropna().astype(str)
    )
    cities = [
        city
        for city in CITY_ORDER
        if city in cities_present
    ]

    if not cities:
        raise RuntimeError(
            f"No requested cities found for {target_var}."
        )

    baseline_lookup = baseline_df.set_index(
        ["city", "quantile"]
    )

    selected_impacts = impact_df[
        impact_df["warming"].isin(WARM_ORDER)
    ]

    x_low = selected_impacts[
        "delta_hdi_low"
    ].min()
    x_high = selected_impacts[
        "delta_hdi_high"
    ].max()

    x_low = min(0.0, x_low)
    x_high = max(0.0, x_high)

    pad = (
        0.08 * (x_high - x_low)
        if x_high > x_low
        else 0.2
    )
    xlim = (
        x_low - pad,
        x_high + pad,
    )

    ROW_HEIGHT = 2.7
    
    fig, axes = plt.subplots(
        nrows=len(cities),
        ncols=len(Q_LEVELS),
        figsize=(
            7.5,
            ROW_HEIGHT * len(cities),
        ),
        sharex=True,
        squeeze=False,
        gridspec_kw={
            "hspace": 0.18,
            "wspace": 0.18,
        },
    )

    panel_index = 0
    WARM_ROW_SPACING = 1.10

    local_y = (
        np.arange(len(WARM_ORDER))[::-1]
        * WARM_ROW_SPACING
    )

    for i, city in enumerate(cities):
        basin = CITY_TO_BASIN[city]

        for j, q in enumerate(Q_LEVELS):
            ax = axes[i, j]

            panel_label = (
                f"({chr(97 + panel_index)})"
            )
            panel_index += 1

            ax.text(
                -0.12,
                1.08,
                panel_label,
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
                ha="left",
            )

            ax.axvline(
                0,
                linestyle="--",
                linewidth=0.8,
                color="0.45",
            )
            ax.set_xlim(xlim)

            sub = impact_df[
                (impact_df["city"] == city)
                & (
                    impact_df["basin_warmed"]
                    == basin
                )
                & (
                    impact_df["quantile"]
                    == float(q)
                )
            ].copy()

            sub = (
                sub.set_index("warming")
                .reindex(WARM_ORDER)
            )

            try:
                baseline_row = baseline_lookup.loc[
                    (city, float(q))
                ]
                baseline_mean = float(
                    baseline_row["baseline_mean"]
                )
            except KeyError:
                baseline_mean = np.nan

            for k, warming in enumerate(WARM_ORDER):
                if (
                    warming not in sub.index
                    or pd.isna(
                        sub.loc[
                            warming,
                            "delta_mean",
                        ]
                    )
                ):
                    continue

                yy = local_y[k]

                delta_mean = float(
                    sub.loc[warming, "delta_mean"]
                )
                delta_low = float(
                    sub.loc[
                        warming,
                        "delta_hdi_low",
                    ]
                )
                delta_high = float(
                    sub.loc[
                        warming,
                        "delta_hdi_high",
                    ]
                )

                ax.errorbar(
                    delta_mean,
                    yy,
                    xerr=[
                        [delta_mean - delta_low],
                        [delta_high - delta_mean],
                    ],
                    fmt="o",
                    color=WARM_COLORS[warming],
                    capsize=3,
                    linewidth=1.2,
                    markersize=5,
                )

                if np.isfinite(baseline_mean):
                    absolute_mean = (
                        baseline_mean + delta_mean
                    )
                    absolute_low = (
                        baseline_mean + delta_low
                    )
                    absolute_high = (
                        baseline_mean + delta_high
                    )

                    if (
                        target_var
                        == "q_at_wbt_daily_peak"
                    ):
                        text = (
                            f"{absolute_mean:.1f}\n"
                            f"[{absolute_low:.1f}, "
                            f"{absolute_high:.1f}]"
                        )
                    else:
                        text = (
                            f"{absolute_mean:.2f}\n"
                            f"[{absolute_low:.2f}, "
                            f"{absolute_high:.2f}]"
                        )

                    ax.text(
                        delta_mean,
                        yy - 0.18,
                        text,
                        ha="center",
                        va="top",
                        fontsize=6.8,
                        color=WARM_COLORS[warming],
                    )

            ax.set_yticks(local_y)

            if j == 0:
                ax.set_yticklabels(
                    WARM_ORDER,
                    fontsize=8,
                )

                basin_label = SEA_LABELS.get(
                    basin,
                    basin.replace(
                        "_",
                        " ",
                    ).title(),
                )

                ax.set_ylabel(
                    (
                        f"{CITY_LABELS[city]}\n"
                        f"({basin_label})"
                    ),
                    fontsize=8.5,
                )
            else:
                ax.set_yticklabels([])

            ax.set_ylim(
                -0.8,
                len(WARM_ORDER) - 0.2,
            )

            if i == 0:
                ax.set_title(
                    f"{int(q * 100)}th percentile",
                    fontsize=9,
                )

            if i == len(cities) - 1:
                ax.set_xlabel(
                    (
                        "Change in extreme $T_w$ "
                        f"({VAR_UNITS[target_var]})"
                    ),
                    fontsize=8,
                )

    st = fig.suptitle(
        (
            f"{VAR_LABELS[target_var]} response to "
            "adjacent-basin SST warming (JJAS)"
        ),
        y=0.995,
        fontsize=10,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.985]
    )

    png_out = os.path.join(
        FIG_DIR,
        (
            f"{target_var}_city_basin_"
            "warming_2panel.png"
        ),
    )

    fig.savefig(
        png_out,
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved PNG: {png_out}")

    st.remove()
    fig.subplots_adjust(top=0.985)

    pdf_out = os.path.join(
        FIG_DIR,
        (
            f"{target_var}_city_basin_"
            "warming_2panel_manuscript.pdf"
        ),
    )

    fig.savefig(
        pdf_out,
        bbox_inches="tight",
    )
    print(f"Saved PDF: {pdf_out}")

    plt.show()
    plt.close(fig)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    for target_var in TARGET_VARS:
        print("\n" + "=" * 80)
        print(f"PROCESSING: {target_var}")
        print("=" * 80)

        try:
            aggregate_baseline_one_var(
                target_var
            )

            impact_df, baseline_df = (
                load_plot_data(target_var)
            )

            plot_one_var(
                target_var,
                impact_df,
                baseline_df,
            )

        except Exception as exc:
            print(
                f"❌ failed processing for "
                f"{target_var}: {exc}"
            )


if __name__ == "__main__":
    main()
