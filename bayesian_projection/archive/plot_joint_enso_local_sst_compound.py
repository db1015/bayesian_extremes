#!/usr/bin/env python3
# coding: utf-8
"""
Plot compound ENSO--IOD and local-basin warming experiments.

The figure contains eight city rows and two columns:

    left:  conditional GPD p95
    right: conditional GPD p99

Each point shows the total change relative to:

    Neutral ENSO + Neutral IOD + 0 degC imposed basin warming

Scenarios include:

    +1 degC basin warming alone
    +1 degC basin warming combined with five La Nina / IOD states
    +2 degC basin warming alone
    +2 degC basin warming combined with five La Nina / IOD states

Internal code retains ``arabian_gulf`` where required by saved data, but all
figure-facing text uses ``Persian Gulf``.
"""

import os

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
BASE_DATA = "../data"
RUN_DIR = os.path.join(
    BASE_DATA,
    "joint_enso_local_sst_city_runs",
)

CSV_PATH = os.path.join(
    RUN_DIR,
    "joint_wbt_daily_peak_enso_iod_local_sst_experiments.csv",
)

FIG_DIR = "../figures/joint_enso_local_sst"
os.makedirs(FIG_DIR, exist_ok=True)


# --------------------------------------------------
# Configuration
# --------------------------------------------------
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

QUANTILES = [0.95, 0.99]

# Each tuple is:
# (display label, N_sd, D_sd, basin_warming_C)
SCENARIOS_TO_PLOT = [
    (r"La Niña + basin warming (-1,0,+1$^\circ$C)", -1.0, 0.0, 1.0),
    (r"La Niña + pIOD + basin warming (-1,+1,+1$^\circ$C)", -1.0, 1.0, 1.0),
    (r"La Niña + nIOD + basin warming (-1,-1,+1$^\circ$C)", -1.0, -1.0, 1.0),
    (r"Strong La Niña + basin warming (-2,0,+1$^\circ$C)", -2.0, 0.0, 1.0),
    (r"Super La Niña + basin warming (-2.5,0,+1$^\circ$C)", -2.5, 0.0, 1.0),

    (r"La Niña + basin warming (-1,0,+2$^\circ$C)", -1.0, 0.0, 2.0),
    (r"La Niña + pIOD + basin warming (-1,+1,+2$^\circ$C)", -1.0, 1.0, 2.0),
    (r"La Niña + nIOD + basin warming (-1,-1,+2$^\circ$C)", -1.0, -1.0, 2.0),
    (r"Strong La Niña + basin warming (-2,0,+2$^\circ$C)", -2.0, 0.0, 2.0),
    (r"Super La Niña + basin warming (-2.5,0,+2$^\circ$C)", -2.5, 0.0, 2.0),
]


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def pretty_city(city):
    return CITY_LABELS.get(
        city,
        city.replace("_", " ").title(),
    )


def select_scenario_row(df, N, D, warming, quantile):
    """Return exactly one matching experiment row."""
    match = df[
        np.isclose(df["N_sd"], N)
        & np.isclose(df["D_sd"], D)
        & np.isclose(df["basin_warming_C"], warming)
        & np.isclose(df["conditional_quantile"], quantile)
    ]

    if len(match) != 1:
        raise RuntimeError(
            "Expected exactly one row for "
            f"N={N}, D={D}, warming={warming}, q={quantile}; "
            f"found {len(match)}."
        )

    return match.iloc[0]


def compute_global_xlim(df):
    selected = []

    for city in CITY_ORDER:
        city_df = df[df["city"] == city]

        for q in QUANTILES:
            for _, N, D, warming in SCENARIOS_TO_PLOT:
                row = select_scenario_row(
                    city_df,
                    N=N,
                    D=D,
                    warming=warming,
                    quantile=q,
                )
                selected.append(
                    (
                        row["delta_total_hdi_low_C"],
                        row["delta_total_hdi_high_C"],
                    )
                )

    selected = np.asarray(selected, dtype=float)

    xmin = np.nanmin(selected[:, 0])
    xmax = np.nanmax(selected[:, 1])

    if not np.isfinite(xmin) or not np.isfinite(xmax):
        return (-1.0, 1.0)

    # Always include zero because all changes are relative to the
    # neutral/no-warming baseline.
    xmin = min(0.0, xmin)
    xmax = max(0.0, xmax)

    span = xmax - xmin
    pad = 0.08 * span if span > 0 else 0.2

    return xmin - pad, xmax + pad


def baseline_level(df, city, quantile):
    """Neutral ENSO/IOD and zero-warming conditional GPD level."""
    row = select_scenario_row(
        df[df["city"] == city],
        N=0.0,
        D=0.0,
        warming=0.0,
        quantile=quantile,
    )
    return float(row["tw_level_mean_C"])


# --------------------------------------------------
# Load and validate
# --------------------------------------------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Experiment CSV not found:\n{CSV_PATH}"
    )

impact_df = pd.read_csv(CSV_PATH)

required_columns = {
    "city",
    "basin",
    "N_sd",
    "D_sd",
    "basin_warming_C",
    "conditional_quantile",
    "tw_level_mean_C",
    "delta_total_mean_C",
    "delta_total_hdi_low_C",
    "delta_total_hdi_high_C",
}

missing_columns = required_columns.difference(impact_df.columns)
if missing_columns:
    raise KeyError(
        "Experiment CSV is missing required columns: "
        + ", ".join(sorted(missing_columns))
    )

cities_present = set(impact_df["city"].dropna().astype(str))
missing_cities = [
    city for city in CITY_ORDER
    if city not in cities_present
]

if missing_cities:
    raise RuntimeError(
        "Experiment CSV is missing requested cities: "
        + ", ".join(missing_cities)
    )


# --------------------------------------------------
# Plot
# --------------------------------------------------
xlim = compute_global_xlim(impact_df)

nrows = len(CITY_ORDER)
ncols = len(QUANTILES)

fig_height = max(2.15 * nrows + 0.8, 17.5)

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(7.5, fig_height),
    sharex=True,
    sharey=False,
)

if nrows == 1 and ncols == 1:
    axes = np.array([[axes]])
elif nrows == 1:
    axes = np.array([axes])
elif ncols == 1:
    axes = axes[:, np.newaxis]

cmap = plt.get_cmap("tab10")
city_colors = {
    city: cmap(i % 10)
    for i, city in enumerate(CITY_ORDER)
}

y_spacing = 4.0
y = np.arange(len(SCENARIOS_TO_PLOT)) * y_spacing

# Visual separator between +1 C and +2 C scenario blocks.
separator_y = (
    y[len(SCENARIOS_TO_PLOT) // 2 - 1]
    + y_spacing / 2
)

panel_index = 0

for i, city in enumerate(CITY_ORDER):
    city_df = impact_df[
        impact_df["city"] == city
    ].copy()

    city_color = city_colors[city]
    basin_name = str(city_df["basin"].iloc[0])

    for j, q in enumerate(QUANTILES):
        ax = axes[i, j]

        means = []
        lows = []
        highs = []

        for _, N, D, warming in SCENARIOS_TO_PLOT:
            row = select_scenario_row(
                city_df,
                N=N,
                D=D,
                warming=warming,
                quantile=q,
            )

            means.append(
                float(row["delta_total_mean_C"])
            )
            lows.append(
                float(row["delta_total_hdi_low_C"])
            )
            highs.append(
                float(row["delta_total_hdi_high_C"])
            )

        means = np.asarray(means)
        lows = np.asarray(lows)
        highs = np.asarray(highs)

        ax.errorbar(
            means,
            y,
            xerr=[
                means - lows,
                highs - means,
            ],
            fmt="o",
            capsize=2.5,
            linewidth=1.1,
            color=city_color,
            ecolor=city_color,
            markersize=4.0,
        )

        ax.axvline(
            0,
            linestyle="--",
            linewidth=0.8,
            color="0.45",
        )

        ax.axhline(
            separator_y,
            linewidth=0.7,
            color="0.75",
        )

        # Posterior mean annotations beneath each point.
        dy = 0.30 * y_spacing

        for k, value in enumerate(means):
            if np.isfinite(value):
                ax.text(
                    value,
                    y[k] + dy,
                    f"{value:.2f}",
                    ha="center",
                    va="top",
                    fontsize=7.2,
                    color=city_color,
                )

        reference_level = baseline_level(
            impact_df,
            city=city,
            quantile=q,
        )

        ax.text(
            0.98,
            0.95,
            (
                f"p{int(q * 100)}: "
                f"{reference_level:.2f}$^\\circ$C\n"
                f"{basin_name}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.2,
            color=city_color,
            bbox=dict(
                facecolor="white",
                edgecolor="0.8",
                alpha=0.95,
                pad=2.0,
            ),
        )

        panel_label = f"({chr(97 + panel_index)})"
        ax.text(
            -0.10,
            1.04,
            panel_label,
            transform=ax.transAxes,
            fontsize=9.5,
            fontweight="bold",
            ha="left",
            va="top",
        )
        panel_index += 1

        ax.set_xlim(xlim)
        ax.set_yticks(y)

        if j == 0:
            ax.set_yticklabels(
                [item[0] for item in SCENARIOS_TO_PLOT],
                fontsize=7.1,
            )
            ax.set_ylabel(
                pretty_city(city),
                fontsize=8.5,
                color=city_color,
            )
        else:
            ax.set_yticklabels([])

        if i == 0:
            ax.set_title(
                f"Conditional GPD p{int(q * 100)}",
                fontsize=9,
            )

        if i == nrows - 1:
            ax.set_xlabel(
                (
                    "($^\\circ$C)"
                ),
                fontsize=8,
            )

        ax.set_ylim(
            y[-1] + 1.1 * y_spacing,
            -1.1 * y_spacing,
        )


# --------------------------------------------------
# Save PNG with title
# --------------------------------------------------
st = fig.suptitle(
    (
        "Compound influence of La Niña, IOD, "
        "and local-basin warming on extreme wet-bulb temperature"
    ),
    fontsize=10,
    y=0.997,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.988]
)

png_out = os.path.join(
    FIG_DIR,
    "joint_enso_local_sst_compound_all_cities.png",
)

fig.savefig(
    png_out,
    dpi=300,
    bbox_inches="tight",
)
print("Saved PNG:", png_out)


# --------------------------------------------------
# Save PDF without title
# --------------------------------------------------
st.remove()
fig.subplots_adjust(top=0.988)

pdf_out = os.path.join(
    FIG_DIR,
    "joint_enso_local_sst_compound_all_cities_manuscript.pdf",
)

fig.savefig(
    pdf_out,
    bbox_inches="tight",
)
print("Saved PDF:", pdf_out)

plt.show()
plt.close(fig)
