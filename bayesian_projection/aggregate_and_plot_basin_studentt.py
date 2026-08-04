#!/usr/bin/env python3
"""
AGGREGATE AND PLOT — BASIN STUDENT-T ENSO/IOD EXPERIMENTS
=========================================================

Purpose
-------
Manuscript: Figure 5, section 2.4
This script combines the former experiment-table and heatmap notebooks/scripts
for Model 3 of 6. It reads the existing basin posterior files, calculates the
posterior basin-mean SST shift for each ENSO/IOD experiment, writes the same
long and wide CSV products, and creates the manuscript heatmap.

Interpretation
--------------
For each posterior draw and wet basin grid cell:

    delta_mu =
        bNp_s  * N_pos
      + bNn_s  * N_neg
      + bD_s   * D
      + bNpD_s * N_pos * D
      + bNnD_s * N_neg * D

The spatial field is averaged across all retained wet grid cells for each
posterior draw. Reported intervals are 94% highest-density intervals of this
basin-mean posterior distribution.

Consistency decisions
---------------------
* Internal data keys and posterior filenames continue to use
  ``arabian_gulf`` so existing files remain valid.
* The manuscript-facing label is ``Persian Gulf``.
* Existing CSV and figure directories and filenames are retained.
* The heatmap includes only the experiment subset used in the previous figure,
  although the saved long and wide tables include every configured experiment.
"""

import argparse
import os

import arviz as az
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate Student-t basin experiments and make heatmap."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the completed figure interactively.",
    )
    return parser.parse_args()


ARGS = parse_args()


# ---------------------------------------------------------------------------
# Existing paths and configuration
# ---------------------------------------------------------------------------
BASINS = [
    "arabian_gulf",
    "red_sea",
    "gulf_oman",
    "gulf_aden",
]

EXPERIMENTS = [
    {"name": "El Niño (+1,0)", "N": +1.0, "D": 0.0},
    {"name": "La Niña (-1,0)", "N": -1.0, "D": 0.0},
    {"name": "pIOD (0,+1)", "N": 0.0, "D": +1.0},
    {"name": "nIOD (0,-1)", "N": 0.0, "D": -1.0},
    {"name": "La Niña + pIOD (-1,+1)", "N": -1.0, "D": +1.0},
    {"name": "La Niña + nIOD (-1,-1)", "N": -1.0, "D": -1.0},
    {"name": "Strong La Niña (-2,0)", "N": -2.0, "D": 0.0},
    {"name": "Super La Niña (-2.5,0)", "N": -2.5, "D": 0.0},
    {"name": "Strong La Niña + nIOD (-2,-1)", "N": -2.0, "D": -1.0},
    {"name": "Strong El Niño (+2,0)", "N": +2.0, "D": 0.0},
    {"name": "El Niño + pIOD (+1,+1)", "N": +1.0, "D": +1.0},
    {"name": "El Niño + nIOD (+1,-1)", "N": +1.0, "D": -1.0},
]

IDATA_TEMPLATE = "../data/sst/studentt_mean_{basin}_roni_dmi_idata.nc"

TABLE_DIR = "../data/sst/studentt_experiment_tables"
os.makedirs(TABLE_DIR, exist_ok=True)

FIG_DIR = "../figures/studentt_mean_shift_heatmaps"
os.makedirs(FIG_DIR, exist_ok=True)

LONG_OUT = os.path.join(
    TABLE_DIR,
    "studentt_mean_shift_experiment_table_long.csv",
)
WIDE_OUT = os.path.join(
    TABLE_DIR,
    "studentt_mean_shift_experiment_table_wide.csv",
)
MEAN_OUT = os.path.join(
    TABLE_DIR,
    "studentt_mean_shift_mean_only.csv",
)
LOW_OUT = os.path.join(
    TABLE_DIR,
    "studentt_mean_shift_hdi_low_only.csv",
)
HIGH_OUT = os.path.join(
    TABLE_DIR,
    "studentt_mean_shift_hdi_high_only.csv",
)

PNG_OUT = os.path.join(
    FIG_DIR,
    "studentt_mean_shift_experiment_heatmap.png",
)
PDF_OUT = os.path.join(
    FIG_DIR,
    "studentt_mean_shift_experiment_heatmap.pdf",
)

HDI_PROB = 0.94

BASIN_ORDER = [
    "arabian_gulf",
    "red_sea",
    "gulf_oman",
    "gulf_aden",
]

BASIN_LABELS = {
    "arabian_gulf": "Persian Gulf",
    "red_sea": "Red Sea",
    "gulf_oman": "Gulf of Oman",
    "gulf_aden": "Gulf of Aden",
}

EXPERIMENT_ORDER = [
    "El Niño (+1,0)",
    "La Niña (-1,0)",
    "pIOD (0,+1)",
    "nIOD (0,-1)",
    "La Niña + nIOD (-1,-1)",
    "La Niña + pIOD (-1,+1)",
    "Strong La Niña (-2,0)",
    "Super La Niña (-2.5,0)",
    "Strong La Niña + nIOD (-2,-1)",
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


# ---------------------------------------------------------------------------
# Posterior experiment calculations
# ---------------------------------------------------------------------------
def stack_draws(da):
    if "chain" in da.dims and "draw" in da.dims:
        return da.stack(sample=("chain", "draw"))
    if "sample" in da.dims:
        return da
    raise ValueError(f"Unexpected dimensions for {da.name}: {da.dims}")


def delta_mu_draws(idata, n_value, d_value):
    posterior = idata.posterior

    bnp_s = stack_draws(posterior["bNp_s"])
    bnn_s = stack_draws(posterior["bNn_s"])
    bd_s = stack_draws(posterior["bD_s"])
    bnpd_s = stack_draws(posterior["bNpD_s"])
    bnnd_s = stack_draws(posterior["bNnD_s"])

    n_pos = max(n_value, 0.0)
    n_neg = min(n_value, 0.0)

    return (
        bnp_s * n_pos
        + bnn_s * n_neg
        + bd_s * d_value
        + bnpd_s * (n_pos * d_value)
        + bnnd_s * (n_neg * d_value)
    )


def basin_mean_and_hdi(field_draws):
    basin_draws = field_draws.mean(dim="space")
    mean_value = float(basin_draws.mean().values)
    low, high = az.hdi(basin_draws.values, hdi_prob=HDI_PROB)
    return mean_value, float(low), float(high)


def build_experiment_table(idata, basin):
    rows = []
    for experiment in EXPERIMENTS:
        draws = delta_mu_draws(
            idata,
            n_value=experiment["N"],
            d_value=experiment["D"],
        )
        mean_value, low, high = basin_mean_and_hdi(draws)
        rows.append(
            {
                "basin": basin,
                "basin_label": BASIN_LABELS[basin],
                "experiment": experiment["name"],
                "N": experiment["N"],
                "D": experiment["D"],
                "mean_dmu": mean_value,
                "hdi_low": low,
                "hdi_high": high,
            }
        )
    return pd.DataFrame(rows)


def format_result(row):
    return (
        f"{row['mean_dmu']:+.3f} "
        f"[{row['hdi_low']:+.3f}, {row['hdi_high']:+.3f}]"
    )


def aggregate_tables():
    tables = []

    required = {"bNp_s", "bNn_s", "bD_s", "bNpD_s", "bNnD_s"}

    for basin in BASINS:
        path = IDATA_TEMPLATE.format(basin=basin)
        if not os.path.exists(path):
            print(f"⚠️ Missing posterior for {basin}: {path}")
            continue

        print(f"Loading: {path}")
        idata = az.from_netcdf(path)

        missing = required - set(idata.posterior.data_vars)
        if missing:
            print(
                f"⚠️ Skipping {basin}; missing posterior variables: "
                f"{sorted(missing)}"
            )
            continue

        tables.append(build_experiment_table(idata, basin))

    if not tables:
        raise FileNotFoundError(
            "No valid Student-t basin posterior files were found."
        )

    final_table = pd.concat(tables, ignore_index=True)
    final_table["result"] = final_table.apply(format_result, axis=1)

    final_table.to_csv(LONG_OUT, index=False)

    formatted_wide = final_table.pivot(
        index="experiment",
        columns="basin",
        values="result",
    ).reindex(columns=BASIN_ORDER)
    formatted_wide.columns = [
        BASIN_LABELS.get(column, column) for column in formatted_wide.columns
    ]
    formatted_wide.to_csv(WIDE_OUT)

    mean_wide = final_table.pivot(
        index="experiment",
        columns="basin",
        values="mean_dmu",
    ).reindex(columns=BASIN_ORDER)
    low_wide = final_table.pivot(
        index="experiment",
        columns="basin",
        values="hdi_low",
    ).reindex(columns=BASIN_ORDER)
    high_wide = final_table.pivot(
        index="experiment",
        columns="basin",
        values="hdi_high",
    ).reindex(columns=BASIN_ORDER)

    for frame in (mean_wide, low_wide, high_wide):
        frame.columns = [
            BASIN_LABELS.get(column, column) for column in frame.columns
        ]

    mean_wide.to_csv(MEAN_OUT)
    low_wide.to_csv(LOW_OUT)
    high_wide.to_csv(HIGH_OUT)

    print(f"✅ saved long table: {LONG_OUT}")
    print(f"✅ saved formatted wide table: {WIDE_OUT}")
    print(f"✅ saved mean-only table: {MEAN_OUT}")
    print(f"✅ saved HDI-low table: {LOW_OUT}")
    print(f"✅ saved HDI-high table: {HIGH_OUT}")

    return final_table


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------
def make_heatmap(final_table):
    plot_df = final_table[
        final_table["basin"].isin(BASIN_ORDER)
        & final_table["experiment"].isin(EXPERIMENT_ORDER)
    ].copy()

    missing_basins = sorted(
        set(BASIN_ORDER) - set(plot_df["basin"].unique())
    )
    missing_experiments = sorted(
        set(EXPERIMENT_ORDER) - set(plot_df["experiment"].unique())
    )

    if missing_basins:
        print(f"⚠️ Missing basins in plotting table: {missing_basins}")
    if missing_experiments:
        print(
            f"⚠️ Missing experiments in plotting table: "
            f"{missing_experiments}"
        )

    mean_matrix = (
        plot_df.pivot(
            index="experiment",
            columns="basin",
            values="mean_dmu",
        )
        .reindex(index=EXPERIMENT_ORDER, columns=BASIN_ORDER)
    )
    low_matrix = (
        plot_df.pivot(
            index="experiment",
            columns="basin",
            values="hdi_low",
        )
        .reindex(index=EXPERIMENT_ORDER, columns=BASIN_ORDER)
    )
    high_matrix = (
        plot_df.pivot(
            index="experiment",
            columns="basin",
            values="hdi_high",
        )
        .reindex(index=EXPERIMENT_ORDER, columns=BASIN_ORDER)
    )

    data = mean_matrix.values.astype(float)
    vmax = np.nanmax(np.abs(data))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    norm = mcolors.TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0.0,
        vmax=vmax,
    )

    with mpl.rc_context(ERL_RC):
        fig, ax = plt.subplots(figsize=(7.1, 5.8))
        image = ax.imshow(
            data,
            aspect="auto",
            cmap="RdBu_r",
            norm=norm,
        )

        ax.set_xticks(np.arange(len(BASIN_ORDER)))
        ax.set_xticklabels(
            [BASIN_LABELS[basin] for basin in BASIN_ORDER],
            fontsize=8,
        )
        ax.set_yticks(np.arange(len(EXPERIMENT_ORDER)))
        ax.set_yticklabels(EXPERIMENT_ORDER, fontsize=8)

        ax.set_xticks(
            np.arange(-0.5, len(BASIN_ORDER), 1),
            minor=True,
        )
        ax.set_yticks(
            np.arange(-0.5, len(EXPERIMENT_ORDER), 1),
            minor=True,
        )
        ax.grid(
            which="minor",
            color="black",
            linestyle="-",
            linewidth=0.5,
        )
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="x", which="major", length=0, pad=4)
        ax.tick_params(axis="y", which="major", length=0, pad=4)

        for row in range(len(EXPERIMENT_ORDER)):
            for column in range(len(BASIN_ORDER)):
                value = mean_matrix.iloc[row, column]
                low = low_matrix.iloc[row, column]
                high = high_matrix.iloc[row, column]

                if pd.isna(value):
                    text = "NA"
                    text_color = "black"
                else:
                    text = (
                        f"{value:+.3f}\n"
                        f"[{low:+.3f}, {high:+.3f}]"
                    )
                    fraction = abs(value) / vmax if vmax > 0 else 0
                    text_color = "white" if fraction > 0.45 else "black"

                ax.text(
                    column,
                    row,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                )

        colorbar = fig.colorbar(
            image,
            ax=ax,
            shrink=0.92,
            pad=0.02,
        )
        colorbar.set_label("Mean Δμ", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)

        fig.tight_layout()
        fig.savefig(
            PNG_OUT,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        fig.savefig(
            PDF_OUT,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )

        if ARGS.show:
            plt.show()
        plt.close(fig)

    print(f"✅ saved PNG: {PNG_OUT}")
    print(f"✅ saved PDF: {PDF_OUT}")


def main():
    final_table = aggregate_tables()
    make_heatmap(final_table)

    print("\n=== Long table ===")
    print(final_table.to_string(index=False))


if __name__ == "__main__":
    main()
