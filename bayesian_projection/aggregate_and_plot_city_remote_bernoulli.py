#!/usr/bin/env python3
# coding: utf-8

'''
============================================================
MODEL 1 OF 6 — POST-PROCESSING FOR MANUSCRIPT SECTION 2.2: Figure 3
Aggregate posterior scenario impacts and reproduce manuscript figure
============================================================

PURPOSE
-------
Figure 3

This script is the single post-processing entry point for the Bernoulli
occurrence model in city_remote_bernoulli_pipeline.py. It does not refit or
modify the Bayesian model. It:
  1. reads the saved InferenceData NetCDF files and model summary CSV;
  2. calculates posterior exceedance probabilities for fixed ENSO/IOD
     scenarios relative to neutral conditions (N=0, D=0);
  3. writes the existing scenario-impact CSV in its existing location; and
  4. reproduces the existing city-panel PNG and manuscript PDF.

POSTERIOR CALCULATION CHOICES
-----------------------------
1. N and D are standardized lagged RONI and DMI values, matching the model.
2. Neutral baseline is N=0 and D=0, so p0 = logistic(intercept).
3. Scenario effects include the fitted N*D interaction.
4. Reported change is p_scenario - p_baseline in absolute probability;
   the figure converts this to percentage points.
5. Risk ratios and odds ratios are also written to the CSV.
6. All uncertainty intervals are 94% highest-density intervals.
7. For the pooled Persian Gulf fit, city parameter positions must match the
   model fit order: Doha, Dubai, Dammam.

EXISTING OUTPUT LOCATIONS RETAINED
----------------------------------
CSV:
  ../data/<var>_daily_city_runs_bernoulli/<var>_bernoulli_scenario_impacts.csv
Figures:
  ../figures/city_roni_dmi_bernoulli/

This script deliberately leaves all model NetCDF files, CSVs, and figures in
their current directories.
============================================================
'''
import argparse
import os

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
        description="Aggregate and plot Bernoulli ENSO/IOD scenario impacts."
    )
    parser.add_argument(
        "--base-data-dir",
        default="../data",
        help="Base data directory; defaults preserve the existing pipeline.",
    )
    parser.add_argument(
        "--fig-dir",
        default="../figures/city_roni_dmi_bernoulli",
        help="Figure directory; defaults preserve existing output locations.",
    )
    parser.add_argument(
        "--vars",
        default="wbt_daily_peak",
        help="Comma-separated target variables.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving.",
    )
    return parser.parse_args()


SCENARIOS = [
    {"name": "El Niño (+1,0)",                "N": +1.0, "D":  0.0},
    {"name": "La Niña (-1,0)",               "N": -1.0, "D":  0.0},
    {"name": "pIOD (0,+1)",                  "N":  0.0, "D": +1.0},
    {"name": "nIOD (0,-1)",                  "N":  0.0, "D": -1.0},
    {"name": "La Niña + pIOD (-1,+1)",       "N": -1.0, "D": +1.0},
    {"name": "La Niña + nIOD (-1,-1)",       "N": -1.0, "D": -1.0},
    {"name": "Strong La Niña (-2,0)",        "N": -2.0, "D":  0.0},
    {"name": "Super La Niña (-2.5,0)",       "N": -2.5, "D":  0.0},
    {"name": "Strong La Niña + nIOD (-2,-1)","N": -2.0, "D": -1.0},
    {"name": "Strong El Niño (+2,0)",        "N": +2.0, "D":  0.0},
    {"name": "El Niño + pIOD (+1,+1)",       "N": +1.0, "D": +1.0},
    {"name": "El Niño + nIOD (+1,-1)",       "N": +1.0, "D": -1.0},
]

SCENARIOS_TO_PLOT = [
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

POOLED_SPACE_ORDER = {
    "gulf_coastal_pooled": ["doha", "dubai", "dammam"],
}

AGGREGATE_CITY_ORDER = [
    "muscat", "doha", "dubai", "dammam", "kuwait_city",
    "basra", "jeddah", "aden", "medina", "riyadh",
]

PLOT_CITY_ORDER = [
    "dammam", "doha", "dubai", "jeddah", "kuwait_city",
    "basra", "riyadh", "medina", "aden", "muscat",
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

VAR_LABELS = {
    "wbt_daily_peak": "Wet-Bulb Temperature exceedance probability",
}


def logistic(x):
    """Numerically stable logistic transformation."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def stack_samples(da):
    return da.stack(sample=("chain", "draw")).values


def get_var_paths(base_data_dir, var):
    run_dir = os.path.join(base_data_dir, f"{var}_daily_city_runs_bernoulli")
    csv_path = os.path.join(
        run_dir, f"{var}_daily_bernoulli_city_roni_dmi_summary.csv"
    )
    return run_dir, csv_path


def load_idata(run_dir, run_id):
    path = os.path.join(run_dir, f"idata_{run_id}.nc")
    if os.path.exists(path):
        return az.from_netcdf(path), path
    return None, path


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
        run_id = record["run_id"]

        if city == "ALL":
            continue

        if ":" in run_id:
            base_run, city_name = run_id.split(":", 1)
            idata, expected_path = load_idata(run_dir, base_run)
            if idata is None:
                print(f"WARNING: missing pooled idata: {expected_path}")
                continue

            lookup_key = pooled_lookup_key(record.get("pooled_group", ""), base_run)
            if lookup_key is None:
                raise ValueError(
                    "Unknown pooled parameter order for "
                    f"pooled_group='{record.get('pooled_group', '')}' and "
                    f"base_run='{base_run}'."
                )

            order = POOLED_SPACE_ORDER[lookup_key]
            if city_name not in order:
                raise ValueError(f"City '{city_name}' not found in pooled order {order}")
            space_index = order.index(city_name)

            posterior = idata.posterior
            a = stack_samples(posterior["a_s"].isel(space=space_index))
            b_n = stack_samples(posterior["bN_s"].isel(space=space_index))
            b_d = stack_samples(posterior["bD_s"].isel(space=space_index))
            b_nd = stack_samples(posterior["bND_s"].isel(space=space_index))
        else:
            idata, expected_path = load_idata(run_dir, run_id)
            if idata is None:
                print(f"WARNING: missing single-city idata: {expected_path}")
                continue

            posterior = idata.posterior
            a = stack_samples(posterior["a"])
            b_n = stack_samples(posterior["bN"])
            b_d = stack_samples(posterior["bD"])
            b_nd = stack_samples(posterior["bND"])

        p_base = logistic(a)
        p_base_low, p_base_high = az.hdi(p_base, hdi_prob=0.94)

        for scenario in SCENARIOS:
            n_value = scenario["N"]
            d_value = scenario["D"]
            interaction_value = n_value * d_value

            eta = a + b_n * n_value + b_d * d_value + b_nd * interaction_value
            p_scenario = logistic(eta)
            delta_p = p_scenario - p_base
            risk_ratio = np.divide(
                p_scenario,
                p_base,
                out=np.full_like(p_scenario, np.nan),
                where=p_base > 0,
            )
            odds_ratio = np.exp(
                b_n * n_value + b_d * d_value + b_nd * interaction_value
            )

            delta_low, delta_high = az.hdi(delta_p, hdi_prob=0.94)
            scenario_low, scenario_high = az.hdi(p_scenario, hdi_prob=0.94)
            risk_low, risk_high = az.hdi(risk_ratio, hdi_prob=0.94)
            odds_low, odds_high = az.hdi(odds_ratio, hdi_prob=0.94)

            rows.append({
                "var": var,
                "city": city,
                "run_id": run_id,
                "scenario": scenario["name"],
                "delta_p_mean": float(delta_p.mean()),
                "delta_p_hdi_low": float(delta_low),
                "delta_p_hdi_high": float(delta_high),
                "p_base_mean": float(p_base.mean()),
                "p_base_hdi_low": float(p_base_low),
                "p_base_hdi_high": float(p_base_high),
                "p_scen_mean": float(p_scenario.mean()),
                "p_scen_hdi_low": float(scenario_low),
                "p_scen_hdi_high": float(scenario_high),
                "risk_ratio_mean": float(np.nanmean(risk_ratio)),
                "risk_ratio_hdi_low": float(risk_low),
                "risk_ratio_hdi_high": float(risk_high),
                "odds_ratio_mean": float(odds_ratio.mean()),
                "odds_ratio_hdi_low": float(odds_low),
                "odds_ratio_hdi_high": float(odds_high),
                "N": n_value,
                "D": d_value,
            })

    if not rows:
        return None

    impact_df = pd.DataFrame(rows)
    impact_df["city"] = pd.Categorical(
        impact_df["city"], categories=AGGREGATE_CITY_ORDER, ordered=True
    )
    impact_df = impact_df.sort_values(["city", "scenario"]).reset_index(drop=True)

    output_csv = os.path.join(run_dir, f"{var}_bernoulli_scenario_impacts.csv")
    impact_df.to_csv(output_csv, index=False)
    print(f"Wrote: {output_csv}")
    return impact_df


def pretty_city(city):
    return CITY_LABELS.get(city, city.replace("_", " ").title())


def compute_global_xlim(df):
    subset = df[df["scenario"].isin(SCENARIOS_TO_PLOT)]
    if subset.empty:
        return -5.0, 5.0

    xmin = np.nanmin(100.0 * subset["delta_p_hdi_low"].to_numpy())
    xmax = np.nanmax(100.0 * subset["delta_p_hdi_high"].to_numpy())
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        return -5.0, 5.0

    span = xmax - xmin
    padding = 0.10 * span if span > 0 else 1.0
    return xmin - padding, xmax + padding


def plot_impacts(impact_df, var, fig_dir, show=False):
    os.makedirs(fig_dir, exist_ok=True)
    impact_df = impact_df[impact_df["scenario"].isin(SCENARIOS_TO_PLOT)].copy()
    if impact_df.empty:
        print(f"No requested scenarios found for {var}")
        return

    cities_present = impact_df["city"].dropna().astype(str).unique().tolist()
    cities = [city for city in PLOT_CITY_ORDER if city in cities_present]
    if not cities:
        print(f"No requested cities found for {var}")
        return

    x_limits = compute_global_xlim(impact_df)
    nrows = len(cities)
    fig_height = max(1.6 * nrows + 0.7, 8.0)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(7.2, fig_height),
        sharex=True,
        sharey=False,
    )
    if nrows == 1:
        axes = np.array([axes])

    cmap = plt.get_cmap("tab10")
    city_colors = {city: cmap(i % 10) for i, city in enumerate(cities)}
    y_spacing = 1.4
    y = np.arange(len(SCENARIOS_TO_PLOT)) * y_spacing

    for i, city in enumerate(cities):
        ax = axes[i]
        panel_label = f"({chr(97 + i)})"
        ax.text(
            -0.08, 1.05, panel_label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
        )

        city_df = impact_df[impact_df["city"].astype(str) == city]
        subset = city_df.set_index("scenario").reindex(SCENARIOS_TO_PLOT)
        color = city_colors[city]

        mean = 100.0 * subset["delta_p_mean"].to_numpy()
        low = 100.0 * subset["delta_p_hdi_low"].to_numpy()
        high = 100.0 * subset["delta_p_hdi_high"].to_numpy()

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

        dy = 0.22 * y_spacing
        for k, value in enumerate(mean):
            if np.isfinite(value):
                ax.text(
                    value,
                    y[k] + dy,
                    f"{value:+.2f}",
                    ha="center",
                    va="top",
                    fontsize=7.2,
                    color=color,
                )

        baseline = 100.0 * float(np.nanmean(subset["p_base_mean"].to_numpy()))
        ax.text(
            0.98,
            0.95,
            f"Baseline: {baseline:.2f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color=color,
            bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.95, pad=2.0),
        )

        ax.set_xlim(x_limits)
        ax.set_yticks(y)
        ax.set_yticklabels(SCENARIOS_TO_PLOT, fontsize=7.5)
        ax.set_ylabel(pretty_city(city), fontsize=8.5, color=color)
        ax.set_ylim(y[-1] + y_spacing, -y_spacing)

        if i == nrows - 1:
            ax.set_xlabel(
                "Change in exceedance probability (percentage points)", fontsize=8
            )

    title = fig.suptitle(
        f"{VAR_LABELS.get(var, var)} response to ENSO / IOD scenarios",
        fontsize=10,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    png_output = os.path.join(
        fig_dir, f"{var}_bernoulli_probability_change_bigpanel.png"
    )
    fig.savefig(png_output, dpi=300, bbox_inches="tight")
    print("Saved PNG:", png_output)

    title.remove()
    fig.subplots_adjust(top=0.985)
    pdf_output = os.path.join(
        fig_dir, f"{var}_bernoulli_probability_change_bigpanel_manuscript.pdf"
    )
    fig.savefig(pdf_output)
    print("Saved PDF:", pdf_output)

    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    target_vars = [value.strip() for value in args.vars.split(",") if value.strip()]

    for var in target_vars:
        print(f"\n=== Aggregating and plotting {var} ===")
        impact_df = compute_impacts_for_var(args.base_data_dir, var)
        if impact_df is not None:
            plot_impacts(impact_df, var, args.fig_dir, show=args.show)


if __name__ == "__main__":
    main()
