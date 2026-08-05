#!/usr/bin/env python3
# coding: utf-8
"""
AGGREGATE AND PLOT — JOINT ENSO–IOD AND LOCAL-BASIN WARMING
===========================================================

Figure 7

This is the complete second stage for Model 6 of 6 and Section 2.6. It loads
the fitted joint posterior, evaluates the full ENSO–IOD-by-basin-warming
factorial experiment grid, writes the existing experiment CSV, and then
creates the compound p95/p99 manuscript figure.

The experiment engine propagates ENSO and IOD through both the fitted basin-SST
submodel and the city GPD scale model. Imposed warming is added to the modeled
basin SST state before evaluating the city's conditional GPD quantile.

Reported changes remain conditional exceedance-magnitude changes. They do not
include changes in threshold-exceedance probability.

Internal keys retain ``arabian_gulf``. All manuscript-facing basin labels use
``Persian Gulf``. Existing CSV and figure paths and filenames are unchanged.
"""

import argparse
import pickle
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Scientific configuration
# ---------------------------------------------------------------------
CITY_ORDER = [
    "muscat",
    "doha",
    "dubai",
    "dammam",
    "kuwait_city",
    "basra",
    "jeddah",
    "aden",
]

CITY_BASINS = {
    "muscat": "gulf_oman",
    "doha": "arabian_gulf",
    "dubai": "arabian_gulf",
    "dammam": "arabian_gulf",
    "kuwait_city": "arabian_gulf",
    "basra": "arabian_gulf",
    "jeddah": "red_sea",
    "aden": "gulf_aden",
}

BASIN_DISPLAY = {
    "gulf_oman": "Gulf of Oman",
    "arabian_gulf": "Persian Gulf",
    "red_sea": "Red Sea",
    "gulf_aden": "Gulf of Aden",
}

ENSO_LEVELS = [0.0, -1.0, -2.0, -2.5]
IOD_LEVELS = [-1.0, 0.0, 1.0]
WARMING_LEVELS = np.arange(0.0, 3.01, 0.5)
THRESHOLD_PROBABILITY = 0.95

# Quantiles of the complete daily WBT distribution.
OVERALL_Q_LEVELS = [0.975, 0.99]
HDI_PROB = 0.94


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate posterior ENSO--IOD and local-basin warming "
            "experiments from the fitted joint GPD model."
        )
    )
    parser.add_argument("--netid", default="k16v981")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional override for the fitted-model output directory.",
    )
    parser.add_argument(
        "--idata",
        default=None,
        help="Optional explicit path to the posterior NetCDF or pickle file.",
    )
    parser.add_argument(
        "--meta",
        default=None,
        help="Optional explicit path to the fitted-model metadata pickle.",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Optional explicit output CSV path.",
    )
    parser.add_argument(
        "--hdi-prob",
        type=float,
        default=HDI_PROB,
    )
    return parser.parse_args()


def resolve_paths(args):
    default_run_dir = Path(
        f"/home/{args.netid}/my_work/code/arabian_peninsula/"
        "bayesian_extremes/data/joint_enso_local_sst_city_runs"
    )
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir

    default_nc = (
        run_dir
        / "idata_joint_wbt_daily_peak_enso_iod_local_sst_JJAS.nc"
    )
    default_pkl = (
        run_dir
        / "idata_joint_wbt_daily_peak_enso_iod_local_sst_JJAS.pkl"
    )
    default_meta = (
        run_dir
        / "meta_joint_wbt_daily_peak_enso_iod_local_sst_JJAS.pkl"
    )
    default_out = (
        run_dir
        / "joint_wbt_daily_peak_enso_iod_local_sst_experiments.csv"
    )

    return {
        "run_dir": run_dir,
        "idata_explicit": Path(args.idata) if args.idata else None,
        "idata_nc": default_nc,
        "idata_pkl": default_pkl,
        "meta": Path(args.meta) if args.meta else default_meta,
        "out_csv": Path(args.out_csv) if args.out_csv else default_out,
    }


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_idata(paths):
    explicit = paths["idata_explicit"]

    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Posterior not found: {explicit}")
        if explicit.suffix == ".nc":
            return az.from_netcdf(explicit), explicit
        with open(explicit, "rb") as handle:
            return pickle.load(handle), explicit

    if paths["idata_nc"].exists():
        return az.from_netcdf(paths["idata_nc"]), paths["idata_nc"]

    if paths["idata_pkl"].exists():
        with open(paths["idata_pkl"], "rb") as handle:
            return pickle.load(handle), paths["idata_pkl"]

    raise FileNotFoundError(
        "No fitted posterior found. Checked:\n"
        f"{paths['idata_nc']}\n"
        f"{paths['idata_pkl']}"
    )


def stack_samples(data_array):
    return data_array.stack(sample=("chain", "draw")).values


def summarize(values, hdi_prob):
    values = np.asarray(values, dtype=float)
    low, high = az.hdi(values, hdi_prob=hdi_prob)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "hdi_low": float(low),
        "hdi_high": float(high),
    }

def overall_to_conditional_probability(
    overall_probability,
    threshold_probability=THRESHOLD_PROBABILITY,
):
    """
    Convert a quantile of the full daily distribution to the corresponding
    quantile of the conditional threshold-exceedance distribution.

    For a p95 POT threshold:
        overall p97.5 -> conditional p50
        overall p99   -> conditional p80
        overall p99.5 -> conditional p90
    """
    if overall_probability < threshold_probability:
        raise ValueError(
            "The requested overall probability lies below the POT threshold."
        )

    if overall_probability >= 1.0:
        raise ValueError("Probability must be less than 1.")

    return (
        overall_probability - threshold_probability
    ) / (
        1.0 - threshold_probability
    )

def gpd_quantile(threshold, sigma, xi, probability, xi_tol=1e-6):
    sigma, xi = np.broadcast_arrays(
        np.asarray(sigma, dtype=float),
        np.asarray(xi, dtype=float),
    )

    out = np.empty_like(sigma, dtype=float)
    near_zero = np.abs(xi) < xi_tol

    out[near_zero] = (
        threshold
        + sigma[near_zero]
        * np.log(1.0 / (1.0 - probability))
    )

    out[~near_zero] = (
        threshold
        + sigma[~near_zero]
        / xi[~near_zero]
        * (
            (1.0 - probability) ** (-xi[~near_zero])
            - 1.0
        )
    )

    return out


def scenario_name(N, D):
    pieces = []

    if N == 0:
        pieces.append("Neutral ENSO")
    elif N == -1:
        pieces.append("La Nina")
    elif N == -2:
        pieces.append("Strong La Nina")
    elif N == -2.5:
        pieces.append("Super La Nina")
    else:
        pieces.append(f"ENSO {N:+g} SD")

    if D == 0:
        pieces.append("Neutral IOD")
    elif D == 1:
        pieces.append("pIOD")
    elif D == -1:
        pieces.append("nIOD")
    else:
        pieces.append(f"IOD {D:+g} SD")

    return " + ".join(pieces)


def ensure_required_variables(posterior):
    required = [
        "xi",
        "sst_a_basin",
        "sst_b_roni_basin",
        "sst_b_dmi_basin",
        "sst_b_roni_dmi_basin",
        "a_city",
        "b_roni_city",
        "b_dmi_city",
        "b_roni_dmi_city",
        "b_sst_city",
        "b_roni_sst_city",
    ]

    missing = [
        name for name in required
        if name not in posterior.data_vars
    ]

    if missing:
        raise KeyError(
            "Posterior is missing required variables: "
            + ", ".join(missing)
        )


def coordinate_index(data_array, dim, label):
    if dim not in data_array.dims:
        raise KeyError(
            f"Dimension '{dim}' is not present in {data_array.name}: "
            f"{data_array.dims}"
        )

    coord_values = [
        str(value)
        for value in data_array.coords[dim].values
    ]

    if label not in coord_values:
        raise KeyError(
            f"Label '{label}' not found in dimension '{dim}'. "
            f"Available: {coord_values}"
        )

    return coord_values.index(label)


# ---------------------------------------------------------------------
# Posterior experiment engine
# ---------------------------------------------------------------------
def build_draw_cache(idata, meta):
    posterior = idata.posterior
    ensure_required_variables(posterior)

    city_coord = [
        str(value)
        for value in posterior["a_city"].coords["city"].values
    ]
    basin_coord = [
        str(value)
        for value in posterior["sst_a_basin"].coords["basin"].values
    ]

    missing_cities = [
        city for city in CITY_ORDER
        if city not in city_coord
    ]
    if missing_cities:
        raise KeyError(
            "Requested cities missing from posterior city coordinate: "
            + ", ".join(missing_cities)
        )

    missing_basins = sorted(
        {
            basin for basin in CITY_BASINS.values()
            if basin not in basin_coord
        }
    )
    if missing_basins:
        raise KeyError(
            "Requested basins missing from posterior basin coordinate: "
            + ", ".join(missing_basins)
        )

    thresholds = meta.get("thresholds", {})
    missing_thresholds = [
        city for city in CITY_ORDER
        if city not in thresholds
    ]
    if missing_thresholds:
        raise KeyError(
            "Threshold metadata missing for: "
            + ", ".join(missing_thresholds)
        )

    cache = {
        "xi": stack_samples(posterior["xi"]),
        "cities": {},
        "basins": {},
        "thresholds": {
            city: float(thresholds[city])
            for city in CITY_ORDER
        },
    }

    for city in CITY_ORDER:
        city_index = coordinate_index(
            posterior["a_city"],
            "city",
            city,
        )

        cache["cities"][city] = {
            "a": stack_samples(
                posterior["a_city"].isel(city=city_index)
            ),
            "b_roni": stack_samples(
                posterior["b_roni_city"].isel(city=city_index)
            ),
            "b_dmi": stack_samples(
                posterior["b_dmi_city"].isel(city=city_index)
            ),
            "b_roni_dmi": stack_samples(
                posterior["b_roni_dmi_city"].isel(city=city_index)
            ),
            "b_sst": stack_samples(
                posterior["b_sst_city"].isel(city=city_index)
            ),
            "b_roni_sst": stack_samples(
                posterior["b_roni_sst_city"].isel(city=city_index)
            ),
        }

    for basin in sorted(set(CITY_BASINS.values())):
        basin_index = coordinate_index(
            posterior["sst_a_basin"],
            "basin",
            basin,
        )

        cache["basins"][basin] = {
            "a": stack_samples(
                posterior["sst_a_basin"].isel(basin=basin_index)
            ),
            "b_roni": stack_samples(
                posterior["sst_b_roni_basin"].isel(
                    basin=basin_index
                )
            ),
            "b_dmi": stack_samples(
                posterior["sst_b_dmi_basin"].isel(
                    basin=basin_index
                )
            ),
            "b_roni_dmi": stack_samples(
                posterior["sst_b_roni_dmi_basin"].isel(
                    basin=basin_index
                )
            ),
        }

    return cache


def evaluate_state(
    cache,
    city,
    N,
    D,
    warming,
    overall_probability,
):
    basin = CITY_BASINS[city]
    basin_draws = cache["basins"][basin]
    city_draws = cache["cities"][city]
    xi = cache["xi"]
    threshold = cache["thresholds"][city]

    ND = N * D

    sst_state = (
        basin_draws["a"]
        + basin_draws["b_roni"] * N
        + basin_draws["b_dmi"] * D
        + basin_draws["b_roni_dmi"] * ND
        + warming
    )

    log_sigma = (
        city_draws["a"]
        + city_draws["b_roni"] * N
        + city_draws["b_dmi"] * D
        + city_draws["b_roni_dmi"] * ND
        + city_draws["b_sst"] * sst_state
        + city_draws["b_roni_sst"] * N * sst_state
    )

    sigma = np.exp(log_sigma)

    conditional_probability = overall_to_conditional_probability(
        overall_probability
    )
    
    tw_level = gpd_quantile(
        threshold,
        sigma,
        xi,
        conditional_probability,
    )

    return {
        "sst_state": sst_state,
        "sigma": sigma,
        "tw_level": tw_level,
        "overall_probability": overall_probability,
        "conditional_probability": conditional_probability,
    }


def run_experiments(cache, hdi_prob):
    state_cache = {}

    # Evaluate every unique posterior state once.
    for city in CITY_ORDER:
        for N in ENSO_LEVELS:
            for D in IOD_LEVELS:
                for warming in WARMING_LEVELS:
                    for overall_probability in OVERALL_Q_LEVELS:
                        key = (
                            city,
                            float(N),
                            float(D),
                            float(warming),
                            float(overall_probability),
                        )
                        state_cache[key] = evaluate_state(
                            cache,
                            city=city,
                            N=float(N),
                            D=float(D),
                            warming=float(warming),
                            overall_probability = float(overall_probability),
                        )

    rows = []

    for city in CITY_ORDER:
        basin = CITY_BASINS[city]
        threshold = cache["thresholds"][city]

        for N in ENSO_LEVELS:
            for D in IOD_LEVELS:
                scen_name = scenario_name(N, D)

                for warming in WARMING_LEVELS:
                    for overall_probability in OVERALL_Q_LEVELS:
                        key = (
                            city,
                            float(N),
                            float(D),
                            float(warming),
                            float(overall_probability),
                        )

                        current = state_cache[key]

                        baseline = state_cache[
                            (
                                city,
                                0.0,
                                0.0,
                                0.0,
                                float(overall_probability),
                            )
                        ]

                        same_mode_no_warming = state_cache[
                            (
                                city,
                                float(N),
                                float(D),
                                0.0,
                                float(overall_probability),
                            )
                        ]

                        neutral_same_warming = state_cache[
                            (
                                city,
                                0.0,
                                0.0,
                                float(warming),
                                float(overall_probability),
                            )
                        ]

                        total_change = (
                            current["tw_level"]
                            - baseline["tw_level"]
                        )

                        warming_effect = (
                            current["tw_level"]
                            - same_mode_no_warming["tw_level"]
                        )

                        mode_effect = (
                            current["tw_level"]
                            - neutral_same_warming["tw_level"]
                        )

                        neutral_warming_effect = (
                            neutral_same_warming["tw_level"]
                            - baseline["tw_level"]
                        )

                        interaction = (
                            warming_effect
                            - neutral_warming_effect
                        )

                        sst_summary = summarize(
                            current["sst_state"],
                            hdi_prob,
                        )
                        sigma_summary = summarize(
                            current["sigma"],
                            hdi_prob,
                        )
                        level_summary = summarize(
                            current["tw_level"],
                            hdi_prob,
                        )
                        total_summary = summarize(
                            total_change,
                            hdi_prob,
                        )
                        warming_summary = summarize(
                            warming_effect,
                            hdi_prob,
                        )
                        mode_summary = summarize(
                            mode_effect,
                            hdi_prob,
                        )
                        interaction_summary = summarize(
                            interaction,
                            hdi_prob,
                        )

                        rows.append(
                            {
                                "city": city,
                                "basin_code": basin,
                                "basin": BASIN_DISPLAY[basin],
                                "scenario": scen_name,
                                "N_sd": float(N),
                                "D_sd": float(D),
                                "basin_warming_C": float(warming),
                                "overall_quantile": float(overall_probability),
                                "conditional_gpd_quantile": float(
                                    current["conditional_probability"]
                                ),
                                "threshold_u_C": threshold,

                                "sst_state_mean_C": sst_summary["mean"],
                                "sst_state_median_C": sst_summary["median"],
                                "sst_state_hdi_low_C": sst_summary["hdi_low"],
                                "sst_state_hdi_high_C": sst_summary["hdi_high"],

                                "gpd_sigma_mean": sigma_summary["mean"],
                                "gpd_sigma_median": sigma_summary["median"],
                                "gpd_sigma_hdi_low": sigma_summary["hdi_low"],
                                "gpd_sigma_hdi_high": sigma_summary["hdi_high"],

                                "tw_level_mean_C": level_summary["mean"],
                                "tw_level_median_C": level_summary["median"],
                                "tw_level_hdi_low_C": level_summary["hdi_low"],
                                "tw_level_hdi_high_C": level_summary["hdi_high"],

                                "delta_total_mean_C": total_summary["mean"],
                                "delta_total_median_C": total_summary["median"],
                                "delta_total_hdi_low_C": total_summary["hdi_low"],
                                "delta_total_hdi_high_C": total_summary["hdi_high"],

                                "delta_warming_mean_C": warming_summary["mean"],
                                "delta_warming_median_C": warming_summary["median"],
                                "delta_warming_hdi_low_C": warming_summary["hdi_low"],
                                "delta_warming_hdi_high_C": warming_summary["hdi_high"],

                                "delta_mode_mean_C": mode_summary["mean"],
                                "delta_mode_median_C": mode_summary["median"],
                                "delta_mode_hdi_low_C": mode_summary["hdi_low"],
                                "delta_mode_hdi_high_C": mode_summary["hdi_high"],

                                "interaction_mean_C": interaction_summary["mean"],
                                "interaction_median_C": interaction_summary["median"],
                                "interaction_hdi_low_C": interaction_summary["hdi_low"],
                                "interaction_hdi_high_C": interaction_summary["hdi_high"],
                            }
                        )

    out = pd.DataFrame(rows)

    out["city"] = pd.Categorical(
        out["city"],
        categories=CITY_ORDER,
        ordered=True,
    )

    out = out.sort_values(
        [
            "city",
            "N_sd",
            "D_sd",
            "basin_warming_C",
            "overall_quantile",
        ]
    ).reset_index(drop=True)

    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def run_experiment_stage():
    args = parse_args()
    paths = resolve_paths(args)

    idata, idata_path = load_idata(paths)

    if not paths["meta"].exists():
        raise FileNotFoundError(
            f"Metadata file not found: {paths['meta']}"
        )

    meta = pd.read_pickle(paths["meta"])

    print("=" * 78)
    print("Joint ENSO--IOD + local SST posterior experiments")
    print(f"Posterior: {idata_path}")
    print(f"Metadata: {paths['meta']}")
    print(f"Output: {paths['out_csv']}")
    print(f"Cities: {CITY_ORDER}")
    print(f"ENSO levels: {ENSO_LEVELS}")
    print(f"IOD levels: {IOD_LEVELS}")
    print(f"Warming levels: {WARMING_LEVELS.tolist()}")
    print(f"Overall daily quantiles: {OVERALL_Q_LEVELS}")
    print(
        "Corresponding conditional GPD quantiles: "
        f"{[overall_to_conditional_probability(q) for q in OVERALL_Q_LEVELS]}"
    )
    print("=" * 78)

    cache = build_draw_cache(idata, meta)
    result = run_experiments(
        cache,
        hdi_prob=args.hdi_prob,
    )

    paths["out_csv"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    result.to_csv(paths["out_csv"], index=False)

    print(f"\nWrote {len(result):,} experiment rows")
    print(f"Saved: {paths['out_csv']}")

    preview_columns = [
        "city",
        "basin",
        "scenario",
        "basin_warming_C",
        "overall_quantile",
        "sst_state_mean_C",
        "tw_level_mean_C",
        "delta_total_mean_C",
        "delta_warming_mean_C",
        "delta_mode_mean_C",
        "interaction_mean_C",
    ]

    print("\nPreview:")
    print(
        result[preview_columns]
        .head(20)
        .to_string(index=False)
    )



# ---------------------------------------------------------------------
# Plot configuration and rendering
# ---------------------------------------------------------------------
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

QUANTILES = [0.975, 0.99]

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

def quantile_label(q):
    percentile = 100.0 * q
    if percentile.is_integer():
        return f"p{int(percentile)}"
    return f"p{percentile:g}"

def select_scenario_row(df, N, D, warming, quantile):
    """Return exactly one matching experiment row."""
    match = df[
        np.isclose(df["N_sd"], N)
        & np.isclose(df["D_sd"], D)
        & np.isclose(df["basin_warming_C"], warming)
        & np.isclose(df["overall_quantile"], quantile)
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




def plot_compound_figure(show=False):
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
        "overall_quantile",
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
                    f"{quantile_label(q)}: "
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
                    f"Overall daily {quantile_label(q)}",
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

    if show:
        plt.show()
    plt.close(fig)


def main():
    run_experiment_stage()
    plot_compound_figure(show=False)


if __name__ == "__main__":
    main()
