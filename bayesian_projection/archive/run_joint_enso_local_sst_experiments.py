#!/usr/bin/env python3
# coding: utf-8
"""
Posterior experiments for the joint ENSO--IOD + local-basin SST GPD model
=======================================================================

This script evaluates a full factorial grid of:

* Eight cities with explicit local-basin SST assignments
* Basin warming from 0.0 to 3.0 degC in 0.5 degC increments
* ENSO levels: 0.0, -1.0, -2.0, -2.5 standard deviations
* IOD levels: -1.0, 0.0, +1.0 standard deviations
* Conditional GPD quantiles: p95 and p99 of exceedance magnitude

For each posterior draw, the local-basin SST state is:

    SST_state =
        sst_a_basin
        + sst_b_roni_basin * N
        + sst_b_dmi_basin * D
        + sst_b_roni_dmi_basin * N * D
        + imposed_basin_warming

That state is then passed through the city GPD scale model:

    log(sigma) =
        a_city
        + b_roni_city * N
        + b_dmi_city * D
        + b_roni_dmi_city * N * D
        + b_sst_city * SST_state
        + b_roni_sst_city * N * SST_state

The script saves posterior summaries for:

* absolute conditional GPD level
* total compound change relative to (N=0, D=0, warming=0)
* basin-warming effect within the same ENSO--IOD state
* ENSO--IOD mode effect at the same warming level
* interaction/nonadditivity between basin warming and ENSO--IOD state

Important
---------
The fitted RONI and DMI predictors were standardized after lagging, so N and D
in this script are standard-deviation units, not degrees Celsius.

The code retains the internal dataset label ``arabian_gulf`` because that is
how the files and posterior coordinates are stored. Manuscript-facing output
uses ``Persian Gulf``.
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
Q_LEVELS = [0.95, 0.99]
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


def evaluate_state(cache, city, N, D, warming, probability):
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

    tw_level = gpd_quantile(
        threshold,
        sigma,
        xi,
        probability,
    )

    return {
        "sst_state": sst_state,
        "sigma": sigma,
        "tw_level": tw_level,
    }


def run_experiments(cache, hdi_prob):
    state_cache = {}

    # Evaluate every unique posterior state once.
    for city in CITY_ORDER:
        for N in ENSO_LEVELS:
            for D in IOD_LEVELS:
                for warming in WARMING_LEVELS:
                    for probability in Q_LEVELS:
                        key = (
                            city,
                            float(N),
                            float(D),
                            float(warming),
                            float(probability),
                        )
                        state_cache[key] = evaluate_state(
                            cache,
                            city=city,
                            N=float(N),
                            D=float(D),
                            warming=float(warming),
                            probability=float(probability),
                        )

    rows = []

    for city in CITY_ORDER:
        basin = CITY_BASINS[city]
        threshold = cache["thresholds"][city]

        for N in ENSO_LEVELS:
            for D in IOD_LEVELS:
                scen_name = scenario_name(N, D)

                for warming in WARMING_LEVELS:
                    for probability in Q_LEVELS:
                        key = (
                            city,
                            float(N),
                            float(D),
                            float(warming),
                            float(probability),
                        )

                        current = state_cache[key]

                        baseline = state_cache[
                            (
                                city,
                                0.0,
                                0.0,
                                0.0,
                                float(probability),
                            )
                        ]

                        same_mode_no_warming = state_cache[
                            (
                                city,
                                float(N),
                                float(D),
                                0.0,
                                float(probability),
                            )
                        ]

                        neutral_same_warming = state_cache[
                            (
                                city,
                                0.0,
                                0.0,
                                float(warming),
                                float(probability),
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
                                "conditional_quantile": float(probability),
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
            "conditional_quantile",
        ]
    ).reset_index(drop=True)

    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
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
    print(f"Conditional quantiles: {Q_LEVELS}")
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
        "conditional_quantile",
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


if __name__ == "__main__":
    main()
