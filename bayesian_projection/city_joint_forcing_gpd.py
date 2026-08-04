#!/usr/bin/env python3
# coding: utf-8
"""
MODEL 6 OF 6 — JOINT ENSO–IOD, BASIN SST, AND CITY EXTREMES
===========================================================

Manuscript role
---------------
This is the joint compound-forcing model used for Section 2.6. It combines a
basin-SST response model with a city-level POT/GPD model so that ENSO and IOD
can influence extreme humid heat both directly and indirectly through the
local basin state.

Model architecture
------------------
A. Basin SST submodel

For each basin b and JJAS month t:

    SST_{b,t} ~ StudentT(nu_sst, mu_{b,t}, sigma_{sst,b})

    mu_{b,t} =
        alpha_{sst,b}
        + beta_{N,b} N_t
        + beta_{D,b} D_t
        + beta_{ND,b} N_t D_t.

B. City exceedance-magnitude submodel

For each city c and exceedance event t:

    z_{c,t} ~ GPD(sigma_{c,t}, xi)

    log(sigma_{c,t}) =
        a_c
        + b_{N,c} N_t
        + b_{D,c} D_t
        + b_{ND,c} N_t D_t
        + b_{S,c} SST_{b(c),t}
        + b_{NS,c} N_t SST_{b(c),t}.

Modeling choices
----------------
1. JJAS only; city-specific empirical p95 thresholds.
2. RONI is lagged two months and DMI one month unless the input indices are
   explicitly declared already lagged.
3. RONI and DMI are standardized after lagging.
4. Doha, Dubai, and Dammam are partially pooled. All other cities are fitted
   independently.
5. Medina and Riyadh have no local-SST or ENSO-by-SST term; their local-SST
   coefficients are fixed to zero.
6. Basin SST coefficients are estimated independently by basin rather than
   hierarchically pooled across basins.
7. One GPD shape parameter is shared across all cities and constrained to
   [-0.3, 0.5].
8. SST affects conditional exceedance magnitude, not threshold-exceedance
   occurrence. The model is not a complete occurrence-and-magnitude model.
9. Exceedances are not declustered. Daily events sharing monthly predictors,
   serial dependence, cross-city dependence, and residual dependence between
   the two submodels are not modeled explicitly.
10. Internal files retain ``arabian_gulf``; manuscript-facing output uses
    ``Persian Gulf``.

Posterior checks
----------------
The fitting script writes:
* R-hat, bulk ESS, tail ESS, divergences, tree-depth, and BFMI diagnostics;
* trace plots for structural parameters;
* a Student-t posterior-predictive check for basin SST;
* an inverse-CDF GPD posterior-predictive check for city exceedances;
* overall and city/basin-specific predictive summaries.

Existing data, posterior, metadata, experiment CSV, and figure paths remain
unchanged.
"""

import argparse
import glob
import pickle
import traceback
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr


# ---------------------------------------------------------------------
# Scientific configuration
# ---------------------------------------------------------------------
CITIES = {
    "muscat": {
        "lat": 23.5880,
        "lon": 58.3829,
        "basin": "gulf_oman",
    },
    "doha": {
        "lat": 25.2854,
        "lon": 51.5310,
        "basin": "arabian_gulf",
    },
    "dubai": {
        "lat": 25.2048,
        "lon": 55.2708,
        "basin": "arabian_gulf",
    },
    "jeddah": {
        "lat": 21.4858,
        "lon": 39.1925,
        "basin": "red_sea",
    },
    "aden": {
        "lat": 12.7855,
        "lon": 45.0187,
        "basin": "gulf_aden",
    },
    "medina": {
        "lat": 24.5247,
        "lon": 39.5692,
        "basin": None,
    },
    "riyadh": {
        "lat": 24.7136,
        "lon": 46.6753,
        "basin": None,
    },
    "dammam": {
        "lat": 26.4207,
        "lon": 50.0888,
        "basin": "arabian_gulf",
    },
    "kuwait_city": {
        "lat": 29.3759,
        "lon": 47.9774,
        "basin": "arabian_gulf",
    },
    "basra": {
        "lat": 30.5085,
        "lon": 47.7835,
        "basin": "arabian_gulf",
    },
}

CITY_LIST = list(CITIES)

POOLED_CITIES = ["doha", "dubai", "dammam"]
INDEPENDENT_CITIES = [city for city in CITY_LIST if city not in POOLED_CITIES]

LOCAL_SST_CITIES = [
    city for city in CITY_LIST if CITIES[city]["basin"] is not None
]
INLAND_CITIES = [
    city for city in CITY_LIST if CITIES[city]["basin"] is None
]

BASIN_LIST = [
    "gulf_oman",
    "arabian_gulf",
    "red_sea",
    "gulf_aden",
]
BASIN_TO_ID = {basin: i for i, basin in enumerate(BASIN_LIST)}
CITY_TO_ID = {city: i for i, city in enumerate(CITY_LIST)}
POOLED_TO_ID = {city: i for i, city in enumerate(POOLED_CITIES)}
INDEPENDENT_TO_ID = {
    city: i for i, city in enumerate(INDEPENDENT_CITIES)
}
LOCAL_SST_TO_ID = {
    city: i for i, city in enumerate(LOCAL_SST_CITIES)
}

VALID_TARGET_VARS = [
    "wbt_daily_peak",
    "tau_at_wbt_daily_peak",
    "t2m_at_wbt_daily_peak",
    "q_at_wbt_daily_peak",
]

XI_LOWER = -0.3
XI_UPPER = 0.5


# ---------------------------------------------------------------------
# CLI and paths
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit the joint ENSO–IOD and local-basin SST POT/GPD model "
            "using the established city and pooling design."
        )
    )
    parser.add_argument("--netid", default="k16v981")
    parser.add_argument(
        "--target-var",
        default="wbt_daily_peak",
        choices=VALID_TARGET_VARS,
    )
    parser.add_argument("--months", default="6,7,8,9")
    parser.add_argument("--q", type=float, default=0.95)
    parser.add_argument("--min-events", type=int, default=50)

    parser.add_argument("--roni-lag", type=int, default=2)
    parser.add_argument("--dmi-lag", type=int, default=1)
    parser.add_argument(
        "--indices-already-lagged",
        action="store_true",
        help="Use only when the CSV already contains the desired lags.",
    )

    parser.add_argument("--data-glob", default=None)
    parser.add_argument("--index-csv", default=None)
    parser.add_argument("--sst-dir", default=None)
    parser.add_argument("--out-dir", default=None)

    parser.add_argument("--draws", type=int, default=1500)
    parser.add_argument("--tune", type=int, default=1500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=94)

    return parser.parse_args()


def resolve_paths(args):
    base = Path(
        f"/home/{args.netid}/my_work/code/arabian_peninsula/"
        "bayesian_extremes/data"
    )

    data_glob = args.data_glob or str(
        base / "DailyPeakState" / "DailyPeakState-*.nc"
    )
    index_csv = args.index_csv or str(
        base / "sst" / "roni_dmi_monthly_1950_2025.csv"
    )
    sst_dir = Path(args.sst_dir or (base / "sst" / "basin_anoms"))
    out_dir = Path(
        args.out_dir or (base / "joint_enso_local_sst_city_runs")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"joint_{args.target_var}_enso_iod_local_sst_JJAS"

    return {
        "data_glob": data_glob,
        "index_csv": index_csv,
        "sst_dir": sst_dir,
        "out_dir": out_dir,
        "idata_nc": out_dir / f"idata_{stem}.nc",
        "idata_pkl": out_dir / f"idata_{stem}.pkl",
        "meta": out_dir / f"meta_{stem}.pkl",
    }


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------
def parse_months(value):
    if value.strip().lower() == "all":
        return None
    months = [int(x.strip()) for x in value.split(",")]
    if any(month < 1 or month > 12 for month in months):
        raise ValueError(f"Invalid month list: {months}")
    return months


def month_key(values):
    return (
        pd.DatetimeIndex(pd.to_datetime(values))
        .to_period("M")
        .to_timestamp()
    )


def pick_col(columns, key):
    exact = {column.lower(): column for column in columns}
    if key.lower() in exact:
        return exact[key.lower()]

    for column in columns:
        if key.lower() in column.lower():
            return column

    return None


def pick_var(ds, target_var):
    if target_var in ds.data_vars:
        return target_var

    lower = {name.lower(): name for name in ds.data_vars}
    if target_var.lower() in lower:
        return lower[target_var.lower()]

    raise KeyError(
        f"Could not find '{target_var}'. "
        f"Available variables: {list(ds.data_vars)}"
    )


def get_latlon_names(obj):
    lat_name = "latitude" if "latitude" in obj.coords else "lat"
    lon_name = "longitude" if "longitude" in obj.coords else "lon"

    if lat_name not in obj.coords or lon_name not in obj.coords:
        raise KeyError(
            f"Could not identify latitude/longitude coordinates: "
            f"{list(obj.coords)}"
        )

    return lat_name, lon_name


def shift_lon_180(ds, lon_name):
    if float(ds[lon_name].max()) > 180:
        lon_new = ((ds[lon_name] + 180) % 360) - 180
        ds = ds.assign_coords({lon_name: lon_new}).sortby(lon_name)

    return ds


def nearest_ij(lat_vals, lon_vals, lat0, lon0):
    i = int(np.argmin(np.abs(lat_vals - lat0)))
    j = int(np.argmin(np.abs(lon_vals - lon0)))
    return i, j


def safe_save_idata(idata, nc_path, pkl_path):
    try:
        az.to_netcdf(idata, nc_path)
        print(f"Saved inference data: {nc_path}")
        return
    except Exception as exc:
        print(f"NetCDF save failed: {exc}")
        traceback.print_exc()

    with open(pkl_path, "wb") as handle:
        pickle.dump(idata, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved inference data as pickle: {pkl_path}")




# ---------------------------------------------------------------------
# Standardized posterior diagnostics
# ---------------------------------------------------------------------
R_HAT_LIMIT = 1.01
ESS_LIMIT = 400
BFMI_LIMIT = 0.30
PPC_MAX_OBS = 2000
PPC_MAX_DRAWS = 500


def diagnostic_directory(paths):
    directory = paths["out_dir"] / "posterior_checks"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def diagnostic_var_names(idata):
    excluded = {
        "sst_mu",
        "gpd_sigma",
        "a_city",
        "b_roni_city",
        "b_dmi_city",
        "b_roni_dmi_city",
        "b_sst_city",
        "b_roni_sst_city",
    }
    return [
        name for name in idata.posterior.data_vars
        if name not in excluded and not name.endswith("_z")
    ]


def write_convergence_checks(idata, paths, run_id):
    out_dir = diagnostic_directory(paths)
    variables = diagnostic_var_names(idata)
    summary = az.summary(
        idata,
        var_names=variables,
        kind="diagnostics",
        round_to=None,
    ).reset_index(names="parameter")

    summary["rhat_flag"] = summary["r_hat"] > R_HAT_LIMIT
    summary["ess_bulk_flag"] = summary["ess_bulk"] < ESS_LIMIT
    summary["ess_tail_flag"] = summary["ess_tail"] < ESS_LIMIT

    stats = idata.sample_stats
    divergences = (
        int(stats["diverging"].sum().values)
        if "diverging" in stats else np.nan
    )
    tree_depth_hits = np.nan
    if "tree_depth" in stats:
        observed_max = np.nanmax(stats["tree_depth"].values)
        tree_depth_hits = int(
            np.sum(stats["tree_depth"].values >= observed_max)
        )
    minimum_bfmi = float(np.nanmin(np.asarray(az.bfmi(idata))))

    summary["divergences"] = divergences
    summary["tree_depth_hits_at_observed_max"] = tree_depth_hits
    summary["minimum_bfmi"] = minimum_bfmi
    summary["overall_flag"] = (
        summary["rhat_flag"]
        | summary["ess_bulk_flag"]
        | summary["ess_tail_flag"]
        | (divergences > 0)
        | (minimum_bfmi < BFMI_LIMIT)
    )

    summary.to_csv(
        out_dir / f"diagnostics_{run_id}.csv",
        index=False,
    )
    pd.DataFrame([{
        "run_id": run_id,
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "min_ess_tail": float(summary["ess_tail"].min()),
        "divergences": divergences,
        "tree_depth_hits_at_observed_max": tree_depth_hits,
        "minimum_bfmi": minimum_bfmi,
        "passed_all_checks": not bool(summary["overall_flag"].any()),
    }]).to_csv(
        out_dir / f"diagnostics_summary_{run_id}.csv",
        index=False,
    )

    axes = az.plot_trace(idata, var_names=variables, compact=True)
    figure = np.asarray(axes).ravel()[0].figure
    figure.tight_layout()
    figure.savefig(
        out_dir / f"trace_{run_id}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def stack_sample(data_array):
    return data_array.stack(sample=("chain", "draw")).values


def stack_sample_dim(data_array, dim):
    return (
        data_array.stack(sample=("chain", "draw"))
        .transpose("sample", dim)
        .values
    )


def sample_gpd(rng, sigma, xi):
    uniform = rng.uniform(
        np.finfo(float).eps,
        1.0 - np.finfo(float).eps,
        size=sigma.shape,
    )
    near_zero = np.abs(xi) < 1e-6
    output = np.empty_like(sigma, dtype=float)
    output[near_zero] = (
        -sigma[near_zero] * np.log(1.0 - uniform[near_zero])
    )
    output[~near_zero] = (
        sigma[~near_zero] / xi[~near_zero]
        * ((1.0 - uniform[~near_zero]) ** (-xi[~near_zero]) - 1.0)
    )
    return output


def summarize_ppc(observed, replicated, scope, component, run_id):
    functions = {
        "mean": np.mean,
        "sd": np.std,
        "median": np.median,
        "p05": lambda values, axis=None: np.quantile(values, 0.05, axis=axis),
        "p95": lambda values, axis=None: np.quantile(values, 0.95, axis=axis),
        "maximum": np.max,
    }
    rows = []
    for name, function in functions.items():
        observed_value = float(function(observed))
        replicated_values = function(replicated, axis=1)
        low, high = az.hdi(replicated_values, hdi_prob=0.94)
        rows.append({
            "run_id": run_id,
            "component": component,
            "scope": scope,
            "statistic": name,
            "observed": observed_value,
            "replicated_mean": float(np.mean(replicated_values)),
            "replicated_hdi_low": float(low),
            "replicated_hdi_high": float(high),
            "observed_inside_94pct_hdi": bool(low <= observed_value <= high),
            "n_observations_checked": int(observed.size),
            "n_posterior_draws_checked": int(replicated.shape[0]),
        })
    return rows


def write_posterior_predictive_checks(
    idata, basin_months, events, paths, run_id, seed
):
    out_dir = diagnostic_directory(paths)
    rng = np.random.default_rng(seed)
    rows = []

    # Basin SST Student-t component
    n_bm = min(PPC_MAX_OBS, len(basin_months))
    bm_idx = np.sort(rng.choice(len(basin_months), n_bm, replace=False))
    sst_mu = (
        idata.posterior["sst_mu"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "basin_month")
        .values
    )
    sigma_basin = stack_sample_dim(
        idata.posterior["sst_sigma_basin"], "basin"
    )
    nu = stack_sample(idata.posterior["sst_nu"])
    basin_ids = basin_months["basin_id"].values.astype(int)

    n_draws = min(PPC_MAX_DRAWS, sst_mu.shape[0])
    draw_idx = np.sort(rng.choice(sst_mu.shape[0], n_draws, replace=False))
    mu = sst_mu[draw_idx][:, bm_idx]
    scales = sigma_basin[draw_idx][:, basin_ids[bm_idx]]
    replicated_sst = (
        mu + scales * rng.standard_t(
            nu[draw_idx, None],
            size=mu.shape,
        )
    )
    observed_sst = basin_months["sst"].values[bm_idx]

    rows.extend(summarize_ppc(
        observed_sst, replicated_sst, "all_basins",
        "basin_sst_student_t", run_id
    ))
    for basin_id, basin in enumerate(BASIN_LIST):
        mask = basin_ids[bm_idx] == basin_id
        if np.any(mask):
            rows.extend(summarize_ppc(
                observed_sst[mask], replicated_sst[:, mask], basin,
                "basin_sst_student_t", run_id
            ))

    # City GPD component
    n_events = min(PPC_MAX_OBS, len(events))
    event_idx = np.sort(rng.choice(len(events), n_events, replace=False))
    gpd_sigma = (
        idata.posterior["gpd_sigma"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "event")
        .values
    )
    xi = stack_sample(idata.posterior["xi"])
    n_gpd_draws = min(PPC_MAX_DRAWS, gpd_sigma.shape[0])
    gpd_draw_idx = np.sort(
        rng.choice(gpd_sigma.shape[0], n_gpd_draws, replace=False)
    )
    sigma = gpd_sigma[gpd_draw_idx][:, event_idx]
    xi_matrix = np.broadcast_to(
        xi[gpd_draw_idx, None], sigma.shape
    )
    replicated_gpd = sample_gpd(rng, sigma, xi_matrix)
    observed_gpd = events["z"].values[event_idx]
    city_ids = events["city_id"].values.astype(int)

    rows.extend(summarize_ppc(
        observed_gpd, replicated_gpd, "all_cities",
        "city_gpd", run_id
    ))
    for city_id, city in enumerate(CITY_LIST):
        mask = city_ids[event_idx] == city_id
        if np.any(mask):
            rows.extend(summarize_ppc(
                observed_gpd[mask], replicated_gpd[:, mask], city,
                "city_gpd", run_id
            ))

    ppc_df = pd.DataFrame(rows)
    ppc_df.to_csv(out_dir / f"ppc_{run_id}.csv", index=False)

    for component, observed, replicated, xlabel in [
        (
            "basin_sst",
            observed_sst,
            replicated_sst,
            "Basin SST anomaly",
        ),
        (
            "city_gpd",
            observed_gpd,
            replicated_gpd,
            "Threshold exceedance magnitude",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        observed_sorted = np.sort(observed)
        ax.plot(
            observed_sorted,
            np.arange(1, observed_sorted.size + 1) / observed_sorted.size,
            linewidth=2,
            label="Observed",
        )
        for draw in replicated[:min(50, replicated.shape[0])]:
            draw_sorted = np.sort(draw)
            ax.plot(
                draw_sorted,
                np.arange(1, draw_sorted.size + 1) / draw_sorted.size,
                alpha=0.12,
                linewidth=0.8,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Empirical cumulative probability")
        ax.set_title(f"Posterior predictive check: {component}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            out_dir / f"ppc_{run_id}_{component}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)



# ---------------------------------------------------------------------
# Lagged indices
# ---------------------------------------------------------------------
def load_monthly_indices(csv_path, roni_lag, dmi_lag, already_lagged):
    frame = pd.read_csv(csv_path)

    if "time" in frame.columns:
        frame["time"] = pd.to_datetime(frame["time"])
    elif {"year", "month"}.issubset(frame.columns):
        frame["time"] = pd.to_datetime(
            dict(
                year=frame["year"],
                month=frame["month"],
                day=1,
            )
        )
    else:
        raise ValueError(
            "Index CSV must contain either 'time' or both "
            "'year' and 'month'."
        )

    roni_col = pick_col(frame.columns, "roni")
    dmi_col = pick_col(frame.columns, "dmi")

    if roni_col is None or dmi_col is None:
        raise ValueError(
            f"Could not identify RONI/DMI columns: "
            f"{frame.columns.tolist()}"
        )

    frame = frame.set_index("time").sort_index()
    frame.index = frame.index.to_period("M").to_timestamp()
    frame = frame[~frame.index.duplicated(keep="last")]

    indices = pd.DataFrame(
        {
            "roni": frame[roni_col].astype(float),
            "dmi": frame[dmi_col].astype(float),
        }
    )

    if not already_lagged:
        # A July day receives May RONI and June DMI under defaults.
        indices["roni"] = indices["roni"].shift(roni_lag)
        indices["dmi"] = indices["dmi"].shift(dmi_lag)

    # Match the established remote scripts: standardize after lagging.
    indices["roni"] = (
        indices["roni"] - indices["roni"].mean()
    ) / indices["roni"].std()
    indices["dmi"] = (
        indices["dmi"] - indices["dmi"].mean()
    ) / indices["dmi"].std()

    indices["roni_dmi"] = indices["roni"] * indices["dmi"]

    return indices


# ---------------------------------------------------------------------
# Basin SST data
# ---------------------------------------------------------------------
def load_one_basin_sst(sst_dir, basin):
    path = sst_dir / f"era5_sst_anom_{basin}_1950_2025.nc"

    if not path.exists():
        raise FileNotFoundError(f"Missing basin SST file: {path}")

    ds = xr.open_dataset(path)

    try:
        var_name = (
            "sst_anom"
            if "sst_anom" in ds.data_vars
            else list(ds.data_vars)[0]
        )
        da = ds[var_name]
        lat_name, lon_name = get_latlon_names(da)

        if float(da[lon_name].max()) > 180:
            lon_new = ((da[lon_name] + 180) % 360) - 180
            da = da.assign_coords({lon_name: lon_new}).sortby(lon_name)

        times = pd.DatetimeIndex(pd.to_datetime(da["time"].values))
        is_month_start = (
            (times.day == 1).all()
            and (times.hour == 0).all()
        )

        if not is_month_start:
            da = da.resample(time="MS").mean(skipna=True)

        series = da.mean(
            dim=[lat_name, lon_name],
            skipna=True,
        ).to_series()

        series.index = (
            pd.DatetimeIndex(pd.to_datetime(series.index))
            .to_period("M")
            .to_timestamp()
        )
        series = (
            series[~series.index.duplicated(keep="last")]
            .sort_index()
        )
        series.name = basin

        return series.astype(float)

    finally:
        ds.close()


def load_all_basin_sst(sst_dir):
    basin_series = [
        load_one_basin_sst(sst_dir, basin)
        for basin in BASIN_LIST
    ]

    return pd.concat(
        basin_series,
        axis=1,
        join="inner",
    ).sort_index()


# ---------------------------------------------------------------------
# Daily city series
# ---------------------------------------------------------------------
def load_city_daily(files, city, target_var, months):
    metadata = CITIES[city]

    values = []
    times = []
    ij = None
    var_name = None
    ref_lat = None
    ref_lon = None

    for filepath in files:
        ds = xr.open_dataset(filepath)

        try:
            if "day" in ds.coords and "time" not in ds.coords:
                ds = ds.rename({"day": "time"})

            lat_name, lon_name = get_latlon_names(ds)
            ds = shift_lon_180(ds, lon_name)

            if var_name is None:
                var_name = pick_var(ds, target_var)
                print(f"[{city}] variable: {var_name}")

            lat_vals = ds[lat_name].values
            lon_vals = ds[lon_name].values

            if ij is None:
                ref_lat = lat_vals.copy()
                ref_lon = lon_vals.copy()
                ij = nearest_ij(
                    lat_vals,
                    lon_vals,
                    metadata["lat"],
                    metadata["lon"],
                )

                i, j = ij
                print(
                    f"[{city}] basin={metadata['basin']} | "
                    f"grid lat={lat_vals[i]:.3f}, "
                    f"lon={lon_vals[j]:.3f}"
                )

            elif not (
                np.array_equal(lat_vals, ref_lat)
                and np.array_equal(lon_vals, ref_lon)
            ):
                raise ValueError(
                    f"{city}: spatial grid changed in {filepath}"
                )

            i, j = ij
            da = ds[var_name].isel(
                {
                    lat_name: i,
                    lon_name: j,
                }
            )

            values.append(
                np.asarray(da.values, dtype="float32")
            )
            times.append(
                pd.to_datetime(da["time"].values)
            )

        finally:
            ds.close()

    y = np.concatenate(values)
    t = pd.DatetimeIndex(np.concatenate(times))

    order = np.argsort(t.values)
    t = t[order]
    y = y[order]

    finite = np.isfinite(y)
    t = t[finite]
    y = y[finite]

    if months is not None:
        keep = t.month.isin(months)
        t = t[keep]
        y = y[keep]

    return t, y


# ---------------------------------------------------------------------
# Construct monthly basin table and city exceedance table
# ---------------------------------------------------------------------
def prepare_data(args, paths):
    months = parse_months(args.months)

    daily_files = sorted(glob.glob(paths["data_glob"]))
    if not daily_files:
        raise FileNotFoundError(
            f"No daily files found: {paths['data_glob']}"
        )

    indices = load_monthly_indices(
        paths["index_csv"],
        roni_lag=args.roni_lag,
        dmi_lag=args.dmi_lag,
        already_lagged=args.indices_already_lagged,
    )

    basin_sst = load_all_basin_sst(paths["sst_dir"])

    monthly = indices.join(
        basin_sst,
        how="inner",
    ).dropna()

    if months is not None:
        monthly = monthly[
            monthly.index.month.isin(months)
        ]

    # Basin-month observations for the SST submodel.
    basin_month_rows = []

    for basin_id, basin in enumerate(BASIN_LIST):
        temp = monthly[
            ["roni", "dmi", "roni_dmi", basin]
        ].copy()

        temp = temp.rename(columns={basin: "sst"})
        temp["basin_id"] = basin_id
        temp["basin"] = basin
        temp["time"] = temp.index

        basin_month_rows.append(
            temp.reset_index(drop=True)
        )

    basin_months = pd.concat(
        basin_month_rows,
        ignore_index=True,
    )

    event_rows = []
    thresholds = {}
    n_days = {}
    n_events = {}

    for city_id, city in enumerate(CITY_LIST):
        basin = CITIES[city]["basin"]

        t, y = load_city_daily(
            daily_files,
            city,
            args.target_var,
            months,
        )

        keys = month_key(t)

        required_columns = [
            "roni",
            "dmi",
            "roni_dmi",
        ]

        if basin is not None:
            required_columns.append(basin)

        required = monthly[required_columns].reindex(keys)
        complete = required.notna().all(axis=1).values

        t = t[complete]
        y = y[complete]
        required = required.iloc[np.where(complete)[0]]

        if y.size == 0:
            raise RuntimeError(
                f"{city}: no observations remain after alignment."
            )

        threshold = float(np.quantile(y, args.q))
        exceed = y > threshold
        z = y[exceed] - threshold

        if z.size < args.min_events:
            raise RuntimeError(
                f"{city}: {z.size} exceedances is below "
                f"--min-events={args.min_events}"
            )

        if basin is None:
            local_sst = np.zeros(z.size, dtype="float32")
            has_local_sst = np.zeros(z.size, dtype="float32")
        else:
            local_sst = required[basin].values[
                exceed
            ].astype("float32")
            has_local_sst = np.ones(z.size, dtype="float32")

        thresholds[city] = threshold
        n_days[city] = int(y.size)
        n_events[city] = int(z.size)

        event_rows.append(
            pd.DataFrame(
                {
                    "z": z.astype("float32"),
                    "city_id": np.full(
                        z.size,
                        city_id,
                        dtype="int32",
                    ),
                    "roni": required["roni"].values[
                        exceed
                    ].astype("float32"),
                    "dmi": required["dmi"].values[
                        exceed
                    ].astype("float32"),
                    "roni_dmi": required["roni_dmi"].values[
                        exceed
                    ].astype("float32"),
                    "local_sst": local_sst,
                    "has_local_sst": has_local_sst,
                }
            )
        )

        print(
            f"{city}: basin={basin}, days={y.size}, "
            f"events={z.size}, u={threshold:.3f}"
        )

    events = pd.concat(
        event_rows,
        ignore_index=True,
    )
    events["roni_sst"] = (
        events["roni"]
        * events["local_sst"]
        * events["has_local_sst"]
    )

    meta = {
        "cities": CITY_LIST,
        "pooled_cities": POOLED_CITIES,
        "independent_cities": INDEPENDENT_CITIES,
        "local_sst_cities": LOCAL_SST_CITIES,
        "inland_cities": INLAND_CITIES,
        "basins": BASIN_LIST,
        "city_metadata": CITIES,
        "thresholds": thresholds,
        "n_days": n_days,
        "n_events": n_events,
        "q": args.q,
        "months": months,
        "target_var": args.target_var,
        "roni_lag": args.roni_lag,
        "dmi_lag": args.dmi_lag,
        "indices_already_lagged": args.indices_already_lagged,
        "monthly_start": monthly.index.min(),
        "monthly_end": monthly.index.max(),
        "data_glob": paths["data_glob"],
        "index_csv": paths["index_csv"],
        "sst_dir": str(paths["sst_dir"]),
    }

    return basin_months, events, meta


# ---------------------------------------------------------------------
# GPD likelihood
# ---------------------------------------------------------------------
def gpd_logp(value, sigma, xi, eps=1e-12, xi_tol=1e-6):
    sigma = sigma + eps
    support = 1.0 + xi * value / sigma

    logp_gpd = (
        -pt.log(sigma)
        - (1.0 + 1.0 / xi) * pt.log(support)
    )
    logp_exp = -pt.log(sigma) - value / sigma

    logp = pt.switch(
        pt.abs(xi) < xi_tol,
        logp_exp,
        logp_gpd,
    )
    logp = pt.switch(
        support > 0,
        logp,
        -np.inf,
    )

    return pt.sum(logp)


# ---------------------------------------------------------------------
# Parameter construction
# ---------------------------------------------------------------------
def build_remote_city_effect(
    name,
    pooled_bar_sigma,
    pooled_sd_sigma,
    independent_sigma,
):
    """
    Construct one city coefficient vector with:
      * partial pooling for Doha, Dubai, and Dammam;
      * independent priors for all other cities.
    """
    pooled_bar = pm.Normal(
        f"{name}_pooled_bar",
        0.0,
        pooled_bar_sigma,
    )
    pooled_sd = pm.HalfNormal(
        f"{name}_pooled_sd",
        pooled_sd_sigma,
    )
    pooled_z = pm.Normal(
        f"{name}_pooled_z",
        0.0,
        1.0,
        dims="pooled_city",
    )
    pooled_values = pm.Deterministic(
        f"{name}_pooled",
        pooled_bar + pooled_sd * pooled_z,
        dims="pooled_city",
    )

    independent_values = pm.Normal(
        f"{name}_independent",
        0.0,
        independent_sigma,
        dims="independent_city",
    )

    full_values = []

    for city in CITY_LIST:
        if city in POOLED_TO_ID:
            value = pooled_values[POOLED_TO_ID[city]]
        else:
            value = independent_values[
                INDEPENDENT_TO_ID[city]
            ]
        full_values.append(value)

    return pm.Deterministic(
        f"{name}_city",
        pt.stack(full_values),
        dims="city",
    )


def build_local_sst_city_effect(
    name,
    pooled_bar_sigma,
    pooled_sd_sigma,
    independent_sigma,
):
    """
    Construct a city coefficient vector for local-SST terms.

    Doha, Dubai, and Dammam are partially pooled.
    Other basin-linked cities are independent.
    Medina and Riyadh are fixed at zero because they have no adjacent basin.
    """
    local_independent_cities = [
        city
        for city in LOCAL_SST_CITIES
        if city not in POOLED_CITIES
    ]
    local_independent_to_id = {
        city: i
        for i, city in enumerate(local_independent_cities)
    }

    pooled_bar = pm.Normal(
        f"{name}_pooled_bar",
        0.0,
        pooled_bar_sigma,
    )
    pooled_sd = pm.HalfNormal(
        f"{name}_pooled_sd",
        pooled_sd_sigma,
    )
    pooled_z = pm.Normal(
        f"{name}_pooled_z",
        0.0,
        1.0,
        dims="pooled_city",
    )
    pooled_values = pm.Deterministic(
        f"{name}_pooled",
        pooled_bar + pooled_sd * pooled_z,
        dims="pooled_city",
    )

    independent_values = pm.Normal(
        f"{name}_independent",
        0.0,
        independent_sigma,
        dims="local_independent_city",
    )

    full_values = []

    for city in CITY_LIST:
        if city in POOLED_TO_ID:
            value = pooled_values[POOLED_TO_ID[city]]
        elif city in local_independent_to_id:
            value = independent_values[
                local_independent_to_id[city]
            ]
        else:
            value = pt.as_tensor_variable(0.0)

        full_values.append(value)

    return pm.Deterministic(
        f"{name}_city",
        pt.stack(full_values),
        dims="city",
    )


# ---------------------------------------------------------------------
# Joint model
# ---------------------------------------------------------------------
def fit_model(args, paths):
    basin_months, events, meta = prepare_data(
        args,
        paths,
    )

    local_independent_cities = [
        city
        for city in LOCAL_SST_CITIES
        if city not in POOLED_CITIES
    ]

    coords = {
        "basin_month": np.arange(len(basin_months)),
        "event": np.arange(len(events)),
        "basin": BASIN_LIST,
        "city": CITY_LIST,
        "pooled_city": POOLED_CITIES,
        "independent_city": INDEPENDENT_CITIES,
        "local_independent_city": local_independent_cities,
    }

    with pm.Model(coords=coords) as model:
        # =============================================================
        # A. Basin SST response to lagged ENSO and IOD
        # =============================================================
        bm_basin_id = pm.Data(
            "bm_basin_id",
            basin_months["basin_id"].values.astype("int32"),
            dims="basin_month",
        )
        bm_roni = pm.Data(
            "bm_roni",
            basin_months["roni"].values.astype("float32"),
            dims="basin_month",
        )
        bm_dmi = pm.Data(
            "bm_dmi",
            basin_months["dmi"].values.astype("float32"),
            dims="basin_month",
        )
        bm_nd = pm.Data(
            "bm_roni_dmi",
            basin_months["roni_dmi"].values.astype("float32"),
            dims="basin_month",
        )
        bm_sst = pm.Data(
            "bm_sst_observed",
            basin_months["sst"].values.astype("float32"),
            dims="basin_month",
        )

        # Basin-specific effects are independent, matching the earlier
        # basin-by-basin SST analyses. No information is pooled across basins.
        sst_a = pm.Normal(
            "sst_a_basin",
            0.0,
            0.5,
            dims="basin",
        )
        sst_b_roni = pm.Normal(
            "sst_b_roni_basin",
            0.0,
            0.5,
            dims="basin",
        )
        sst_b_dmi = pm.Normal(
            "sst_b_dmi_basin",
            0.0,
            0.5,
            dims="basin",
        )
        sst_b_nd = pm.Normal(
            "sst_b_roni_dmi_basin",
            0.0,
            0.3,
            dims="basin",
        )

        sst_sigma_basin = pm.HalfNormal(
            "sst_sigma_basin",
            0.5,
            dims="basin",
        )
        sst_nu_minus_two = pm.Exponential(
            "sst_nu_minus_two",
            1.0 / 10.0,
        )
        sst_nu = pm.Deterministic(
            "sst_nu",
            sst_nu_minus_two + 2.0,
        )

        sst_mu = pm.Deterministic(
            "sst_mu",
            sst_a[bm_basin_id]
            + sst_b_roni[bm_basin_id] * bm_roni
            + sst_b_dmi[bm_basin_id] * bm_dmi
            + sst_b_nd[bm_basin_id] * bm_nd,
            dims="basin_month",
        )

        pm.StudentT(
            "sst_like",
            nu=sst_nu,
            mu=sst_mu,
            sigma=sst_sigma_basin[bm_basin_id],
            observed=bm_sst,
            dims="basin_month",
        )

        # =============================================================
        # B. City POT/GPD response
        # =============================================================
        z_obs = pm.Data(
            "z_observed",
            events["z"].values.astype("float32"),
            dims="event",
        )
        event_city_id = pm.Data(
            "event_city_id",
            events["city_id"].values.astype("int32"),
            dims="event",
        )
        event_roni = pm.Data(
            "event_roni",
            events["roni"].values.astype("float32"),
            dims="event",
        )
        event_dmi = pm.Data(
            "event_dmi",
            events["dmi"].values.astype("float32"),
            dims="event",
        )
        event_nd = pm.Data(
            "event_roni_dmi",
            events["roni_dmi"].values.astype("float32"),
            dims="event",
        )
        event_sst = pm.Data(
            "event_local_sst",
            events["local_sst"].values.astype("float32"),
            dims="event",
        )
        event_roni_sst = pm.Data(
            "event_roni_sst",
            events["roni_sst"].values.astype("float32"),
            dims="event",
        )

        xi = pm.TruncatedNormal(
            "xi",
            mu=0.05,
            sigma=0.15,
            lower=XI_LOWER,
            upper=XI_UPPER,
        )

        # Remote coefficients follow the established pooling choices.
        a_city = build_remote_city_effect(
            "a",
            pooled_bar_sigma=1.0,
            pooled_sd_sigma=0.8,
            independent_sigma=1.0,
        )
        b_roni_city = build_remote_city_effect(
            "b_roni",
            pooled_bar_sigma=0.5,
            pooled_sd_sigma=0.3,
            independent_sigma=0.5,
        )
        b_dmi_city = build_remote_city_effect(
            "b_dmi",
            pooled_bar_sigma=0.5,
            pooled_sd_sigma=0.3,
            independent_sigma=0.5,
        )
        b_nd_city = build_remote_city_effect(
            "b_roni_dmi",
            pooled_bar_sigma=0.3,
            pooled_sd_sigma=0.2,
            independent_sigma=0.3,
        )

        # Local-SST coefficients use the same Gulf pooling choice.
        b_sst_city = build_local_sst_city_effect(
            "b_sst",
            pooled_bar_sigma=0.5,
            pooled_sd_sigma=0.3,
            independent_sigma=0.5,
        )
        b_roni_sst_city = build_local_sst_city_effect(
            "b_roni_sst",
            pooled_bar_sigma=0.3,
            pooled_sd_sigma=0.2,
            independent_sigma=0.3,
        )

        log_sigma = (
            a_city[event_city_id]
            + b_roni_city[event_city_id] * event_roni
            + b_dmi_city[event_city_id] * event_dmi
            + b_nd_city[event_city_id] * event_nd
            + b_sst_city[event_city_id] * event_sst
            + b_roni_sst_city[event_city_id]
            * event_roni_sst
        )

        gpd_sigma = pm.Deterministic(
            "gpd_sigma",
            1e-6 + pt.exp(log_sigma),
            dims="event",
        )

        pm.DensityDist(
            "gpd_like",
            gpd_sigma,
            xi,
            logp=lambda value, sigma, shape: gpd_logp(
                value,
                sigma,
                shape,
            ),
            observed=z_obs,
            dims="event",
        )

        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
            random_seed=args.seed,
            return_inferencedata=True,
        )

    safe_save_idata(
        idata,
        paths["idata_nc"],
        paths["idata_pkl"],
    )
    pd.to_pickle(meta, paths["meta"])
    print(f"Saved metadata: {paths['meta']}")

    run_id = f"joint_{args.target_var}_enso_iod_local_sst_JJAS"
    write_convergence_checks(idata, paths, run_id)
    write_posterior_predictive_checks(
        idata=idata,
        basin_months=basin_months,
        events=events,
        paths=paths,
        run_id=run_id,
        seed=args.seed,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    paths = resolve_paths(args)

    print("=" * 78)
    print("Joint ENSO–IOD + local-basin SST POT/GPD fit")
    print(f"Target variable: {args.target_var}")
    print(f"Cities: {CITY_LIST}")
    print(f"Pooled cities: {POOLED_CITIES}")
    print(f"Independent cities: {INDEPENDENT_CITIES}")
    print(f"Local-SST cities: {LOCAL_SST_CITIES}")
    print(f"Inland cities without local SST: {INLAND_CITIES}")
    print(f"Daily files: {paths['data_glob']}")
    print(f"Index CSV: {paths['index_csv']}")
    print(f"Basin SST directory: {paths['sst_dir']}")
    print(f"Output directory: {paths['out_dir']}")
    print("=" * 78)

    fit_model(args, paths)


if __name__ == "__main__":
    main()