#!/usr/bin/env python3
"""
MODEL 5 OF 6 — CITY EXTREME-STATE RESPONSE TO ADJACENT-BASIN SST
================================================================

Manuscript role
---------------
This pipeline supports the local-basin warming analysis associated with
Section 2.5. It fits basin-specific hierarchical POT/GPD models linking the
magnitude of city-level extreme daily atmospheric states to monthly SST
anomalies in each city's adjacent basin.

Pipeline modes
--------------
1. ``fit_one``
   Fits one target-variable and basin combination and saves its posterior and
   metadata.

2. ``aggregate_one_var``
   Loads all available basin fits for one target variable and evaluates the
   configured +0.5, +1.0, +1.5, and +2.0 degree Celsius basin-warming
   experiments.

Scientific response
-------------------
For each city c, a city-specific empirical JJAS p95 threshold u_c is calculated
from the selected daily target variable. The fitted response is the positive
exceedance magnitude

    z_{c,t} = Y_{c,t} - u_c, for Y_{c,t} > u_c.

Supported target variables are daily maximum wet-bulb temperature and the
temperature, specific humidity, and tau_MSE values occurring at that daily
wet-bulb maximum.

Modeling choices
----------------
1. Season:
   Only June–September observations are retained.

2. Local SST covariate:
   The predictor is the spatial mean monthly SST anomaly of the basin assigned
   to each city. Every daily exceedance within a month receives the same
   monthly basin-SST anomaly.

3. Threshold and event count:
   Thresholds are city-specific empirical p95 values. A city must retain at
   least 50 exceedances.

4. GPD structure:
   Exceedance magnitudes follow a generalized Pareto distribution with

       log(sigma_{c,t}) = a_c + b_c SST_{b(c),t}.

   Basin SST affects the GPD scale only. It does not affect the threshold-
   exceedance probability and does not affect the GPD shape parameter.

5. Shape parameter:
   One xi parameter is shared by all cities included in a basin fit and is
   constrained to [-0.3, 0.5].

6. Partial pooling:
   Doha, Dubai, and Dammam are partially pooled when fitting the Persian Gulf
   basin. Kuwait City and Basra remain independent. Cities assigned to the
   other basins are fit independently because each basin contains only one
   modeled city.

7. Warming experiments:
   Counterfactual warming values are additive changes to the basin-SST anomaly
   covariate. The reported p95 and p99 changes are conditional GPD quantile
   changes unless a separate unconditional POT calculation is explicitly used.

8. Dependence not modeled:
   Exceedances are not declustered. Repeated monthly SST values, serial
   dependence among daily extremes, and cross-city dependence are not modeled
   explicitly.

9. Basin naming:
   The internal key ``arabian_gulf`` is retained because existing data,
   metadata, and posterior filenames depend on it. Manuscript-facing text uses
   ``Persian Gulf``.

Posterior checks
----------------
Every ``fit_one`` run now writes:
  * parameter-level R-hat, bulk ESS, and tail ESS;
  * divergence, tree-depth, and BFMI summaries;
  * trace plots for fitted basin-level and independent-city parameters;
  * inverse-CDF GPD posterior-predictive checks;
  * pooled and city-specific observed-versus-replicated summaries.

Existing data, posterior, metadata, aggregate CSV, and figure locations are
retained.
"""

import os
import glob
import argparse
import traceback
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
NETID = "k16v981"

CITIES = {
    "muscat":      {"lat": 23.5880, "lon": 58.3829},
    "doha":        {"lat": 25.2854, "lon": 51.5310},
    "dubai":       {"lat": 25.2048, "lon": 55.2708},
    "kuwait_city": {"lat": 29.3759, "lon": 47.9774},
    "basra":       {"lat": 30.5085, "lon": 47.7835},
    "jeddah":      {"lat": 21.4858, "lon": 39.1925},
    "aden":        {"lat": 12.7855, "lon": 45.0187},
    "dammam":       {"lat": 26.4207, "lon": 50.0888},
}

CITY_LIST = list(CITIES.keys())

CITY_TO_BASIN = {
    "muscat":      "gulf_oman",
    "doha":        "arabian_gulf",
    "dubai":       "arabian_gulf",
    "kuwait_city": "arabian_gulf",
    "basra":       "arabian_gulf",
    "dammam":      "arabian_gulf",
    "jeddah":      "red_sea",
    "aden":        "gulf_aden",
}

VALID_BASINS = [
    "gulf_oman",
    "arabian_gulf",
    "red_sea",
    "gulf_aden",
]

POOLED_GULF_CITIES = [
    "doha",
    "dubai",
    "dammam",
]

VALID_TARGET_VARS = [
    "wbt_daily_peak",
    "tau_at_wbt_daily_peak",
    "t2m_at_wbt_daily_peak",
    "q_at_wbt_daily_peak",
]

WBT_DIR = f"/home/{NETID}/my_work/code/arabian_peninsula/bayesian_extremes/data/DailyPeakState/"
WBT_GLOB = os.path.join(WBT_DIR, "DailyPeakState-*.nc")

SST_DIR = f"/home/{NETID}/my_work/code/arabian_peninsula/bayesian_extremes/data/sst/basin_anoms"
OUT_DIR = f"/home/{NETID}/my_work/code/arabian_peninsula/bayesian_extremes/data/wbt_sst_city_runs"
os.makedirs(OUT_DIR, exist_ok=True)

MONTHS = [6, 7, 8, 9]   # JJAS
Q = 0.95
MIN_EVENTS = 50

XI_LOWER = -0.3
XI_UPPER = 0.5

RANDOM_SEED = 58
DRAWS = 1500
TUNE = 1500
CHAINS = 4
CORES = 4
TARGET_ACCEPT = 0.98

WARMING_EXPTS = {
    "+0.5C": 0.5,
    "+1C": 1.0,
    "+1.5C": 1.5,
    "+2C": 2.0,
}
Q_LEVELS = [0.95, 0.99]

# -----------------------
# Helpers
# -----------------------
def get_latlon_names(ds):
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    return lat_name, lon_name

def shift_lon_180(ds, lon_name):
    lon = ds[lon_name]
    if float(lon.max()) > 180:
        lon_new = ((lon + 180) % 360) - 180
        ds = ds.assign_coords({lon_name: lon_new}).sortby(lon_name)
    return ds

def nearest_ij(lat_vals, lon_vals, lat0, lon0):
    i = int(np.argmin(np.abs(lat_vals - lat0)))
    j = int(np.argmin(np.abs(lon_vals - lon0)))
    return i, j

def pick_var(ds, target_var):
    if target_var in ds.data_vars:
        return target_var
    for v in ds.data_vars:
        if v.lower() == target_var.lower():
            return v
    raise KeyError(
        f"Could not find target_var='{target_var}' in dataset. "
        f"Available vars={list(ds.data_vars)}"
    )

def month_key_daily(t_daily):
    return pd.to_datetime(t_daily).to_period("M").to_timestamp()

def gpd_logp(z, sigma, xi, eps=1e-12, xi_tol=1e-6):
    sigma = sigma + eps
    t = 1 + xi * z / sigma
    logp_gpd = -pt.log(sigma) - (1 + 1/xi) * pt.log(t)
    logp_exp = -pt.log(sigma) - z / sigma
    logp = pt.switch(pt.abs(xi) < xi_tol, logp_exp, logp_gpd)
    logp = pt.switch(t > 0, logp, -np.inf)
    return pt.sum(logp)

def safe_save_idata(idata, out_path):
    try:
        az.to_netcdf(idata, out_path)
        print(f"✅ saved idata: {out_path}")
        return
    except Exception as e:
        print("⚠️ az.to_netcdf failed:", e)
        traceback.print_exc()

    try:
        pkl_path = out_path.replace(".nc", ".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(idata, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✅ pickled idata: {pkl_path}")
    except Exception as e:
        print("❌ pickle also failed:", e)



# -----------------------
# Standardized posterior checks
# -----------------------
CHECK_DIR = os.path.join(OUT_DIR, "posterior_checks")
os.makedirs(CHECK_DIR, exist_ok=True)

R_HAT_LIMIT = 1.01
ESS_LIMIT = 400
BFMI_LIMIT = 0.30
PPC_MAX_EVENTS = 2000
PPC_MAX_DRAWS = 500


def _diagnostic_var_names(idata):
    """Retain fitted scalar/vector parameters while omitting large event arrays."""
    excluded = {
        "a_z",
        "b_z",
        "a_pooled",
        "b_pooled",
        "a_city",
        "b_city",
        "sigma",
    }
    return [
        name
        for name in idata.posterior.data_vars
        if name not in excluded
    ]


def write_convergence_checks(idata, run_id):
    """Write numerical MCMC diagnostics and a compact trace plot."""
    var_names = _diagnostic_var_names(idata)

    summary = az.summary(
        idata,
        var_names=var_names,
        kind="diagnostics",
        round_to=None,
    )
    summary.index.name = "parameter"
    summary = summary.reset_index()

    summary["rhat_flag"] = summary["r_hat"] > R_HAT_LIMIT
    summary["ess_bulk_flag"] = summary["ess_bulk"] < ESS_LIMIT
    summary["ess_tail_flag"] = summary["ess_tail"] < ESS_LIMIT

    sample_stats = idata.sample_stats
    divergences = (
        int(sample_stats["diverging"].sum().values)
        if "diverging" in sample_stats
        else np.nan
    )

    tree_depth_hits = np.nan
    if "tree_depth" in sample_stats:
        observed_max = np.nanmax(sample_stats["tree_depth"].values)
        tree_depth_hits = int(
            np.sum(sample_stats["tree_depth"].values >= observed_max)
        )

    minimum_bfmi = float(np.nanmin(np.asarray(az.bfmi(idata))))

    summary["divergences"] = divergences
    summary["tree_depth_hits_at_observed_max"] = tree_depth_hits
    summary["minimum_bfmi"] = minimum_bfmi
    summary["divergence_flag"] = divergences > 0
    summary["bfmi_flag"] = minimum_bfmi < BFMI_LIMIT
    summary["overall_flag"] = (
        summary["rhat_flag"]
        | summary["ess_bulk_flag"]
        | summary["ess_tail_flag"]
        | summary["divergence_flag"]
        | summary["bfmi_flag"]
    )

    detail_path = os.path.join(
        CHECK_DIR,
        f"diagnostics_{run_id}.csv",
    )
    summary.to_csv(detail_path, index=False)

    aggregate = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "max_rhat": float(summary["r_hat"].max()),
                "min_ess_bulk": float(summary["ess_bulk"].min()),
                "min_ess_tail": float(summary["ess_tail"].min()),
                "divergences": divergences,
                "tree_depth_hits_at_observed_max": tree_depth_hits,
                "minimum_bfmi": minimum_bfmi,
                "passed_all_checks": not bool(summary["overall_flag"].any()),
            }
        ]
    )
    aggregate.to_csv(
        os.path.join(
            CHECK_DIR,
            f"diagnostics_summary_{run_id}.csv",
        ),
        index=False,
    )

    axes = az.plot_trace(
        idata,
        var_names=var_names,
        compact=True,
    )
    figure = np.asarray(axes).ravel()[0].figure
    figure.tight_layout()
    figure.savefig(
        os.path.join(CHECK_DIR, f"trace_{run_id}.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"✅ wrote posterior diagnostics: {detail_path}")


def _stack_sample_city(data_array):
    return (
        data_array
        .stack(sample=("chain", "draw"))
        .transpose("sample", "city")
        .values
    )


def _stack_sample(data_array):
    return data_array.stack(sample=("chain", "draw")).values


def _sample_gpd(rng, sigma, xi):
    """Generate GPD random variates by inverse CDF."""
    uniform = rng.uniform(
        np.finfo(float).eps,
        1.0 - np.finfo(float).eps,
        size=sigma.shape,
    )
    near_zero = np.abs(xi) < 1e-6
    draws = np.empty_like(sigma, dtype=float)

    draws[near_zero] = (
        -sigma[near_zero] * np.log(1.0 - uniform[near_zero])
    )
    draws[~near_zero] = (
        sigma[~near_zero]
        / xi[~near_zero]
        * (
            (1.0 - uniform[~near_zero]) ** (-xi[~near_zero])
            - 1.0
        )
    )
    return draws


def write_posterior_predictive_checks(
    idata,
    z_all,
    sst_all,
    city_ids,
    city_names,
    run_id,
):
    """
    Compare observed exceedance magnitudes with replicated GPD draws.

    A reproducible subset limits diagnostic memory use. Checks are calculated
    for the complete basin fit and separately for every included city.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    n_events = min(PPC_MAX_EVENTS, len(z_all))
    event_index = np.sort(
        rng.choice(len(z_all), size=n_events, replace=False)
    )

    a_city = _stack_sample_city(idata.posterior["a_city"])
    b_city = _stack_sample_city(idata.posterior["b_city"])
    xi = _stack_sample(idata.posterior["xi"])

    n_available = a_city.shape[0]
    n_draws = min(PPC_MAX_DRAWS, n_available)
    draw_index = np.sort(
        rng.choice(n_available, size=n_draws, replace=False)
    )

    selected_city = city_ids[event_index]
    log_sigma = (
        a_city[draw_index][:, selected_city]
        + b_city[draw_index][:, selected_city]
        * sst_all[event_index][None, :]
    )
    sigma = np.exp(log_sigma) + 1e-6
    xi_matrix = np.broadcast_to(
        xi[draw_index, None],
        sigma.shape,
    )
    replicated = _sample_gpd(rng, sigma, xi_matrix)
    observed = z_all[event_index]

    statistic_functions = {
        "mean": np.mean,
        "sd": np.std,
        "median": np.median,
        "p95": lambda values, axis=None: np.quantile(
            values,
            0.95,
            axis=axis,
        ),
        "maximum": np.max,
    }

    rows = []

    def summarize_scope(scope, observed_scope, replicated_scope):
        for statistic, function in statistic_functions.items():
            observed_value = float(function(observed_scope))
            replicated_values = function(
                replicated_scope,
                axis=1,
            )
            low, high = az.hdi(
                replicated_values,
                hdi_prob=0.94,
            )
            rows.append(
                {
                    "run_id": run_id,
                    "scope": scope,
                    "statistic": statistic,
                    "observed": observed_value,
                    "replicated_mean": float(
                        np.mean(replicated_values)
                    ),
                    "replicated_hdi_low": float(low),
                    "replicated_hdi_high": float(high),
                    "observed_inside_94pct_hdi": bool(
                        low <= observed_value <= high
                    ),
                    "n_events_checked": int(
                        observed_scope.size
                    ),
                    "n_posterior_draws_checked": n_draws,
                }
            )

    summarize_scope(
        "all_cities",
        observed,
        replicated,
    )

    for city_index, city_name in enumerate(city_names):
        mask = selected_city == city_index
        if not np.any(mask):
            continue
        summarize_scope(
            city_name,
            observed[mask],
            replicated[:, mask],
        )

    ppc_df = pd.DataFrame(rows)
    ppc_path = os.path.join(
        CHECK_DIR,
        f"ppc_{run_id}.csv",
    )
    ppc_df.to_csv(ppc_path, index=False)

    observed_sorted = np.sort(observed)
    observed_ecdf = (
        np.arange(1, observed_sorted.size + 1)
        / observed_sorted.size
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(
        observed_sorted,
        observed_ecdf,
        linewidth=2,
        label="Observed",
    )

    for draw in replicated[:min(50, replicated.shape[0])]:
        draw_sorted = np.sort(draw)
        draw_ecdf = (
            np.arange(1, draw_sorted.size + 1)
            / draw_sorted.size
        )
        ax.plot(
            draw_sorted,
            draw_ecdf,
            alpha=0.12,
            linewidth=0.8,
        )

    ax.set_xlabel("Threshold exceedance magnitude")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_title(f"Posterior predictive check: {run_id}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(CHECK_DIR, f"ppc_{run_id}.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"✅ wrote posterior-predictive checks: {ppc_path}")


def basin_nc_path(basin):
    return os.path.join(SST_DIR, f"era5_sst_anom_{basin}_1950_2025.nc")

def idata_path(target_var, basin):
    return os.path.join(OUT_DIR, f"idata_city_hier_{target_var}_vs_sst_{basin}_JJAS.nc")

def meta_path(target_var, basin):
    return os.path.join(OUT_DIR, f"meta_city_hier_{target_var}_vs_sst_{basin}_JJAS.pkl")

def aggregate_csv_path(target_var):
    return os.path.join(
        OUT_DIR,
        f"{target_var}_city_response_to_adjacent_basin_warming_ALLBASINS_JJAS.csv"
    )

def stack_samples(da):
    return da.stack(sample=("chain", "draw")).values

def gpd_quantile(u, sigma, xi, q, xi_tol=1e-6):
    xi = np.asarray(xi)
    sigma = np.asarray(sigma)
    out = np.empty(np.broadcast(xi, sigma).shape, dtype=float)
    near0 = np.abs(xi) < xi_tol
    out[~near0] = u + (sigma[~near0] / xi[~near0]) * ((1.0 - q) ** (-xi[~near0]) - 1.0)
    out[near0] = u + sigma[near0] * np.log(1.0 / (1.0 - q))
    return out

def load_basin_sst_monthly(basin):
    nc = basin_nc_path(basin)
    dsS = xr.open_dataset(nc)
    daS = dsS["sst_anom"]

    latn = "latitude" if "latitude" in daS.coords else "lat"
    lonn = "longitude" if "longitude" in daS.coords else "lon"

    if float(daS[lonn].max()) > 180:
        lon = daS[lonn]
        lon_new = ((lon + 180) % 360) - 180
        daS = daS.assign_coords({lonn: lon_new}).sortby(lonn)

    t0 = pd.DatetimeIndex(pd.to_datetime(daS["time"].values))
    is_ms = (t0.day == 1).all() and (t0.hour == 0).all()
    if not is_ms:
        daS = daS.resample(time="MS").mean(skipna=True)

    sst_m = daS.mean(dim=[latn, lonn], skipna=True).to_series()
    sst_m.index = pd.to_datetime(sst_m.index)

    dsS.close()

    if MONTHS is not None:
        sst_m = sst_m[sst_m.index.month.isin(MONTHS)]

    return sst_m

def load_city_daily(files, city, lat0, lon0, target_var):
    ys, ts = [], []
    ij = None
    var_name = None
    ref_lat = None
    ref_lon = None

    for fp in files:
        ds = xr.open_dataset(fp)

        # Fix DailyPeakState time coordinate
        if "day" in ds.coords and "time" not in ds.coords:
            ds = ds.rename({"day": "time"})
        lat_name, lon_name = get_latlon_names(ds)
        ds = shift_lon_180(ds, lon_name)

        if var_name is None:
            var_name = pick_var(ds, target_var)

        lat_vals = ds[lat_name].values
        lon_vals = ds[lon_name].values

        if ref_lat is None:
            ref_lat = lat_vals.copy()
            ref_lon = lon_vals.copy()
            ij = nearest_ij(lat_vals, lon_vals, lat0, lon0)
        else:
            if not (np.array_equal(lat_vals, ref_lat) and np.array_equal(lon_vals, ref_lon)):
                raise ValueError(f"{city}: grid changed in file {fp}")

        i, j = ij
        da = ds[var_name].isel({lat_name: i, lon_name: j})
        t = pd.to_datetime(da["time"].values)
        y = da.values.astype("float32")
        ds.close()

        ys.append(y)
        ts.append(t)

    y_all = np.concatenate(ys)
    t_all = pd.DatetimeIndex(np.concatenate(ts))
    o = np.argsort(t_all.values)
    t_all = t_all[o]
    y_all = y_all[o]

    valid = np.isfinite(y_all)
    t_all = t_all[valid]
    y_all = y_all[valid]

    if MONTHS is not None:
        m = t_all.month.isin(MONTHS)
        t_all = t_all[m]
        y_all = y_all[m]

    return t_all, y_all


def load_idata_any(target_var, basin):
    nc_path = idata_path(target_var, basin)
    pkl_path = nc_path.replace(".nc", ".pkl")

    # First try NetCDF
    if os.path.exists(nc_path):
        try:
            return az.from_netcdf(nc_path, engine="netcdf4"), nc_path
        except Exception as e:
            print(f"⚠️ failed to open NetCDF for basin='{basin}': {e}")

    # Then try pickle fallback
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                idata = pickle.load(f)
            return idata, pkl_path
        except Exception as e:
            print(f"⚠️ failed to open pickle for basin='{basin}': {e}")

    raise FileNotFoundError(
        f"No readable idata found for target_var='{target_var}', basin='{basin}'.\n"
        f"Tried:\n  {nc_path}\n  {pkl_path}"
    )


# -----------------------
# Stage 1: fit one (target_var, basin)
# -----------------------
def fit_one(target_var, basin):
    if target_var not in VALID_TARGET_VARS:
        raise ValueError(f"target_var must be one of {VALID_TARGET_VARS}")
    if basin not in VALID_BASINS:
        raise ValueError(f"basin must be one of {VALID_BASINS}")

    # Only include cities assigned to this basin
    basin_cities = [
        city for city in CITY_LIST
        if CITY_TO_BASIN[city] == basin
    ]

    # Only Doha, Dubai, and Dammam partially pool
    pooled_cities = [
        city for city in basin_cities
        if city in POOLED_GULF_CITIES
    ]

    # Kuwait City and Basra remain independent
    independent_cities = [
        city for city in basin_cities
        if city not in pooled_cities
    ]

    print("=" * 72)
    print("Running fit_one")
    print(f"TARGET_VAR       = {target_var}")
    print(f"BASIN            = {basin}")
    print(f"MONTHS           = {MONTHS}")
    print(f"BASIN CITIES     = {basin_cities}")
    print(f"POOLED CITIES    = {pooled_cities}")
    print(f"INDEPENDENT      = {independent_cities}")
    print("=" * 72)

    files = sorted(glob.glob(WBT_GLOB))
    if not files:
        raise FileNotFoundError(
            f"No DailyPeakState files found: {WBT_GLOB}"
        )

    sst_m = load_basin_sst_monthly(basin)

    z_list = []
    sst_list = []
    city_id_list = []

    u_by_city = {}
    n_days_by_city = {}
    n_exc_by_city = {}

    for ci, city in enumerate(basin_cities):
        t, y = load_city_daily(
            files,
            city,
            CITIES[city]["lat"],
            CITIES[city]["lon"],
            target_var,
        )

        n_days_by_city[city] = int(len(y))

        u = float(np.nanquantile(y, Q))
        exc = y > u
        z = (y[exc] - u).astype("float32")

        if z.size < MIN_EVENTS:
            raise RuntimeError(
                f"{city}: too few exceedances "
                f"({z.size} < {MIN_EVENTS}). "
                "Lower Q or MIN_EVENTS."
            )

        mk = month_key_daily(t)
        sst_day = sst_m.reindex(mk).values.astype("float32")

        if np.isnan(sst_day).any():
            bad = np.isnan(sst_day)
            raise ValueError(
                f"{city}: missing SST for some JJAS days. "
                f"First missing date={t[bad][0]}"
            )

        sst_e = sst_day[exc]

        z_list.append(z)
        sst_list.append(sst_e)
        city_id_list.append(
            np.full(z.size, ci, dtype="int32")
        )

        u_by_city[city] = u
        n_exc_by_city[city] = int(z.size)

        print(
            f"✅ {city}: days={len(y)} "
            f"exc={z.size} "
            f"frac={z.size / len(y):.3f} "
            f"u={u:.3f}"
        )

    z_all = np.concatenate(z_list).astype("float32")
    sst_all = np.concatenate(sst_list).astype("float32")
    cid_all = np.concatenate(city_id_list).astype("int32")

    print(
        f"\n✅ Built event table: "
        f"E={z_all.size} exceedances across "
        f"S={len(basin_cities)} cities"
    )

    print(
        f"✅ SST covariate range: "
        f"[{np.nanmin(sst_all):.3f}, "
        f"{np.nanmax(sst_all):.3f}]"
    )

    coords = {
        "event": np.arange(z_all.size),
        "city": basin_cities,
    }

    if pooled_cities:
        coords["pooled_city"] = pooled_cities

    if independent_cities:
        coords["independent_city"] = independent_cities

    with pm.Model(coords=coords) as model:
        z_obs = pm.ConstantData("z", z_all, dims="event")
        sst_e = pm.ConstantData("sst", sst_all, dims="event")
        c_id = pm.ConstantData("c_id", cid_all, dims="event")

        xi = pm.TruncatedNormal("xi", mu=0.05, sigma=0.15, lower=XI_LOWER, upper=XI_UPPER)

        city_to_index = {
            city: i for i, city in enumerate(basin_cities)
        }

        pooled_to_index = {
            city: i for i, city in enumerate(pooled_cities)
        }

        independent_to_index = {
            city: i for i, city in enumerate(independent_cities)
        }

        a_values = [None] * len(basin_cities)
        b_values = [None] * len(basin_cities)

        # -------------------------------------------
        # Partially pooled: Doha, Dubai, Dammam
        # -------------------------------------------
        if pooled_cities:
            a_bar = pm.Normal(
                "a_bar",
                0.0,
                1.0,
            )
            b_bar = pm.Normal(
                "b_bar",
                0.0,
                0.5,
            )

            a_sd = pm.HalfNormal(
                "a_sd",
                0.8,
            )
            b_sd = pm.HalfNormal(
                "b_sd",
                0.3,
            )

            a_z = pm.Normal(
                "a_z",
                0.0,
                1.0,
                dims="pooled_city",
            )
            b_z = pm.Normal(
                "b_z",
                0.0,
                1.0,
                dims="pooled_city",
            )

            a_pooled = pm.Deterministic(
                "a_pooled",
                a_bar + a_sd * a_z,
                dims="pooled_city",
            )
            b_pooled = pm.Deterministic(
                "b_pooled",
                b_bar + b_sd * b_z,
                dims="pooled_city",
            )

            for city in pooled_cities:
                city_i = city_to_index[city]
                pool_i = pooled_to_index[city]

                a_values[city_i] = a_pooled[pool_i]
                b_values[city_i] = b_pooled[pool_i]

        # -------------------------------------------
        # Independent: Kuwait City, Basra,
        # plus all single-city basin fits
        # -------------------------------------------
        if independent_cities:
            a_independent = pm.Normal(
                "a_independent",
                0.0,
                1.0,
                dims="independent_city",
            )
            b_independent = pm.Normal(
                "b_independent",
                0.0,
                0.5,
                dims="independent_city",
            )

            for city in independent_cities:
                city_i = city_to_index[city]
                independent_i = independent_to_index[city]

                a_values[city_i] = a_independent[independent_i]
                b_values[city_i] = b_independent[independent_i]

        a_city = pm.Deterministic(
            "a_city",
            pt.stack(a_values),
            dims="city",
        )

        b_city = pm.Deterministic(
            "b_city",
            pt.stack(b_values),
            dims="city",
        )

        log_sigma = a_city[c_id] + b_city[c_id] * sst_e
        sigma = pm.Deterministic("sigma", 1e-6 + pt.exp(log_sigma), dims="event")

        pm.DensityDist(
            "z_like",
            sigma, xi,
            logp=lambda value, sigma, xi: gpd_logp(value, sigma, xi),
            observed=z_all,
            dims="event",
        )

        idata = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            cores=CORES,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
        )

    safe_save_idata(idata, idata_path(target_var, basin))

    run_id = (
        f"city_hier_{target_var}_vs_sst_{basin}_JJAS"
    )
    write_convergence_checks(idata, run_id)
    write_posterior_predictive_checks(
        idata=idata,
        z_all=z_all,
        sst_all=sst_all,
        city_ids=cid_all,
        city_names=basin_cities,
        run_id=run_id,
    )

    meta = {
        "cities": basin_cities,
        "cities_meta": {
            city: CITIES[city]
            for city in basin_cities
        },
        "pooled_cities": pooled_cities,
        "independent_cities": independent_cities,
        "u_by_city": u_by_city,
        "n_days_by_city": n_days_by_city,
        "n_exc_by_city": n_exc_by_city,
        "Q": Q,
        "months": MONTHS,
        "basin": basin,
        "target_var": target_var,
    }
    pd.Series(meta).to_pickle(meta_path(target_var, basin))
    print(f"✅ saved meta: {meta_path(target_var, basin)}")

# -----------------------
# Stage 2: aggregate one variable across all basins
# -----------------------
def aggregate_one_var(target_var):
    if target_var not in VALID_TARGET_VARS:
        raise ValueError(f"target_var must be one of {VALID_TARGET_VARS}")

    print("=" * 72)
    print(f"Running aggregate_one_var")
    print(f"TARGET_VAR = {target_var}")
    print("=" * 72)

    basin_to_cities = defaultdict(list)
    for city, basin in CITY_TO_BASIN.items():
        basin_to_cities[basin].append(city)

    all_rows = []

    for basin, target_cities in basin_to_cities.items():
        ipath = idata_path(target_var, basin)
        pkl_path = ipath.replace(".nc", ".pkl")
        mpath = meta_path(target_var, basin)

        if (not os.path.exists(ipath) and not os.path.exists(pkl_path)) or not os.path.exists(mpath):
            print(f"⚠️ missing run for basin='{basin}', target_var='{target_var}'.")
            print(f"   expected nc : {ipath}")
            print(f"   expected pkl: {pkl_path}")
            print(f"   expected meta: {mpath}")
            continue

        try:
            idata, used_path = load_idata_any(target_var, basin)
            print(f"✅ loaded idata from: {used_path}")
        except Exception as e:
            print(f"⚠️ could not load idata for basin='{basin}': {e}")
            continue

        meta = pd.read_pickle(mpath)

        meta_target_var = meta.get("target_var", None)
        if meta_target_var is not None and meta_target_var != target_var:
            print(f"⚠️ target_var mismatch for basin='{basin}': meta has '{meta_target_var}', expected '{target_var}'")
            continue

        cities = list(meta["cities"])
        cities_keep = [c for c in cities if c in target_cities]

        if len(cities_keep) == 0:
            print(f"⚠️ basin='{basin}' run has cities={cities} but none match target_cities={target_cities}")
            continue

        u_by_city = meta["u_by_city"]
        n_days_by_city = meta["n_days_by_city"]
        n_exc_by_city = meta["n_exc_by_city"]

        post = idata.posterior
        xi = stack_samples(post["xi"])
        a_city = post["a_city"].stack(sample=("chain", "draw"))
        b_city = post["b_city"].stack(sample=("chain", "draw"))

        city_to_i = {c: i for i, c in enumerate(cities)}

        for city in cities_keep:
            i = city_to_i[city]
            u = float(u_by_city[city])

            a = a_city.isel(city=i).values
            b = b_city.isel(city=i).values
            sigma0 = np.exp(a)

            for label, dS in WARMING_EXPTS.items():
                sigma1 = sigma0 * np.exp(b * dS)

                for q in Q_LEVELS:
                    x0 = gpd_quantile(u, sigma0, xi, q)
                    x1 = gpd_quantile(u, sigma1, xi, q)
                    dx = x1 - x0

                    mean = float(dx.mean())
                    lo, hi = az.hdi(dx, hdi_prob=0.94)

                    all_rows.append({
                        "target_var": target_var,
                        "basin_warmed": basin,
                        "city": city,
                        "warming": label,
                        "dS_C": dS,
                        "quantile": q,
                        "delta_mean": mean,
                        "delta_hdi_low": float(lo),
                        "delta_hdi_high": float(hi),
                        "n_days": int(n_days_by_city[city]),
                        "n_exc": int(n_exc_by_city[city]),
                        "u": u,
                        "months": "".join(str(m) for m in meta["months"]),
                    })

    impact_all = pd.DataFrame(all_rows)
    out_csv = aggregate_csv_path(target_var)
    impact_all.to_csv(out_csv, index=False)
    print(f"✅ wrote merged adjacent-basin impacts: {out_csv}")

# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p_fit = sub.add_parser("fit_one")
    p_fit.add_argument("--target_var", required=True, choices=VALID_TARGET_VARS)
    p_fit.add_argument("--basin", required=True, choices=VALID_BASINS)

    p_agg = sub.add_parser("aggregate_one_var")
    p_agg.add_argument("--target_var", required=True, choices=VALID_TARGET_VARS)

    args = parser.parse_args()

    if args.mode == "fit_one":
        fit_one(args.target_var, args.basin)
    elif args.mode == "aggregate_one_var":
        aggregate_one_var(args.target_var)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()