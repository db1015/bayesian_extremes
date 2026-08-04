#!/usr/bin/env python3
# coding: utf-8
'''
============================================================
MODEL 2 OF 6 — SUPPLEMENTAL ANALYSIS SUPPORTING SECTION 2.2
Daily POT/GPD exceedance magnitude as a function of RONI and DMI
============================================================

SCIENTIFIC PURPOSE
------------------
Estimate how lagged ENSO and Indian Ocean Dipole conditions alter the
magnitude distribution of daily maximum wet-bulb temperature exceedances,
conditional on an exceedance having occurred. This complements the
Section 2.2 Bernoulli occurrence model and supports the supplemental figure.

MODELING CHOICES
----------------
1. POT response: z_t = wbt_daily_peak_t - u_c for days above a
   city-specific JJAS threshold u_c. The default threshold is the p95 of
   valid, lag-aligned daily values and is treated as fixed.
2. Conditional interpretation: this model addresses exceedance magnitude,
   not exceedance occurrence. The Bernoulli model separately addresses
   occurrence probability.
3. Spatial sampling: nearest model grid cell to each city coordinate.
4. Season: June-September by default; configurable with --months.
5. Remote covariates: monthly RONI lagged 2 months and DMI lagged 1 month.
   Both are standardized after lagging. Each exceedance event inherits the
   corresponding monthly values.
6. GPD parameterization:
      z_t ~ GPD(sigma_t, xi)
      log(sigma_t) = a + bN*N_t + bD*D_t + bND*(N_t*D_t)
   Covariates affect scale only; shape xi is constant within each fit.
7. Shape prior and support: xi ~ TruncatedNormal(0.05, 0.15),
   constrained to [-0.3, 0.5]. The likelihood explicitly enforces
   1 + xi*z/sigma > 0.
8. Scale priors: intercept Normal(0,1); standardized-covariate slopes
   Normal(0,0.5). The log link guarantees positive scale.
9. Partial pooling: Doha, Dubai, and Dammam are modeled jointly with
   non-centered hierarchical city effects and one shared xi. Other cities
   are fit independently. This is a scientific pooling choice based on their
   shared Persian Gulf coastal setting.
10. Declustering: OFF by default, matching the manuscript analysis. With
    --decluster, a runs rule retains the first exceedance separated by more
    than --run-length-days. No residual temporal dependence is otherwise
    modeled, and monthly covariates repeat across events in the same month.
11. Minimum sample: fits require at least --min-events exceedances.
12. Posterior intervals: 94% highest-density intervals.
13. Scenario interpretation: post-processing changes the fitted GPD scale
    under fixed standardized RONI/DMI combinations, holds xi fixed within
    each posterior draw, and reports changes in conditional GPD p95 and p99.

OUTPUTS — EXISTING LOCATIONS RETAINED
-------------------------------------
Existing:
  idata_<run_id>.nc
  <var>_daily_gpd_city_roni_dmi_summary.csv
Added under OUT_DIR/posterior_checks/:
  posterior_diagnostics_summary.csv
  posterior_predictive_summary.csv
  diagnostics_<run_id>.csv
  ppc_<run_id>.csv
  trace_<run_id>.png
  ppc_<run_id>.png

Posterior checks include R-hat, bulk/tail ESS, divergences, tree-depth hits,
BFMI, and inverse-CDF GPD posterior-predictive checks of exceedance mean,
median, p95, and maximum. Diagnostic flags identify runs requiring review;
they do not replace scientific inspection of trace and predictive plots.
============================================================
'''
import os
import glob
import argparse
import numpy as np
import pandas as pd
import xarray as xr

import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--netid", type=str, default="k16v981")
    p.add_argument("--var", type=str, required=True,
                   help="Variable key, e.g. wbt, hi, tau")
    p.add_argument("--data-dir", type=str, default=None,
                   help="Optional override for variable data directory")
    p.add_argument("--glob", type=str, default=None,
                   help="Optional override for input glob")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Optional override for output directory")
    p.add_argument("--months", type=str, default="6,7,8,9",
                   help="Comma-separated months, or 'all'")
    p.add_argument("--q", type=float, default=0.95)
    p.add_argument("--min-events", type=int, default=30)
    p.add_argument("--decluster", action="store_true")
    p.add_argument("--run-length-days", type=int, default=3)
    p.add_argument("--draws", type=int, default=1500)
    p.add_argument("--tune", type=int, default=1500)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--cores", type=int, default=4)
    p.add_argument("--target-accept", type=float, default=0.995)
    p.add_argument("--seed", type=int, default=72)
    p.add_argument("--max-rhat", type=float, default=1.01)
    p.add_argument("--min-ess", type=float, default=400)
    p.add_argument("--min-bfmi", type=float, default=0.30)
    p.add_argument("--ppc-draws", type=int, default=500,
                   help="Posterior draws used for inverse-CDF predictive checks")
    return p.parse_args()

args = parse_args()

# -----------------------
# Config
# -----------------------
NETID = args.netid
VAR = args.var.lower()

BASE_DIR = f"/home/{NETID}/my_work/code/arabian_peninsula/bayesian_extremes/data"

DAILY_STATE_DIR = os.path.join(BASE_DIR, "DailyPeakState")
DATA_DIR = args.data_dir or DAILY_STATE_DIR
DATA_GLOB = args.glob or os.path.join(DATA_DIR, "DailyPeakState-*.nc")

VALID_TARGET_VARS = ["wbt_daily_peak"]

if VAR not in VALID_TARGET_VARS:
    raise ValueError(f"Unknown --var={VAR}. Supported: {VALID_TARGET_VARS}")

IDX_CSV = os.path.join(BASE_DIR, "sst", "roni_dmi_monthly_1950_2025.csv")

OUT_DIR = args.out_dir or os.path.join(BASE_DIR, f"{VAR}_daily_city_runs")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, f"{VAR}_daily_gpd_city_roni_dmi_summary.csv")

CHECK_DIR = os.path.join(OUT_DIR, "posterior_checks")
os.makedirs(CHECK_DIR, exist_ok=True)
DIAGNOSTICS_CSV = os.path.join(CHECK_DIR, "posterior_diagnostics_summary.csv")
PPC_CSV = os.path.join(CHECK_DIR, "posterior_predictive_summary.csv")

MAX_RHAT = args.max_rhat
MIN_ESS = args.min_ess
MIN_BFMI = args.min_bfmi
PPC_DRAWS = args.ppc_draws

RANDOM_SEED = args.seed

# POT
Q = args.q
MIN_EVENTS = args.min_events
DECLUSTER = args.decluster
RUN_LENGTH_DAYS = args.run_length_days

# GPD
XI_LOWER = -0.3
XI_UPPER = 0.5

# Season restriction
if args.months.strip().lower() == "all":
    MONTHS = None
else:
    MONTHS = [int(x) for x in args.months.split(",")]

# Sampling
DRAWS = args.draws
TUNE = args.tune
CHAINS = args.chains
CORES = args.cores
TARGET_ACCEPT = args.target_accept

# -----------------------
# City definitions
# -----------------------
CITIES = {
    "muscat":       {"lat": 23.5880, "lon": 58.3829},
    "doha":         {"lat": 25.2854, "lon": 51.5310},
    "dubai":        {"lat": 25.2048, "lon": 55.2708},
    "jeddah":       {"lat": 21.4858, "lon": 39.1925},
    "aden":         {"lat": 12.7855, "lon": 45.0187},

    # NEW
    "medina":       {"lat": 24.5247, "lon": 39.5692},
    "riyadh":       {"lat": 24.7136, "lon": 46.6753},
    "dammam":       {"lat": 26.4207, "lon": 50.0888},
    "kuwait_city":  {"lat": 29.3759, "lon": 47.9774},
    "basra":        {"lat": 30.5085, "lon": 47.7835},
}

POOLED_GROUPS = {
    "gulf_coastal_pooled": ["doha", "dubai", "dammam"]
}

SCENARIOS = [
    ("ElNino(+1,0)",    1.0,  0.0),
    ("LaNina(-1,0)",   -1.0,  0.0),
    ("pIOD(0,+1)",      0.0,  1.0),
    ("nIOD(0,-1)",      0.0, -1.0),
    ("Joint(+1,+1)",    1.0,  1.0),
    ("Opposing(+1,-1)", 1.0, -1.0),
    ("Joint(-1,-1)",   -1.0, -1.0),
]

# -----------------------
# Helpers
# -----------------------
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

def month_key_daily(t_daily):
    return pd.to_datetime(t_daily).to_period("M").to_timestamp()

def decluster_events(exceed_mask, run_len_days=3):
    idx = np.where(exceed_mask)[0]
    if idx.size == 0:
        return idx

    keep = [idx[0]]
    last = idx[0]
    for k in idx[1:]:
        if (k - last) > run_len_days:
            keep.append(k)
        last = k
    return np.array(keep, dtype=int)

def gpd_logp(z, sigma, xi, eps=1e-12, xi_tol=1e-6):
    sigma = sigma + eps
    t = 1 + xi * z / sigma
    logp_gpd = -pt.log(sigma) - (1 + 1 / xi) * pt.log(t)
    logp_exp = -pt.log(sigma) - z / sigma
    logp = pt.switch(pt.abs(xi) < xi_tol, logp_exp, logp_gpd)
    logp = pt.switch(t > 0, logp, -np.inf)
    return pt.sum(logp)

def summarize_param(post, name, hdi=0.94):
    arr = post[name].values
    mean = float(arr.mean())
    lo, hi = az.hdi(arr, hdi_prob=hdi)
    return mean, float(lo), float(hi)


# -----------------------
# Standardized posterior checks
# -----------------------
def core_parameter_names(idata):
    candidates = [
        "xi", "a", "bN", "bD", "bND",
        "a_bar", "bN_bar", "bD_bar", "bND_bar",
        "a_sd", "bN_sd", "bD_sd", "bND_sd",
        "a_s", "bN_s", "bD_s", "bND_s",
    ]
    return [name for name in candidates if name in idata.posterior]


def scalar_diagnostic_summary(idata, run_id):
    var_names = core_parameter_names(idata)
    summary = az.summary(
        idata,
        var_names=var_names,
        kind="diagnostics",
        round_to=None,
    )

    max_rhat = float(np.nanmax(summary["r_hat"].to_numpy()))
    min_ess_bulk = float(np.nanmin(summary["ess_bulk"].to_numpy()))
    min_ess_tail = float(np.nanmin(summary["ess_tail"].to_numpy()))

    stats = idata.sample_stats
    divergences = int(stats["diverging"].sum().values)

    if "reached_max_treedepth" in stats:
        treedepth_hits = int(stats["reached_max_treedepth"].sum().values)
    else:
        treedepth_hits = 0

    min_bfmi = float(np.nanmin(np.asarray(az.bfmi(idata), dtype=float)))

    flags = []
    if max_rhat > MAX_RHAT:
        flags.append(f"rhat>{MAX_RHAT}")
    if min_ess_bulk < MIN_ESS:
        flags.append(f"bulk_ess<{MIN_ESS:g}")
    if min_ess_tail < MIN_ESS:
        flags.append(f"tail_ess<{MIN_ESS:g}")
    if divergences > 0:
        flags.append("divergences")
    if treedepth_hits > 0:
        flags.append("max_treedepth")
    if min_bfmi < MIN_BFMI:
        flags.append(f"bfmi<{MIN_BFMI}")

    summary.to_csv(os.path.join(CHECK_DIR, f"diagnostics_{run_id}.csv"))
    return {
        "run_id": run_id,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "min_ess_tail": min_ess_tail,
        "divergences": divergences,
        "max_treedepth_hits": treedepth_hits,
        "min_bfmi": min_bfmi,
        "diagnostic_status": "PASS" if not flags else "REVIEW",
        "diagnostic_flags": ";".join(flags),
    }


def gpd_random_from_posterior(sigma, xi, rng):
    """Generate GPD variates by inverse CDF for arrays [draw,event]."""
    sigma = np.asarray(sigma, dtype=float)
    xi = np.asarray(xi, dtype=float)
    u = rng.uniform(np.finfo(float).eps, 1.0 - np.finfo(float).eps, size=sigma.shape)
    xi_event = np.broadcast_to(xi[:, None], sigma.shape)

    near_zero = np.abs(xi_event) < 1e-6
    out = np.empty_like(sigma)
    out[near_zero] = -sigma[near_zero] * np.log1p(-u[near_zero])
    out[~near_zero] = (
        sigma[~near_zero] / xi_event[~near_zero]
        * ((1.0 - u[~near_zero]) ** (-xi_event[~near_zero]) - 1.0)
    )
    return out


def predictive_stat_rows(run_id, group, observed, replicated):
    stats = {
        "mean": lambda x: np.mean(x, axis=-1),
        "median": lambda x: np.median(x, axis=-1),
        "p95": lambda x: np.quantile(x, 0.95, axis=-1),
        "maximum": lambda x: np.max(x, axis=-1),
    }
    rows = []
    for stat_name, func in stats.items():
        obs_value = float(func(observed[None, :])[0])
        rep_values = np.asarray(func(replicated), dtype=float)
        lo, hi = az.hdi(rep_values, hdi_prob=0.94)
        rows.append({
            "run_id": run_id,
            "group": group,
            "statistic": stat_name,
            "n_events": int(observed.size),
            "observed": obs_value,
            "replicated_mean": float(rep_values.mean()),
            "replicated_hdi_low": float(lo),
            "replicated_hdi_high": float(hi),
            "observed_in_94pct_hdi": bool(float(lo) <= obs_value <= float(hi)),
            "posterior_predictive_p_ge_observed": float(np.mean(rep_values >= obs_value)),
        })
    return rows


def posterior_predictive_rows(idata, run_id):
    observed = np.asarray(idata.observed_data["z_like"].values, dtype=float)
    sigma = (
        idata.posterior["sigma"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "event")
        .values
    )
    xi = idata.posterior["xi"].stack(sample=("chain", "draw")).values

    total_draws = sigma.shape[0]
    if PPC_DRAWS < total_draws:
        select_rng = np.random.default_rng(RANDOM_SEED)
        selected = np.sort(select_rng.choice(total_draws, size=PPC_DRAWS, replace=False))
        sigma = sigma[selected, :]
        xi = xi[selected]

    rng = np.random.default_rng(RANDOM_SEED + 1009)
    replicated = gpd_random_from_posterior(sigma, xi, rng)

    groups = [("ALL", np.arange(observed.size))]
    if "s_id" in idata.constant_data:
        s_id = np.asarray(idata.constant_data["s_id"].values, dtype=int)
        pooled_cities = next(iter(POOLED_GROUPS.values()))
        groups.extend(
            (city, np.where(s_id == s)[0])
            for s, city in enumerate(pooled_cities)
        )

    rows = []
    for group, idx in groups:
        rows.extend(
            predictive_stat_rows(
                run_id,
                group,
                observed[idx],
                replicated[:, idx],
            )
        )
    return rows, observed, replicated


def save_trace_plot(idata, run_id):
    axes = az.plot_trace(
        idata,
        var_names=core_parameter_names(idata),
        compact=True,
    )
    fig = np.asarray(axes).ravel()[0].figure
    fig.suptitle(run_id, fontsize=11)
    fig.tight_layout()
    fig.savefig(
        os.path.join(CHECK_DIR, f"trace_{run_id}.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_ppc_plot(observed, replicated, run_id):
    """ECDF comparison avoids relying on a random method for DensityDist."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    obs_sorted = np.sort(observed)
    obs_y = np.arange(1, obs_sorted.size + 1) / obs_sorted.size
    ax.step(obs_sorted, obs_y, where="post", linewidth=2.0, label="Observed")

    n_show = min(100, replicated.shape[0])
    show_idx = np.linspace(0, replicated.shape[0] - 1, n_show, dtype=int)
    for i in show_idx:
        rep_sorted = np.sort(replicated[i])
        rep_y = np.arange(1, rep_sorted.size + 1) / rep_sorted.size
        ax.step(rep_sorted, rep_y, where="post", linewidth=0.5, alpha=0.10)

    ax.set_xlabel("Exceedance magnitude")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_title(f"GPD posterior predictive ECDF: {run_id}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        os.path.join(CHECK_DIR, f"ppc_{run_id}.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def run_all_posterior_checks(summary_df):
    diagnostics_rows = []
    ppc_rows = []

    base_run_ids = sorted({
        str(run_id).split(":", 1)[0]
        for run_id in summary_df["run_id"]
    })

    for run_id in base_run_ids:
        nc_path = os.path.join(OUT_DIR, f"idata_{run_id}.nc")
        if not os.path.exists(nc_path):
            print(f"WARNING: posterior checks skipped; missing {nc_path}")
            continue

        print(f"Running posterior checks: {run_id}")
        idata = az.from_netcdf(nc_path)
        diagnostics_rows.append(scalar_diagnostic_summary(idata, run_id))

        run_rows, observed, replicated = posterior_predictive_rows(idata, run_id)
        ppc_rows.extend(run_rows)
        pd.DataFrame(run_rows).to_csv(
            os.path.join(CHECK_DIR, f"ppc_{run_id}.csv"),
            index=False,
        )

        save_trace_plot(idata, run_id)
        save_ppc_plot(observed, replicated, run_id)

    diagnostics_df = pd.DataFrame(diagnostics_rows)
    ppc_df = pd.DataFrame(ppc_rows)
    diagnostics_df.to_csv(DIAGNOSTICS_CSV, index=False)
    ppc_df.to_csv(PPC_CSV, index=False)

    print("Wrote posterior diagnostics:", DIAGNOSTICS_CSV)
    print("Wrote posterior predictive summary:", PPC_CSV)
    if not diagnostics_df.empty:
        print("\nPosterior diagnostic status:")
        print(
            diagnostics_df[
                ["run_id", "diagnostic_status", "diagnostic_flags"]
            ].to_string(index=False)
        )


# -----------------------
# Load monthly indices
# -----------------------
idx = pd.read_csv(IDX_CSV)

if "time" in idx.columns:
    idx["time"] = pd.to_datetime(idx["time"])
elif {"year", "month"}.issubset(idx.columns):
    idx["time"] = pd.to_datetime(dict(year=idx["year"], month=idx["month"], day=1))
else:
    raise ValueError(f"Index CSV needs time or year/month columns. Found: {idx.columns.tolist()}")

def pick_col(cols, key):
    cols_l = {c.lower(): c for c in cols}
    for cl, orig in cols_l.items():
        if cl == key or key in cl:
            return orig
    return None

roni_col = pick_col(idx.columns, "roni")
dmi_col = pick_col(idx.columns, "dmi")
if roni_col is None or dmi_col is None:
    raise ValueError(f"Could not find RONI/DMI columns in {idx.columns.tolist()}")

idx = idx.set_index("time").sort_index()
idx = idx[~idx.index.duplicated(keep="last")]

# Assign earlier index months to the later humid-heat month:
# July receives May RONI and June DMI.
N_m = idx[roni_col].astype("float32").shift(2)
D_m = idx[dmi_col].astype("float32").shift(1)

# Standardize after applying the lags.
N_m = (N_m - N_m.mean()) / N_m.std()
D_m = (D_m - D_m.mean()) / D_m.std()
ND_m = (N_m * D_m).astype("float32")

# -----------------------
# Load daily files
# -----------------------
files = sorted(glob.glob(DATA_GLOB))
if not files:
    raise FileNotFoundError(f"No files found: {DATA_GLOB}")

print(f"Found {len(files)} files for {VAR}:")
print(files[:3], "..." if len(files) > 3 else "")

def load_city_series(city, lat0, lon0):
    ys = []
    ts = []
    var_name = None
    ij = None

    for fp in files:
        ds = xr.open_dataset(fp)

        if "day" in ds.coords and "time" not in ds.coords:
            ds = ds.rename({"day": "time"})
        lat_name, lon_name = get_latlon_names(ds)
        ds = shift_lon_180(ds, lon_name)

        if var_name is None:
            var_name = pick_var(ds, VAR)
            print(f"[{city}] using variable '{var_name}'")

        if ij is None:
            lat_vals = ds[lat_name].values
            lon_vals = ds[lon_name].values
            ij = nearest_ij(lat_vals, lon_vals, lat0, lon0)

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

    finite = np.isfinite(y_all)
    t_all = t_all[finite]
    y_all = y_all[finite]

    if MONTHS is not None:
        m = t_all.month.isin(MONTHS)
        t_all = t_all[m]
        y_all = y_all[m]

    return t_all, y_all

# -----------------------
# Build event table
# -----------------------
def build_event_table(t_daily, y_daily):
    mk = month_key_daily(t_daily)

    Nm = N_m.reindex(mk).values.astype("float32")
    Dm = D_m.reindex(mk).values.astype("float32")
    NDm_ = ND_m.reindex(mk).values.astype("float32")

    # Remove days without complete lagged index data.
    valid = np.isfinite(Nm) & np.isfinite(Dm) & np.isfinite(NDm_)

    t_daily = t_daily[valid]
    y_daily = y_daily[valid]
    Nm = Nm[valid]
    Dm = Dm[valid]
    NDm_ = NDm_[valid]

    if len(y_daily) == 0:
        raise RuntimeError(
            "No daily observations remain after lag alignment."
        )

    # Threshold is calculated from all valid daily observations.
    u = float(np.nanquantile(y_daily, Q))
    exc = y_daily > u

    if DECLUSTER:
        exc_idx = decluster_events(
            exc,
            run_len_days=RUN_LENGTH_DAYS,
        )
        exc = np.zeros_like(exc, dtype=bool)
        exc[exc_idx] = True

    # GPD response consists only of positive exceedance magnitudes.
    z = (y_daily[exc] - u).astype("float32")

    if z.size < MIN_EVENTS:
        raise RuntimeError(
            f"Too few exceedances: {z.size} "
            f"(<{MIN_EVENTS}). Lower Q or MIN_EVENTS."
        )

    return {
        "z": z,
        "N": Nm[exc],
        "D": Dm[exc],
        "ND": NDm_[exc],
        "u": u,
        "n_days": int(len(y_daily)),
        "n_exc": int(z.size),
    }

# -----------------------
# Models
# -----------------------
def fit_single_city(run_id, city_name, tbl, out_dir):
    z = tbl["z"]
    N = tbl["N"]
    D = tbl["D"]
    ND = tbl["ND"]
    E = z.size

    coords = {"event": np.arange(E)}

    with pm.Model(coords=coords) as model:
        z_obs = pm.ConstantData("z", z, dims="event")
        N_t = pm.ConstantData("N_t", N, dims="event")
        D_t = pm.ConstantData("D_t", D, dims="event")
        ND_t = pm.ConstantData("ND_t", ND, dims="event")

        xi = pm.TruncatedNormal("xi", mu=0.05, sigma=0.15, lower=XI_LOWER, upper=XI_UPPER)

        a = pm.Normal("a", 0.0, 1.0)
        bN = pm.Normal("bN", 0.0, 0.5)
        bD = pm.Normal("bD", 0.0, 0.5)
        bND = pm.Normal("bND", 0.0, 0.5)

        log_sigma = a + bN * N_t + bD * D_t + bND * ND_t
        sigma = pm.Deterministic("sigma", 1e-6 + pt.exp(log_sigma), dims="event")

        pm.DensityDist(
            "z_like",
            sigma, xi,
            logp=lambda z, sigma, xi: gpd_logp(z, sigma, xi),
            observed=z_obs,
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

    out_nc = os.path.join(out_dir, f"idata_{run_id}.nc")
    az.to_netcdf(idata, out_nc)
    print("✅ saved", out_nc)

    post = idata.posterior.stack(sample=("chain", "draw"))

    row = {
        "run_id": run_id,
        "var": VAR,
        "city": city_name,
        "pooled_group": "",
        "Q": Q,
        "months": "ALL" if MONTHS is None else "".join([str(m) for m in MONTHS]),
        "decluster": int(DECLUSTER),
        "run_length_days": RUN_LENGTH_DAYS if DECLUSTER else 0,
        "n_days": tbl["n_days"],
        "n_exc": tbl["n_exc"],
        "u": tbl["u"],
    }

    for name in ["xi", "a", "bN", "bD", "bND"]:
        m, lo, hi = summarize_param(post, name)
        row[f"{name}_mean"] = m
        row[f"{name}_hdi_low"] = lo
        row[f"{name}_hdi_high"] = hi

    for lab, Nv, Dv in SCENARIOS:
        dlog = (post["bN"] * Nv + post["bD"] * Dv + post["bND"] * (Nv * Dv)).values
        row[f"dlogsig_{lab}_mean"] = float(dlog.mean())
        lo, hi = az.hdi(dlog, hdi_prob=0.94)
        row[f"dlogsig_{lab}_hdi_low"] = float(lo)
        row[f"dlogsig_{lab}_hdi_high"] = float(hi)
        row[f"sigfactor_{lab}_mean"] = float(np.exp(dlog).mean())

    return row

def fit_pooled_pair(run_id, cities, tables, out_dir):
    z_list, N_list, D_list, ND_list, s_list = [], [], [], [], []
    n_days_list, n_exc_list = [], []

    for s, cname in enumerate(cities):
        tbl = tables[cname]
        E = tbl["z"].size
        z_list.append(tbl["z"])
        N_list.append(tbl["N"])
        D_list.append(tbl["D"])
        ND_list.append(tbl["ND"])
        s_list.append(np.full(E, s, dtype="int32"))
        n_days_list.append(tbl["n_days"])
        n_exc_list.append(tbl["n_exc"])

    z = np.concatenate(z_list).astype("float32")
    N = np.concatenate(N_list).astype("float32")
    D = np.concatenate(D_list).astype("float32")
    ND = np.concatenate(ND_list).astype("float32")
    s_id = np.concatenate(s_list).astype("int32")

    E_all = z.size
    S = len(cities)

    coords = {"event": np.arange(E_all), "space": np.arange(S)}

    with pm.Model(coords=coords) as model:
        z_obs = pm.ConstantData("z", z, dims="event")
        N_t = pm.ConstantData("N_t", N, dims="event")
        D_t = pm.ConstantData("D_t", D, dims="event")
        ND_t = pm.ConstantData("ND_t", ND, dims="event")
        s_idx = pm.ConstantData("s_id", s_id, dims="event")

        xi = pm.TruncatedNormal("xi", mu=0.05, sigma=0.15, lower=XI_LOWER, upper=XI_UPPER)

        a_bar = pm.Normal("a_bar", 0.0, 1.0)
        bN_bar = pm.Normal("bN_bar", 0.0, 0.5)
        bD_bar = pm.Normal("bD_bar", 0.0, 0.5)
        bND_bar = pm.Normal("bND_bar", 0.0, 0.5)

        a_sd = pm.HalfNormal("a_sd", 0.8)
        bN_sd = pm.HalfNormal("bN_sd", 0.3)
        bD_sd = pm.HalfNormal("bD_sd", 0.3)
        bND_sd = pm.HalfNormal("bND_sd", 0.2)

        a_z = pm.Normal("a_z", 0, 1, dims="space")
        bN_z = pm.Normal("bN_z", 0, 1, dims="space")
        bD_z = pm.Normal("bD_z", 0, 1, dims="space")
        bND_z = pm.Normal("bND_z", 0, 1, dims="space")

        a_s = pm.Deterministic("a_s", a_bar + a_sd * a_z, dims="space")
        bN_s = pm.Deterministic("bN_s", bN_bar + bN_sd * bN_z, dims="space")
        bD_s = pm.Deterministic("bD_s", bD_bar + bD_sd * bD_z, dims="space")
        bND_s = pm.Deterministic("bND_s", bND_bar + bND_sd * bND_z, dims="space")

        log_sigma = a_s[s_idx] + bN_s[s_idx] * N_t + bD_s[s_idx] * D_t + bND_s[s_idx] * ND_t
        sigma = pm.Deterministic("sigma", 1e-6 + pt.exp(log_sigma), dims="event")

        pm.DensityDist(
            "z_like",
            sigma, xi,
            logp=lambda z, sigma, xi: gpd_logp(z, sigma, xi),
            observed=z_obs,
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

    out_nc = os.path.join(out_dir, f"idata_{run_id}.nc")
    az.to_netcdf(idata, out_nc)
    print("✅ saved", out_nc)

    post = idata.posterior.stack(sample=("chain", "draw"))

    row = {
        "run_id": run_id,
        "var": VAR,
        "city": "ALL",
        "pooled_group": ",".join(cities),
        "Q": Q,
        "months": "ALL" if MONTHS is None else "".join([str(m) for m in MONTHS]),
        "decluster": int(DECLUSTER),
        "run_length_days": RUN_LENGTH_DAYS if DECLUSTER else 0,
        "n_days": int(np.sum(n_days_list)),
        "n_exc": int(np.sum(n_exc_list)),
        "u": np.nan,
    }

    for name in ["xi", "a_bar", "bN_bar", "bD_bar", "bND_bar", "a_sd", "bN_sd", "bD_sd", "bND_sd"]:
        m, lo, hi = summarize_param(post, name)
        row[f"{name}_mean"] = m
        row[f"{name}_hdi_low"] = lo
        row[f"{name}_hdi_high"] = hi

    for lab, Nv, Dv in SCENARIOS:
        dlog = (post["bN_bar"] * Nv + post["bD_bar"] * Dv + post["bND_bar"] * (Nv * Dv)).values
        row[f"dlogsig_{lab}_mean"] = float(dlog.mean())
        lo, hi = az.hdi(dlog, hdi_prob=0.94)
        row[f"dlogsig_{lab}_hdi_low"] = float(lo)
        row[f"dlogsig_{lab}_hdi_high"] = float(hi)
        row[f"sigfactor_{lab}_mean"] = float(np.exp(dlog).mean())

    rows_city = []
    for s, cname in enumerate(cities):
        r = {
            "run_id": f"{run_id}:{cname}",
            "var": VAR,
            "city": cname,
            "pooled_group": run_id,
            "Q": Q,
            "months": row["months"],
            "decluster": row["decluster"],
            "run_length_days": row["run_length_days"],
            "n_days": tables[cname]["n_days"],
            "n_exc": tables[cname]["n_exc"],
            "u": tables[cname]["u"],
        }

        for name in ["a_s", "bN_s", "bD_s", "bND_s"]:
            arr = post[name].isel(space=s).values
            r[f"{name}_mean"] = float(arr.mean())
            lo, hi = az.hdi(arr, hdi_prob=0.94)
            r[f"{name}_hdi_low"] = float(lo)
            r[f"{name}_hdi_high"] = float(hi)

        for lab, Nv, Dv in SCENARIOS:
            dlog = (
                post["bN_s"].isel(space=s) * Nv
                + post["bD_s"].isel(space=s) * Dv
                + post["bND_s"].isel(space=s) * (Nv * Dv)
            ).values
            r[f"dlogsig_{lab}_mean"] = float(dlog.mean())
            lo, hi = az.hdi(dlog, hdi_prob=0.94)
            r[f"dlogsig_{lab}_hdi_low"] = float(lo)
            r[f"dlogsig_{lab}_hdi_high"] = float(hi)
            r[f"sigfactor_{lab}_mean"] = float(np.exp(dlog).mean())

        rows_city.append(r)

    return [row] + rows_city

# -----------------------
# Main
# -----------------------
def main():
    print(f"\n=== Running VAR={VAR} ===")
    print(f"Input glob: {DATA_GLOB}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Output CSV: {OUT_CSV}\n")

    all_rows = []

    # 1) pooled Doha + Dubai first
    for run_id, members in POOLED_GROUPS.items():
        tables = {}
        for cname in members:
            t, y = load_city_series(cname, CITIES[cname]["lat"], CITIES[cname]["lon"])
            tables[cname] = build_event_table(t, y)
            print(f"✅ built events {cname}: n_exc={tables[cname]['n_exc']}  u={tables[cname]['u']:.3f}")

        pooled_run_id = f"{VAR}_{run_id}_roni_dmi"
        rows = fit_pooled_pair(pooled_run_id, members, tables, OUT_DIR)
        all_rows.extend(rows)

    # 2) single cities excluding pooled members
    pooled_members = set(sum(POOLED_GROUPS.values(), []))
    for cname, meta in CITIES.items():
        if cname in pooled_members:
            continue

        t, y = load_city_series(cname, meta["lat"], meta["lon"])
        tbl = build_event_table(t, y)
        print(f"✅ built events {cname}: n_exc={tbl['n_exc']}  u={tbl['u']:.3f}")

        run_id = f"{VAR}_{cname}_roni_dmi"
        row = fit_single_city(run_id, cname, tbl, OUT_DIR)
        all_rows.append(row)

    # 3) write combined CSV
    df = pd.DataFrame(all_rows)

    front = [
        "run_id", "var", "city", "pooled_group", "Q", "months",
        "decluster", "run_length_days", "n_days", "n_exc", "u"
    ]
    cols = front + [c for c in df.columns if c not in front]
    df = df[cols]

    df.to_csv(OUT_CSV, index=False)
    print("✅ wrote summary CSV:", OUT_CSV)

    # 4) standardized convergence and posterior-predictive checks
    run_all_posterior_checks(df)

if __name__ == "__main__":
    main() 