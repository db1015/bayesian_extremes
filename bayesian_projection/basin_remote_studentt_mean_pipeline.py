#!/usr/bin/env python3
"""
MODEL 3 OF 6 — BASIN-MEAN SST RESPONSE TO ENSO AND IOD
======================================================

Manuscript role
---------------
This model supports the basin-scale SST analysis used to interpret the remote
ENSO/IOD influence discussed alongside Section 2.2. It estimates how monthly
JJAS sea-surface-temperature anomalies within one basin shift under lagged
RONI and DMI states.

Scientific response
-------------------
The response is monthly ERA5 basin SST anomaly at every valid wet grid cell.
The model is fit in long form, with one observation for every valid
month-by-grid-cell combination.

Modeling choices
----------------
1. Season:
   Only June–September (JJAS) months are retained.

2. Predictor lags:
   RONI is lagged by two months and DMI by one month. Thus, for example, July
   SST is paired with May RONI and June DMI.

3. Predictor scaling:
   Lagged RONI and DMI are standardized to mean zero and unit standard
   deviation after temporal alignment.

4. ENSO asymmetry:
   Standardized RONI is split into positive and negative components:
       N_pos = max(N, 0)
       N_neg = min(N, 0)
   This allows El Niño-like and La Niña-like states to have different slopes.

5. IOD interactions:
   DMI is retained as a linear predictor and interacts separately with the
   positive and negative ENSO components.

6. Hierarchical spatial structure:
   Every wet basin grid cell has its own intercept and slopes. These
   cell-specific parameters are partially pooled through basin-wide means and
   hierarchical standard deviations using non-centered parameterizations.

7. Likelihood:
   A Student-t likelihood is used to make the mean-shift model robust to
   unusually large monthly SST anomalies. The residual scale and degrees of
   freedom are shared across basin grid cells.

8. Response scaling:
   By default, SST anomalies are modeled in their original units
   (STANDARDIZE_Y=False). Prior scales are therefore based on the empirical
   standard deviation of the basin response. Setting --standardize-y changes
   the fitted parameter units but not the model structure.

9. Dependence not modeled:
   Monthly observations and neighboring grid cells are treated as
   conditionally independent given the hierarchical mean model. No explicit
   temporal autocorrelation or spatial covariance process is included.

10. Basin naming:
    The internal key ``arabian_gulf`` is retained because it is embedded in
    existing data and posterior filenames. Manuscript figures label this basin
    as ``Persian Gulf``.

Posterior checks
----------------
Each fit writes:
  * parameter-level R-hat, bulk ESS, and tail ESS;
  * divergence, tree-depth, and BFMI summaries;
  * trace plots for basin-level parameters;
  * a posterior-predictive check on a reproducible subset of observations;
  * observed-versus-replicated summaries for mean, standard deviation,
    median, p05, p95, and maximum.

The predictive check is generated after sampling from the fitted Student-t
distribution. A subset is used to prevent the full long-form basin field from
creating an unnecessarily large posterior-predictive object.

Existing data and posterior paths are retained.
"""

from pathlib import Path
import argparse
import pickle
import traceback

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit the hierarchical Student-t basin SST model."
    )
    parser.add_argument(
        "--netid",
        default="k16v981",
        help="Tempest NetID used to construct existing absolute data paths.",
    )
    parser.add_argument(
        "--basin",
        default="gulf_aden",
        choices=["arabian_gulf", "red_sea", "gulf_oman", "gulf_aden"],
    )
    parser.add_argument("--enso-lag", type=int, default=2)
    parser.add_argument("--iod-lag", type=int, default=1)
    parser.add_argument("--standardize-y", action="store_true")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument(
        "--ppc-observations",
        type=int,
        default=2000,
        help="Maximum number of observations used in posterior-predictive checks.",
    )
    parser.add_argument(
        "--ppc-draws",
        type=int,
        default=500,
        help="Maximum number of posterior samples used for predictive checks.",
    )
    return parser.parse_args()


ARGS = parse_args()

NETID = ARGS.netid
BASIN = ARGS.basin
ENSO_LAG = ARGS.enso_lag
IOD_LAG = ARGS.iod_lag
STANDARDIZE_Y = ARGS.standardize_y
RANDOM_SEED = ARGS.seed

BASE_DIR = Path(
    f"/home/{NETID}/my_work/code/arabian_peninsula/bayesian_extremes/data"
)
BASIN_NC = (
    BASE_DIR
    / "sst"
    / "basin_anoms"
    / f"era5_sst_anom_{BASIN}_1950_2025.nc"
)
IDX_CSV = BASE_DIR / "sst" / "roni_dmi_monthly_1950_2025.csv"
OUT_IDATA = BASE_DIR / "sst" / f"studentt_mean_{BASIN}_roni_dmi_idata.nc"

CHECK_DIR = BASE_DIR / "sst" / "posterior_checks" / "studentt_basin_mean"
CHECK_DIR.mkdir(parents=True, exist_ok=True)

R_HAT_LIMIT = 1.01
ESS_LIMIT = 400
BFMI_LIMIT = 0.30


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def pick_col(columns, key):
    lower = {column.lower(): column for column in columns}
    for candidate, original in lower.items():
        if candidate == key or key in candidate:
            return original
    return None


def safe_save_idata(idata, out_path):
    out_path = Path(out_path)
    try:
        az.to_netcdf(idata, out_path)
        print(f"✅ ArviZ NetCDF saved: {out_path}")
        return out_path
    except Exception as exc:
        print("⚠️ az.to_netcdf failed.")
        print(exc)
        traceback.print_exc()

    pkl_path = out_path.with_suffix(".pkl")
    try:
        with pkl_path.open("wb") as file_obj:
            pickle.dump(idata, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✅ Pickle saved: {pkl_path}")
        return pkl_path
    except Exception as exc:
        print("❌ Pickle save failed.")
        print(exc)
        raise


def diagnostic_variables(idata):
    excluded = {
        "a_z",
        "bNp_z",
        "bNn_z",
        "bD_z",
        "bNpD_z",
        "bNnD_z",
        "a_s",
        "bNp_s",
        "bNn_s",
        "bD_s",
        "bNpD_s",
        "bNnD_s",
    }
    return [
        name
        for name in idata.posterior.data_vars
        if name not in excluded
    ]


def write_convergence_checks(idata, run_name):
    var_names = diagnostic_variables(idata)
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
        tree_depth = sample_stats["tree_depth"].values
        maximum = np.nanmax(tree_depth)
        tree_depth_hits = int(np.sum(tree_depth >= maximum))

    bfmi_values = np.asarray(az.bfmi(idata), dtype=float)
    minimum_bfmi = float(np.nanmin(bfmi_values))

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

    out_csv = CHECK_DIR / f"diagnostics_{run_name}.csv"
    summary.to_csv(out_csv, index=False)

    aggregate = pd.DataFrame(
        [
            {
                "run_id": run_name,
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
        CHECK_DIR / f"diagnostics_summary_{run_name}.csv",
        index=False,
    )

    axes = az.plot_trace(idata, var_names=var_names, compact=True)
    figure = np.asarray(axes).ravel()[0].figure
    figure.tight_layout()
    figure.savefig(
        CHECK_DIR / f"trace_{run_name}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"✅ wrote diagnostics: {out_csv}")
    return aggregate


def _stack_sample_space(da):
    return da.stack(sample=("chain", "draw")).transpose("sample", "space").values


def _stack_samples(da):
    return da.stack(sample=("chain", "draw")).values


def posterior_predictive_check(
    idata,
    y_obs,
    s_obs,
    npos_obs,
    nneg_obs,
    d_obs,
    nposd_obs,
    nnegd_obs,
    run_name,
):
    rng = np.random.default_rng(RANDOM_SEED)

    n_obs_total = len(y_obs)
    n_obs_check = min(ARGS.ppc_observations, n_obs_total)
    obs_idx = np.sort(
        rng.choice(n_obs_total, size=n_obs_check, replace=False)
    )

    a_s = _stack_sample_space(idata.posterior["a_s"])
    bnp_s = _stack_sample_space(idata.posterior["bNp_s"])
    bnn_s = _stack_sample_space(idata.posterior["bNn_s"])
    bd_s = _stack_sample_space(idata.posterior["bD_s"])
    bnpd_s = _stack_sample_space(idata.posterior["bNpD_s"])
    bnnd_s = _stack_sample_space(idata.posterior["bNnD_s"])
    sigma = _stack_samples(idata.posterior["sigma"])
    nu = _stack_samples(idata.posterior["nu"])

    n_samples_total = a_s.shape[0]
    n_samples_check = min(ARGS.ppc_draws, n_samples_total)
    sample_idx = np.sort(
        rng.choice(n_samples_total, size=n_samples_check, replace=False)
    )

    s_sub = s_obs[obs_idx]
    mu = (
        a_s[sample_idx][:, s_sub]
        + bnp_s[sample_idx][:, s_sub] * npos_obs[obs_idx][None, :]
        + bnn_s[sample_idx][:, s_sub] * nneg_obs[obs_idx][None, :]
        + bd_s[sample_idx][:, s_sub] * d_obs[obs_idx][None, :]
        + bnpd_s[sample_idx][:, s_sub] * nposd_obs[obs_idx][None, :]
        + bnnd_s[sample_idx][:, s_sub] * nnegd_obs[obs_idx][None, :]
    )

    replicated = (
        mu
        + sigma[sample_idx, None]
        * rng.standard_t(nu[sample_idx, None], size=mu.shape)
    )
    observed = y_obs[obs_idx]

    statistic_functions = {
        "mean": np.mean,
        "sd": np.std,
        "median": np.median,
        "p05": lambda values, axis=None: np.quantile(values, 0.05, axis=axis),
        "p95": lambda values, axis=None: np.quantile(values, 0.95, axis=axis),
        "maximum": np.max,
    }

    rows = []
    for statistic, function in statistic_functions.items():
        observed_value = float(function(observed))
        replicated_values = function(replicated, axis=1)
        lo, hi = az.hdi(replicated_values, hdi_prob=0.94)
        rows.append(
            {
                "run_id": run_name,
                "statistic": statistic,
                "observed": observed_value,
                "replicated_mean": float(np.mean(replicated_values)),
                "replicated_hdi_low": float(lo),
                "replicated_hdi_high": float(hi),
                "observed_inside_94pct_hdi": bool(lo <= observed_value <= hi),
                "n_observations_checked": n_obs_check,
                "n_posterior_draws_checked": n_samples_check,
            }
        )

    ppc_df = pd.DataFrame(rows)
    ppc_df.to_csv(CHECK_DIR / f"ppc_{run_name}.csv", index=False)

    obs_sorted = np.sort(observed)
    obs_ecdf = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(obs_sorted, obs_ecdf, linewidth=2, label="Observed")

    plot_draws = min(50, replicated.shape[0])
    for draw in replicated[:plot_draws]:
        draw_sorted = np.sort(draw)
        draw_ecdf = np.arange(1, len(draw_sorted) + 1) / len(draw_sorted)
        ax.plot(draw_sorted, draw_ecdf, alpha=0.12, linewidth=0.8)

    ax.set_xlabel("SST anomaly")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_title(f"Posterior predictive check: {run_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        CHECK_DIR / f"ppc_{run_name}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"✅ wrote posterior-predictive checks for {run_name}")
    return ppc_df


# ---------------------------------------------------------------------------
# Load basin SST anomalies
# ---------------------------------------------------------------------------
ds = xr.open_dataset(BASIN_NC)
da = ds["sst_anom"]

lat_name = "latitude" if "latitude" in da.coords else "lat"
lon_name = "longitude" if "longitude" in da.coords else "lon"

if float(da[lon_name].max()) > 180:
    longitude = da[lon_name]
    shifted = ((longitude + 180) % 360) - 180
    da = da.assign_coords({lon_name: shifted}).sortby(lon_name)

time_initial = pd.DatetimeIndex(pd.to_datetime(da["time"].values))
is_month_start_midnight = (
    (time_initial.day == 1).all()
    and (time_initial.hour == 0).all()
    and (time_initial.minute == 0).all()
)

if not is_month_start_midnight:
    print("⚠️ SST time is not monthly. Resampling to monthly month-start means.")
    da = da.resample(time="MS").mean(skipna=True)

jjas = pd.DatetimeIndex(pd.to_datetime(da["time"].values)).month.isin(
    [6, 7, 8, 9]
)
da = da.isel(time=jjas)

da_stacked = da.stack(space=(lat_name, lon_name))
valid_space = np.isfinite(da_stacked).any("time").values
da_stacked = da_stacked.isel(space=valid_space)

Y = da_stacked.values.astype("float32")
time = pd.to_datetime(da_stacked["time"].values)

T, S = Y.shape
print(f"✅ SST loaded: T={T}, S={S}")

ds.close()


# ---------------------------------------------------------------------------
# Load and align ENSO/IOD indices
# ---------------------------------------------------------------------------
indices = pd.read_csv(IDX_CSV)

if "time" in indices.columns:
    indices["time"] = pd.to_datetime(indices["time"])
elif {"year", "month"}.issubset(indices.columns):
    indices["time"] = pd.to_datetime(
        dict(year=indices["year"], month=indices["month"], day=1)
    )
else:
    raise ValueError(
        "Index CSV needs either 'time' or ('year','month') columns. "
        f"Found: {indices.columns.tolist()}"
    )

roni_col = pick_col(indices.columns, "roni")
dmi_col = pick_col(indices.columns, "dmi")
if roni_col is None or dmi_col is None:
    raise ValueError(
        f"Could not identify RONI/DMI columns. Columns: {indices.columns.tolist()}"
    )

indices = indices.set_index("time").sort_index()

if indices.index.duplicated().any():
    print("⚠️ Duplicate index times found; keeping the final occurrence.")
    indices = indices[~indices.index.duplicated(keep="last")]

if pd.Index(time).duplicated().any():
    raise ValueError("SST monthly time axis contains duplicate values.")

lagged = pd.DataFrame(
    {
        "N_lag": indices[roni_col].shift(ENSO_LAG),
        "D_lag": indices[dmi_col].shift(IOD_LAG),
    }
).sort_index()

aligned = lagged.reindex(time)
if aligned[["N_lag", "D_lag"]].isna().any().any():
    missing = aligned[
        aligned["N_lag"].isna() | aligned["D_lag"].isna()
    ]
    raise ValueError(
        "Missing lagged index values after alignment.\n"
        f"First missing rows:\n{missing.head()}\n"
        f"Index range: {indices.index.min()} to {indices.index.max()}\n"
        f"SST range: {time.min()} to {time.max()}\n"
        f"ENSO_LAG={ENSO_LAG}, IOD_LAG={IOD_LAG}"
    )

N = aligned["N_lag"].astype("float32").values
D = aligned["D_lag"].astype("float32").values

N = (N - N.mean()) / N.std()
D = (D - D.mean()) / D.std()

N_pos = np.maximum(N, 0.0).astype("float32")
N_neg = np.minimum(N, 0.0).astype("float32")
D = D.astype("float32")
NposD = (N_pos * D).astype("float32")
NnegD = (N_neg * D).astype("float32")

print(f"✅ Data aligned: T={T} months, S={S} wet points")
print(f"✅ RONI lag={ENSO_LAG}; DMI lag={IOD_LAG}")


# ---------------------------------------------------------------------------
# Response handling and long-form observations
# ---------------------------------------------------------------------------
if STANDARDIZE_Y:
    y_mean = np.nanmean(Y)
    y_sd = np.nanstd(Y)
    Y_model = (Y - y_mean) / y_sd
else:
    y_mean = 0.0
    y_sd = np.nanstd(Y)
    Y_model = Y.copy()

print(f"✅ y_sd used for prior scaling: {y_sd:.3f}")

mask = np.isfinite(Y_model)
t_idx, s_idx = np.where(mask)

y_obs = Y_model[t_idx, s_idx].astype("float32")
Npos_obs = N_pos[t_idx].astype("float32")
Nneg_obs = N_neg[t_idx].astype("float32")
D_obs = D[t_idx].astype("float32")
NposD_obs = NposD[t_idx].astype("float32")
NnegD_obs = NnegD[t_idx].astype("float32")
s_obs = s_idx.astype("int32")

E = len(y_obs)
print(f"✅ Long table built: n_obs={E}, S={S}")

coords = {"obs": np.arange(E), "space": np.arange(S)}


# ---------------------------------------------------------------------------
# Hierarchical Student-t mean-shift model
# ---------------------------------------------------------------------------
with pm.Model(coords=coords) as model:
    y_t = pm.ConstantData("y_t", y_obs, dims="obs")
    Npos_t = pm.ConstantData("Npos_t", Npos_obs, dims="obs")
    Nneg_t = pm.ConstantData("Nneg_t", Nneg_obs, dims="obs")
    D_t = pm.ConstantData("D_t", D_obs, dims="obs")
    NposD_t = pm.ConstantData("NposD_t", NposD_obs, dims="obs")
    NnegD_t = pm.ConstantData("NnegD_t", NnegD_obs, dims="obs")
    s_id = pm.ConstantData("s_id", s_obs, dims="obs")

    mean_scale = 1.0 if STANDARDIZE_Y else y_sd
    beta_scale = 0.5 if STANDARDIZE_Y else 0.5 * y_sd
    sigma_scale = 0.5 if STANDARDIZE_Y else 0.5 * y_sd

    a_bar = pm.Normal("a_bar", mu=0.0, sigma=mean_scale)
    a_sd = pm.HalfNormal("a_sd", sigma=0.5 * mean_scale)
    a_z = pm.Normal("a_z", mu=0.0, sigma=1.0, dims="space")
    a_s = pm.Deterministic("a_s", a_bar + a_sd * a_z, dims="space")

    bNp_bar = pm.Normal("bNp_bar", mu=0.0, sigma=beta_scale)
    bNn_bar = pm.Normal("bNn_bar", mu=0.0, sigma=beta_scale)
    bD_bar = pm.Normal("bD_bar", mu=0.0, sigma=beta_scale)
    bNpD_bar = pm.Normal("bNpD_bar", mu=0.0, sigma=beta_scale)
    bNnD_bar = pm.Normal("bNnD_bar", mu=0.0, sigma=beta_scale)

    bNp_sd = pm.HalfNormal("bNp_sd", sigma=0.5 * beta_scale)
    bNn_sd = pm.HalfNormal("bNn_sd", sigma=0.5 * beta_scale)
    bD_sd = pm.HalfNormal("bD_sd", sigma=0.5 * beta_scale)
    bNpD_sd = pm.HalfNormal("bNpD_sd", sigma=0.5 * beta_scale)
    bNnD_sd = pm.HalfNormal("bNnD_sd", sigma=0.5 * beta_scale)

    bNp_z = pm.Normal("bNp_z", mu=0.0, sigma=1.0, dims="space")
    bNn_z = pm.Normal("bNn_z", mu=0.0, sigma=1.0, dims="space")
    bD_z = pm.Normal("bD_z", mu=0.0, sigma=1.0, dims="space")
    bNpD_z = pm.Normal("bNpD_z", mu=0.0, sigma=1.0, dims="space")
    bNnD_z = pm.Normal("bNnD_z", mu=0.0, sigma=1.0, dims="space")

    bNp_s = pm.Deterministic(
        "bNp_s", bNp_bar + bNp_sd * bNp_z, dims="space"
    )
    bNn_s = pm.Deterministic(
        "bNn_s", bNn_bar + bNn_sd * bNn_z, dims="space"
    )
    bD_s = pm.Deterministic(
        "bD_s", bD_bar + bD_sd * bD_z, dims="space"
    )
    bNpD_s = pm.Deterministic(
        "bNpD_s", bNpD_bar + bNpD_sd * bNpD_z, dims="space"
    )
    bNnD_s = pm.Deterministic(
        "bNnD_s", bNnD_bar + bNnD_sd * bNnD_z, dims="space"
    )

    mu = (
        a_s[s_id]
        + bNp_s[s_id] * Npos_t
        + bNn_s[s_id] * Nneg_t
        + bD_s[s_id] * D_t
        + bNpD_s[s_id] * NposD_t
        + bNnD_s[s_id] * NnegD_t
    )

    sigma = pm.HalfNormal("sigma", sigma=sigma_scale)
    nu_minus_two = pm.Exponential("nu_minus_two", lam=1 / 10)
    nu = pm.Deterministic("nu", 2.0 + nu_minus_two)

    pm.StudentT(
        "y_like",
        nu=nu,
        mu=mu,
        sigma=sigma,
        observed=y_t,
        dims="obs",
    )

    idata = pm.sample(
        draws=ARGS.draws,
        tune=ARGS.tune,
        chains=ARGS.chains,
        cores=ARGS.cores,
        target_accept=ARGS.target_accept,
        random_seed=RANDOM_SEED,
    )


# ---------------------------------------------------------------------------
# Save and check
# ---------------------------------------------------------------------------
safe_save_idata(idata, OUT_IDATA)

run_name = f"studentt_mean_{BASIN}_roni_dmi"
write_convergence_checks(idata, run_name)
posterior_predictive_check(
    idata=idata,
    y_obs=y_obs,
    s_obs=s_obs,
    npos_obs=Npos_obs,
    nneg_obs=Nneg_obs,
    d_obs=D_obs,
    nposd_obs=NposD_obs,
    nnegd_obs=NnegD_obs,
    run_name=run_name,
)

print("\n=== Completed basin Student-t model ===")
print(f"Basin key: {BASIN}")
print(f"Posterior: {OUT_IDATA}")
print(f"Checks: {CHECK_DIR}")
