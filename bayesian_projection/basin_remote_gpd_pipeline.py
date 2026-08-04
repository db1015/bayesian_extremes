#!/usr/bin/env python3
"""
MODEL 4 OF 6 — SPATIAL BASIN SST EXTREMES VS ENSO AND IOD
=========================================================

Manuscript role
---------------
This model supports the Supplementary Information analysis associated with
Section 2.4. It estimates how lagged ENSO and IOD states alter the magnitude
of extreme monthly JJAS sea-surface-temperature anomalies at individual grid
cells within a selected basin.

Response and POT construction
-----------------------------
For every retained wet grid cell s, a cell-specific threshold u_s is defined
as the empirical 0.95 quantile of monthly JJAS SST anomalies. Positive
exceedance magnitudes are

    z_{t,s} = Y_{t,s} - u_s,  for Y_{t,s} > u_s.

Cells with fewer than MIN_EVENTS exceedances are removed before fitting.

Modeling choices
----------------
1. Season:
   Only June–September (JJAS) months are retained before threshold estimation.

2. Predictor lags:
   RONI is lagged by two months and DMI by one month.

3. Predictor scaling:
   Lagged RONI and DMI are standardized after temporal alignment. Their product
   is included as an interaction.

4. Extreme-value model:
   Exceedance magnitudes follow a generalized Pareto distribution. ENSO, IOD,
   and their interaction affect the logarithm of the GPD scale parameter.

5. Spatial hierarchy:
   Each retained basin grid cell receives its own intercept and slopes. These
   parameters are partially pooled through basin-wide means and hierarchical
   standard deviations using non-centered parameterizations.

6. Shape parameter:
   One GPD shape parameter xi is shared across all retained grid cells and is
   constrained to [-0.3, 0.5].

7. Dependence:
   Exceedances are not declustered. The likelihood treats events as
   conditionally independent and does not include explicit temporal
   autocorrelation or spatial covariance. The hierarchy pools parameters but
   is not a spatial Gaussian process.

8. Spatial coordinates:
   Kilometer coordinates are calculated for bookkeeping and possible future
   spatial priors, but they do not enter the current likelihood.

9. Basin naming:
   The internal key ``arabian_gulf`` is retained because existing data and
   posterior filenames use it. Manuscript-facing outputs label it
   ``Persian Gulf``.

Posterior checks
----------------
Each fit writes:
  * parameter-level R-hat, bulk ESS, and tail ESS;
  * divergence, tree-depth, and BFMI summaries;
  * trace plots for basin-level parameters;
  * inverse-CDF GPD posterior-predictive checks;
  * pooled and cell-level observed-versus-replicated summaries.

Existing input and posterior output paths are retained.
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
import pytensor.tensor as pt
import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit spatial hierarchical POT/GPD basin SST model."
    )
    parser.add_argument("--netid", default="k16v981")
    parser.add_argument(
        "--basin",
        default="gulf_oman",
        choices=["arabian_gulf", "red_sea", "gulf_oman", "gulf_aden"],
    )
    parser.add_argument("--enso-lag", type=int, default=2)
    parser.add_argument("--iod-lag", type=int, default=1)
    parser.add_argument("--q", type=float, default=0.95)
    parser.add_argument("--min-events", type=int, default=5)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.97)
    parser.add_argument("--max-treedepth", type=int, default=15)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--ppc-events", type=int, default=2000)
    parser.add_argument("--ppc-draws", type=int, default=500)
    return parser.parse_args()


ARGS = parse_args()

NETID = ARGS.netid
BASIN = ARGS.basin
ENSO_LAG = ARGS.enso_lag
IOD_LAG = ARGS.iod_lag
Q = ARGS.q
MIN_EVENTS = ARGS.min_events
RANDOM_SEED = ARGS.seed

XI_LOWER = -0.3
XI_UPPER = 0.5

BASE_DIR = Path(
    f"/home/{NETID}/my_work/code/arabian_peninsula/bayesian_extremes/data"
)
BASIN_NC = (
    BASE_DIR / "sst" / "basin_anoms"
    / f"era5_sst_anom_{BASIN}_1950_2025.nc"
)
IDX_CSV = BASE_DIR / "sst" / "roni_dmi_monthly_1950_2025.csv"
OUT_IDATA = BASE_DIR / "sst" / f"gpd_{BASIN}_roni_dmi_idata.nc"

CHECK_DIR = BASE_DIR / "sst" / "posterior_checks" / "spatial_basin_gpd"
CHECK_DIR.mkdir(parents=True, exist_ok=True)

R_HAT_LIMIT = 1.01
ESS_LIMIT = 400
BFMI_LIMIT = 0.30


def pick_col(columns, key):
    lower = {column.lower(): column for column in columns}
    for candidate, original in lower.items():
        if candidate == key or key in candidate:
            return original
    return None


def gpd_logp(z, sigma, xi, eps=1e-12, xi_tol=1e-6):
    sigma = sigma + eps
    support = 1 + xi * z / sigma
    logp_gpd = -pt.log(sigma) - (1 + 1 / xi) * pt.log(support)
    logp_exp = -pt.log(sigma) - z / sigma
    logp = pt.switch(pt.abs(xi) < xi_tol, logp_exp, logp_gpd)
    logp = pt.switch(support > 0, logp, -np.inf)
    return pt.sum(logp)


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

    try:
        posterior_path = out_path.with_name(
            f"{out_path.stem}_posterior.nc"
        )
        idata.posterior.to_netcdf(posterior_path)
        print(f"✅ Posterior-only fallback saved: {posterior_path}")
    except Exception as exc:
        print(f"⚠️ Posterior-only fallback failed: {exc}")

    pickle_path = out_path.with_suffix(".pkl")
    with pickle_path.open("wb") as file_obj:
        pickle.dump(idata, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ Pickle fallback saved: {pickle_path}")
    return pickle_path


def diagnostic_variables(idata):
    excluded = {
        "a_z", "bN_z", "bD_z", "bND_z",
        "a_s", "bN_s", "bD_s", "bND_s", "sigma",
    }
    return [
        name for name in idata.posterior.data_vars
        if name not in excluded
    ]


def write_convergence_checks(idata, run_name):
    variables = diagnostic_variables(idata)
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

    treedepth_hits = np.nan
    if "tree_depth" in stats:
        treedepth_hits = int(
            np.sum(stats["tree_depth"].values >= ARGS.max_treedepth)
        )

    minimum_bfmi = float(np.nanmin(np.asarray(az.bfmi(idata))))
    summary["divergences"] = divergences
    summary["treedepth_hits"] = treedepth_hits
    summary["minimum_bfmi"] = minimum_bfmi
    summary["divergence_flag"] = divergences > 0
    summary["treedepth_flag"] = (
        treedepth_hits > 0 if np.isfinite(treedepth_hits) else False
    )
    summary["bfmi_flag"] = minimum_bfmi < BFMI_LIMIT
    summary["overall_flag"] = (
        summary["rhat_flag"]
        | summary["ess_bulk_flag"]
        | summary["ess_tail_flag"]
        | summary["divergence_flag"]
        | summary["treedepth_flag"]
        | summary["bfmi_flag"]
    )

    summary_path = CHECK_DIR / f"diagnostics_{run_name}.csv"
    summary.to_csv(summary_path, index=False)

    aggregate = pd.DataFrame([{
        "run_id": run_name,
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "min_ess_tail": float(summary["ess_tail"].min()),
        "divergences": divergences,
        "treedepth_hits": treedepth_hits,
        "minimum_bfmi": minimum_bfmi,
        "passed_all_checks": not bool(summary["overall_flag"].any()),
    }])
    aggregate.to_csv(
        CHECK_DIR / f"diagnostics_summary_{run_name}.csv",
        index=False,
    )

    axes = az.plot_trace(idata, var_names=variables, compact=True)
    figure = np.asarray(axes).ravel()[0].figure
    figure.tight_layout()
    figure.savefig(
        CHECK_DIR / f"trace_{run_name}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"✅ wrote convergence checks: {summary_path}")


def _stack_sample_space(da):
    return (
        da.stack(sample=("chain", "draw"))
        .transpose("sample", "space")
        .values
    )


def _stack_samples(da):
    return da.stack(sample=("chain", "draw")).values


def sample_gpd(rng, sigma, xi, size):
    uniform = rng.uniform(
        np.finfo(float).eps,
        1.0 - np.finfo(float).eps,
        size=size,
    )
    near_zero = np.abs(xi) < 1e-6
    output = np.empty(size, dtype=float)
    output[near_zero] = -sigma[near_zero] * np.log(
        1.0 - uniform[near_zero]
    )
    output[~near_zero] = (
        sigma[~near_zero] / xi[~near_zero]
        * ((1.0 - uniform[~near_zero]) ** (-xi[~near_zero]) - 1.0)
    )
    return output


def posterior_predictive_checks(
    idata, z, n_event, d_event, nd_event, s_event, run_name
):
    rng = np.random.default_rng(RANDOM_SEED)

    n_events = min(ARGS.ppc_events, len(z))
    event_idx = np.sort(
        rng.choice(len(z), size=n_events, replace=False)
    )

    a_s = _stack_sample_space(idata.posterior["a_s"])
    b_n = _stack_sample_space(idata.posterior["bN_s"])
    b_d = _stack_sample_space(idata.posterior["bD_s"])
    b_nd = _stack_sample_space(idata.posterior["bND_s"])
    xi = _stack_samples(idata.posterior["xi"])

    n_available = a_s.shape[0]
    n_draws = min(ARGS.ppc_draws, n_available)
    draw_idx = np.sort(
        rng.choice(n_available, size=n_draws, replace=False)
    )

    spaces = s_event[event_idx]
    log_sigma = (
        a_s[draw_idx][:, spaces]
        + b_n[draw_idx][:, spaces] * n_event[event_idx][None, :]
        + b_d[draw_idx][:, spaces] * d_event[event_idx][None, :]
        + b_nd[draw_idx][:, spaces] * nd_event[event_idx][None, :]
    )
    sigma = np.exp(log_sigma) + 1e-6
    xi_matrix = np.broadcast_to(
        xi[draw_idx, None],
        sigma.shape,
    )
    replicated = sample_gpd(
        rng,
        sigma=sigma,
        xi=xi_matrix,
        size=sigma.shape,
    )
    observed = z[event_idx]

    functions = {
        "mean": np.mean,
        "sd": np.std,
        "median": np.median,
        "p95": lambda values, axis=None: np.quantile(
            values, 0.95, axis=axis
        ),
        "maximum": np.max,
    }

    rows = []
    for name, function in functions.items():
        observed_value = float(function(observed))
        replicated_values = function(replicated, axis=1)
        low, high = az.hdi(replicated_values, hdi_prob=0.94)
        rows.append({
            "run_id": run_name,
            "scope": "all_retained_cells",
            "statistic": name,
            "observed": observed_value,
            "replicated_mean": float(np.mean(replicated_values)),
            "replicated_hdi_low": float(low),
            "replicated_hdi_high": float(high),
            "observed_inside_94pct_hdi": bool(
                low <= observed_value <= high
            ),
            "n_events_checked": n_events,
            "n_posterior_draws_checked": n_draws,
        })

    ppc = pd.DataFrame(rows)
    ppc.to_csv(CHECK_DIR / f"ppc_{run_name}.csv", index=False)

    observed_sorted = np.sort(observed)
    observed_ecdf = (
        np.arange(1, len(observed_sorted) + 1) / len(observed_sorted)
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(
        observed_sorted,
        observed_ecdf,
        linewidth=2,
        label="Observed",
    )
    for draw in replicated[:min(50, len(replicated))]:
        sorted_draw = np.sort(draw)
        draw_ecdf = np.arange(1, len(draw) + 1) / len(draw)
        ax.plot(sorted_draw, draw_ecdf, alpha=0.12, linewidth=0.8)
    ax.set_xlabel("SST-anomaly exceedance magnitude")
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


# ---------------------------------------------------------------------------
# Load basin SST anomalies
# ---------------------------------------------------------------------------
dataset = xr.open_dataset(BASIN_NC)
sst = dataset["sst_anom"]

lat_name = "latitude" if "latitude" in sst.coords else "lat"
lon_name = "longitude" if "longitude" in sst.coords else "lon"

if float(sst[lon_name].max()) > 180:
    longitude = sst[lon_name]
    shifted = ((longitude + 180) % 360) - 180
    sst = sst.assign_coords({lon_name: shifted}).sortby(lon_name)

initial_time = pd.DatetimeIndex(pd.to_datetime(sst["time"].values))
monthly = (
    (initial_time.day == 1).all()
    and (initial_time.hour == 0).all()
    and (initial_time.minute == 0).all()
)
if not monthly:
    print("⚠️ SST time is not monthly; resampling to month-start means.")
    sst = sst.resample(time="MS").mean(skipna=True)

jjas = pd.DatetimeIndex(
    pd.to_datetime(sst["time"].values)
).month.isin([6, 7, 8, 9])
sst = sst.isel(time=jjas)

stacked = sst.stack(space=(lat_name, lon_name))
valid_space = np.isfinite(stacked.isel(time=0).values)
stacked = stacked.isel(space=valid_space)

Y = stacked.values.astype("float32")
time = pd.to_datetime(stacked["time"].values)
T, S = Y.shape

space_index = stacked["space"].to_index()
lats = np.array([item[0] for item in space_index], dtype="float32")
lons = np.array([item[1] for item in space_index], dtype="float32")

dataset.close()
print(f"✅ Restricted to JJAS: T={T}, S={S} wet cells")


# ---------------------------------------------------------------------------
# Load, lag, align, and standardize indices
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
        "Index CSV needs 'time' or both 'year' and 'month'."
    )

roni_col = pick_col(indices.columns, "roni")
dmi_col = pick_col(indices.columns, "dmi")
if roni_col is None or dmi_col is None:
    raise ValueError(
        f"Could not identify RONI/DMI columns: {indices.columns.tolist()}"
    )

indices = indices.set_index("time").sort_index()
indices = indices[~indices.index.duplicated(keep="last")]

lagged = pd.DataFrame({
    "N": indices[roni_col].shift(ENSO_LAG),
    "D": indices[dmi_col].shift(IOD_LAG),
})
aligned = lagged.reindex(time)

if aligned.isna().any().any():
    raise ValueError(
        "Missing lagged index values after alignment:\n"
        f"{aligned[aligned.isna().any(axis=1)].head()}"
    )

N = aligned["N"].astype("float32").values
D = aligned["D"].astype("float32").values
N = (N - N.mean()) / N.std()
D = (D - D.mean()) / D.std()
ND = (N * D).astype("float32")

print(f"✅ RONI lag={ENSO_LAG}; DMI lag={IOD_LAG}")


# ---------------------------------------------------------------------------
# POT extraction
# ---------------------------------------------------------------------------
threshold = np.nanquantile(Y, Q, axis=0).astype("float32")
exceedance_mask = Y > threshold[None, :]
time_idx, original_space_idx = np.where(exceedance_mask)

z = (
    Y[time_idx, original_space_idx]
    - threshold[original_space_idx]
).astype("float32")
N_event = N[time_idx].astype("float32")
D_event = D[time_idx].astype("float32")
ND_event = ND[time_idx].astype("float32")

counts = np.bincount(original_space_idx, minlength=S)
keep_space = counts >= MIN_EVENTS
if keep_space.sum() < 5:
    raise RuntimeError(
        f"Only {keep_space.sum()} cells have >= {MIN_EVENTS} exceedances."
    )

keep_event = keep_space[original_space_idx]
z = z[keep_event]
N_event = N_event[keep_event]
D_event = D_event[keep_event]
ND_event = ND_event[keep_event]
old_space_event = original_space_idx[keep_event].astype("int32")

old_to_new = -np.ones(S, dtype="int32")
old_to_new[np.where(keep_space)[0]] = np.arange(
    keep_space.sum(), dtype="int32"
)
space_event = old_to_new[old_space_event]

E = len(z)
S_kept = int(keep_space.sum())
print(
    f"✅ POT built: Q={Q}, retained cells={S_kept}, "
    f"exceedances={E}, average={E / S_kept:.1f} per cell"
)

coords = {"event": np.arange(E), "space": np.arange(S_kept)}


# ---------------------------------------------------------------------------
# Hierarchical spatial-cell GPD
# ---------------------------------------------------------------------------
with pm.Model(coords=coords) as model:
    z_obs = pm.ConstantData("z", z, dims="event")
    N_t = pm.MutableData("N_t", N_event, dims="event")
    D_t = pm.MutableData("D_t", D_event, dims="event")
    ND_t = pm.MutableData("ND_t", ND_event, dims="event")
    s_id = pm.ConstantData("s_id", space_event, dims="event")

    xi = pm.TruncatedNormal(
        "xi",
        mu=0.05,
        sigma=0.15,
        lower=XI_LOWER,
        upper=XI_UPPER,
    )

    a_bar = pm.Normal("a_bar", 0.0, 1.0)
    bN_bar = pm.Normal("bN_bar", 0.0, 0.5)
    bD_bar = pm.Normal("bD_bar", 0.0, 0.5)
    bND_bar = pm.Normal("bND_bar", 0.0, 0.5)

    a_sd = pm.HalfNormal("a_sd", 0.8)
    bN_sd = pm.HalfNormal("bN_sd", 0.3)
    bD_sd = pm.HalfNormal("bD_sd", 0.3)
    bND_sd = pm.HalfNormal("bND_sd", 0.2)

    a_z = pm.Normal("a_z", 0.0, 1.0, dims="space")
    bN_z = pm.Normal("bN_z", 0.0, 1.0, dims="space")
    bD_z = pm.Normal("bD_z", 0.0, 1.0, dims="space")
    bND_z = pm.Normal("bND_z", 0.0, 1.0, dims="space")

    a_s = pm.Deterministic(
        "a_s", a_bar + a_sd * a_z, dims="space"
    )
    bN_s = pm.Deterministic(
        "bN_s", bN_bar + bN_sd * bN_z, dims="space"
    )
    bD_s = pm.Deterministic(
        "bD_s", bD_bar + bD_sd * bD_z, dims="space"
    )
    bND_s = pm.Deterministic(
        "bND_s", bND_bar + bND_sd * bND_z, dims="space"
    )

    log_sigma = (
        a_s[s_id]
        + bN_s[s_id] * N_t
        + bD_s[s_id] * D_t
        + bND_s[s_id] * ND_t
    )
    sigma = pm.Deterministic(
        "sigma", 1e-6 + pt.exp(log_sigma), dims="event"
    )

    pm.DensityDist(
        "z_like",
        sigma,
        xi,
        logp=lambda value, scale, shape: gpd_logp(
            value, scale, shape
        ),
        observed=z_obs,
        dims="event",
    )

    idata = pm.sample(
        draws=ARGS.draws,
        tune=ARGS.tune,
        chains=ARGS.chains,
        cores=ARGS.cores,
        target_accept=ARGS.target_accept,
        max_treedepth=ARGS.max_treedepth,
        random_seed=RANDOM_SEED,
    )


safe_save_idata(idata, OUT_IDATA)
run_name = f"gpd_{BASIN}_roni_dmi"
write_convergence_checks(idata, run_name)
posterior_predictive_checks(
    idata=idata,
    z=z,
    n_event=N_event,
    d_event=D_event,
    nd_event=ND_event,
    s_event=space_event,
    run_name=run_name,
)

print("\n=== Completed spatial basin GPD model ===")
print(f"Basin key: {BASIN}")
print(f"Posterior: {OUT_IDATA}")
print(f"Checks: {CHECK_DIR}")
