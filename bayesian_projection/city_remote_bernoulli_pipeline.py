#!/usr/bin/env python
# coding: utf-8

# ============================================================
# Daily exceedance Bernoulli vs RONI & DMI (monthly covariates), by city
#   - single-cell per city
#   - partial pooling ONLY for Doha + Dubai + Dammam
#   - variable-agnostic structure, currently set for wbt_daily_peak
#   - outputs: one CSV with posterior summaries for each run
# ============================================================

import os
import glob
import argparse
import numpy as np
import pandas as pd
import xarray as xr

import pymc as pm
import pytensor.tensor as pt
import arviz as az

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--netid", type=str, default="k16v981")
    p.add_argument("--var", type=str, required=True,
                   help="Variable key, e.g. wbt_daily_peak")
    p.add_argument("--data-dir", type=str, default=None,
                   help="Optional override for variable data directory")
    p.add_argument("--glob", type=str, default=None,
                   help="Optional override for input glob")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Optional override for output directory")
    p.add_argument("--months", type=str, default="6,7,8,9",
                   help="Comma-separated months, or 'all'")
    p.add_argument("--q", type=float, default=0.95,
                   help="Quantile threshold for exceedance")
    p.add_argument("--draws", type=int, default=1500)
    p.add_argument("--tune", type=int, default=1500)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--cores", type=int, default=4)
    p.add_argument("--target-accept", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=72)
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

OUT_DIR = args.out_dir or os.path.join(BASE_DIR, f"{VAR}_daily_city_runs_bernoulli")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, f"{VAR}_daily_bernoulli_city_roni_dmi_summary.csv")

RANDOM_SEED = args.seed
Q = args.q

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

def summarize_param(post, name, hdi=0.94):
    arr = post[name].values
    mean = float(arr.mean())
    lo, hi = az.hdi(arr, hdi_prob=hdi)
    return mean, float(lo), float(hi)

def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

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

N_m = idx[roni_col].astype("float32")
D_m = idx[dmi_col].astype("float32")
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
# Build Bernoulli table
# -----------------------
def build_bernoulli_table(t_daily, y_daily):
    mk = month_key_daily(t_daily)

    Nm = N_m.reindex(mk).values.astype("float32")
    Dm = D_m.reindex(mk).values.astype("float32")
    NDm_ = ND_m.reindex(mk).values.astype("float32")

    if np.isnan(Nm).any() or np.isnan(Dm).any():
        bad = np.isnan(Nm) | np.isnan(Dm)
        raise ValueError(f"Missing RONI/DMI values for some days. First missing date: {t_daily[bad][0]}")

    u = float(np.nanquantile(y_daily, Q))
    exc = (y_daily > u).astype("int8")

    return {
        "exc": exc,
        "N": Nm,
        "D": Dm,
        "ND": NDm_,
        "u": u,
        "n_days": int(len(y_daily)),
        "n_exc": int(exc.sum()),
        "exc_rate": float(exc.mean()),
    }

# -----------------------
# Models
# -----------------------
def fit_single_city(run_id, city_name, tbl, out_dir):
    exc = tbl["exc"]
    N = tbl["N"]
    D = tbl["D"]
    ND = tbl["ND"]
    T = len(exc)

    coords = {"day": np.arange(T)}

    with pm.Model(coords=coords) as model:
        y_obs = pm.ConstantData("y_obs", exc, dims="day")
        N_t = pm.ConstantData("N_t", N, dims="day")
        D_t = pm.ConstantData("D_t", D, dims="day")
        ND_t = pm.ConstantData("ND_t", ND, dims="day")

        # intercept centered at low exceedance rate
        a = pm.Normal("a", mu=-3.0, sigma=2.0)
        bN = pm.Normal("bN", 0.0, 1.0)
        bD = pm.Normal("bD", 0.0, 1.0)
        bND = pm.Normal("bND", 0.0, 1.0)

        logit_p = a + bN * N_t + bD * D_t + bND * ND_t
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p), dims="day")

        pm.Bernoulli("exc_like", p=p, observed=y_obs, dims="day")

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
        "n_days": tbl["n_days"],
        "n_exc": tbl["n_exc"],
        "exc_rate": tbl["exc_rate"],
        "u": tbl["u"],
    }

    for name in ["a", "bN", "bD", "bND"]:
        m, lo, hi = summarize_param(post, name)
        row[f"{name}_mean"] = m
        row[f"{name}_hdi_low"] = lo
        row[f"{name}_hdi_high"] = hi

    # baseline probability
    base_logit = post["a"].values
    base_p = logistic(base_logit)
    row["p_base_mean"] = float(base_p.mean())
    lo, hi = az.hdi(base_p, hdi_prob=0.94)
    row["p_base_hdi_low"] = float(lo)
    row["p_base_hdi_high"] = float(hi)

    for lab, Nv, Dv in SCENARIOS:
        eta = (
            post["a"].values
            + post["bN"].values * Nv
            + post["bD"].values * Dv
            + post["bND"].values * (Nv * Dv)
        )
        p_scen = logistic(eta)
        row[f"p_{lab}_mean"] = float(p_scen.mean())
        lo, hi = az.hdi(p_scen, hdi_prob=0.94)
        row[f"p_{lab}_hdi_low"] = float(lo)
        row[f"p_{lab}_hdi_high"] = float(hi)

        odds_ratio = np.exp(
            post["bN"].values * Nv
            + post["bD"].values * Dv
            + post["bND"].values * (Nv * Dv)
        )
        row[f"oddsratio_{lab}_mean"] = float(odds_ratio.mean())
        lo, hi = az.hdi(odds_ratio, hdi_prob=0.94)
        row[f"oddsratio_{lab}_hdi_low"] = float(lo)
        row[f"oddsratio_{lab}_hdi_high"] = float(hi)

    return row

def fit_pooled_group(run_id, cities, tables, out_dir):
    exc_list, N_list, D_list, ND_list, s_list = [], [], [], [], []
    n_days_list, n_exc_list = [], []
    u_list, exc_rate_list = [], []

    for s, cname in enumerate(cities):
        tbl = tables[cname]
        T = len(tbl["exc"])

        exc_list.append(tbl["exc"])
        N_list.append(tbl["N"])
        D_list.append(tbl["D"])
        ND_list.append(tbl["ND"])
        s_list.append(np.full(T, s, dtype="int32"))

        n_days_list.append(tbl["n_days"])
        n_exc_list.append(tbl["n_exc"])
        u_list.append(tbl["u"])
        exc_rate_list.append(tbl["exc_rate"])

    exc = np.concatenate(exc_list).astype("int8")
    N = np.concatenate(N_list).astype("float32")
    D = np.concatenate(D_list).astype("float32")
    ND = np.concatenate(ND_list).astype("float32")
    s_id = np.concatenate(s_list).astype("int32")

    T_all = len(exc)
    S = len(cities)

    coords = {"day": np.arange(T_all), "space": np.arange(S)}

    with pm.Model(coords=coords) as model:
        y_obs = pm.ConstantData("y_obs", exc, dims="day")
        N_t = pm.ConstantData("N_t", N, dims="day")
        D_t = pm.ConstantData("D_t", D, dims="day")
        ND_t = pm.ConstantData("ND_t", ND, dims="day")
        s_idx = pm.ConstantData("s_id", s_id, dims="day")

        a_bar = pm.Normal("a_bar", mu=-3.0, sigma=2.0)
        bN_bar = pm.Normal("bN_bar", 0.0, 1.0)
        bD_bar = pm.Normal("bD_bar", 0.0, 1.0)
        bND_bar = pm.Normal("bND_bar", 0.0, 1.0)

        a_sd = pm.HalfNormal("a_sd", 1.0)
        bN_sd = pm.HalfNormal("bN_sd", 0.7)
        bD_sd = pm.HalfNormal("bD_sd", 0.7)
        bND_sd = pm.HalfNormal("bND_sd", 0.7)

        a_z = pm.Normal("a_z", 0, 1, dims="space")
        bN_z = pm.Normal("bN_z", 0, 1, dims="space")
        bD_z = pm.Normal("bD_z", 0, 1, dims="space")
        bND_z = pm.Normal("bND_z", 0, 1, dims="space")

        a_s = pm.Deterministic("a_s", a_bar + a_sd * a_z, dims="space")
        bN_s = pm.Deterministic("bN_s", bN_bar + bN_sd * bN_z, dims="space")
        bD_s = pm.Deterministic("bD_s", bD_bar + bD_sd * bD_z, dims="space")
        bND_s = pm.Deterministic("bND_s", bND_bar + bND_sd * bND_z, dims="space")

        logit_p = a_s[s_idx] + bN_s[s_idx] * N_t + bD_s[s_idx] * D_t + bND_s[s_idx] * ND_t
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p), dims="day")

        pm.Bernoulli("exc_like", p=p, observed=y_obs, dims="day")

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
        "n_days": int(np.sum(n_days_list)),
        "n_exc": int(np.sum(n_exc_list)),
        "exc_rate": float(np.sum(n_exc_list) / np.sum(n_days_list)),
        "u": np.nan,
    }

    for name in ["a_bar", "bN_bar", "bD_bar", "bND_bar", "a_sd", "bN_sd", "bD_sd", "bND_sd"]:
        m, lo, hi = summarize_param(post, name)
        row[f"{name}_mean"] = m
        row[f"{name}_hdi_low"] = lo
        row[f"{name}_hdi_high"] = hi

    base_logit = post["a_bar"].values
    base_p = logistic(base_logit)
    row["p_base_mean"] = float(base_p.mean())
    lo, hi = az.hdi(base_p, hdi_prob=0.94)
    row["p_base_hdi_low"] = float(lo)
    row["p_base_hdi_high"] = float(hi)

    for lab, Nv, Dv in SCENARIOS:
        eta = (
            post["a_bar"].values
            + post["bN_bar"].values * Nv
            + post["bD_bar"].values * Dv
            + post["bND_bar"].values * (Nv * Dv)
        )
        p_scen = logistic(eta)
        row[f"p_{lab}_mean"] = float(p_scen.mean())
        lo, hi = az.hdi(p_scen, hdi_prob=0.94)
        row[f"p_{lab}_hdi_low"] = float(lo)
        row[f"p_{lab}_hdi_high"] = float(hi)

        odds_ratio = np.exp(
            post["bN_bar"].values * Nv
            + post["bD_bar"].values * Dv
            + post["bND_bar"].values * (Nv * Dv)
        )
        row[f"oddsratio_{lab}_mean"] = float(odds_ratio.mean())
        lo, hi = az.hdi(odds_ratio, hdi_prob=0.94)
        row[f"oddsratio_{lab}_hdi_low"] = float(lo)
        row[f"oddsratio_{lab}_hdi_high"] = float(hi)

    rows_city = []
    for s, cname in enumerate(cities):
        r = {
            "run_id": f"{run_id}:{cname}",
            "var": VAR,
            "city": cname,
            "pooled_group": run_id,
            "Q": Q,
            "months": row["months"],
            "n_days": tables[cname]["n_days"],
            "n_exc": tables[cname]["n_exc"],
            "exc_rate": tables[cname]["exc_rate"],
            "u": tables[cname]["u"],
        }

        for name in ["a_s", "bN_s", "bD_s", "bND_s"]:
            arr = post[name].isel(space=s).values
            r[f"{name}_mean"] = float(arr.mean())
            lo, hi = az.hdi(arr, hdi_prob=0.94)
            r[f"{name}_hdi_low"] = float(lo)
            r[f"{name}_hdi_high"] = float(hi)

        base_logit = post["a_s"].isel(space=s).values
        base_p = logistic(base_logit)
        r["p_base_mean"] = float(base_p.mean())
        lo, hi = az.hdi(base_p, hdi_prob=0.94)
        r["p_base_hdi_low"] = float(lo)
        r["p_base_hdi_high"] = float(hi)

        for lab, Nv, Dv in SCENARIOS:
            eta = (
                post["a_s"].isel(space=s).values
                + post["bN_s"].isel(space=s).values * Nv
                + post["bD_s"].isel(space=s).values * Dv
                + post["bND_s"].isel(space=s).values * (Nv * Dv)
            )
            p_scen = logistic(eta)
            r[f"p_{lab}_mean"] = float(p_scen.mean())
            lo, hi = az.hdi(p_scen, hdi_prob=0.94)
            r[f"p_{lab}_hdi_low"] = float(lo)
            r[f"p_{lab}_hdi_high"] = float(hi)

            odds_ratio = np.exp(
                post["bN_s"].isel(space=s).values * Nv
                + post["bD_s"].isel(space=s).values * Dv
                + post["bND_s"].isel(space=s).values * (Nv * Dv)
            )
            r[f"oddsratio_{lab}_mean"] = float(odds_ratio.mean())
            lo, hi = az.hdi(odds_ratio, hdi_prob=0.94)
            r[f"oddsratio_{lab}_hdi_low"] = float(lo)
            r[f"oddsratio_{lab}_hdi_high"] = float(hi)

        rows_city.append(r)

    return [row] + rows_city

# -----------------------
# Main
# -----------------------
def main():
    print(f"\n=== Running Bernoulli exceedance model for VAR={VAR} ===")
    print(f"Input glob: {DATA_GLOB}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Output CSV: {OUT_CSV}\n")

    all_rows = []

    # 1) pooled Doha + Dubai + Dammam first
    for run_id, members in POOLED_GROUPS.items():
        tables = {}
        for cname in members:
            t, y = load_city_series(cname, CITIES[cname]["lat"], CITIES[cname]["lon"])
            tables[cname] = build_bernoulli_table(t, y)
            print(
                f"✅ built Bernoulli table {cname}: "
                f"n_days={tables[cname]['n_days']}  "
                f"n_exc={tables[cname]['n_exc']}  "
                f"u={tables[cname]['u']:.3f}  "
                f"rate={tables[cname]['exc_rate']:.4f}"
            )

        pooled_run_id = f"{VAR}_{run_id}_roni_dmi_bernoulli"
        rows = fit_pooled_group(pooled_run_id, members, tables, OUT_DIR)
        all_rows.extend(rows)

    # 2) single cities excluding pooled members
    pooled_members = set(sum(POOLED_GROUPS.values(), []))
    for cname, meta in CITIES.items():
        if cname in pooled_members:
            continue

        t, y = load_city_series(cname, meta["lat"], meta["lon"])
        tbl = build_bernoulli_table(t, y)
        print(
            f"✅ built Bernoulli table {cname}: "
            f"n_days={tbl['n_days']}  "
            f"n_exc={tbl['n_exc']}  "
            f"u={tbl['u']:.3f}  "
            f"rate={tbl['exc_rate']:.4f}"
        )

        run_id = f"{VAR}_{cname}_roni_dmi_bernoulli"
        row = fit_single_city(run_id, cname, tbl, OUT_DIR)
        all_rows.append(row)

    # 3) write combined CSV
    df = pd.DataFrame(all_rows)

    front = [
        "run_id", "var", "city", "pooled_group", "Q", "months",
        "n_days", "n_exc", "exc_rate", "u"
    ]
    cols = front + [c for c in df.columns if c not in front]
    df = df[cols]

    df.to_csv(OUT_CSV, index=False)
    print("✅ wrote summary CSV:", OUT_CSV)

if __name__ == "__main__":
    main()

