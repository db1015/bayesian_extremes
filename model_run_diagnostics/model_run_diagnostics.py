#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
Generate MCMC diagnostic summaries for the Bayesian models used in the
Arabian Peninsula humid-heat analysis.

The script evaluates four model families:

1. City HHE exceedance probability ~ ENSO/IOD
2. Local basin SST ~ ENSO/IOD
3. City HHE magnitude ~ local SST
4. Joint ENSO/IOD + local SST ~ city HHE magnitude

For each model/location, the publication-facing summary reports:

    - maximum R-hat
    - minimum bulk effective sample size (ESS)
    - minimum tail effective sample size (ESS)
    - minimum Bayesian fraction of missing information (BFMI)

Additional diagnostics, including Monte Carlo standard error and divergent
transitions, are retained in the full diagnostic CSV files.

Expected inputs are saved ArviZ InferenceData NetCDF files produced by the
model-fitting scripts in ``bayesian_projection/``.

Example
-------
From the repository root:

    python model_run_diagnostics/model_run_diagnostics.py

Or with explicit paths:

    python model_run_diagnostics/model_run_diagnostics.py \
        --data-dir /path/to/model/output/data \
        --output-dir /path/to/diagnostic/output
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd


# ============================================================
# DISPLAY NAMES
# ============================================================

CITY_NAMES = {
    "kuwait_city": "Kuwait City",
    "riyadh": "Riyadh",
    "medina": "Medina",
    "muscat": "Muscat",
    "jeddah": "Jeddah",
    "aden": "Aden",
    "basra": "Basra",
    "doha": "Doha",
    "dubai": "Dubai",
    "dammam": "Dammam",
}

BASIN_NAMES = {
    "arabian_gulf": "Persian Gulf",
    "gulf_aden": "Gulf of Aden",
    "gulf_oman": "Gulf of Oman",
    "red_sea": "Red Sea",
}


# ============================================================
# HELPERS
# ============================================================

def clean_city_name(name: str) -> str:
    """Convert internal city identifiers to publication-facing names."""
    return CITY_NAMES.get(
        name.lower(),
        name.replace("_", " ").title(),
    )


def clean_basin_name(name: str) -> str:
    """Convert internal basin identifiers to publication-facing names."""
    return BASIN_NAMES.get(
        name,
        name.replace("_", " ").title(),
    )


def diagnostic_summary(
    idata: az.InferenceData,
    var_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return parameter-level MCMC diagnostics.

    If variable names are not supplied, observation-level deterministic
    quantities are excluded automatically.
    """
    posterior = idata.posterior

    if var_names is None:
        var_names = []

        excluded_dims = {
            "day",
            "obs",
            "observation",
            "time",
        }

        for name, da in posterior.data_vars.items():
            if any(dim in excluded_dims for dim in da.dims):
                continue
            var_names.append(name)

    if not var_names:
        raise RuntimeError(
            "No suitable posterior parameters were found for diagnostics."
        )

    return az.summary(
        idata,
        var_names=var_names,
        kind="diagnostics",
        round_to=None,
    )


def collapse_diagnostics(summary: pd.DataFrame) -> dict[str, float]:
    """Collapse parameter-level diagnostics to conservative run-level values."""
    return {
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "min_ess_tail": float(summary["ess_tail"].min()),
        "max_mcse_mean": float(summary["mcse_mean"].max()),
        "max_mcse_sd": float(summary["mcse_sd"].max()),
    }


def get_divergences(idata: az.InferenceData) -> float | int:
    """Return total number of divergent transitions if available."""
    if "sample_stats" not in idata.groups():
        return np.nan

    if "diverging" not in idata.sample_stats:
        return np.nan

    return int(idata.sample_stats["diverging"].sum().values)


def get_min_bfmi(idata: az.InferenceData) -> float:
    """Return the minimum BFMI across MCMC chains."""
    try:
        values = np.asarray(az.bfmi(idata), dtype=float)
        return float(np.nanmin(values))
    except Exception:
        return np.nan


def scalar_city_diagnostics(
    idata: az.InferenceData,
    city_dim: str,
    city_index: int,
) -> pd.DataFrame:
    """
    Calculate diagnostics for scalar city-specific posterior parameters.

    Observation-level and latent non-centered parameters are excluded.
    """
    pieces = []

    for parameter, da in idata.posterior.data_vars.items():

        if city_dim not in da.dims:
            continue

        if parameter.endswith("_z"):
            continue

        city_da = da.isel({city_dim: city_index})

        remaining_dims = [
            dim
            for dim in city_da.dims
            if dim not in {"chain", "draw"}
        ]

        if remaining_dims:
            continue

        tmp = az.InferenceData(
            posterior=city_da.to_dataset(name=parameter)
        )

        pieces.append(
            az.summary(
                tmp,
                var_names=[parameter],
                kind="diagnostics",
                round_to=None,
            )
        )

    if not pieces:
        raise RuntimeError(
            f"No scalar city-specific parameters found for index {city_index}."
        )

    return pd.concat(pieces)


def identify_city_dimension(idata: az.InferenceData) -> str | None:
    """Identify the city dimension used by a posterior."""
    for dim in ("city", "space", "cities"):
        if dim in idata.posterior.dims:
            return dim

    return None


def coordinate_city_names(
    idata: az.InferenceData,
    city_dim: str,
) -> list[str] | None:
    """Recover city names from posterior coordinates when available."""
    coord = idata.posterior.coords.get(city_dim)

    if coord is None:
        return None

    values = coord.values

    if values.dtype == object or np.issubdtype(values.dtype, np.str_):
        return [str(value) for value in values]

    return None


# ============================================================
# MODEL 1: CITY HHE ~ ENSO/IOD
# ============================================================

def bernoulli_diagnostics(data_dir: Path) -> pd.DataFrame:
    """Diagnostics for city-level Bernoulli ENSO/IOD models."""
    model_dir = data_dir / "wbt_daily_peak_daily_city_runs_bernoulli"

    files = sorted(model_dir.glob("idata_*.nc"))

    if not files:
        raise FileNotFoundError(
            f"No Bernoulli posterior files found in {model_dir}"
        )

    rows = []

    for path in files:

        print(f"  {path.name}")

        idata = az.from_netcdf(path)
        name = path.stem

        if "gulf_coastal_pooled" in name:

            preferred = [
                "a_bar",
                "bN_bar",
                "bD_bar",
                "bND_bar",
                "a_sd",
                "bN_sd",
                "bD_sd",
                "bND_sd",
                "a_s",
                "bN_s",
                "bD_s",
                "bND_s",
            ]

            var_names = [
                var
                for var in preferred
                if var in idata.posterior
            ]

            summary = diagnostic_summary(idata, var_names)

            rows.append({
                "city_or_group": "Persian Gulf hierarchical",
                "posterior_file": path.name,
                **collapse_diagnostics(summary),
                "divergences": get_divergences(idata),
                "min_bfmi": get_min_bfmi(idata),
            })

            city_dim = identify_city_dimension(idata)

            if city_dim is None:
                raise RuntimeError(
                    f"Could not identify city dimension in {path.name}"
                )

            city_names = coordinate_city_names(idata, city_dim)

            if city_names is None:
                # This ordering is fixed by the corresponding model script.
                city_names = ["Doha", "Dubai", "Dammam"]

            if len(city_names) != idata.posterior.sizes[city_dim]:
                raise RuntimeError(
                    f"City-label count does not match posterior dimension "
                    f"in {path.name}"
                )

            for index, city in enumerate(city_names):

                summary = scalar_city_diagnostics(
                    idata=idata,
                    city_dim=city_dim,
                    city_index=index,
                )

                rows.append({
                    "city_or_group": clean_city_name(str(city)),
                    "posterior_file": path.name,
                    **collapse_diagnostics(summary),
                    "divergences": get_divergences(idata),
                    "min_bfmi": get_min_bfmi(idata),
                })

        else:

            city = (
                name
                .replace("idata_wbt_daily_peak_", "")
                .replace("_roni_dmi_bernoulli", "")
            )

            preferred = [
                "a",
                "bN",
                "bD",
                "bND",
            ]

            var_names = [
                var
                for var in preferred
                if var in idata.posterior
            ]

            summary = diagnostic_summary(idata, var_names)

            rows.append({
                "city_or_group": clean_city_name(city),
                "posterior_file": path.name,
                **collapse_diagnostics(summary),
                "divergences": get_divergences(idata),
                "min_bfmi": get_min_bfmi(idata),
            })

    return (
        pd.DataFrame(rows)
        .sort_values("city_or_group")
        .reset_index(drop=True)
    )


# ============================================================
# MODEL 2: LOCAL SST ~ ENSO/IOD
# ============================================================

def basin_sst_diagnostics(data_dir: Path) -> pd.DataFrame:
    """Diagnostics for basin-scale Student-t ENSO/IOD models."""
    model_dir = data_dir / "sst"

    files = sorted(
        model_dir.glob("studentt_mean_*_roni_dmi_idata.nc")
    )

    if not files:
        raise FileNotFoundError(
            f"No basin SST posterior files found in {model_dir}"
        )

    rows = []

    preferred = [
        "a_bar",
        "a_sd",
        "bNp_bar",
        "bNn_bar",
        "bD_bar",
        "bNpD_bar",
        "bNnD_bar",
        "bNp_sd",
        "bNn_sd",
        "bD_sd",
        "bNpD_sd",
        "bNnD_sd",
        "sigma",
        "nu",
        "nu_minus_two",
    ]

    for path in files:

        print(f"  {path.name}")

        idata = az.from_netcdf(path)

        basin_key = (
            path.stem
            .replace("studentt_mean_", "")
            .replace("_roni_dmi_idata", "")
        )

        var_names = [
            var
            for var in preferred
            if var in idata.posterior
        ]

        summary = diagnostic_summary(idata, var_names)

        rows.append({
            "basin": clean_basin_name(basin_key),
            "posterior_file": path.name,
            **collapse_diagnostics(summary),
            "divergences": get_divergences(idata),
            "min_bfmi": get_min_bfmi(idata),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("basin")
        .reset_index(drop=True)
    )


# ============================================================
# MODEL 3: CITY HHE ~ LOCAL SST
# ============================================================

def local_sst_diagnostics(data_dir: Path) -> pd.DataFrame:
    """Diagnostics for city-level local-SST GPD models."""
    model_dir = data_dir / "wbt_sst_city_runs"

    files = sorted(
        model_dir.glob(
            "idata_city_hier_wbt_daily_peak_vs_sst_*_JJAS.nc"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No city/local-SST posterior files found in {model_dir}"
        )

    rows = []

    for path in files:

        print(f"  {path.name}")

        idata = az.from_netcdf(path)

        basin_key = (
            path.stem
            .replace(
                "idata_city_hier_wbt_daily_peak_vs_sst_",
                "",
            )
            .replace("_JJAS", "")
        )

        basin = clean_basin_name(basin_key)

        meta_path = model_dir / (
            "meta_city_hier_wbt_daily_peak_vs_sst_"
            f"{basin_key}_JJAS.pkl"
        )

        city_names = None

        if meta_path.exists():

            with open(meta_path, "rb") as file:
                metadata = pickle.load(file)

            if isinstance(metadata, dict):
                for key in (
                    "cities",
                    "city_names",
                    "city_order",
                    "CITY_NAMES",
                ):
                    if key in metadata:
                        city_names = list(metadata[key])
                        break

        overall_summary = diagnostic_summary(idata)

        rows.append({
            "basin": basin,
            "city": "ALL / hierarchical fit",
            "posterior_file": path.name,
            **collapse_diagnostics(overall_summary),
            "divergences": get_divergences(idata),
            "min_bfmi": get_min_bfmi(idata),
        })

        city_dim = identify_city_dimension(idata)

        if city_dim is None:
            continue

        n_cities = idata.posterior.sizes[city_dim]

        if city_names is None:
            city_names = coordinate_city_names(idata, city_dim)

        if city_names is None:
            raise RuntimeError(
                f"Could not recover city labels for {path.name}. "
                "City labels must be stored in metadata or posterior "
                "coordinates to avoid ambiguous reporting."
            )

        if len(city_names) != n_cities:
            raise RuntimeError(
                f"City-label count does not match posterior dimension "
                f"in {path.name}"
            )

        for index, city in enumerate(city_names):

            summary = scalar_city_diagnostics(
                idata=idata,
                city_dim=city_dim,
                city_index=index,
            )

            rows.append({
                "basin": basin,
                "city": clean_city_name(str(city)),
                "posterior_file": path.name,
                **collapse_diagnostics(summary),
                "divergences": get_divergences(idata),
                "min_bfmi": get_min_bfmi(idata),
            })

    return (
        pd.DataFrame(rows)
        .sort_values(["basin", "city"])
        .reset_index(drop=True)
    )


# ============================================================
# MODEL 4: JOINT ENSO/IOD + LOCAL SST ~ CITY HHE
# ============================================================

def joint_diagnostics(data_dir: Path) -> pd.DataFrame:
    """Diagnostics for the joint ENSO/IOD + local-SST HHE model."""
    posterior_file = (
        data_dir
        / "joint_enso_local_sst_city_runs"
        / "idata_joint_wbt_daily_peak_enso_iod_local_sst_JJAS.nc"
    )

    if not posterior_file.exists():
        raise FileNotFoundError(
            f"Joint-model posterior not found: {posterior_file}"
        )

    print(f"  {posterior_file.name}")

    idata = az.from_netcdf(posterior_file)

    overall_summary = diagnostic_summary(idata)

    rows = [{
        "city_or_group": "ALL / joint hierarchical fit",
        "posterior_file": posterior_file.name,
        **collapse_diagnostics(overall_summary),
        "divergences": get_divergences(idata),
        "min_bfmi": get_min_bfmi(idata),
    }]

    city_dim = identify_city_dimension(idata)

    if city_dim is None:
        raise RuntimeError(
            "Could not identify the city dimension in the joint posterior."
        )

    city_names = coordinate_city_names(idata, city_dim)

    if city_names is None:
        raise RuntimeError(
            "Joint posterior does not contain explicit city labels. "
            "Store city names as posterior coordinates in the model-fitting "
            "script before publishing diagnostics."
        )

    for index, city in enumerate(city_names):

        summary = scalar_city_diagnostics(
            idata=idata,
            city_dim=city_dim,
            city_index=index,
        )

        rows.append({
            "city_or_group": clean_city_name(str(city)),
            "posterior_file": posterior_file.name,
            **collapse_diagnostics(summary),
            "divergences": get_divergences(idata),
            "min_bfmi": get_min_bfmi(idata),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("city_or_group")
        .reset_index(drop=True)
    )


# ============================================================
# SI TABLES
# ============================================================

def compact_si_table(
    df: pd.DataFrame,
    name_col: str,
    drop_patterns: list[str] | None = None,
) -> pd.DataFrame:
    """Create the compact diagnostic table reported in the Supplement."""
    output = df.copy()

    if drop_patterns:

        mask = pd.Series(False, index=output.index)

        for pattern in drop_patterns:
            mask |= (
                output[name_col]
                .astype(str)
                .str.contains(
                    pattern,
                    case=False,
                    regex=False,
                    na=False,
                )
            )

        output = output.loc[~mask].copy()

    output = output[
        [
            name_col,
            "max_rhat",
            "min_ess_bulk",
            "min_ess_tail",
            "min_bfmi",
        ]
    ].copy()

    output = output.rename(
        columns={
            name_col: "Location",
            "max_rhat": "Max R-hat",
            "min_ess_bulk": "Min ESS bulk",
            "min_ess_tail": "Min ESS tail",
            "min_bfmi": "Min BFMI",
        }
    )

    output["Max R-hat"] = (
        output["Max R-hat"]
        .astype(float)
        .round(3)
    )

    output["Min ESS bulk"] = (
        output["Min ESS bulk"]
        .astype(float)
        .round()
        .astype("Int64")
    )

    output["Min ESS tail"] = (
        output["Min ESS tail"]
        .astype(float)
        .round()
        .astype("Int64")
    )

    output["Min BFMI"] = (
        output["Min BFMI"]
        .astype(float)
        .round(3)
    )

    return output.reset_index(drop=True)


def save_table(
    table: pd.DataFrame,
    stem: str,
    output_dir: Path,
) -> None:
    """Save a diagnostic table as both CSV and LaTeX."""
    table.to_csv(
        output_dir / f"{stem}.csv",
        index=False,
    )

    table.to_latex(
        output_dir / f"{stem}.tex",
        index=False,
        escape=False,
    )


# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Generate Bayesian model-run diagnostic summaries."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "data",
        help=(
            "Directory containing saved model posterior outputs. "
            "Default: <repository>/data"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help=(
            "Directory for diagnostic CSV and LaTeX tables. "
            "Default: model_run_diagnostics/outputs"
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("BAYESIAN MODEL DIAGNOSTICS")
    print("=" * 72)
    print(f"Input directory : {data_dir}")
    print(f"Output directory: {output_dir}")

    print("\n[1/4] City HHE ~ ENSO/IOD")
    bern_table = bernoulli_diagnostics(data_dir)

    print("\n[2/4] Local basin SST ~ ENSO/IOD")
    sst_remote_table = basin_sst_diagnostics(data_dir)

    print("\n[3/4] City HHE ~ local SST")
    city_local_sst_table = local_sst_diagnostics(data_dir)

    print("\n[4/4] Joint ENSO/IOD + local SST ~ city HHE")
    joint_table = joint_diagnostics(data_dir)

    # --------------------------------------------------------
    # Save complete QC tables
    # --------------------------------------------------------

    bern_table.to_csv(
        output_dir / "full_diagnostics_city_vs_remote.csv",
        index=False,
    )

    sst_remote_table.to_csv(
        output_dir / "full_diagnostics_sst_vs_remote.csv",
        index=False,
    )

    city_local_sst_table.to_csv(
        output_dir / "full_diagnostics_city_vs_local_sst.csv",
        index=False,
    )

    joint_table.to_csv(
        output_dir / "full_diagnostics_joint_model.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Publication-facing SI tables
    # --------------------------------------------------------

    si_bernoulli = compact_si_table(
        bern_table,
        name_col="city_or_group",
        drop_patterns=[
            "hierarchical",
            "pooled",
        ],
    )

    si_sst_remote = compact_si_table(
        sst_remote_table,
        name_col="basin",
    )

    si_city_local = compact_si_table(
        city_local_sst_table,
        name_col="city",
        drop_patterns=[
            "ALL / hierarchical fit",
            "hierarchical",
        ],
    )

    si_joint = compact_si_table(
        joint_table,
        name_col="city_or_group",
        drop_patterns=[
            "ALL / joint hierarchical fit",
            "hierarchical",
        ],
    )

    save_table(
        si_bernoulli,
        "si_diagnostics_city_vs_remote",
        output_dir,
    )

    save_table(
        si_sst_remote,
        "si_diagnostics_sst_vs_remote",
        output_dir,
    )

    save_table(
        si_city_local,
        "si_diagnostics_city_vs_local_sst",
        output_dir,
    )

    save_table(
        si_joint,
        "si_diagnostics_joint_model",
        output_dir,
    )

    print("\n" + "=" * 72)
    print("SI DIAGNOSTIC TABLES")
    print("=" * 72)

    for title, table in (
        ("City HHE ~ ENSO/IOD", si_bernoulli),
        ("Local SST ~ ENSO/IOD", si_sst_remote),
        ("City HHE ~ local SST", si_city_local),
        ("Joint ENSO/IOD + local SST", si_joint),
    ):
        print(f"\n{title}")
        print("-" * len(title))
        print(table.to_string(index=False))

    print("\nSaved diagnostic products to:")
    print(output_dir)


if __name__ == "__main__":
    main()


# In[ ]:




