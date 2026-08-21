# ENSO, IOD, and Local SST Forcing of Humid-Heat Extremes in the Arabian Peninsula

Code accompanying the manuscript:

> **El Niño–Southern Oscillation, Indian Ocean Dipole, and Local Sea-Surface Temperature Forcing of Humid-Heat Extremes in the Arabian Peninsula**

This repository contains the data-processing, composite-analysis, Bayesian extreme-value modeling, diagnostic, and figure-generation workflows used to investigate how remote ocean variability and local sea-surface temperatures (SSTs) influence humid-heat extremes across the Arabian Peninsula (AP).

The analysis focuses on two related mechanisms:

1. **Remote climate variability**, represented by the El Niño–Southern Oscillation (ENSO) and Indian Ocean Dipole (IOD), which alters the frequency, magnitude, and spatial structure of humid-heat extremes.
2. **Local SST forcing**, particularly warming of the Persian Gulf, Red Sea, Gulf of Aden, and Gulf of Oman, which regulates atmospheric moisture availability and the upper tail of wet-bulb temperature.

These mechanisms are first evaluated independently and then combined in Bayesian peaks-over-threshold (POT) experiments to examine compound effects on extreme humid heat.

---

## Repository Structure

```text
.
├── bayesian_projection/
│   ├── city_remote_bernoulli_pipeline.py
│   ├── city_remote_hier_gpd_pipeline.py
│   ├── basin_remote_studentt_mean_pipeline.py
│   ├── basin_remote_gpd_pipeline.py
│   ├── city_local_hier_gpd_pipeline.py
│   ├── city_joint_forcing_gpd.py
│   ├── aggregate_and_plot_*.py
│   └── README.md
│
├── data_creation/
│   ├── create_daily_peak_state.py
│   ├── create_basin_sst_anomalies.py
│   ├── create_roni_dmi_indices.py
│   └── create_city_wbt_lagged_indices.py
│
├── enso_iod_analysis/
│   └── make_enso_iod_phase_composites.py
│
├── map_generation/
│   └── reference_map.py
│
├── map_fig_generation/
│   └── plot_all_four_city_inset_maps_with_baselines.py
│
├── model_run_diagnostics/
│   └── model_run_diagnostics.py
│
└── figures/
```

Developmental scripts and notebooks are stored in `archive/` directories and are not part of the final manuscript workflow.

---

# Analysis Workflow

The principal workflow is:

```text
ERA5 atmospheric fields + SST
              │
              ▼
       Data preparation
              │
              ├──────────────► RONI / DMI indices
              │
              ├──────────────► Basin SST anomalies
              │
              └──────────────► Daily peak-Tw atmospheric state
                                      │
                                      ▼
                           ENSO / IOD spatial composites
                                      │
                                      ▼
                            Bayesian POT analyses
                                      │
                  ┌───────────────────┼────────────────────┐
                  ▼                   ▼                    ▼
             Remote ENSO/IOD      Local basin SST      Joint forcing
                  │                   │                    │
                  └───────────────────┼────────────────────┘
                                      ▼
                           Posterior diagnostics
                                      │
                                      ▼
                         Manuscript figures/tables
```

---

# 1. Data Creation

Scripts in `data_creation/` generate the primary derived datasets used by the analysis.

## `create_daily_peak_state.py`

Processes hourly ERA5 atmospheric data over the Arabian Peninsula and identifies the atmospheric state associated with maximum daily wet-bulb temperature (`Tw`).

Derived variables include:

- daily maximum `Tw`
- air temperature at maximum daily `Tw`
- specific humidity at maximum daily `Tw`
- relative humidity at maximum daily `Tw`
- thermodynamic stickiness diagnostics
- timing of daily maxima

The wet-bulb and thermodynamic calculations depend on routines developed for the associated humid-heat analysis and are not distributed in this repository.

## `create_basin_sst_anomalies.py`

Constructs ERA5 SST anomalies for the regional ocean basins surrounding the Arabian Peninsula using a 1991–2020 climatology.

The analysis considers the:

- Persian Gulf
- Red Sea
- Gulf of Aden
- Gulf of Oman

## `create_roni_dmi_indices.py`

Constructs monthly SST-based climate indices from ERA5:

- Relative Oceanic Niño Index (RONI)
- Dipole Mode Index (DMI)

The Niño 3.4, tropical SST, western tropical Indian Ocean, and southeastern tropical Indian Ocean regions are calculated from area-weighted SST averages.

Monthly anomalies are referenced to the 1991–2020 climatology.

## `create_city_wbt_lagged_indices.py`

Extracts city-level daily peak `Tw`, aggregates monthly statistics, and merges these observations with lagged RONI and DMI.

Lags from 0–6 months are retained to support evaluation of the temporal relationship between remote SST variability and Arabian Peninsula humid heat.

---

# 2. ENSO–IOD Spatial Composite Analysis

```text
enso_iod_analysis/make_enso_iod_phase_composites.py
```

Produces the spatial composite analysis of humid-heat extremes under:

- El Niño
- La Niña
- positive IOD
- negative IOD

The analysis evaluates atmospheric conditions during extreme `Tw` days, including changes in temperature, atmospheric moisture, circulation, and related atmospheric fields.

Statistical significance is evaluated using block-bootstrap resampling.

This workflow generates the spatial ENSO/IOD composite figure used in the manuscript.

---

# 3. Bayesian Extreme-Value Analysis

The primary statistical analysis is contained in:

```text
bayesian_projection/
```

The Bayesian models evaluate three related components of humid-heat variability:

### Remote forcing

How ENSO and IOD affect the probability and magnitude of extreme `Tw`.

### Local SST forcing

How SST variability in adjacent ocean basins affects the upper tail of city-level `Tw`.

### Compound forcing

How ENSO/IOD variability and local basin warming combine to alter extreme humid heat.

The models include:

- Bayesian Bernoulli exceedance models
- hierarchical Generalized Pareto Distribution (GPD) models
- Student-t SST models
- spatial basin GPD models
- compound ENSO/IOD/local-SST GPD experiments

Each model is paired with an aggregation/plotting script that converts posterior samples into the quantities and figures presented in the manuscript.

See:

```text
bayesian_projection/README.md
```

for full model descriptions, assumptions, scenario definitions, and model-to-figure relationships.

---

# 4. Posterior Diagnostics

```text
model_run_diagnostics/model_run_diagnostics.py
```

Evaluates convergence and sampling diagnostics for the Bayesian models used in the manuscript.

Diagnostics include:

- R-hat
- bulk effective sample size (ESS)
- tail ESS
- divergences
- Bayesian fraction of missing information (BFMI)

The script also generates compact diagnostic tables used in the Supplementary Information.

---

# 5. Figure Generation

## Reference map

```text
map_generation/reference_map.py
```

Generates the geographic reference figure showing:

- Arabian Peninsula study region
- analysis cities
- adjacent ocean basins
- Niño regions
- IOD western and eastern poles

## Manuscript figure assembly

```text
map_fig_generation/plot_all_four_city_inset_maps_with_baselines.py
```

Converts model outputs into the final manuscript-ready multi-panel city figures.

Additional figure generation associated directly with individual Bayesian models is handled by the `aggregate_and_plot_*.py` scripts in `bayesian_projection/`.

---

# Data Availability

Raw and intermediate climate data are **not distributed through this GitHub repository** because of their size.

The analysis primarily uses ERA5 atmospheric and SST data. Users wishing to reproduce the complete workflow should obtain the corresponding source data independently and configure the data-creation scripts to point to their local files.

Derived datasets generated by the scripts are expected under:

```text
data/
```

The `data/` directory and large model outputs are excluded from version control.

Likewise, developmental code stored in `archive/` directories is excluded from the public repository.

---

# Reproducing the Analysis

The approximate workflow is:

```bash
# 1. Generate daily peak atmospheric-state data
python data_creation/create_daily_peak_state.py

# 2. Generate regional SST anomalies
python data_creation/create_basin_sst_anomalies.py \
    --sst-glob "/path/to/era5_sst/era5_sst_*.nc"

# 3. Generate RONI and DMI
python data_creation/create_roni_dmi_indices.py \
    --sst-glob "/path/to/era5_sst/era5_sst_*.nc"

# 4. Generate city WBT / lagged-index tables
python data_creation/create_city_wbt_lagged_indices.py

# 5. Generate ENSO/IOD composites
python enso_iod_analysis/make_enso_iod_phase_composites.py

# 6. Run Bayesian models
cd bayesian_projection
python <model_pipeline>.py

# 7. Aggregate posterior experiments and generate figures
python <aggregate_and_plot_script>.py

# 8. Evaluate posterior diagnostics
cd ..
python model_run_diagnostics/model_run_diagnostics.py
```

Individual scripts may require configuration of external ERA5 input paths before execution.

---

# Main Analysis Locations

The primary cities analyzed are:

- Aden
- Basra
- Dammam
- Doha
- Dubai
- Jeddah
- Kuwait City
- Medina
- Muscat
- Riyadh

The primary adjacent ocean basins are the Persian Gulf, Red Sea, Gulf of Aden, and Gulf of Oman.

---

# Software

The analysis was conducted in Python.

Principal dependencies include:

```text
numpy
pandas
xarray
scipy
pymc
arviz
pytensor
matplotlib
cartopy
geopandas
regionmask
dask
h5netcdf
netCDF4
cmocean
```

Python 3.11 or later is recommended.

---

# Citation

If using this code, please cite the accompanying manuscript once available:

> Bose, D., et al.  
> *El Niño–Southern Oscillation, Indian Ocean Dipole, and Local Sea-Surface Temperature Forcing of Humid-Heat Extremes in the Arabian Peninsula.*

A complete citation will be added following publication.

---

# Authors

**Daniel Bose**  
Department of Earth Sciences  
Montana State University

**Cascade Tuholske**  
Montana State University

Additional manuscript coauthors are listed in the accompanying publication.

---

# License

See the repository `LICENSE` file for reuse and distribution terms.

---

# Contact

For questions regarding the analysis or code:

**Daniel Bose**  
Department of Earth Sciences  
Montana State University  
daniel.bose1@montana.edu