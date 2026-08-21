# Bayesian Projection Pipeline for Remote Climate Variability and Local SST Forcing of Humid-Heat Extremes

This repository contains the Bayesian statistical models used to investigate how large-scale climate variability (ENSO and the Indian Ocean Dipole) and adjacent basin sea-surface temperature (SST) anomalies influence humid-heat extremes across the Arabian Peninsula.

The repository accompanies:

> Bose, D., Raymond, C., Parks, R. & Tuholske, C. *(in preparation)*

The analysis is organized as a sequence of independent Bayesian models. Each model has a corresponding aggregation and plotting script that reproduces the figures and tables used in the manuscript.

---

# Repository Structure

```
bayesian_projection/

├── city_remote_bernoulli_pipeline.py
├── aggregate_and_plot_city_remote_bernoulli.py

├── city_remote_hier_gpd_pipeline.py
├── aggregate_and_plot_city_remote_gpd.py

├── basin_remote_studentt_mean_pipeline.py
├── aggregate_and_plot_basin_studentt.py

├── basin_remote_gpd_pipeline.py
├── aggregate_and_plot_basin_spatial_gpd.py

├── city_local_hier_gpd_pipeline.py
├── aggregate_and_plot_city_basin_warming.py

├── city_joint_forcing_gpd.py
├── aggregate_and_plot_joint_local_experiments.py

└── archive/
```

Each Bayesian model follows the same workflow:

```
Input data
      │
      ▼
Bayesian model fit
      │
      ▼
Posterior (.nc)
      │
      ▼
Posterior diagnostics
      │
      ▼
Scenario experiments
      │
      ▼
CSV summary tables
      │
      ▼
Publication-quality figures
```

---

# Models

## 1. Remote variability and exceedance probability (Section 2.2)

**Model**

```
city_remote_bernoulli_pipeline.py
```

Fits Bayesian Bernoulli logistic models describing the probability that a daily humid-heat extreme exceeds the historical threshold as a function of lagged ENSO (RONI) and IOD (DMI).

**Outputs**

```
aggregate_and_plot_city_remote_bernoulli.py
```

Computes posterior scenario impacts and reproduces the manuscript figure.

---

## 2. Remote variability and extreme magnitude (Section 2.2)

**Model**

```
city_remote_hier_gpd_pipeline.py
```

Fits hierarchical Generalized Pareto (GPD) models describing changes in the magnitude of humid-heat extremes conditioned on ENSO and IOD.

**Outputs**

```
aggregate_and_plot_city_remote_gpd.py
```

Computes posterior changes in p95 and p99 extremes and reproduces the supplementary figures.

---

## 3. Remote variability and basin SST (Section 2.4)

**Model**

```
basin_remote_studentt_mean_pipeline.py
```

Fits hierarchical Student-t models describing monthly SST anomalies within each basin as a function of ENSO and IOD.

**Outputs**

```
aggregate_and_plot_basin_studentt.py
```

Computes basin-mean SST responses and generates the supplementary heatmaps.

---

## 4. Remote variability and basin SST extremes (Section 2.4)

**Model**

```
basin_remote_gpd_pipeline.py
```

Fits spatial hierarchical GPD models describing extreme monthly SST anomalies within each basin.

**Outputs**

```
aggregate_and_plot_basin_spatial_gpd.py
```

Evaluates ENSO/IOD experiments across basin grid cells and generates spatial response maps.

---

## 5. Local basin warming (Section 2.5)

**Model**

```
city_local_hier_gpd_pipeline.py
```

Fits hierarchical GPD models relating city-scale humid-heat extremes to SST anomalies within each city's adjacent basin.

Counterfactual basin warming experiments (+0.5°C to +2.0°C) are evaluated after fitting.

**Outputs**

```
aggregate_and_plot_city_basin_warming.py
```

Produces the manuscript figures showing projected changes in conditional p95 and p99 humid-heat extremes under local basin warming.

---

## 6. Joint remote forcing and local SST forcing (Section 2.6)

**Model**

```
city_joint_forcing_gpd.py
```

Fits the final hierarchical model combining

- remote ENSO forcing,
- remote IOD forcing,
- local basin SST anomalies,
- ENSO × local SST interactions,

to quantify the combined effects of climate variability and local ocean warming.

**Outputs**

```
aggregate_and_plot_joint_local_experiments.py
```

Evaluates compound ENSO/IOD/basin warming scenarios and reproduces the manuscript figures.

---

# Input Data

The models use the following datasets.

## Daily atmospheric state

```
DailyPeakState/
```

Contains daily values evaluated at the time of maximum wet-bulb temperature, including

- wet-bulb temperature
- air temperature
- specific humidity
- τMSE

---

## Basin SST anomalies

```
sst/
```

Monthly SST anomalies averaged over

- Persian Gulf
- Gulf of Oman
- Gulf of Aden
- Red Sea

---

## Climate indices

```
roni_dmi_monthly_1950_2025.csv
```

Monthly

- Relative Oceanic Niño Index (RONI)
- Dipole Mode Index (DMI)

---

# Common Modeling Choices

Unless otherwise stated, all Bayesian models use

- June–September (JJAS) only
- historical p95 threshold
- Peaks-over-threshold framework
- minimum 50 exceedances per fitted location
- no declustering
- RONI lag = 2 months
- DMI lag = 1 month
- standardized predictors
- 94% highest-density intervals (HDIs)
- four MCMC chains
- non-centered hierarchical parameterizations where appropriate

Posterior diagnostics include

- R̂
- bulk ESS
- tail ESS
- divergences
- BFMI
- posterior predictive checks

---

# Outputs

Each model produces

```
Posterior (.nc)

Posterior diagnostic tables

Posterior predictive checks

Experiment CSV files

Publication-quality PNG figures

Publication-quality PDF figures
```

---

# Running the Models

Each analysis consists of two steps.

Example:

```
python city_remote_bernoulli_pipeline.py

python aggregate_and_plot_city_remote_bernoulli.py
```

The same workflow applies to every model in the repository.

---

# Archive

```
archive/
```

Contains prototype and developmental models that were used during method development.

These scripts are retained for provenance but were **not** used to generate the results presented in the manuscript.

---

# Dependencies

Python ≥ 3.11

Required packages

```
numpy
pandas
xarray
scipy
pymc
arviz
matplotlib
cartopy
pytensor
```

---

# Citation

If this repository contributes to your research, please cite the accompanying manuscript once published.

```
Bose, D., Raymond, C., Parks, R., & Tuholske, C.
El Niño-Southern Oscillation, Indian Ocean Dipole, and Local Sea-Surface Temperature Forcing of Humid-Heat Extremes in the Arabian Peninsula
```

---

# Contact

Daniel Bose

Department of Earth Sciences

Montana State University

daniel.bose1@student.montana.edu
daniel.bose1@montana.edu