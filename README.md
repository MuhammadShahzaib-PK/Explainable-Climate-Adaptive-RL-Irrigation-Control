# Explainable RL for Climate-Adaptive Garden/Farm Irrigation Control

## Overview
A PPO reinforcement learning agent is trained to control irrigation timing across a 22-year,
4-country, 3-crop panel (Pakistan, India, Brazil, United States; wheat, rice, cotton; 2003-2024),
built from FAOSTAT yield records merged with NASA GISS temperature anomaly, CO2 (OCO-2 satellite
blended with Mauna Loa), and MERRA-2 aerosol optical depth. The trained policy's decisions are then
audited using an XGBoost surrogate model and SHAP, to identify what actually drives its irrigation
choices.

**This repository documents a real methodological correction made during the project, and reports
the corrected result plainly rather than only the cleaner original claim.**

## The corrected finding
An initial version of this analysis (see "Data provenance" below) reported a simple negative
result: the trained policy's decisions were dominated by internal simulation state rather than
climate signal. That directional finding survives a subsequent bug fix that scaled the training
data from 8 years to the full 22-year, 4-country panel — but the corrected analysis is more precise
and, on one dimension, more positive than the original:

1. **The trained policy statistically significantly outperforms all three naive baselines**
   (never irrigate, always light, always heavy) on the reward it was trained on. 95% bootstrap
   confidence intervals do not overlap:

   | Condition | Mean reward | 95% CI |
   |---|---|---|
   | PPO (trained) | -42.96 | [-44.69, -41.15] |
   | Never irrigate | -277.04 | [-282.12, -272.81] |
   | Always light | -240.68 | [-240.79, -240.47] |
   | Always heavy | -249.34 | [-249.34, -249.33] |

2. **Real climate signal exists in the corrected 22-year data.** Granger causality: temperature
   anomaly is significant (p=0.015), CO2 highly significant (p=0.0001), aerosol not significant
   (p=0.096).

3. **But the policy's actual decisions still don't use that climate signal.** SHAP on the corrected
   policy shows `soil_moisture` and `day_progress` (both internal simulation state) dominate the
   action choice, while `co2_normalized` and `temp_anomaly` (real, Granger-confirmed climate
   drivers) matter far less.

**The precise, defensible claim is therefore:** the reward function only requires tracking soil
moisture to score well, so the policy learned to do exactly that and nothing more, despite the
climate covariates it under-uses being demonstrably real drivers of yield. This is a sharper,
better-evidenced version of "reward under-specification" than a single SHAP plot alone would
support — it's now backed by a genuine outperformance result and an independent Granger causality
check on the same corrected data.

## Data provenance and the bug fix, stated plainly
During development, an early version of the training pipeline built its climate-merged panel with
an unqualified `dropna()` that silently dropped any row missing *any* climate covariate, capping
the usable Pakistan-wheat series at 8 years (2015-2022) even though 22 years of yield data existed.
This was caught, root-caused, and fixed by: (a) extending CO2 coverage back to 2003 using Mauna Loa
annual means blended with OCO-2 satellite data where available, (b) linearly interpolating the small
remaining aerosol gaps, and (c) rebuilding the merge from the raw yield table rather than patching
the already-capped one. The model in this repository (`models/ppo_farm_unified_v3.zip`) is the
**post-fix** model, trained on the full 22-year, 264-row panel. An earlier model
(`ppo_farm_pakistan_wheat`) and an intermediate one (`ppo_farm_unified_v2`, trained on a
multi-country but still year-truncated 96-row panel) exist from before this fix and are not
included here, to avoid presenting superseded results as current.

## Verification
This notebook does not just describe the corrected pipeline -- it actually loads the real,
originally-trained model (`ppo_farm_unified_v3.zip`) and reruns the full evaluation (baseline
comparison, Granger causality, surrogate + SHAP) from the underlying data files. Re-running it
reproduced the original recorded numbers almost exactly (mean rewards matched to two decimal
places; SHAP driver ranking identical), which is a meaningful integrity check: it confirms the
model file, the data files, and the environment code are mutually consistent, not just individually
plausible.

## Repository structure
```
├── notebooks/
│   └── explainable_rl_irrigation_pipeline.ipynb   # full pipeline, executable end to end
├── models/
│   └── ppo_farm_unified_v3.zip                    # the real, final trained PPO policy
├── data/
│   ├── faostat_yield_filtered.csv                 # raw yield panel, 4 countries x 3 crops x 22 years
│   ├── climate_yearly_context_extended.csv        # temperature, CO2 (blended), aerosol by year
│   ├── climate_yearly_context.csv                 # earlier climate table (aerosol not yet interpolated)
│   └── agri_climate_merged.csv                    # an earlier merge, kept for provenance/comparison
├── figures/
│   ├── fig_baseline_comparison.png
│   └── fig_shap_driver_importance.png
├── results/
│   ├── results_summary.json
│   ├── baseline_comparison_v2.csv
│   └── baseline_comparison_summary.csv
└── README.md
```

## Environment and reward design
Two environment versions are defined; the notebook trains and evaluates using
`FarmIrrigationEnvV2`, which drives soil-moisture dynamics from an FAO-56-style evapotranspiration
relationship rather than arbitrary constants. **Some constants (base ET0, the reward weighting) are
reasonable placeholders, not literature-derived values** -- this is stated here as plainly as it was
in the original analysis, and should be replaced with citable regional values (FAO-56 tables,
regional agromet references) before this work is presented as a validated deployable system. It is
not currently presented as one.

## How to run
1. Clone the repository.
2. Install dependencies: `pip install pandas numpy gymnasium stable-baselines3 statsmodels xgboost shap scikit-learn matplotlib jupyter`
3. From the repository root, open and run `notebooks/explainable_rl_irrigation_pipeline.ipynb` top to bottom.
4. It will load the real saved model and reproduce the results above; it does not retrain from scratch.

## Status
This is an active research project; the manuscript is not yet submitted. A companion computer-vision
and planning pipeline (garden image segmentation, FAO-56 water-balance calculation, sensor-limited
exploration robot) exists but is not yet included in this repository.

## Author
**Muhammad Shahzaib**
Mechatronics Engineer | Independent Researcher, Causal & Explainable Machine Learning for Climate
and Agricultural Systems
