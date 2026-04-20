# Phase 2 - MLflow Experiment Tracking and Model Registry

## Project context

This phase extends the Phase 1 fuel-price regression workflow with local MLflow tracking.
The dataset used in all experiments is `global_fuel_prices_2020_2026.csv`, and the prediction
target is `petrol_usd_liter`.

## Objective

The goal of Phase 2 is to:

- track multiple regression experiments in MLflow
- log parameters, metrics, tags, and artifacts
- compare model performance
- register the best-performing model in the MLflow Model Registry
- compute confidence intervals as a bonus step with `statsmodels`

## Dataset and preprocessing

The preprocessing logic is reused from Phase 1 through `pipeline.py`.
The notebook applies the same core steps:

- automatic date feature extraction
- dropping leakage-related target alternatives
- numeric imputation and scaling
- one-hot encoding for categorical variables
- identical train/test split strategy

This guarantees consistency between Phase 1, Phase 2, and the later Phase 3 dashboard.

## Tracked models

The Phase 2 notebook compares five configurations:

1. LinearRegression
2. Ridge with alpha = 0.5
3. Ridge with alpha = 5.0
4. Lasso with alpha = 0.001
5. RandomForestRegressor with 300 trees and max depth 12

The notebook evaluates all runs under the same preprocessing pipeline.

## MLflow logging

### Logged parameters

Examples of logged parameters:

- `test_size`
- `random_state`
- number of numeric features
- number of categorical features
- model-specific hyperparameters such as `alpha`, `n_estimators`, or `max_depth`

### Logged metrics

The notebook logs at least the following evaluation metrics for every run:

- `MAE`
- `RMSE`
- `R2`

### Logged tags

The following tags are explicitly written:

- `algorithm`
- `dataset_version`
- `target`

### Logged artifacts

The notebook logs multiple artifacts, including:

- residual distribution plots
- actual-vs-predicted plots
- `conf_intervals.csv`
- `prediction_ci.csv`
- confidence interval visualizations

## Bonus: confidence intervals

As requested in the assignment, the notebook fits a `statsmodels` OLS model on the scaled numeric
features and computes 95 percent confidence intervals.

Generated files include:

- `conf_intervals.csv`
- `prediction_ci.csv`

These artifacts are intended for later analysis and for reuse in Phase 3.

## Best model and registry

According to `model_meta.json`, the best model selected in Phase 2 is:

- `RandomForest_300trees`

The notebook then registers the best model in the MLflow Model Registry using:

- `BestFuelRegressor_v1`

and transitions the newest version to:

- `Production`

## Result summary

Phase 2 is largely implemented and fulfills the coding requirements of the assignment:

- local MLflow experiment setup
- multiple tracked runs
- parameter logging
- metric logging
- artifact logging
- tag logging
- model registration
- production-stage transition
- bonus confidence interval logic

## Submission status

The following deliverables are present in code or generated-output form:

- `phase2.ipynb`
- `pipeline.py`
- `mlflow.db`
- `conf_intervals.csv`
- `prediction_ci.csv`
- `model_meta.json`

The following parts are still manual submission tasks:

- `phase2.pdf` screenshots of the MLflow UI
- insertion of explicit screenshots if required by the teacher

## Screenshot placeholders

Insert the following screenshots into the final submission if needed:

1. MLflow experiment overview page
2. run comparison table
3. registered model page
4. production model version page

## Conclusion

Phase 2 successfully extends the Phase 1 solution into an MLOps workflow with experiment tracking,
model comparison, model registration, and uncertainty estimation. The notebook and generated
artifacts provide a solid basis for the final Streamlit dashboard in Phase 3.
