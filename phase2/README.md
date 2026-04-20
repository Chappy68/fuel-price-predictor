# Phase 2 Status

## Contents moved into this folder

- `phase2.ipynb`
- `pipeline.py`
- `mlflow.db`
- `conf_intervals.csv`
- `prediction_ci.csv`
- `model_meta.json`
- `requirements.txt`

## Quick requirement check

### Completed

- `phase2.ipynb` exists and contains MLflow experiment setup.
- Multiple ML models are compared in MLflow runs.
- Parameters are logged with `mlflow.log_param(...)`.
- At least two metrics are logged: `MAE`, `RMSE`, and `R2`.
- Artifacts are logged:
  - residual plots
  - actual-vs-predicted plots
  - confidence interval CSV files
- Tags are logged:
  - `algorithm`
  - `dataset_version`
  - `target`
- The best model is registered in the MLflow Model Registry.
- The best model is transitioned to the `Production` stage.
- Bonus confidence intervals with `statsmodels` are implemented.
- `model_meta.json` is generated for Phase 3.

### Still missing / not found in the folder right now

- `phase2.pdf` is still missing.
- `ci_plot.png` is referenced by the notebook, but no file was found here after the move.
- `model_comparison.png` is referenced by the notebook, but no file was found here after the move.
- No screenshot export from the MLflow UI was found yet.

## Conclusion

Phase 2 is mostly implemented in code and notebook form.
The main missing deliverables are the documentation/export artifacts:

- `phase2.pdf`
- MLflow UI screenshots
- the missing generated PNG files if you want them included as standalone files
