# Phase 1 - Linear Regression in scikit-learn + Static Web Deployment

## Overview

This phase delivers an end-to-end linear regression workflow:

- Training in Python with scikit-learn
- Model export to JSON for browser inference
- Static prediction interface built with plain HTML, CSS, and JavaScript
- Submission-ready notebook and dataset file

The prediction target is **`petrol_usd_liter`**.

## Dataset

- Included submission file: `global_fuel_prices_2020_2026.csv`
- Public reference dataset: [Kaggle - Global Gas Prices](https://www.kaggle.com/datasets/kazishafinalam/global-gas-prices)

The local dataset used in this project contains weekly fuel-price observations with the following relevant fields:

- `date`
- `country`
- `region`
- `income_level`
- `subsidy_level`
- `petrol_usd_liter`
- `diesel_usd_liter`
- `lpg_usd_liter`
- `brent_crude_usd`
- `tax_percentage`

## Model Approach

The model is trained in `phase1/pipeline.py` using **Linear Regression** with a preprocessing pipeline:

- Numeric features: median imputation + standard scaling
- Categorical features: most-frequent imputation + one-hot encoding with `drop="first"`
- Automatic date feature engineering: year, month, ISO week

To avoid leakage, the non-target fuel price columns are removed before training.

## Final Phase 1 Result

Default target:

- `petrol_usd_liter`

Evaluation metrics on the holdout test split:

- `MAE = 0.1677`
- `RMSE = 0.2600`
- `R^2 = 0.9724`

## Files

- `phase1/phase1.ipynb`: notebook submission
- `phase1/pipeline.py`: reproducible training and export pipeline
- `phase1/docs/model.json`: exported browser-ready model
- `phase1/webapp/index.html`: static UI
- `phase1/webapp/styles.css`: styling
- `phase1/webapp/app.js`: browser prediction logic
- `global_fuel_prices_2020_2026.csv`: dataset file

## How To Run The Python Pipeline

Recommended environment:

- `Python (nn_env)`

Run:

```powershell
c:\Users\maxch\anaconda3\envs\nn_env\python.exe phase1\pipeline.py --no-plots
```

This regenerates:

- `phase1/docs/model.json`
- `phase1/webapp/model.json`

## How To Use The Static Web App Locally

Because the app loads `model.json`, it should be served through a local web server instead of opening `index.html` directly as a file.

Example:

```powershell
cd phase1\webapp
python -m http.server 8000
```

Then open:

- `http://localhost:8000`

## GitHub Pages Deployment

The app is deployment-ready, but a live GitHub Pages URL could not be created from this workspace because no Git repository / GitHub remote was available here.

Suggested deployment flow:

1. Create or open the target GitHub repository.
2. Publish the contents of `phase1/webapp` via GitHub Pages.
3. Keep `model.json`, `index.html`, `styles.css`, `app.js`, and `.nojekyll` together in the published folder.

After publishing, add the live URL here:

- `GitHub Pages URL: <insert-your-pages-link-here>`

## Browser Inference Logic

The web app reproduces the exported scikit-learn pipeline in JavaScript:

- Numeric inputs are standardized with the exported `mean` and `scale`
- Categorical inputs use the exported one-hot encoding layout
- The final prediction is computed from:

```text
prediction = intercept + sum(feature_contributions)
```

## Submission Note

For Phase 1 submission, include:

- `phase1/phase1.ipynb`
- `phase1` source files
- `global_fuel_prices_2020_2026.csv`
- the static `webapp` folder
