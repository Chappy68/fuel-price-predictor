# Phase 3 - Streamlit Dashboard

## Goal

This folder contains the local Streamlit application for Phase 3.

The dashboard:

- loads the best model registered in MLflow during Phase 2
- allows the user to enter feature values through sliders and dropdowns
- predicts `petrol_usd_liter`
- optionally shows a 95% confidence interval using `statsmodels`
- visualizes model insights and dataset trends

## Files

- `app.py` - Streamlit dashboard
- `pipeline.py` - wrapper that reuses `phase2/pipeline.py`

## Required inputs from Phase 2

The app expects these files to exist in `../phase2/`:

- `model_meta.json`
- `pipeline.py`
- the MLflow-tracked model registry entry

## How to run

### 1. Start the local MLflow server

Use the same local MLflow setup you used in Phase 2. The app expects:

- `http://localhost:5000`

### 2. Start the Streamlit app

From this folder:

```powershell
cd c:\Users\maxch\OneDrive\Desktop\HTL\5CHIF\DSAI\MLOPS_Project\phase3
streamlit run app.py
```

## Current status

### Implemented

- user input section
- prediction output
- bonus confidence interval via `statsmodels`
- feature importance / coefficient plot
- dataset distribution plot
- yearly trend plot
- integration with `phase2/model_meta.json`
- registry-based model loading through MLflow

### Still manual

- `phase3.pdf`
- screenshots of the running Streamlit app for submission
