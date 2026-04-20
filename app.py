"""
MLOps Project – Phase 3: Streamlit Prediction Dashboard
Dataset : global_fuel_prices_2020_2026.csv
Target  : petrol_usd_liter

Loads the best model registered in MLflow Phase 2 and provides:
  - Dynamic input form (numeric sliders + categorical dropdowns)
  - Prediction output with 95% confidence interval (bonus)
  - Key insights: feature coefficients / importance, price distribution
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fuel Price Predictor",
    page_icon="⛽",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
TRACKING_URI = "http://localhost:5000"
META_FILE    = "model_meta.json"

# ── Load metadata written by Phase 2 ─────────────────────────────────────────
@st.cache_resource
def load_meta():
    try:
        with open(META_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(
            f"`{META_FILE}` not found. Run the Phase 2 notebook first, "
            "then restart this app from the same directory."
        )
        st.stop()

meta                 = load_meta()
TARGET               = meta["target_column"]
NUMERIC_FEATURES     = meta["numeric_features"]
CATEGORICAL_FEATURES = meta["categorical_features"]
REGISTRY_NAME        = meta["registry_name"]
DATASET_FILE         = meta["dataset_file"]

# ── Load MLflow model ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model from MLflow registry…")
def load_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.sklearn.load_model(f"models:/{REGISTRY_NAME}/Production")

# ── Load raw dataset (for dropdowns, OLS, and plots) ─────────────────────────
@st.cache_resource(show_spinner="Loading dataset…")
def load_dataset():
    # Try to locate the CSV the same way pipeline.py does
    sys.path.insert(0, str(Path.cwd()))
    try:
        from pipeline import find_project_root
        csv_path = find_project_root() / DATASET_FILE
    except Exception:
        csv_path = Path(DATASET_FILE)

    df = pd.read_csv(csv_path)

    # Re-apply the same preprocessing from pipeline.py
    from pipeline import (
        convert_numeric_like_objects, add_date_features,
        resolve_target_column, infer_columns_to_drop,
    )
    working = convert_numeric_like_objects(df)
    date_cols = add_date_features(working)
    target = resolve_target_column(working)
    drop   = infer_columns_to_drop(working, target, date_cols)
    working = working.drop(columns=drop, errors="ignore")
    working = working.dropna(subset=[target]).copy()
    y = pd.to_numeric(working[target], errors="coerce")
    X = working.drop(columns=[target]).copy()
    mask = y.notna()
    return X.loc[mask].copy(), y.loc[mask].copy(), df

# ── Fit OLS for confidence intervals (numeric features only, like Phase 2) ────
@st.cache_resource(show_spinner="Fitting OLS for confidence intervals…")
def load_ols(X, y):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split

    X_tr, _, y_tr, _ = train_test_split(
        X, y,
        test_size=meta.get("test_size", 0.2),
        random_state=meta.get("random_state", 42),
    )
    if not NUMERIC_FEATURES:
        return None, None

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    X_tr_num = num_pipe.fit_transform(X_tr[NUMERIC_FEATURES])
    X_tr_sm  = sm.add_constant(X_tr_num)
    ols      = sm.OLS(y_tr.values, X_tr_sm).fit()
    return ols, num_pipe

# ── Collect unique options for each categorical feature ───────────────────────
@st.cache_resource
def get_cat_options(X):
    opts = {}
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            opts[col] = sorted(X[col].dropna().unique().tolist())
    return opts


# =============================================================================
# Load everything
# =============================================================================
X_all, y_all, df_raw = load_dataset()
ols_model, num_pipe  = load_ols(X_all, y_all)
cat_options          = get_cat_options(X_all)

# =============================================================================
# SIDEBAR – user inputs
# =============================================================================
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/"
    "House_icon.svg/240px-House_icon.svg.png",
    width=0,   # hidden — just keeping import alive; replace with fuel icon if desired
)
st.sidebar.title("⛽ Feature Inputs")
st.sidebar.caption("Set values to predict petrol price (USD/liter).")

user_inputs = {}

# Numeric sliders
for feat in NUMERIC_FEATURES:
    col_data = pd.to_numeric(X_all[feat], errors="coerce").dropna()
    mn  = float(col_data.min())
    mx  = float(col_data.max())
    med = float(col_data.median())
    label = feat.replace("_", " ").title()

    # Date-derived fields: integer step; others: float
    if "year" in feat.lower():
        user_inputs[feat] = float(st.sidebar.slider(
            label, min_value=int(mn), max_value=int(mx), value=int(med), step=1))
    elif "month" in feat.lower() or "week" in feat.lower():
        user_inputs[feat] = float(st.sidebar.slider(
            label, min_value=int(mn), max_value=int(mx), value=int(med), step=1))
    else:
        step = round((mx - mn) / 100, 4) or 0.01
        user_inputs[feat] = st.sidebar.slider(
            label, min_value=mn, max_value=mx, value=med, step=step,
            format="%.3f")

# Categorical dropdowns
for feat in CATEGORICAL_FEATURES:
    options = cat_options.get(feat, [])
    label   = feat.replace("_", " ").title()
    if options:
        # Pick the most frequent value as default
        mode_val = X_all[feat].mode().iloc[0] if not X_all[feat].mode().empty else options[0]
        default_idx = options.index(mode_val) if mode_val in options else 0
        user_inputs[feat] = st.sidebar.selectbox(label, options=options, index=default_idx)
    else:
        user_inputs[feat] = st.sidebar.text_input(label, value="")

predict_btn = st.sidebar.button("🔮 Predict Price", type="primary", use_container_width=True)

# =============================================================================
# MAIN
# =============================================================================
st.title("⛽ Global Fuel Price Prediction Dashboard")
st.markdown(
    "Predict **petrol price (USD/liter)** using the best model registered in "
    "**MLflow Phase 2**. Confidence intervals are computed via OLS (bonus ✅)."
)

tab_predict, tab_insights, tab_about = st.tabs(
    ["📈 Prediction", "🔍 Key Insights", "ℹ️ About"]
)

# ── Tab 1 – Prediction ────────────────────────────────────────────────────────
with tab_predict:
    input_df = pd.DataFrame([user_inputs])

    if predict_btn:
        try:
            model      = load_model()
            prediction = float(model.predict(input_df)[0])

            # Bonus: CI from OLS on numeric features
            has_ci = False
            if ols_model is not None and NUMERIC_FEATURES and num_pipe is not None:
                try:
                    x_num    = num_pipe.transform(input_df[NUMERIC_FEATURES])
                    x_sm     = sm.add_constant(x_num, has_constant="add")
                    pred_sm  = ols_model.get_prediction(x_sm)
                    ci_frame = pred_sm.summary_frame(alpha=0.05)
                    ols_pred = float(ci_frame["mean"].iloc[0])
                    ci_lower = float(ci_frame["mean_ci_lower"].iloc[0])
                    ci_upper = float(ci_frame["mean_ci_upper"].iloc[0])
                    has_ci   = True
                except Exception:
                    pass

            st.success("Prediction ready!")

            c1, c2, c3 = st.columns(3)
            c1.metric("🎯 Predicted Price (ML model)", f"${prediction:.4f} / liter")
            if has_ci:
                c2.metric("📉 95% CI Lower", f"${max(ci_lower, 0):.4f}")
                c3.metric("📈 95% CI Upper", f"${ci_upper:.4f}")

                st.info(
                    f"**OLS estimate:** ${ols_pred:.4f} / liter  |  "
                    f"**95% CI:** ${max(ci_lower,0):.4f} – ${ci_upper:.4f}"
                )

                # Visual: predicted vs CI band
                fig, ax = plt.subplots(figsize=(7, 2.8))
                ax.barh(["ML Pipeline"], [prediction],
                        color="#2196F3", height=0.35,
                        label=f"ML model: ${prediction:.4f}")
                err_low  = max(ols_pred - ci_lower, 0)
                err_high = ci_upper - ols_pred
                ax.barh(["OLS + 95% CI"], [ols_pred],
                        xerr=[[err_low], [err_high]],
                        color="#4CAF50", height=0.35, capsize=8,
                        error_kw={"elinewidth": 2},
                        label=f"OLS: ${ols_pred:.4f}")
                ax.set_xlabel("petrol_usd_liter")
                ax.set_title("ML Prediction vs OLS 95% Confidence Interval",
                             fontweight="bold")
                ax.legend(fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info(f"Predicted price: **${prediction:.4f} / liter**")

        except Exception as e:
            st.error(
                "Could not load the MLflow model. Make sure the MLflow server "
                f"is running (`mlflow ui`) and Phase 2 has been executed.  \n\n`{e}`"
            )
    else:
        st.info("👈 Adjust the inputs in the sidebar, then click **Predict Price**.")

    with st.expander("Current input values"):
        st.dataframe(
            pd.DataFrame(user_inputs, index=["value"]).T.rename(columns={"value": "Input"}),
            use_container_width=True,
        )

# ── Tab 2 – Key Insights ──────────────────────────────────────────────────────
with tab_insights:

    # ── Plot 1: feature coefficients / importance ─────────────────────────────
    st.subheader("Feature Importance / Coefficients")

    try:
        model = load_model()
        est   = model.named_steps.get("regressor", model.named_steps.get("model", None))
        preprocessor = model.named_steps["preprocessor"]

        if est is not None and hasattr(est, "feature_importances_"):
            # Tree-based model
            feat_names   = preprocessor.get_feature_names_out()
            importances  = est.feature_importances_
            title        = f"Feature Importance ({meta['best_model_name']})"
            ylabel       = "Gini Importance"
        elif est is not None and hasattr(est, "coef_"):
            # Linear model — use absolute standardised coefficients
            feat_names  = preprocessor.get_feature_names_out()
            importances = np.abs(est.coef_)
            importances = importances / importances.sum()
            title       = f"Normalised |Coefficient| ({meta['best_model_name']})"
            ylabel      = "Relative magnitude"
        else:
            feat_names  = np.array(NUMERIC_FEATURES)
            importances = np.ones(len(NUMERIC_FEATURES)) / max(len(NUMERIC_FEATURES), 1)
            title       = "Feature weights (uniform fallback)"
            ylabel      = ""

        top_n  = min(20, len(feat_names))
        idx    = np.argsort(importances)[::-1][:top_n]
        colors = ["#4CAF50" if i == idx[0] else "#2196F3" for i in range(top_n)]

        fig, ax = plt.subplots(figsize=(12, 4))
        bars = ax.bar(range(top_n),
                      [importances[i] for i in idx],
                      color=colors, edgecolor="white")
        ax.set_xticks(range(top_n))
        ax.set_xticklabels([feat_names[i] for i in idx],
                           rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        for bar, val in zip(bars, [importances[i] for i in idx]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7.5)
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("🟩 Highest-importance feature  🟦 Other features")

    except Exception as e:
        st.warning(f"Could not render feature importance: {e}")

    # ── Plot 2: target price distribution ─────────────────────────────────────
    st.subheader(f"Distribution of `{TARGET}` in Training Data")
    try:
        fig2, ax2 = plt.subplots(figsize=(9, 3.5))
        ax2.hist(y_all, bins=50, color="#2196F3", edgecolor="white", alpha=0.85)
        ax2.axvline(y_all.median(), color="tomato", linestyle="--",
                    label=f"Median: ${y_all.median():.3f}")
        ax2.axvline(y_all.mean(), color="orange", linestyle=":",
                    label=f"Mean: ${y_all.mean():.3f}")
        ax2.set_xlabel("Price (USD/liter)")
        ax2.set_ylabel("Count")
        ax2.set_title("petrol_usd_liter Distribution", fontweight="bold")
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)
    except Exception as e:
        st.warning(f"Distribution plot error: {e}")

    # ── Plot 3: price over time (if date_year available) ──────────────────────
    if "date_year" in X_all.columns:
        st.subheader("Average Petrol Price by Year")
        try:
            yearly = pd.DataFrame({"year": X_all["date_year"], "price": y_all})
            yearly = yearly.groupby("year")["price"].mean().reset_index()
            fig3, ax3 = plt.subplots(figsize=(9, 3.5))
            ax3.plot(yearly["year"], yearly["price"], "o-", color="#2196F3", lw=2, ms=6)
            ax3.set_xlabel("Year")
            ax3.set_ylabel("Avg petrol_usd_liter")
            ax3.set_title("Global Average Petrol Price Trend", fontweight="bold")
            ax3.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig3)
        except Exception as e:
            st.warning(f"Time trend plot error: {e}")

# ── Tab 3 – About ─────────────────────────────────────────────────────────────
with tab_about:
    st.markdown(f"""
## MLOps Project – Phase 3

| Item | Detail |
|------|--------|
| **Dataset** | `{DATASET_FILE}` (Global Fuel Prices 2020–2026) |
| **Target** | `{TARGET}` |
| **Best model** | `{meta.get('best_model_name', '—')}` |
| **Registry** | `{REGISTRY_NAME}` (Production) |
| **Phase 1** | Scikit-learn → ML.js → GitHub Pages |
| **Phase 2** | MLflow experiment tracking + model registry |
| **Phase 3** | This Streamlit app |
| **Bonus** | 95% prediction CIs via statsmodels OLS |

### Architecture

```
global_fuel_prices_2020_2026.csv
        │
        ├── pipeline.py  (Phase 1 preprocessing, shared)
        │
        └── phase2.ipynb
                │
                ├── MLflow Tracking Server  (localhost:5000)
                │       └── Model Registry: {REGISTRY_NAME}/Production
                │
                └── model_meta.json  ──►  app.py  (this file)
                                              │
                                   mlflow.sklearn.load_model()
                                              │
                               sidebar inputs ──► predict() ──► st.metric + CI chart
```

### How to run

```bash
# Terminal 1 – MLflow server
mlflow ui

# Terminal 2 – this app (from the same directory as pipeline.py)
streamlit run app.py
```
""")
