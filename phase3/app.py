"""
Phase 3 Streamlit app for the MLOps project.

This dashboard:
- loads the best model from the local MLflow Model Registry
- provides user inputs for all model features
- predicts petrol price in USD per liter
- optionally shows a 95% confidence interval using statsmodels OLS
- visualizes model insights and dataset trends
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - dependency check
    mlflow = None
    MLFLOW_IMPORT_ERROR = exc

try:
    import statsmodels.api as sm
    STATSMODELS_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - dependency check
    sm = None
    STATSMODELS_IMPORT_ERROR = exc


st.set_page_config(page_title="Fuel Price Predictor", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
PHASE2_DIR = PROJECT_DIR / "phase2"
META_PATH = PHASE2_DIR / "model_meta.json"
TRACKING_URI = "http://localhost:5000"

if str(PHASE2_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE2_DIR))

from pipeline import (  # type: ignore  # noqa: E402
    DATA_FILENAME,
    add_date_features,
    convert_numeric_like_objects,
    find_project_root,
    infer_columns_to_drop,
    resolve_target_column,
)


@st.cache_data
def load_meta() -> dict[str, Any]:
    with META_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


meta = load_meta()
TARGET = meta["target_column"]
NUMERIC_FEATURES = meta["numeric_features"]
CATEGORICAL_FEATURES = meta["categorical_features"]
REGISTRY_NAME = meta["registry_name"]
DATASET_FILE = meta.get("dataset_file", DATA_FILENAME)
BEST_MODEL_NAME = meta.get("best_model_name", "Unknown")


def mlflow_ready() -> tuple[bool, str]:
    if MLFLOW_IMPORT_ERROR is not None:
        return False, f"mlflow import failed: {MLFLOW_IMPORT_ERROR}"
    return True, ""


def statsmodels_ready() -> tuple[bool, str]:
    if STATSMODELS_IMPORT_ERROR is not None:
        return False, f"statsmodels import failed: {STATSMODELS_IMPORT_ERROR}"
    return True, ""


@st.cache_data
def load_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    project_root = find_project_root()
    csv_path = project_root / DATASET_FILE

    raw_df = pd.read_csv(csv_path)
    working_df = convert_numeric_like_objects(raw_df)
    date_columns = add_date_features(working_df)

    target_col = resolve_target_column(working_df)
    columns_to_drop = infer_columns_to_drop(working_df, target_col, date_columns)
    working_df = working_df.drop(columns=columns_to_drop, errors="ignore")
    working_df = working_df.dropna(subset=[target_col]).copy()

    y = pd.to_numeric(working_df[target_col], errors="coerce")
    X = working_df.drop(columns=[target_col]).copy()
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()
    return X, y, raw_df


@st.cache_resource
def load_model():
    ready, message = mlflow_ready()
    if not ready:
        raise RuntimeError(message)
    assert mlflow is not None
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.sklearn.load_model(f"models:/{REGISTRY_NAME}/Production")


@st.cache_resource
def fit_ols_for_ci(X: pd.DataFrame, y: pd.Series):
    ready, _ = statsmodels_ready()
    if not ready or not NUMERIC_FEATURES:
        return None, None

    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=meta.get("test_size", 0.2),
        random_state=meta.get("random_state", 42),
    )

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    X_train_numeric = numeric_pipe.fit_transform(X_train[NUMERIC_FEATURES])
    X_train_sm = sm.add_constant(X_train_numeric)
    ols_model = sm.OLS(y_train.values, X_train_sm).fit()
    return ols_model, numeric_pipe


@st.cache_data
def categorical_options(X: pd.DataFrame) -> dict[str, list[Any]]:
    options: dict[str, list[Any]] = {}
    for feature in CATEGORICAL_FEATURES:
        if feature in X.columns:
            values = X[feature].dropna().unique().tolist()
            options[feature] = sorted(values)
    return options


def build_input_frame(X: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("User Inputs")
    st.sidebar.caption("Adjust the feature values and generate a local prediction.")

    user_values: dict[str, Any] = {}

    for feature in NUMERIC_FEATURES:
        series = pd.to_numeric(X[feature], errors="coerce").dropna()
        label = feature.replace("_", " ").title()
        minimum = float(series.min())
        maximum = float(series.max())
        default = float(series.median())

        if feature.startswith("date_"):
            user_values[feature] = float(
                st.sidebar.slider(
                    label,
                    min_value=int(minimum),
                    max_value=int(maximum),
                    value=int(default),
                    step=1,
                )
            )
        else:
            step = round((maximum - minimum) / 100, 4) or 0.01
            user_values[feature] = st.sidebar.slider(
                label,
                min_value=minimum,
                max_value=maximum,
                value=default,
                step=step,
                format="%.3f",
            )

    options = categorical_options(X)
    for feature in CATEGORICAL_FEATURES:
        values = options.get(feature, [])
        label = feature.replace("_", " ").title()
        if values:
            mode = X[feature].mode().iloc[0] if not X[feature].mode().empty else values[0]
            default_index = values.index(mode) if mode in values else 0
            user_values[feature] = st.sidebar.selectbox(label, values, index=default_index)
        else:
            user_values[feature] = st.sidebar.text_input(label, value="")

    return pd.DataFrame([user_values])


def render_prediction(model, input_df: pd.DataFrame, ols_model, numeric_pipe) -> None:
    prediction = float(model.predict(input_df)[0])

    ci_available = False
    ci_lower = ci_upper = ols_estimate = None
    if ols_model is not None and numeric_pipe is not None and NUMERIC_FEATURES:
        try:
            transformed = numeric_pipe.transform(input_df[NUMERIC_FEATURES])
            transformed_sm = sm.add_constant(transformed, has_constant="add")
            summary = ols_model.get_prediction(transformed_sm).summary_frame(alpha=0.05)
            ols_estimate = float(summary["mean"].iloc[0])
            ci_lower = float(summary["mean_ci_lower"].iloc[0])
            ci_upper = float(summary["mean_ci_upper"].iloc[0])
            ci_available = True
        except Exception:
            ci_available = False

    st.success("Prediction created successfully.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted price", f"${prediction:.4f} / liter")
    if ci_available and ci_lower is not None and ci_upper is not None:
        col2.metric("95% CI lower", f"${max(ci_lower, 0):.4f}")
        col3.metric("95% CI upper", f"${ci_upper:.4f}")
        st.info(
            f"OLS estimate: ${ols_estimate:.4f} / liter | "
            f"95% CI: ${max(ci_lower, 0):.4f} to ${ci_upper:.4f}"
        )

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.barh(["MLflow model"], [prediction], color="#1f77b4", height=0.35)
        ax.barh(
            ["OLS with 95% CI"],
            [ols_estimate],
            xerr=[[max(ols_estimate - ci_lower, 0)], [ci_upper - ols_estimate]],
            color="#2ca02c",
            height=0.35,
            capsize=8,
        )
        ax.set_xlabel("petrol_usd_liter")
        ax.set_title("Prediction and confidence interval")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Confidence interval unavailable. Install statsmodels to enable the bonus output.")


def render_feature_importance(model) -> None:
    st.subheader("Feature importance or coefficient overview")

    estimator = model.named_steps.get("regressor", None)
    preprocessor = model.named_steps["preprocessor"]
    feature_names = np.array(preprocessor.get_feature_names_out())

    if hasattr(estimator, "feature_importances_"):
        values = np.array(estimator.feature_importances_)
        title = f"Feature importance ({BEST_MODEL_NAME})"
        ylabel = "Importance"
    elif hasattr(estimator, "coef_"):
        coefficients = np.abs(np.array(estimator.coef_))
        values = coefficients / coefficients.sum()
        title = f"Normalized absolute coefficients ({BEST_MODEL_NAME})"
        ylabel = "Relative magnitude"
    else:
        values = np.ones(len(feature_names)) / max(len(feature_names), 1)
        title = "Fallback feature weights"
        ylabel = "Weight"

    top_n = min(20, len(feature_names))
    indices = np.argsort(values)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#2ca02c" if i == indices[0] else "#1f77b4" for i in range(top_n)]
    bars = ax.bar(range(top_n), values[indices], color=colors, edgecolor="white")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(feature_names[indices], rotation=35, ha="right", fontsize=8)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    for bar, value in zip(bars, values[indices]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01, f"{value:.4f}",
                ha="center", va="bottom", fontsize=7.5)
    plt.tight_layout()
    st.pyplot(fig)


def render_dataset_insights(X: pd.DataFrame, y: pd.Series) -> None:
    st.subheader(f"Distribution of {TARGET}")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.hist(y, bins=50, color="#1f77b4", edgecolor="white", alpha=0.85)
    ax.axvline(y.median(), color="tomato", linestyle="--", label=f"Median: ${y.median():.3f}")
    ax.axvline(y.mean(), color="orange", linestyle=":", label=f"Mean: ${y.mean():.3f}")
    ax.set_xlabel("Price (USD/liter)")
    ax.set_ylabel("Count")
    ax.set_title("Target distribution", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    if "date_year" in X.columns:
        st.subheader("Average petrol price by year")
        yearly = pd.DataFrame({"year": X["date_year"], "price": y}).groupby("year")["price"].mean().reset_index()
        fig2, ax2 = plt.subplots(figsize=(9, 3.5))
        ax2.plot(yearly["year"], yearly["price"], "o-", color="#1f77b4", lw=2, ms=6)
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Average petrol_usd_liter")
        ax2.set_title("Global average petrol price trend", fontweight="bold")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)


def render_about_tab() -> None:
    mlflow_ok, mlflow_message = mlflow_ready()
    stats_ok, stats_message = statsmodels_ready()

    st.markdown("## Phase 3 Overview")
    st.markdown(
        f"""
| Item | Detail |
|------|--------|
| Dataset | `{DATASET_FILE}` |
| Target | `{TARGET}` |
| Best model | `{BEST_MODEL_NAME}` |
| Registry entry | `{REGISTRY_NAME}` / Production |
| Metadata file | `{META_PATH}` |
| MLflow import | `{'OK' if mlflow_ok else 'Missing'}` |
| statsmodels import | `{'OK' if stats_ok else 'Missing'}` |
"""
    )

    st.markdown("### How to run")
    st.code(
        "cd c:\\Users\\maxch\\OneDrive\\Desktop\\HTL\\5CHIF\\DSAI\\MLOPS_Project\\phase3\n"
        "streamlit run app.py",
        language="powershell",
    )

    st.markdown("### Notes")
    st.markdown(
        "- The app loads the model from the local MLflow Model Registry.\n"
        "- The MLflow server must be available on `http://localhost:5000`.\n"
        "- Confidence intervals are optional and depend on `statsmodels`."
    )

    if not mlflow_ok:
        st.error(f"MLflow dependency missing: {mlflow_message}")
    if not stats_ok:
        st.warning(f"statsmodels dependency missing: {stats_message}")


def main() -> None:
    st.title("Global Fuel Price Prediction Dashboard")
    st.markdown(
        "This Streamlit app uses the best model registered in MLflow during Phase 2 and "
        "predicts `petrol_usd_liter` from interactive feature inputs."
    )

    X_all, y_all, _ = load_dataset()
    input_df = build_input_frame(X_all)
    ols_model, numeric_pipe = fit_ols_for_ci(X_all, y_all)

    predict_tab, insights_tab, about_tab = st.tabs(["Prediction", "Key Insights", "About"])

    with predict_tab:
        if st.sidebar.button("Predict price", type="primary", use_container_width=True):
            try:
                model = load_model()
                render_prediction(model, input_df, ols_model, numeric_pipe)
            except Exception as exc:
                st.error(
                    "Could not load the registered MLflow model. Make sure the local MLflow "
                    f"server is running and the model registry entry exists.\n\n{exc}"
                )
        else:
            st.info("Adjust the sidebar inputs and click `Predict price`.")

        with st.expander("Current input values"):
            st.dataframe(
                input_df.T.rename(columns={0: "Input"}),
                use_container_width=True,
            )

    with insights_tab:
        try:
            model = load_model()
            render_feature_importance(model)
        except Exception as exc:
            st.warning(f"Feature importance unavailable: {exc}")
        render_dataset_insights(X_all, y_all)

    with about_tab:
        render_about_tab()


if __name__ == "__main__":
    main()
