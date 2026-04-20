import argparse
import shutil
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILENAME = "global_fuel_prices_2020_2026.csv"
DEFAULT_TARGET_COLUMN = "petrol_usd_liter"
FUEL_PRICE_COLUMNS = {"petrol_usd_liter", "diesel_usd_liter", "lpg_usd_liter"}


def find_project_root(start_path: Path | None = None) -> Path:
    current = (start_path or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / DATA_FILENAME).exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {DATA_FILENAME} starting from {current}. "
        "Open the notebook inside the project or run the script from the project tree."
    )


def resolve_notebook_dir(project_dir: Path) -> Path:
    phase1_dir = project_dir / "phase1"
    return phase1_dir if phase1_dir.exists() else project_dir


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)


def add_date_features(df: pd.DataFrame) -> list[str]:
    date_columns = [column for column in df.columns if "date" in column.lower()]
    for date_column in date_columns:
        converted = pd.to_datetime(df[date_column], errors="coerce")
        if converted.notna().any():
            df[date_column] = converted
            df[f"{date_column}_year"] = converted.dt.year
            df[f"{date_column}_month"] = converted.dt.month
            df[f"{date_column}_weekofyear"] = converted.dt.isocalendar().week.astype("float64")
    return date_columns


def convert_numeric_like_objects(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    converted_df = df.copy()
    object_like_columns = converted_df.select_dtypes(include=["object", "string"]).columns
    for column in object_like_columns:
        converted = pd.to_numeric(converted_df[column], errors="coerce")
        if converted.notna().mean() >= threshold:
            converted_df[column] = converted
    return converted_df


def resolve_target_column(df: pd.DataFrame, requested: str | None = None) -> str:
    columns = df.columns.tolist()
    if requested:
        if requested not in columns:
            available = ", ".join(columns)
            raise ValueError(f"Target column '{requested}' does not exist. Available columns: {available}")
        return requested

    for candidate in (DEFAULT_TARGET_COLUMN, "diesel_usd_liter", "lpg_usd_liter"):
        if candidate in columns:
            return candidate

    numeric_candidates = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    if numeric_candidates:
        return numeric_candidates[0]

    raise ValueError("No suitable target column could be inferred from the dataset.")


def infer_columns_to_drop(df: pd.DataFrame, target_column: str, date_columns: list[str]) -> list[str]:
    columns_to_drop = list(date_columns)
    for fuel_column in sorted(FUEL_PRICE_COLUMNS - {target_column}):
        if fuel_column in df.columns:
            columns_to_drop.append(fuel_column)
    return columns_to_drop


def build_model(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_features:
        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", make_one_hot_encoder()),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_features))

    if not transformers:
        raise ValueError("The dataset does not contain usable input features after preprocessing.")

    preprocessor = ColumnTransformer(transformers)
    return Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])


def export_model(result: dict) -> Path:
    docs_dir = result["docs_dir"]
    docs_dir.mkdir(parents=True, exist_ok=True)
    webapp_dir = result["notebook_dir"] / "webapp"

    model = result["model"]
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]
    feature_frame = result["X"]

    export_data = {
        "modelName": "Global Fuel Price Linear Regression",
        "targetColumn": result["target_column"],
        "intercept": float(regressor.intercept_),
        "metrics": result["metrics"],
        "dataset": {
            "fileName": DATA_FILENAME,
            "rows": int(len(result["df"])),
            "trainingRows": int(len(result["X_train"])),
            "testRows": int(len(result["X_test"])),
        },
        "inputSchema": {
            "numeric": [],
            "categorical": [],
        },
        "numericFeatures": [],
        "categoricalFeatures": [],
        "coefficients": {},
    }

    feature_names = preprocessor.get_feature_names_out()
    for name, coefficient in zip(feature_names, regressor.coef_):
        export_data["coefficients"][name] = float(coefficient)

    if result["numeric_features"]:
        numeric_transformer = preprocessor.named_transformers_["num"]
        num_imputer = numeric_transformer.named_steps["imputer"]
        scaler = numeric_transformer.named_steps["scaler"]
        for index, column in enumerate(result["numeric_features"]):
            statistic = num_imputer.statistics_[index]
            series = pd.to_numeric(feature_frame[column], errors="coerce")
            default_value = float(series.median()) if series.notna().any() else float(statistic or 0.0)
            export_data["numericFeatures"].append(
                {
                    "name": column,
                    "imputer": float(statistic) if pd.notna(statistic) else 0.0,
                    "mean": float(scaler.mean_[index]),
                    "scale": float(scaler.scale_[index]) if scaler.scale_[index] != 0 else 1.0,
                }
            )
            export_data["inputSchema"]["numeric"].append(
                {
                    "name": column,
                    "label": column.replace("_", " ").title(),
                    "min": float(series.min()) if series.notna().any() else default_value,
                    "max": float(series.max()) if series.notna().any() else default_value,
                    "default": default_value,
                    "step": 1.0 if column.startswith("date_") else 0.1,
                }
            )

    if result["categorical_features"]:
        categorical_transformer = preprocessor.named_transformers_["cat"]
        cat_imputer = categorical_transformer.named_steps["imputer"]
        encoder = categorical_transformer.named_steps["encoder"]
        cat_modes = cat_imputer.statistics_.tolist()

        for index, column in enumerate(result["categorical_features"]):
            all_categories = [str(value) for value in encoder.categories_[index].tolist()]
            base_category = all_categories[0] if all_categories else None
            kept_categories = all_categories[1:] if len(all_categories) > 1 else []
            export_data["categoricalFeatures"].append(
                {
                    "name": column,
                    "mode": str(cat_modes[index]) if pd.notna(cat_modes[index]) else (base_category or ""),
                    "baseCategory": base_category,
                    "keptCategories": kept_categories,
                }
            )
            export_data["inputSchema"]["categorical"].append(
                {
                    "name": column,
                    "label": column.replace("_", " ").title(),
                    "default": str(cat_modes[index]) if pd.notna(cat_modes[index]) else (base_category or ""),
                    "options": all_categories,
                }
            )

    export_path = docs_dir / "model.json"
    with export_path.open("w", encoding="utf-8") as file:
        json.dump(export_data, file, indent=2, ensure_ascii=False)

    if webapp_dir.exists():
        webapp_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(export_path, webapp_dir / "model.json")

    return export_path


def create_diagnostic_plots(result: dict) -> None:
    y_test = result["y_test"]
    y_pred = result["y_pred"]
    residuals = y_test - y_pred

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_analysis(
    target_column: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    create_plots: bool = True,
    display_summary: bool = True,
) -> dict:
    project_dir = find_project_root()
    notebook_dir = resolve_notebook_dir(project_dir)
    data_path = project_dir / DATA_FILENAME
    docs_dir = notebook_dir / "docs"

    df = pd.read_csv(data_path)
    working_df = convert_numeric_like_objects(df)
    date_columns = add_date_features(working_df)

    target_column = resolve_target_column(working_df, target_column)
    columns_to_drop = infer_columns_to_drop(working_df, target_column, date_columns)

    working_df = working_df.drop(columns=columns_to_drop, errors="ignore")
    working_df = working_df.dropna(subset=[target_column]).copy()

    y = pd.to_numeric(working_df[target_column], errors="coerce")
    X = working_df.drop(columns=[target_column]).copy()

    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    all_missing_columns = [column for column in X.columns if X[column].isna().all()]
    X = X.drop(columns=all_missing_columns, errors="ignore")

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    if X.empty:
        raise ValueError("No training data is left after preprocessing.")
    if not numeric_features and not categorical_features:
        raise ValueError("No usable features are available for training.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = build_model(numeric_features, categorical_features)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }

    result = {
        "project_dir": project_dir,
        "notebook_dir": notebook_dir,
        "data_path": data_path,
        "docs_dir": docs_dir,
        "target_column": target_column,
        "columns_to_drop": columns_to_drop,
        "df": df,
        "working_df": working_df,
        "X": X,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "model": model,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "metrics": metrics,
    }

    export_path = export_model(result)
    result["export_path"] = export_path

    if display_summary:
        print(f"Project dir: {project_dir}")
        print(f"Data path   : {data_path}")
        print(f"Target      : {target_column}")
        print(f"Dropped     : {columns_to_drop}")
        print(f"Rows used   : {len(X)}")
        print(f"Numeric     : {numeric_features}")
        print(f"Categorical : {categorical_features}")
        print(f"MAE         : {metrics['mae']:.4f}")
        print(f"RMSE        : {metrics['rmse']:.4f}")
        print(f"R2          : {metrics['r2']:.4f}")
        print(f"Saved model : {export_path}")

    if create_plots:
        create_diagnostic_plots(result)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 1 fuel price regression pipeline.")
    parser.add_argument("--target", default=None, help="Optional target column to predict.")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib plots. Useful for automated runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_analysis(target_column=arguments.target, create_plots=not arguments.no_plots)
