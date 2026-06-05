import os
import numpy as np

import joblib
import mlflow
import mlflow.sklearn

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error

from scipy.stats import randint
from src.data.load_data import load_data

def feature_engineering(df):
    df = df.copy()

    if "GrLivArea" in df.columns and "GarageArea" in df.columns:
        df["TotalArea"] = df["GrLivArea"] + df["GarageArea"]

    if "YearBuilt" in df.columns:
        df["HouseAge"] = 2010 - df["YearBuilt"]

    # log skewed features (safe check)
    for col in ["GrLivArea", "GarageArea", "TotalArea"]:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    return df

def missing_values_handling(df):
    df = df.copy()
    # Drop features with more than 50% missing values
    missing_percent = df.isnull().sum() / len(df)
    df = df.drop(columns=missing_percent[missing_percent > 0.5].index)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def tune_model(name, pipeline, param_grid, X_train, y_train):
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=10,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    search.fit(X_train, y_train)

    return search.best_estimator_, -search.best_score_


def train():

    # Set up MLflow experiment
    mlflow.set_experiment("house_price_prediction")
    
    # Load Train and Test data
    df = load_data()
    
    # Feature engineering and missing value handling
    df = missing_values_handling(df)
    df = feature_engineering(df)
    
    TARGET = "SalePrice"

    # Training data
    y = np.log1p(df[TARGET])
    X = df.drop(columns=[TARGET])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    # Define preprocessing pipelines for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
     ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # Define models and their hyperparameter grids for tuning
    models = {
        "random_forest": (
            Pipeline([("preprocessor", preprocessor),
                      ("model", RandomForestRegressor(random_state=42))]),
            {
                "model__n_estimators": randint(100, 500),
                "model__max_depth": [None, 10, 20, 30],
                "model__min_samples_split": randint(2, 10)
            }
        ),

        "gradient_boosting": (
            Pipeline([("preprocessor", preprocessor),
                      ("model", GradientBoostingRegressor(random_state=42))]),
            {
                "model__n_estimators": randint(100, 400),
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": randint(2, 5)
            }
        ),

        "xgboost": (
            Pipeline([("preprocessor", preprocessor),
                      ("model", XGBRegressor(
                          random_state=42,
                          verbosity=0,
                          n_jobs=-1
                      ))]),
            {
                "model__n_estimators": randint(100, 400),
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": randint(3, 8)
            }
        )
    }

    # Initialize variables to track the best model and its performance
    best_model = None
    best_rmse = float("inf")
    best_name = None
    best_pipelines = {}

    # Model directory
    os.makedirs("models", exist_ok=True)

    # Hyperparameter tuning
    for name, (model, params) in models.items():

        with mlflow.start_run(run_name=name):

            best_pipeline, cv_rmse = tune_model(
                name, model, params, X_train, y_train
            )            

            preds = best_pipeline.predict(X_val)
            rmse = root_mean_squared_error(y_val, preds)

            # MLflow logging
            mlflow.log_param("model", name)
            mlflow.log_metric("cv_rmse", cv_rmse)
            mlflow.log_metric("test_rmse", rmse)

            mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

            print(f"\n====================")
            print(f"Tuned {name} model:")
            print(f"CV RMSE: {cv_rmse:.3f} | Test RMSE: {rmse:.3f}")
            print(f"====================\n")

            best_pipelines[name] = (best_pipeline, rmse)

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = best_pipeline
                best_name = name

    for name, (best_pipeline, rmse) in best_pipelines.items():
        best_pipeline.fit(X, y)
        output_path = os.path.join("models", f"{name}.pkl")
        training_rmse = root_mean_squared_error(y, best_pipeline.predict(X)) # Train RMSE
        joblib.dump(best_pipeline, output_path)

        print("\n====================")
        print(f"Saved retrained {name} model to: {output_path}")
        print(f"Training RMSE: {training_rmse:.3f}")
        print("====================\n")


    with mlflow.start_run(run_name="best_model_summary"):
        mlflow.log_param("best_model", best_name)
        mlflow.log_metric("best_rmse", best_rmse)
            
    print("\n====================")
    print(f"Best model: {best_name}")
    print(f"Best RMSE: {best_rmse:.3f}")
    print("Saved all tuned models to: models/")
    print("====================\n")

if __name__ == "__main__":
    train()