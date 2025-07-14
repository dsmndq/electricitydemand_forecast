import pandas as pd
import numpy as np
import xgboost as xgb
import statsmodels.api as sm
import pickle
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_featured_data(path: Path) -> pd.DataFrame:
    """
    Loads the featured dataset, parses the datetime index, and sorts it.

    Args:
        path (Path): The file path to the featured data CSV.

    Returns:
        pd.DataFrame: DataFrame with a DatetimeIndex, ready for modeling.
    """
    print(f"Loading featured data from {path}...")
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found at {path}. "
            "Please run 'feature_engineering.py' first to generate it."
        )
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    df = df.sort_index()
    print("Featured data loaded successfully.")
    return df


def train_sarima(train_data: pd.Series):
    """
    Fits a SARIMA model to the training data.

    Note: SARIMA can be very slow on large datasets. This function uses
    daily resampled data for faster training.

    Args:
        train_data (pd.Series): The training time series data (e.g., demand_mw).

    Returns:
        SARIMAXResults: The fitted SARIMA model object.
    """
    print("\n--- Training SARIMA Model ---")
    print("Fitting SARIMA(1,1,1)(1,1,0,7)... This may take a few minutes.")
    # A simple SARIMA model order (p,d,q)(P,D,Q,m) for daily data with weekly seasonality.
    model = sm.tsa.SARIMAX(
        train_data,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 0, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    print("SARIMA model training complete.")
    print(results.summary())
    return results


def train_xgboost(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Trains an XGBoost regression model using a feature-rich dataset.

    Args:
        train_df (pd.DataFrame): The training dataframe with features and target.
        test_df (pd.DataFrame): The testing dataframe for early stopping validation.

    Returns:
        xgb.XGBRegressor: The trained XGBoost model object.
    """
    print("\n--- Training XGBoost Model ---")

    features = [col for col in train_df.columns if col != 'demand_mw']
    target = 'demand_mw'

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        objective='reg:squarederror',
        early_stopping_rounds=50,
        n_jobs=-1  # Use all available CPU cores
    )

    print("Fitting XGBoost model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100
    )
    print("XGBoost model training complete.")
    return model


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    """
    Calculates and prints MAE and RMSE for a model's predictions.

    Args:
        y_true (np.ndarray): The true target values.
        y_pred (np.ndarray): The predicted values from the model.
        model_name (str): The name of the model for clear reporting.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n--- {model_name} Evaluation ---")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("-" * (len(model_name) + 22))


def save_model(model, path: Path):
    """
    Saves a trained model object to a file using pickle.

    Args:
        model: The trained model object (e.g., from Scikit-learn, XGBoost).
        path (Path): The file path where the model will be saved.
    """
    print(f"Saving model to {path}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully.")


def main():
    """
    Main function to run the model training and evaluation pipeline.
    """
    input_path = Path("featured_electricity_data.csv")
    models_dir = Path("models")

    df = load_featured_data(input_path)
    df_clean = df.dropna()
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Data shape after dropping NaNs: {df_clean.shape}")

    # --- Data Splitting (Time-based) ---
    split_date = df_clean.index.max() - pd.DateOffset(months=1)
    train_df = df_clean.loc[df_clean.index <= split_date].copy()
    test_df = df_clean.loc[df_clean.index > split_date].copy()
    print(f"\nTrain data: {train_df.index.min()} to {train_df.index.max()}")
    print(f"Test data:  {test_df.index.min()} to {test_df.index.max()}")

    # --- SARIMA Model Workflow (on daily data) ---
    sarima_train_daily = train_df['demand_mw'].resample('D').mean()
    sarima_test_daily = test_df['demand_mw'].resample('D').mean()
    sarima_model = train_sarima(sarima_train_daily)
    sarima_pred = sarima_model.get_forecast(steps=len(sarima_test_daily)).predicted_mean
    evaluate_model(sarima_test_daily, sarima_pred, "SARIMA (Daily)")
    save_model(sarima_model, models_dir / "sarima_model.pkl")

    # --- XGBoost Model Workflow (on hourly data) ---
    xgb_model = train_xgboost(train_df, test_df)
    features = [col for col in test_df.columns if col != 'demand_mw']
    xgb_pred = xgb_model.predict(test_df[features])
    evaluate_model(test_df['demand_mw'], xgb_pred, "XGBoost (Hourly)")
    save_model(xgb_model, models_dir / "xgboost_model.pkl")


if __name__ == "__main__":
    main()