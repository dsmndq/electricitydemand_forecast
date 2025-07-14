import pandas as pd
import pickle
from pathlib import Path

# Import feature engineering functions from the other script.
# This ensures that the features are created in exactly the same way.
from feature_engineering import (
    create_time_features,
    create_lag_features,
    create_rolling_features,
    create_weather_features,
)


def load_model(path: Path):
    """
    Loads a trained model from a pickle file.

    Args:
        path (Path): The file path to the saved model.

    Returns:
        The loaded model object.
    """
    print(f"Loading model from {path}...")
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. "
            "Please run 'train_models.py' first to generate it."
        )
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
    return model


def create_future_dataframe(last_timestamp: pd.Timestamp, days: int) -> pd.DataFrame:
    """
    Creates a future DataFrame for a given number of days.

    Args:
        last_timestamp (pd.Timestamp): The last timestamp from the historical data.
        days (int): The number of future days to create.

    Returns:
        pd.DataFrame: A DataFrame with a future DatetimeIndex.
    """
    print(f"Creating future DataFrame for the next {days} days...")
    future_dates = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=days * 24,
        freq='h'
    )
    future_df = pd.DataFrame(index=future_dates)
    future_df.index.name = 'timestamp'
    return future_df


def predict_future(model, historical_df: pd.DataFrame, weather_df: pd.DataFrame, days_to_predict: int = 14):
    """
    Creates features for future dates and predicts demand using the trained model.

    Args:
        model: The trained XGBoost model.
        historical_df (pd.DataFrame): DataFrame with historical featured data.
        weather_df (pd.DataFrame): DataFrame with original weather data for forecasting.
        days_to_predict (int): Number of future days to predict.

    Returns:
        pd.DataFrame: A DataFrame containing the future predictions.
    """
    print("\n--- Starting Future Prediction Workflow ---")
    last_ts = historical_df.index.max()
    future_df = create_future_dataframe(last_ts, days=days_to_predict)

    # --- Feature Engineering for the Future ---
    # To create lag and rolling features, we need to combine historical data
    # with the future dataframe. The longest lag is 1 week (168 hours).
    required_history_len = 24 * 7

    # 1. Time Features
    future_with_features = create_time_features(future_df.copy())

    # 2. Weather Features (using last year's data as a forecast proxy)
    print("Creating future weather features (using data from the previous year)...")
    forecast_weather_dates = future_with_features.index - pd.DateOffset(years=1)
    weather_forecast = weather_df.reindex(forecast_weather_dates, method='nearest').set_index(future_with_features.index)

    future_with_features['temperature_c'] = weather_forecast['temperature_c']
    future_with_features['humidity_percent'] = weather_forecast['humidity_percent']
    future_with_features = create_weather_features(future_with_features)

    # 3. Lag and Rolling Features
    # The target column 'demand_mw' is needed for the functions to run.
    # We will fill the future 'demand_mw' with NaNs, as we don't know it yet.
    df_for_features = pd.concat([historical_df.tail(required_history_len), future_with_features])

    df_for_features = create_lag_features(df_for_features, lags=[24, 24 * 7])
    df_for_features = create_rolling_features(df_for_features, window_size=24)

    # Select only the future rows which now have all the engineered features
    future_df_final = df_for_features.loc[future_with_features.index]

    # --- Prediction ---
    # Ensure the feature order is the same as in the training script
    features = [col for col in model.feature_names_in_]
    print(f"\nPredicting with {len(features)} features...")

    predictions = model.predict(future_df_final[features])

    # Create the final forecast DataFrame
    forecast_df = pd.DataFrame({
        'timestamp': future_df_final.index,
        'demand_forecast_mw': predictions
    }).set_index('timestamp')

    print("Prediction complete.")
    return forecast_df


def main():
    """Main function to load model and generate predictions."""
    model_path = Path("models/xgboost_model.pkl")
    historical_data_path = Path("featured_electricity_data.csv")
    original_data_path = Path("electricity_and_weather_data.csv")
    output_path = Path("forecast.csv")

    model = load_model(model_path)
    historical_df = pd.read_csv(historical_data_path, index_col='timestamp', parse_dates=True)
    weather_df = pd.read_csv(original_data_path, index_col='timestamp', parse_dates=True)

    forecast = predict_future(model, historical_df, weather_df, days_to_predict=14)

    print(f"\nSaving 14-day forecast to {output_path}...")
    forecast.to_csv(output_path)
    print("Forecast saved successfully.")
    print("\nForecast head:")
    print(forecast.head())


if __name__ == "__main__":
    main()