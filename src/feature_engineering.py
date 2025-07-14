import pandas as pd
import numpy as np
from pathlib import Path


def load_data(path: Path) -> pd.DataFrame:
    """
    Loads the combined electricity and weather data CSV, parses the datetime column,
    and sets it as the index.

    Args:
        path (Path): The file path to the data CSV.

    Returns:
        pd.DataFrame: DataFrame with a DatetimeIndex.
    """
    print(f"Loading data from {path}...")
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found at {path}. "
            "Please run 'analyze_electricity_demand.py' first to generate it."
        )
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    print("Data loaded successfully.")
    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time-based features from the datetime index.

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex.

    Returns:
        pd.DataFrame: DataFrame with new time features.
    """
    print("Creating time features...")
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek  # Monday=0, Sunday=6
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    print("Time features created: hour, dayofweek, quarter, month, year, dayofyear.")
    return df


def create_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """
    Creates lag features for the 'demand_mw' column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        lags (list[int]): A list of lag periods to create (in hours).

    Returns:
        pd.DataFrame: DataFrame with new lag features.
    """
    print(f"Creating lag features for lags: {lags} hours...")
    for lag in lags:
        df[f'demand_lag_{lag}h'] = df['demand_mw'].shift(lag)
    print("Lag features created.")
    return df


def create_rolling_features(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Creates rolling window features for the 'demand_mw' column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        window_size (int): The size of the rolling window in hours.

    Returns:
        pd.DataFrame: DataFrame with new rolling features.
    """
    print(f"Creating rolling features with window size: {window_size} hours...")
    df[f'demand_rolling_mean_{window_size}h'] = df['demand_mw'].rolling(window=window_size).mean()
    df[f'demand_rolling_std_{window_size}h'] = df['demand_mw'].rolling(window=window_size).std()
    print("Rolling features (mean, std) created.")
    return df


def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features based on weather data (temperature and humidity).

    Args:
        df (pd.DataFrame): DataFrame with 'temperature_c' and 'humidity_percent' columns.

    Returns:
        pd.DataFrame: DataFrame with new weather-based features.
    """
    print("Creating weather features...")
    df['temp_squared'] = df['temperature_c'] ** 2
    df['temp_x_humidity'] = df['temperature_c'] * df['humidity_percent']
    print("Weather features created: temp_squared, temp_x_humidity.")
    return df


def main():
    """
    Main function to run the feature engineering pipeline.
    """
    input_data_path = Path("electricity_and_weather_data.csv")
    output_data_path = Path("featured_electricity_data.csv")

    df = load_data(input_data_path)

    df = create_time_features(df)
    df = create_lag_features(df, lags=[24, 24 * 7])  # 1 day and 1 week ago
    df = create_rolling_features(df, window_size=24) # 24-hour rolling window
    df = create_weather_features(df)

    print(f"\nSaving featured data to {output_data_path}...")
    df.to_csv(output_data_path)
    print("Featured data saved successfully.")


if __name__ == "__main__":
    main()