import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
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
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    print("Data loaded successfully.")
    return df


def create_dummy_data(output_path: Path):
    """
    Creates a single dummy CSV file with electricity and weather data.
    """
    print("Creating dummy data file...")
    # Create a date range for 2 years of hourly data
    dates = pd.date_range(start='2021-01-01', end='2022-12-31', freq='h')
    n = len(dates)

    # --- Create Electricity Data ---
    # Trend: linearly increasing demand over time
    trend = np.linspace(start=100, stop=120, num=n)
    # Seasonality (Yearly): higher in summer and winter
    yearly_seasonality = 15 * np.sin(2 * np.pi * dates.dayofyear / 365.25 - np.pi / 2) + 15
    # Seasonality (Weekly): lower on weekends
    weekly_seasonality = -10 * np.sin(2 * np.pi * dates.dayofweek / 7)
    # Noise
    noise = np.random.normal(0, 5, n)
    # Combine components for demand
    demand = trend + yearly_seasonality + weekly_seasonality + noise

    # --- Create Weather Data ---
    # Temperature with a clear yearly pattern
    temp_seasonality = 15 * -np.cos(2 * np.pi * dates.dayofyear / 365.25) + 10
    temp_noise = np.random.normal(0, 2, n)
    temperature = temp_seasonality + temp_noise

    # --- Create Humidity Data (inversely related to temperature for simplicity) ---
    humidity_seasonality = -25 * -np.cos(2 * np.pi * dates.dayofyear / 365.25) + 60
    humidity_noise = np.random.normal(0, 5, n)
    humidity = np.clip(humidity_seasonality + humidity_noise, 20, 100)

    # --- Combine into a single DataFrame ---
    combined_df = pd.DataFrame({
        'timestamp': dates,
        'demand_mw': demand,
        'temperature_c': temperature,
        'humidity_percent': humidity
    })
    combined_df.to_csv(output_path, index=False)
    print(f"Dummy file '{output_path}' created.")


def main():
    """Main function to run the analysis pipeline."""
    # Define file paths
    data_path = Path("electricity_and_weather_data.csv")
    output_folder = Path("plots")
    output_folder.mkdir(exist_ok=True)

    # --- Step 1: Create data if it doesn't exist ---
    if not data_path.exists():
        create_dummy_data(data_path)

    # --- Step 2: Load and Merge Data ---
    df = load_data(data_path)

    # For this analysis, we will focus only on the electricity demand.
    # We will resample to daily averages to make long-term patterns clearer.
    demand_series = df['demand_mw'].resample('D').mean()

    # --- Step 3: Plot Electricity Demand Over Time ---
    print("Plotting overall electricity demand...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))
    demand_series.plot(ax=ax)
    ax.set_title('Daily Average Electricity Demand Over Time', fontsize=16)
    ax.set_xlabel('Date')
    ax.set_ylabel('Demand (MW)')
    ax.grid(True)
    plt.tight_layout()
    plot_path = output_folder / "daily_average_demand.png"
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")

    print(
        "Observations from the plot:\n"
        "- Trend: There appears to be a slight upward trend in demand over the two years.\n"
        "- Seasonality: Clear annual peaks are visible, likely corresponding to summer/winter.\n"
        "- Anomalies: No major anomalies are present in this smooth, generated data."
    )

    # --- Step 4: Time-Series Decomposition ---
    print("\nPerforming time-series decomposition...")
    # A multiplicative model is often suitable for demand data where seasonal
    # fluctuations grow with the trend. The period is 365 for annual seasonality.
    decomposition = sm.tsa.seasonal_decompose(
        demand_series.dropna(),
        model='multiplicative',
        period=365
    )

    # Plot the decomposed components
    fig = decomposition.plot()
    fig.set_size_inches(14, 10)
    fig.suptitle('Time-Series Decomposition of Electricity Demand', y=1.02, fontsize=16)
    plt.tight_layout()
    plot_path = output_folder / "demand_decomposition.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Decomposition plot saved to {plot_path}")

    print(
        "Decomposition plots explained:\n"
        "- Observed: The original daily average data.\n"
        "- Trend: The underlying long-term movement in the data, capturing the growth.\n"
        "- Seasonal: The repeating annual pattern of electricity demand.\n"
        "- Resid: The residual (or noise) component, which is what's left after removing the trend and seasonal components."
    )


if __name__ == "__main__":
    main()
