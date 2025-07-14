import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import seaborn as sns 
def visualize_forecast_separate(forecast_csv_path: str | Path, historical_csv_path: str | Path, output_path: str | Path | None = None):
    """
    Reads electricity demand data, splits it into historical and forecast periods,
    and creates two separate plots: one for the overall view and one zoomed-in view.

    The last 7 days of data are considered the forecast period.

    Args:
        forecast_csv_path (str | Path): The path to the forecast CSV file.
        historical_csv_path (str | Path): The path to the original historical data CSV.
        output_path (str | Path | None): The path to save the plot images. If None, plots are displayed.
    """
    try:
        # Load the data from the CSV file
        df = pd.read_csv(forecast_csv_path)
        historical_df = pd.read_csv(historical_csv_path)
    except FileNotFoundError:
        print(f"Error: A data file was not found.")
        return

    # --- 1. Data Preparation ---
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Dynamically find the split date by finding the last date in the historical data
    historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
    split_date = historical_df['timestamp'].max()
    print(f"Historical data ends on: {split_date}")
    print(f"Forecast data starts on: {split_date + pd.Timedelta(hours=1)}")

    historical_data = df.loc[df.index <= split_date]
    forecast_data = df.loc[df.index > split_date]

    plt.style.use('seaborn-v0_8-whitegrid')

    # --- 2. Create the Main Plot (Overall View) ---
    fig_main, ax_main = plt.subplots(figsize=(16, 8))
    ax_main.plot(historical_data.index, historical_data['demand_forecast_mw'], color='royalblue', label='Historical Data', linewidth=2)
    ax_main.plot(forecast_data.index, forecast_data['demand_forecast_mw'], color='darkorange', label='Forecast Data', linewidth=2)
    ax_main.axvline(x=split_date, color='black', linestyle='--', label='Forecast Start')
    ax_main.set_title('Electricity Demand: Historical vs. 30-Day Forecast', fontsize=18, fontweight='bold')
    ax_main.set_xlabel('Date', fontsize=14)
    ax_main.set_ylabel('Demand Forecast (MW)', fontsize=14)
    ax_main.legend(fontsize=12)
    ax_main.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
    ax_main.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    fig_main.autofmt_xdate()
    plt.tight_layout()

    # --- 3. Create the Zoomed-In Plot ---
    fig_zoom, ax_zoom = plt.subplots(figsize=(12, 6))
    zoom_start = split_date - pd.Timedelta(days=14) # Show last 14 days of history
    zoom_end = forecast_data.index.max() # Show the full forecast
    zoom_hist_data = historical_data.loc[historical_data.index >= zoom_start]
    zoom_forecast_data = forecast_data.loc[forecast_data.index <= zoom_end]
    ax_zoom.plot(zoom_hist_data.index, zoom_hist_data['demand_forecast_mw'], color='royalblue', label='Historical Data')
    ax_zoom.plot(zoom_forecast_data.index, zoom_forecast_data['demand_forecast_mw'], color='darkorange', label='Forecast Data')
    ax_zoom.axvline(x=split_date, color='black', linestyle='--', label='Forecast Start')
    ax_zoom.set_title('Zoomed-In View: Forecast Start', fontsize=16, fontweight='bold')
    ax_zoom.set_xlabel('Date and Time', fontsize=12)
    ax_zoom.set_ylabel('Demand Forecast (MW)', fontsize=12)
    ax_zoom.legend()
    ax_zoom.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%b-%d'))
    ax_zoom.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    fig_zoom.autofmt_xdate()
    plt.tight_layout()

    # --- 4. Save or Display Plots ---
    if output_path:
        p = Path(output_path)
        zoom_output_path = p.with_stem(f"{p.stem}_zoom")
        fig_main.savefig(output_path, dpi=300, bbox_inches='tight')
        fig_zoom.savefig(zoom_output_path, dpi=300, bbox_inches='tight')
        print(f"Main plot successfully saved to '{output_path}'")
        print(f"Zoomed plot successfully saved to '{zoom_output_path}'")
    else:
        plt.show()

    plt.close(fig_main)
    plt.close(fig_zoom)


def analyze_forecast_patterns(csv_path: str | Path = 'forecast.csv', output_path: str | Path | None = None):
    """
    Analyzes the forecast data to show average demand patterns by hour and day of the week.

    This helps validate if the model has learned key business cycles (e.g., daily peaks,
    weekday vs. weekend differences).

    Args:
        csv_path (str | Path): The path to the forecast CSV file.
        output_path (str | Path | None): The path to save the plot image. If None, the plot is displayed.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return

    # --- 1. Data Preparation & Feature Engineering ---
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # We only want to analyze the patterns within the *actual* forecast period.
    # The forecast data starts after the last historical data point.
    # Note: This assumes the forecast CSV contains historical data as well.
    # A more robust way would be to get the split_date from the historical file,
    # but for this analysis, a 7-day forecast window is reasonable.
    split_date = df.index.max() - pd.Timedelta(days=7)
    forecast_data = df.loc[df.index > split_date].copy()

    # --- 2. Feature Engineering & Aggregation ---
    # Extract time features from the index
    forecast_data['hour'] = forecast_data.index.hour
    forecast_data['day_of_week'] = forecast_data.index.day_name()

    # Calculate the average demand for each hour and day
    hourly_avg = forecast_data.groupby('hour')['demand_forecast_mw'].mean()
    daily_avg = forecast_data.groupby('day_of_week')['demand_forecast_mw'].mean()

    # Ensure days are plotted in the correct order (Mon -> Sun)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg = daily_avg.reindex(day_order)

    # --- 3. Visualization ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Analysis of Forecasted Demand Patterns', fontsize=20, fontweight='bold')

    # Subplot 1: Average Demand by Hour of Day 🕒
    axes[0].plot(hourly_avg.index, hourly_avg.values, marker='o', linestyle='-', color='dodgerblue')
    axes[0].set_title('Average Demand by Hour of Day', fontsize=14)
    axes[0].set_xlabel('Hour of Day (0-23)', fontsize=12)
    axes[0].set_ylabel('Average Demand (MW)', fontsize=12)
    axes[0].set_xticks(range(0, 24, 2))  # Show ticks every 2 hours for clarity
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Subplot 2: Average Demand by Day of Week 📅
    sns.barplot(x=daily_avg.index, y=daily_avg.values, ax=axes[1], palette='viridis', hue=daily_avg.index, legend=False)
    axes[1].set_title('Average Demand by Day of Week', fontsize=14)
    axes[1].set_xlabel('Day of Week', fontsize=12)
    axes[1].set_ylabel('') # Y-axis label is shared with the left plot
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")

    # --- 4. Save or Display Plot ---
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for the suptitle

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Pattern analysis plot saved to '{output_path}'")
    else:
        plt.show()

    plt.close(fig)


def main():
    """Main function to generate and save all forecast visualizations."""
    # Define the output directory for plots and ensure it exists
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Define input and output file paths
    forecast_csv_path = Path("forecast.csv")
    historical_csv_path = Path("electricity_and_weather_data.csv")
    vis_output_path = plots_dir / "forecast_visualization.png"
    patterns_output = plots_dir / "forecast_patterns.png"

    # Generate the main forecast visualization plots (overall and zoomed)
    print("\n--- Generating Forecast Visualization Plots ---")
    visualize_forecast_separate(forecast_csv_path=forecast_csv_path, historical_csv_path=historical_csv_path, output_path=vis_output_path)

    # Generate the pattern analysis plot
    print("\n--- Generating Forecast Pattern Analysis Plot ---")
    analyze_forecast_patterns(csv_path=forecast_csv_path, output_path=patterns_output)


if __name__ == '__main__':
    main()