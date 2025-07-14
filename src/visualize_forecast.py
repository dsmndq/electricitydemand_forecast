import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

def plot_forecast_vs_historical(historical_df, forecast_df, output_folder):
    """
    Plots the forecast against a recent slice of historical data.
    This is "The Essential Plot".
    """
    print("Generating Plot 1: Forecast vs. Historical Data...")
    
    # Get the last 30 days of historical data before the forecast starts
    last_hist_month = historical_df.loc[historical_df.index < forecast_df.index.min()].last('30D')
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Plot historical and forecast data
    ax.plot(last_hist_month.index, last_hist_month['demand_mw'], color='royalblue', label='Historical Demand')
    ax.plot(forecast_df.index, forecast_df['demand_forecast_mw'], color='darkorange', label='Forecasted Demand')
    ax.axvline(forecast_df.index.min(), color='crimson', linestyle='--', label='Forecast Start')
    
    # Formatting
    formatter = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(formatter)
    locator = mdates.DayLocator(interval=5)
    ax.xaxis.set_major_locator(locator)
    fig.autofmt_xdate()
    
    ax.set_title('Electricity Demand: Historical vs. Forecast', fontsize=16, weight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Demand (MW)', fontsize=12)
    ax.legend()
    ax.autoscale(axis='x', tight=True)
    
    # Save the plot
    plot_path = output_folder / "01_forecast_vs_historical.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"-> Saved to {plot_path}")

def plot_detailed_zoom(forecast_df, output_folder):
    """
    Plots a detailed, zoomed-in view of the first few days of the forecast.
    This is "The Detailed View".
    """
    print("Generating Plot 2: Detailed Forecast Zoom-In...")
    
    # Get the first 3 days of the forecast
    detail_view = forecast_df.first('3D')
    
    fig, ax = plt.subplots(figsize=(15, 7))
    
    ax.plot(detail_view.index, detail_view['demand_forecast_mw'], color='darkorange', 
            marker='o', linestyle='-', label='Hourly Forecast')

    # Formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a, %H:%M'))
    ax.grid(which='major', linestyle='--', linewidth='0.5')
    
    ax.set_title('Detailed Forecast: First 72 Hours', fontsize=16, weight='bold')
    ax.set_xlabel('Date and Time', fontsize=12)
    ax.set_ylabel('Predicted Demand (MW)', fontsize=12)
    
    # Save the plot
    plot_path = output_folder / "02_detailed_zoom.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"-> Saved to {plot_path}")

def plot_seasonal_patterns(forecast_df, output_folder):
    """
    Analyzes and plots the predicted average hourly and daily patterns.
    This is "The Analytical View".
    """
    print("Generating Plot 3: Analysis of Seasonal Patterns...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Analysis of Predicted Seasonal Patterns', fontsize=18, weight='bold')
    
    # Plot 1: Average Hourly Demand Profile
    hourly_avg = forecast_df.groupby(forecast_df.index.hour)['demand_forecast_mw'].mean()
    ax1.plot(hourly_avg.index, hourly_avg.values, color='navy', marker='o')
    ax1.set_title('Predicted Average Hourly Profile', fontsize=14)
    ax1.set_xlabel('Hour of Day', fontsize=12)
    ax1.set_ylabel('Average Predicted Demand (MW)', fontsize=12)
    ax1.set_xticks(range(0, 24, 2))
    ax1.grid(True)

    # Plot 2: Average Daily Demand Profile
    daily_avg = forecast_df.groupby(forecast_df.index.dayofweek)['demand_forecast_mw'].mean()
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    sns.barplot(x=day_names, y=daily_avg.values, ax=ax2, palette='viridis', hue=day_names, legend=False)
    ax2.set_title('Predicted Average Daily Profile', fontsize=14)
    ax2.set_xlabel('Day of Week', fontsize=12)
    ax2.set_ylabel('') # Label is shared with left plot
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the plot
    plot_path = output_folder / "03_seasonal_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"-> Saved to {plot_path}")

def main():
    """
    Main function to orchestrate the creation of all forecast visualizations.
    """
    # Define paths
    forecast_path = Path("forecast.csv")
    historical_data_path = Path("electricity_and_weather_data.csv")
    output_folder = Path("plots")
    output_folder.mkdir(exist_ok=True)

    # Load data
    print(f"Loading forecast from {forecast_path}...")
    try:
        forecast_df = pd.read_csv(forecast_path, index_col='timestamp', parse_dates=True)
        historical_df = pd.read_csv(historical_data_path, index_col='timestamp', parse_dates=True)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure forecast.csv and electricity_and_weather_data.csv are present.")
        return
        
    print("Data loaded successfully.\n")

    # Generate all plots
    plot_forecast_vs_historical(historical_df, forecast_df, output_folder)
    plot_detailed_zoom(forecast_df, output_folder)
    plot_seasonal_patterns(forecast_df, output_folder)
    
    print("\nAll visualizations have been generated successfully.")

if __name__ == "__main__":
    main()
