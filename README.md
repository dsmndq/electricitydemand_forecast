# Electricity Demand Forecasting

This project implements a complete pipeline to forecast electricity demand. It uses historical electricity demand and weather data to train machine learning models (SARIMA and XGBoost) and generate future demand predictions. The pipeline includes steps for data analysis, feature engineering, model training, prediction, and visualization.

## Features

- **End-to-End Pipeline**: A single script (`run_pipeline.py`) orchestrates the entire workflow from data analysis to final visualization.
- **Dual-Model Approach**: Implements both a classical statistical model (SARIMA) for daily trends and a powerful gradient boosting model (XGBoost) for hourly, feature-rich forecasting.
- **Comprehensive Feature Engineering**: Creates a wide range of features including time-based (hour, day of week, etc.), lag, rolling window, and weather-based interaction features.
- **Automated Visualization**: Generates insightful plots comparing the forecast against historical data, a detailed hourly view, and analysis of predicted seasonal patterns.
- **Dummy Data Generation**: If no input data is found, the pipeline automatically generates a realistic dummy dataset to allow for a complete run.

## Pipeline Workflow

The pipeline is executed by `run_pipeline.py`, which runs the following scripts in order:

1.  **`src/analyze_electricity_demand.py`**:
    - Creates a dummy `electricity_and_weather_data.csv` if it doesn't exist.
    - Loads the data and performs an initial time-series analysis.
    - Generates plots for overall demand and time-series decomposition (trend, seasonality, residuals).

2.  **`src/feature_engineering.py`**:
    - Loads the data from the previous step.
    - Creates various features: time-based, lag, rolling statistics, and weather interactions.
    - Saves the augmented data to `featured_electricity_data.csv`.

3.  **`src/train_models.py`**:
    - Loads the featured data.
    - Splits the data into training and testing sets.
    - Trains a SARIMA model on daily aggregated data.
    - Trains an XGBoost model on the hourly featured data.
    - Evaluates both models and saves them to the `models/` directory.

4.  **`src/predict.py`**:
    - Loads the trained XGBoost model.
    - Creates a future dataframe for the next 14 days.
    - Engineers features for the future dates, using historical data for lags/rolling features and last year's weather as a proxy for the forecast.
    - Generates demand predictions and saves them to `forecast.csv`.

5.  **`src/visualize_forecast.py`**:
    - Loads the historical data and the new forecast.
    - Creates and saves three key visualizations in the `plots/` directory:
        - Forecast vs. Historical Data.
        - Detailed 72-Hour Forecast Zoom-In.
        - Analysis of Predicted Seasonal Patterns (hourly and daily profiles).

## How to Run

### Prerequisites

- Python 3.x
- Required Python packages. You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn statsmodels xgboost scikit-learn
```

It is recommended to create a `requirements.txt` file with the above packages for easier setup.

### Execution

To run the entire pipeline, execute the main runner script from the project's root directory:

```bash
python run_pipeline.py
```

The script will handle all steps, from data generation (if needed) to creating the final plots.

## Project Structure

```
.
├── models/
│   ├── sarima_model.pkl
│   └── xgboost_model.pkl
├── plots/
│   ├── 01_forecast_vs_historical.png
│   ├── 02_detailed_zoom.png
│   ├── 03_seasonal_analysis.png
│   ├── daily_average_demand.png
│   └── demand_decomposition.png
├── src/
│   ├── analyze_electricity_demand.py
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── train_models.py
│   └── visualize_forecast.py
├── electricity_and_weather_data.csv  (Generated or user-provided)
├── featured_electricity_data.csv     (Generated)
├── forecast.csv                      (Generated)
└── run_pipeline.py
```

## Output

- **Models**: Trained models are saved in the `models/` directory.
- **Data**: The final forecast is saved as `forecast.csv`. Intermediate data files are also created in the root directory.
- **Visualizations**: All generated plots are saved in the `plots/` directory.