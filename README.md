# Electricity Demand Forecasting Pipeline

This project demonstrates a complete, end-to-end pipeline for forecasting electricity demand. It covers dummy data generation, exploratory data analysis, feature engineering, model training (SARIMA and XGBoost), prediction, and visualization.

The entire workflow is automated and can be executed with a single command.

## Features

*   **Modular Scripts:** Each step of the pipeline (analysis, feature engineering, training, etc.) is a separate, well-documented Python script, promoting clarity and maintainability.
*   **Dummy Data Generation:** If no data is present, a realistic two-year hourly dataset is created automatically, allowing the pipeline to run out-of-the-box.
*   **Time-Series Analysis:** Includes time-series decomposition to identify trend, seasonality, and residuals in the demand data.
*   **Advanced Feature Engineering:** Creates a rich feature set including:
    *   Time-based features (hour, day of week, month, etc.).
    *   Lag features (demand from 1 day and 1 week ago).
    *   Rolling window statistics (24-hour mean and standard deviation).
    *   Weather-based interaction features.
*   **Dual Model Approach:** Trains and evaluates both a classical statistical model (SARIMA) for a baseline and a modern machine learning model (XGBoost) for the final forecast.
*   **Automated Pipeline:** A single Python script (`run_pipeline.py`) orchestrates the entire workflow from data generation to final forecast visualization.
*   **Clear Visualizations:** Generates plots for the initial data analysis and the final forecast results.

## Project Structure

```
.
├── models/                # Stores trained model artifacts (.pkl)
├── plots/                 # Stores analysis plots
├── src/                   # Source code for the pipeline steps
│   ├── analyze_electricity_demand.py
│   ├── feature_engineering.py
│   ├── predict.py
│   └── train_models.py
├── .gitignore
├── electricity_and_weather_data.csv  # Generated raw data
├── featured_electricity_data.csv     # Data with engineered features
├── forecast.csv                      # Final 30-day forecast data
├── forecast_overview.png             # Visualization of the full forecast
├── forecast_zoom.png                 # Zoomed-in visualization of the forecast
├── README.md                         # This file
├── requirements.txt                  # Project dependencies
└── run_pipeline.py                   # Main pipeline execution script
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd electricitydemand_forecast
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Linux/macOS
    python3 -m venv .venv
    source .venv/bin/activate

    # For Windows
    python -m venv .venv
    .venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the entire pipeline, simply execute the `run_pipeline.py` script from the root directory of the project:

```bash
python run_pipeline.py
```

This command will:
1.  Check for existing data. If `electricity_and_weather_data.csv` is not found, it will be generated.
2.  Execute each script in the pipeline in the correct order.
3.  Print progress messages to the console for each step.
4.  Save all outputs (data files, models, and plots) to the appropriate directories.

## Pipeline Steps Explained

The `run_pipeline.py` script executes the following modules in sequence:

1.  **`src/analyze_electricity_demand.py`**:
    *   Generates `electricity_and_weather_data.csv` if it doesn't exist.
    *   Loads the data and performs a time-series decomposition to analyze trend and seasonality.
    *   Saves analysis plots (`daily_average_demand.png`, `demand_decomposition.png`) to the `plots/` directory.

2.  **`src/feature_engineering.py`**:
    *   Loads the raw data from `electricity_and_weather_data.csv`.
    *   Engineers a comprehensive set of features for the models.
    *   Saves the augmented dataset to `featured_electricity_data.csv`.

3.  **`src/train_models.py`**:
    *   Loads the featured data.
    *   Splits the data into training and testing sets.
    *   Trains a SARIMA model (on daily data) and an XGBoost model (on hourly data).
    *   Evaluates both models and prints their performance metrics (MAE and RMSE).
    *   Saves the trained models (`sarima_model.pkl`, `xgboost_model.pkl`) to the `models/` directory.

4.  **`src/predict.py`**:
    *   Loads the trained XGBoost model.
    *   Creates a future dataframe for the next 30 days.
    *   Engineers the necessary features for the future data, using historical data as a basis for lags and weather proxies.
    *   Generates a 30-day hourly forecast and saves it to `forecast.csv`.

5.  **`visualize_forecast.py`**:
    *   Loads the historical data and the new forecast from `forecast.csv`.
    *   Creates and saves two plots to the root directory:
        *   `forecast_overview.png`: Shows the full history and the 30-day forecast.
        *   `forecast_zoom.png`: Shows the last few months of history and the forecast for a more detailed view.

## Outputs

After a successful run, the following files will be generated:

*   **Data:** `electricity_and_weather_data.csv`, `featured_electricity_data.csv`, `forecast.csv`
*   **Models:** `models/sarima_model.pkl`, `models/xgboost_model.pkl`
*   **Plots:** `plots/daily_average_demand.png`, `plots/demand_decomposition.png`, `forecast_overview.png`, `forecast_zoom.png`

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

*This project was created to demonstrate a robust MLOps pipeline for time-series forecasting.*