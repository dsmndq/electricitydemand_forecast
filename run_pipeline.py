import subprocess
import sys
import os

def run_script(script_path):
    """Executes a Python script and checks for errors."""
    # Ensure the script path is valid
    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}", file=sys.stderr)
        sys.exit(1)

    print(f"--- Running {script_path} ---")
    try:
        # Using sys.executable ensures we use the same python interpreter
        # that is running this script.
        process = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if process.stdout:
            print(process.stdout.strip())
        if process.stderr:
            # Some tools write progress to stderr, so it's not always an error
            print("--- Stderr ---", file=sys.stderr)
            print(process.stderr.strip(), file=sys.stderr)
        print(f"--- Finished {script_path} successfully ---\n")
    except subprocess.CalledProcessError as e:
        print(f"--- Error running {script_path} ---", file=sys.stderr)
        if e.stdout:
            print("--- Stdout ---")
            print(e.stdout.strip())
        if e.stderr:
            print("--- Stderr ---", file=sys.stderr)
            print(e.stderr.strip(), file=sys.stderr)
        print(f"--- Pipeline failed at {script_path} ---", file=sys.stderr)
        sys.exit(1)

def main():
    """Runs the entire electricity demand forecast pipeline."""
    print("--- Starting Electricity Demand Forecast Pipeline ---")

    # The order of scripts is based on run.sh
    scripts_to_run = [
        "src/analyze_electricity_demand.py",
        "src/feature_engineering.py",
        "src/train_models.py",
        "src/predict.py",
        "src/visualize_forecast.py"
    ]

    for script in scripts_to_run:
        run_script(script)

    print("--- Pipeline finished successfully! ---")
    print("Check the 'plots/' directory for your forecast visualizations.")

if __name__ == "__main__":
    main()