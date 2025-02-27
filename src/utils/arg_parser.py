import argparse
import yaml
import os

# Automatically detect the root directory of the ML project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "parameters.yml")

def load_config():
    """Loads parameters from a YAML configuration file and converts relative paths to absolute."""
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    # Convert relative paths to absolute paths
    for key, value in config.items():
        if isinstance(value, str) and ("/" in value or "\\" in value):  # Check if it's a file/directory path
            if not os.path.isabs(value):  # Ensure it's relative to the project root
                config[key] = os.path.abspath(os.path.join(PROJECT_ROOT, value))

    return config

def get_input_args():
    """Parses command-line arguments with defaults loaded from a YAML file."""
    config = load_config()

    parser = argparse.ArgumentParser(description="Command-line arguments for CNN training and prediction")

    parser.add_argument('--alpha', type=float, default=config['alpha'], help='Alpha value for model')

    parser.add_argument('--fit_intercept', type=bool, default=config['fit_intercept'], help='Fit_intercept value for model')

    parser.add_argument('--solver', type=str, default=config['solver'], help='Solver value for model')

    parser.add_argument('--data_directory', type=str, default=config["data_directory"], help='Path to goauto data')

    args = parser.parse_args()

    return args