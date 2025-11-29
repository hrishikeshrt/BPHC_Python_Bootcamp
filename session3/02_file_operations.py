#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Operations Module
Basic file handling and data organization functions

@author: Hrishikesh Terdalkar
"""

###############################################################################

import os
import csv
import json
from datetime import datetime

import pandas as pd

###############################################################################


def create_sample_experiment_data():
    """Create sample experimental data for demonstration"""
    # Temperature and pressure readings over time
    time_points = [0, 5, 10, 15, 20, 25, 30]  # minutes
    temperatures = [25.0, 26.5, 28.2, 27.8, 26.9, 25.5, 24.8]  # deg C
    pressures = [101.3, 101.5, 101.8, 101.6, 101.4, 101.2, 101.1]  # kPa

    return time_points, temperatures, pressures


def save_data_to_csv(filename, times, temps, pressures):
    """Save experimental data to CSV file manually"""
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["Time_min", "Temperature_C", "Pressure_kPa"])
        # Write data rows
        for i in range(len(times)):
            writer.writerow([times[i], temps[i], pressures[i]])
    print(f"Data saved to {filename}")


def read_data_from_csv(filename):
    """Read experimental data from CSV file manually"""
    times, temps, pressures = [], [], []

    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            times.append(float(row[0]))
            temps.append(float(row[1]))
            pressures.append(float(row[2]))

    return times, temps, pressures


def use_pandas_for_data_handling(times, temps, pressures):
    """Demonstrate pandas for easier data handling"""
    # Create DataFrame
    df = pd.DataFrame(
        {"Time_min": times, "Temperature_C": temps, "Pressure_kPa": pressures}
    )

    print("DataFrame created:")
    print(df)

    # Basic statistics
    print("\nBasic statistics:")
    print(df.describe())

    return df


def handle_json_configuration():
    """Work with JSON files for experiment configuration"""
    experiment_info = {
        "experiment_id": "thermal_study_001",
        "researcher": "PhD Student",
        "date": datetime.now().isoformat(),
        "equipment": {
            "sensor": "Thermocouple Type K",
            "data_logger": "Arduino Uno",
            "sampling_rate": "1 sample/minute",
        },
        "conditions": {"ambient_temperature": 25.0, "humidity": 45.0},
    }

    # Save to JSON file
    with open("experiment_config.json", "w") as jsonfile:
        json.dump(experiment_info, jsonfile, indent=2)
    print("Experiment configuration saved to JSON")

    # Read from JSON file
    with open("experiment_config.json", "r") as jsonfile:
        loaded_config = json.load(jsonfile)

    print("\nExperiment configuration loaded:")
    for key, value in loaded_config.items():
        print(f"{key}: {value}")


def demonstrate_file_operations():
    """Demonstrate all file operations"""
    print("=== FILE OPERATIONS DEMONSTRATION ===")

    # Create sample data
    times, temps, pressures = create_sample_experiment_data()

    # Save and read CSV manually
    save_data_to_csv("manual_data.csv", times, temps, pressures)
    times_read, temps_read, pressures_read = read_data_from_csv(
        "manual_data.csv"
    )
    print(f"Data read back: {len(times_read)} measurements")

    # Use pandas
    df = use_pandas_for_data_handling(times, temps, pressures)
    df.to_csv("pandas_data.csv", index=False)
    df_read = pd.read_csv("pandas_data.csv")
    print(f"Pandas data read back: {df_read.shape}")

    # JSON configuration
    handle_json_configuration()


if __name__ == "__main__":
    demonstrate_file_operations()
