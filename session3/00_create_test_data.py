#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Data Creation
Create sample experiment data for testing the analyzers

@author: Hrishikesh Terdalkar
"""

###############################################################################

import os

import pandas as pd
import numpy as np

RNG_SEED = 42
np.random.seed(RNG_SEED)

###############################################################################


def create_engineering_test_data():
    """Create realistic engineering test data"""
    # Time points (0 to 120 minutes, every 5 minutes)
    time_points = np.arange(0, 121, 5)

    # Simulate different engineering parameters with realistic patterns
    temperature = (
        25
        + 8 * np.sin(2 * np.pi * time_points / 60)
        + np.random.normal(0, 1, len(time_points))
    )
    pressure = (
        101.3
        + 2 * np.cos(2 * np.pi * time_points / 40)
        + np.random.normal(0, 0.2, len(time_points))
    )
    flow_rate = (
        5
        + 2 * np.sin(2 * np.pi * time_points / 30)
        + np.random.normal(0, 0.1, len(time_points))
    )
    vibration = (
        0.1
        + 0.05 * np.sin(2 * np.pi * time_points / 20)
        + np.random.normal(0, 0.01, len(time_points))
    )

    df = pd.DataFrame(
        {
            "Time_min": time_points,
            "Temperature_C": temperature,
            "Pressure_kPa": pressure,
            "Flow_Rate_Lmin": flow_rate,
            "Vibration_mm": vibration,
        }
    )

    return df


def create_multiple_experiments():
    """Create multiple experiment files for batch processing"""
    experiments_dir = "experiments"
    os.makedirs(experiments_dir, exist_ok=True)

    experiment_types = [
        ("thermal_study", "Temperature", "C", 20, 35),
        ("pressure_test", "Pressure", "kPa", 100, 110),
        ("flow_analysis", "FlowRate", "L/min", 1, 10),
        ("vibration_test", "Vibration", "mm", 0.05, 0.15),
    ]

    for exp_type, param, unit, min_val, max_val in experiment_types:
        for i in range(1, 4):  # Create 3 of each type
            exp_name = f"{exp_type}_{i:02d}"
            time_points = np.arange(0, 61, 2)  # 1 hour of data

            # Different patterns for different experiments
            if exp_type == "thermal_study":
                pattern = np.sin(2 * np.pi * time_points / 30)
            elif exp_type == "pressure_test":
                pattern = np.cos(2 * np.pi * time_points / 20)
            elif exp_type == "flow_analysis":
                pattern = np.sin(2 * np.pi * time_points / 15)
            else:  # vibration_test
                pattern = np.sin(2 * np.pi * time_points / 10)

            measurements = min_val + (max_val - min_val) * pattern
            measurements += np.random.normal(
                0, (max_val - min_val) * 0.05, len(time_points)
            )

            df = pd.DataFrame(
                {
                    "Time_min": time_points,
                    f"{param}_{unit}": measurements,
                    "Experiment_ID": exp_name,
                }
            )

            filename = os.path.join(experiments_dir, f"{exp_name}.csv")
            df.to_csv(filename, index=False)
            print(f"Created: {filename}")


def main():
    """Main function to create test data"""
    print("Creating test data for Session 3 examples...")

    # Create single comprehensive test file
    comprehensive_data = create_engineering_test_data()
    comprehensive_data.to_csv("engineering_test_data.csv", index=False)
    print("Created: engineering_test_data.csv")

    # Create multiple experiment files for batch processing
    create_multiple_experiments()
    print("Created multiple experiment files in 'experiments/' directory")

    print("\nTest data creation complete!")
    print("\nUsage examples:")
    print(
        "  python basic_analyzer.py engineering_test_data.csv --stats --verbose"
    )
    print(
        "  python advanced_analyzer.py -i engineering_test_data.csv --stats --trends --plot"
    )
    print(
        "  python batch_processor.py --input-dir experiments/ --output-dir batch_results --summary"
    )


if __name__ == "__main__":
    main()
