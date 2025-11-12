#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Experiment Processor
Process multiple experiment files automatically

@author: Hrishikesh Terdalkar
"""

###############################################################################

import os
import sys
import json
import glob
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

RNG_SEED = 42
np.random.seed(RNG_SEED)

###############################################################################


class BatchExperimentProcessor:
    """Batch processing for multiple experiments"""

    def __init__(self):
        self.parser = self.setup_parser()

    def setup_parser(self):
        """Setup batch processing parser"""
        parser = argparse.ArgumentParser(
            description="Batch Process Multiple Experiments",
            epilog=(
                "Example: python session3/02_batch_processor.py "
                "--input-dir experiments/ --output-dir results/"
            ),
        )

        parser.add_argument(
            "--input-dir",
            "-i",
            required=True,
            help="Directory containing experiment files",
        )
        parser.add_argument(
            "--output-dir",
            "-o",
            required=True,
            help="Output directory for results",
        )
        parser.add_argument(
            "--file-pattern",
            default="*.csv",
            help="File pattern to match (default: *.csv)",
        )
        parser.add_argument(
            "--summary", action="store_true", help="Generate summary report"
        )
        parser.add_argument(
            "--verbose", "-v", action="store_true", help="Verbose output"
        )

        return parser

    def find_experiment_files(self, input_dir, pattern):
        """Find all experiment files matching pattern"""
        search_pattern = os.path.join(input_dir, pattern)
        files = glob.glob(search_pattern)

        if not files:
            print(f"No files found matching {search_pattern}")
            return []

        if self.args.verbose:
            print(f"Found {len(files)} experiment files")
            for file in files:
                print(f"  {os.path.basename(file)}")

        return files

    def process_single_experiment(self, file_path, output_dir):
        """Process a single experiment file"""
        try:
            exp_name = os.path.splitext(os.path.basename(file_path))[0]
            exp_output_dir = os.path.join(output_dir, exp_name)
            os.makedirs(exp_output_dir, exist_ok=True)

            # Load data
            df = pd.read_csv(file_path)

            # Basic analysis
            results = {
                "experiment": exp_name,
                "file": file_path,
                "timestamp": datetime.now().isoformat(),
                "data_shape": df.shape,
                "columns": list(df.columns),
            }

            # Calculate metrics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            metrics = {}

            for col in numeric_cols:
                metrics[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                }

            results["metrics"] = metrics

            # Save results
            output_file = os.path.join(exp_output_dir, "analysis.json")
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

            if self.args.verbose:
                print(f"[OK] Processed {exp_name}")

            return results

        except Exception as e:
            print(f"[ERROR] Processing {file_path}: {e}")
            return {"error": str(e), "file": file_path}

    def generate_batch_summary(self, all_results, output_dir):
        """Generate summary report for batch processing"""
        successful = [r for r in all_results if "error" not in r]
        failed = [r for r in all_results if "error" in r]

        summary = {
            "batch_date": datetime.now().isoformat(),
            "total_files": len(all_results),
            "successful": len(successful),
            "failed": len(failed),
            "experiments": [],
        }

        for result in successful:
            summary["experiments"].append(
                {
                    "name": result["experiment"],
                    "data_points": result["data_shape"][0],
                    "variables": result["data_shape"][1],
                }
            )

        # Save summary
        summary_file = os.path.join(output_dir, "batch_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print("\n" + "=" * 50)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total files: {summary['total_files']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")

        if successful:
            print("\nSuccessful experiments:")
            for exp in summary["experiments"][:5]:  # Show first 5
                print(f"  {exp['name']}: {exp['data_points']} points")

        if failed:
            print("\nFailed experiments:")
            for failure in failed[:3]:  # Show first 3 failures
                print(f"  {failure['file']}: {failure['error']}")

    def run_batch_processing(self, args):
        """Run batch processing"""
        self.args = args

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Find experiment files
        experiment_files = self.find_experiment_files(
            args.input_dir, args.file_pattern
        )

        if not experiment_files:
            return False

        # Process each experiment
        all_results = []

        for file_path in experiment_files:
            result = self.process_single_experiment(file_path, args.output_dir)
            all_results.append(result)

        # Generate summary if requested
        if args.summary:
            self.generate_batch_summary(all_results, args.output_dir)

        print(
            f"\n[OK] Batch processing complete! Results in {args.output_dir}"
        )
        return True


def create_test_experiments():
    """Create test experiment files for demonstration"""
    test_dir = "test_experiments"
    os.makedirs(test_dir, exist_ok=True)

    experiments = [
        ("thermal_001", "Temperature", "C", 20, 35),
        ("pressure_002", "Pressure", "kPa", 100, 110),
        ("flow_003", "FlowRate", "L/min", 1, 10),
    ]

    for exp_name, param, unit, min_val, max_val in experiments:
        time_points = np.arange(0, 60, 2)
        measurements = min_val + (max_val - min_val) * np.sin(
            2 * np.pi * time_points / 30
        )
        measurements += np.random.normal(
            0, (max_val - min_val) * 0.05, len(time_points)
        )

        df = pd.DataFrame(
            {"Time_min": time_points, f"{param}_{unit}": measurements}
        )

        filename = os.path.join(test_dir, f"{exp_name}.csv")
        df.to_csv(filename, index=False)
        print(f"Created: {filename}")

    return test_dir


def main():
    """Main function"""
    # Create test data if running directly
    if not os.path.exists("test_experiments"):
        print("Creating test experiment files...")
        create_test_experiments()

    processor = BatchExperimentProcessor()
    args = processor.parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found")
        sys.exit(1)

    success = processor.run_batch_processing(args)

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
