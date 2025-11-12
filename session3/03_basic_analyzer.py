#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic Research Data Analyzer
Simple command-line tool for data analysis

@author: Hrishikesh Terdalkar
"""

###############################################################################

import os
import sys
import json
import argparse
from datetime import datetime

import pandas as pd
import numpy as np

###############################################################################


def calculate_basic_stats(data):
    """Calculate basic statistics for a dataset"""
    stats = {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std_dev": float(np.std(data)),
        "min": float(min(data)),
        "max": float(max(data)),
        "range": float(max(data) - min(data)),
    }
    return stats


def setup_basic_parser():
    """Setup basic argument parser"""
    parser = argparse.ArgumentParser(
        description="Basic Research Data Analyzer",
        epilog=(
            "Example: python session3/03_basic_analyzer.py "
            "engineering_test_data.csv --stats --output results"
        ),
    )

    parser.add_argument(
        "input", help="Input CSV file containing experimental data"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="analysis_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Calculate basic statistics"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    return parser


def analyze_data(args):
    """Main analysis function"""
    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    try:
        # Load data
        df = pd.read_csv(args.input)

        if args.verbose:
            print(f"Loaded data from {args.input}")
            print(f"Data shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")

        results = {
            "analysis_date": datetime.now().isoformat(),
            "input_file": args.input,
            "data_shape": df.shape,
            "columns": list(df.columns),
        }

        # Calculate statistics if requested
        if args.stats:
            stats_results = {}
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                stats_results[col] = calculate_basic_stats(df[col])

            results["statistics"] = stats_results

            if args.verbose:
                print("Calculated statistics for numeric columns")

        # Save results
        output_file = os.path.join(args.output, "analysis.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        if args.verbose:
            print(f"Results saved to {output_file}")

        # Print summary
        print_summary(results)

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def print_summary(results):
    """Print analysis summary"""
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Input file: {results['input_file']}")
    print(f"Data shape: {results['data_shape']}")

    if "statistics" in results:
        print("\nStatistical Summary:")
        for col, stats in results["statistics"].items():
            print(f"  {col}:")
            print(f"    Mean: {stats['mean']:.2f}")
            print(f"    Std: {stats['std_dev']:.2f}")
            print(f"    Range: {stats['min']:.2f} to {stats['max']:.2f}")


def main():
    """Main function"""
    parser = setup_basic_parser()
    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)

    # Run analysis
    success = analyze_data(args)

    if success:
        print(f"\n[OK] Analysis complete! Results in {args.output}/")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
